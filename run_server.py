import os
import sys
import time
import json
import torch
import logging
import hashlib
import argparse
import threading
import multiprocessing
from datetime import datetime, timedelta, timezone
import uuid
import shutil

# Import FastAPI and related components for the web server
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from server import *
shared_state = SharedState()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log")
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

app = FastAPI(
    title="2048-Zero Trainer Server",
    description="API server for distributed 2048-Zero training",
    version="1.0.0"
)

# Define global storage adapter variable with proper type annotation
storage_adapter: StorageAdapter = None

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.get("/state")
async def get_state(token: str = Depends(get_token_from_auth)):
    """Get the current server state"""
    return shared_state.to_dict()

@app.get("/weights/{revision}")
async def get_weights(revision: int, token: str = Depends(get_token_from_auth)):
    """Serve weights file for a specific revision"""
    weights_dir = os.path.join("weights")
    weights_path = os.path.join(weights_dir, f"r{revision}.pt")
    
    if not os.path.exists(weights_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Weights for revision {revision} not found"
        )
    
    return FileResponse(
        weights_path,
        media_type="application/octet-stream",
        filename=f"r{revision}.pt"
    )

@app.post("/upload")
async def upload_games(request: Request, token: str = Depends(get_token_from_auth)):
    """Upload games to the server"""
    # Get revision from header
    revision_header = request.headers.get("X-Revision")
    if not revision_header:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Revision header"
        )
    
    try:
        batch_revision = int(revision_header)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid X-Revision header"
        )
    
    # Check revision matches
    server_state = shared_state.to_dict()
    if batch_revision != server_state["revision"]:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": f"Revision mismatch: expected {server_state['revision']}, got {batch_revision}"}
        )
    
    # Check if server is training
    if server_state["is_training"]:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "Server is in training mode, not accepting uploads"}
        )
    
    # Process request body
    try:
        body = await request.body()
        content_encoding = request.headers.get("Content-Encoding", "")
        expected_hash = request.headers.get("X-Content-SHA256", "")
        
        if content_encoding == "zstd":
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            uncompressed_body = dctx.decompress(body)
        else:
            uncompressed_body = body
        
        # Verify SHA-256 hash if provided
        if expected_hash:
            # Calculate SHA-256 hash of the uncompressed data
            sha256 = hashlib.sha256()
            sha256.update(uncompressed_body)
            actual_hash = sha256.hexdigest()
            
            # Compare hashes
            if actual_hash != expected_hash:
                # logger.warning(f"SHA-256 hash mismatch: expected {expected_hash}, got {actual_hash}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Data integrity error: SHA-256 hash mismatch"}
                )
            else:
                pass
                # logger.info("SHA-256 hash verified successfully")
        
        batch = json.loads(uncompressed_body)
        
        # Add games to queue
        success, message = shared_state.add_games(batch)
        if not success:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"detail": message}
            )
        
        return {"status": "accepted"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing request: {str(e)}"
        )

def process_and_train(batch_data, current_model, config):
    """Process game data and train the model using PyTorch Lightning"""
    from zero.game import GameRules
    from zero.lightning_trainer import LightningDistributedTrainer
    
    # Create game rules
    rules = GameRules()
    
    # Configure trainer with hyper-parameters from config
    trainer = LightningDistributedTrainer(
        model=current_model,
        rules=rules,
        device=config.get("device", None),  # Auto-detect if not specified
        learning_rate=config.get("learning_rate", 0.001),
        batch_size=config.get("batch_size", 64),
        weight_decay=config.get("weight_decay", 1e-4),
        epochs=config.get("epochs", 10),
        momentum=config.get("momentum", 0.9),
        scheduler_step_size=config.get("scheduler_step_size", 20),
        scheduler_gamma=config.get("scheduler_gamma", 0.5),
        num_workers=config.get("num_workers", 4),
        accelerator=config.get("accelerator", "auto"),
        val_split=config.get("val_split", 0.1)
    )
    
    logger.info("Starting training with PyTorch Lightning")
    
    # Train model on the batch data
    result = trainer.train_on_game_data(batch_data)
    
    # Log training metrics
    metrics = result["metrics"]
    logger.info(f"Training completed on {metrics['samples']} samples:")
    logger.info(f"  Loss: {metrics['loss']:.4f}")
    logger.info(f"  Policy Loss: {metrics['policy_loss']:.4f}")
    logger.info(f"  Value Loss: {metrics['value_loss']:.4f}")
    logger.info(f"  Final Learning Rate: {metrics['learning_rate']:.6f}")
    
    # Return the updated model
    return result["model"]

def initialize_model(config):
    """Initialize model from scratch or checkpoint"""
    import torch
    import os
    import re
    from zero.game import GameRules
    from zero.zeromodel import ZeroNetwork
    global storage_adapter

    model_config = config.get("model", {})
    k_channels = model_config.get("k_channels", 16)
    filters = model_config.get("filters", 128)
    blocks = model_config.get("blocks", 10)
    
    # Create rules to get board dimensions
    rules = GameRules()
    
    # Create model architecture
    model = ZeroNetwork(
        rules.height,
        rules.width,
        k_channels,
        filters=filters,
        blocks=blocks
    )
    
    # Check if we're resuming from a checkpoint
    starting_revision = 0
    if config.get("resume", False) and config.get("resume_from"):
        resume_path = config.get("resume_from")
        logger.info(f"Attempting to resume from checkpoint: {resume_path}")
        
        if resume_path.startswith("r2://"):
            try:
                logger.info(f"Downloading checkpoint from R2: {resume_path}")
                
                if storage_adapter is None:
                    logger.error("Storage adapter not initialized before model initialization!")
                    raise RuntimeError("Storage adapter must be initialized before model")
                
                if not config.get("use_r2", False):
                    logger.error("R2 URL specified but R2 storage backend not configured")
                    logger.error("Please run with --use-r2 and R2 credentials")
                    config["resume"] = False
                else:
                    _, bucket_key = resume_path.split("://", 1)
                    bucket, key = bucket_key.split("/", 1)
                    
                    config_bucket = config.get("r2_bucket", "")
                    if bucket != config_bucket:
                        logger.warning(f"R2 bucket in URL ({bucket}) doesn't match configured bucket ({config_bucket})")
                    
                    if hasattr(storage_adapter.backend, "s3"):
                        signed_url = storage_adapter.backend.s3.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': bucket, 'Key': key},
                            ExpiresIn=3600
                        )
                        
                        weights_dir = config.get("weights_dir", "weights")
                        os.makedirs(weights_dir, exist_ok=True)
                        local_filename = os.path.join(weights_dir, os.path.basename(key))
                        
                        logger.info(f"Downloading from R2 to {local_filename}...")
                        import requests
                        response = requests.get(signed_url, stream=True, timeout=60)
                        response.raise_for_status()
                        
                        with open(local_filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        logger.info(f"Successfully downloaded to: {local_filename}")
                        resume_path = local_filename
                    else:
                        logger.error("Storage adapter doesn't have R2/S3 client configured correctly")
                        config["resume"] = False
            except Exception as e:
                logger.error(f"Failed to download checkpoint from R2: {e}")
                logger.error("Starting fresh")
                config["resume"] = False
        
        elif resume_path.startswith(("http://", "https://")):
            try:
                logger.info(f"Downloading checkpoint from URL: {resume_path}")
                
                weights_dir = config.get("weights_dir", "weights")
                os.makedirs(weights_dir, exist_ok=True)
                local_filename = os.path.join(weights_dir, os.path.basename(resume_path))
                
                import requests
                logger.info(f"Downloading to {local_filename}...")
                response = requests.get(resume_path, stream=True, timeout=60)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(local_filename, 'wb') as f:
                    if total_size == 0:
                        logger.info("Unknown file size, downloading...")
                        f.write(response.content)
                    else:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if downloaded % (1024*1024) == 0:
                                    progress = downloaded / total_size * 100
                                    logger.info(f"Download progress: {progress:.1f}% ({downloaded/(1024*1024):.1f} MB)")
                
                logger.info(f"Successfully downloaded to: {local_filename}")
                resume_path = local_filename
                
                try:
                    run_url = resume_path.replace(".pt", "_run.json")
                    run_path = os.path.join(weights_dir, os.path.basename(run_url))
                    
                    run_response = requests.get(run_url, timeout=10)
                    if run_response.status_code == 200:
                        with open(run_path, 'wb') as f:
                            f.write(run_response.content)
                        logger.info(f"Also downloaded run data to {run_path}")
                except Exception as e:
                    logger.info(f"Could not download run data: {e}")
            except Exception as e:
                logger.error(f"Error downloading checkpoint: {e}")
                config["resume"] = False
        
        # Make sure we can access the checkpoint file
        if config.get("resume", False) and not os.path.exists(resume_path):
            logger.error(f"Checkpoint file does not exist: {resume_path}")
            config["resume"] = False
        
        # Load the checkpoint if we have a valid file
        if config.get("resume", False):
            try:
                # Determine the revision number to start from
                revision = config.get("resume_revision")
                if revision is None:
                    # Try to extract revision from the filename
                    pattern = r'r(\d+)\.pt'
                    match = re.search(pattern, resume_path)
                    if match:
                        revision = int(match.group(1))
                        logger.info(f"Extracted revision {revision} from filename")
                    else:
                        # Get the highest existing revision number and increment by 1
                        weights_dir = config.get("weights_dir", "weights")
                        if os.path.exists(weights_dir):
                            revisions = []
                            for filename in os.listdir(weights_dir):
                                if filename.startswith("r") and filename.endswith(".pt"):
                                    try:
                                        rev = int(filename[1:-3])
                                        revisions.append(rev)
                                    except ValueError:
                                        continue
                            revision = max(revisions) + 1 if revisions else 0
                
                # Set starting revision
                starting_revision = revision
                logger.info(f"Loading checkpoint and starting from revision {starting_revision}")
                
                # Load the model state
                device = config.get("device", "cpu")
                if device is None or device == "auto":
                    device = "cpu"  # Default to CPU for loading
                
                # Load weights with appropriate device mapping
                state_dict = torch.load(resume_path, map_location=device)
                model.load_state_dict(state_dict)
                logger.info(f"Successfully loaded model weights from {resume_path}")
                
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.error("Creating fresh model instead")
                starting_revision = 0
                config["resume"] = False
    
    # Make sure storage adapter is initialized
    if storage_adapter is None:
        logger.error("Storage adapter not initialized before model initialization!")
        raise RuntimeError("Storage adapter must be initialized before model")
    
    # Save initial model using the storage adapter
    logger.info(f"Saving model (revision {starting_revision})")
    weights_path, weights_url, weights_sha256 = storage_adapter.save_model(model, starting_revision)
    
    # Update shared state
    shared_state.update_model(starting_revision, weights_path, weights_url, weights_sha256)
    
    logger.info(f"Initialized model: revision={starting_revision}, path={weights_path}, url={weights_url}")
    return model, starting_revision

def main_training_loop(config):
    """Main training loop"""
    logger.info("Starting main training loop")
    
    # Initialize model - now returns both model and starting revision
    current_model, current_revision = initialize_model(config)
    logger.info(f"Starting training loop with revision {current_revision}")
    
    # Main loop
    while True:
        # 1. Wait for enough games or deadline to pass
        current_state = shared_state.to_dict()
        current_time = datetime.now(timezone.utc)
        deadline = datetime.fromisoformat(current_state["deadline"])
        min_games = current_state["min_games"]

        if current_state["current_queue_size"] < min_games and current_time < deadline:
            # Not enough games yet and deadline hasn't passed
            logger.info(f"Waiting for more games: {current_state['current_queue_size']}/{min_games}")
            time.sleep(config.get("check_interval_seconds", 10))
            continue

        # 2. Check if we have enough games
        if current_state["current_queue_size"] < min_games:
            # Deadline passed but not enough games
            logger.info(f"Deadline passed but not enough games: {current_state['current_queue_size']}/{min_games}")

            # Extend deadline
            with shared_state.lock:
                shared_state.deadline = datetime.now(timezone.utc) + timedelta(minutes=config.get("training_deadline_minutes", 30))

            time.sleep(config.get("check_interval_seconds", 10))
            continue

        # 3. Set training mode
        logger.info("Starting training process")
        shared_state.set_training(True)

        # 4. Drain queue
        batch_data = shared_state.drain_queue()

        # 5. Process and train
        new_model = process_and_train(batch_data, current_model, config)

        # 6. Save model using storage adapter
        current_revision += 1
        weights_path, weights_url, weights_sha256 = storage_adapter.save_model(new_model, current_revision)

        # 7. Update shared state with new model
        shared_state.update_model(current_revision, weights_path, weights_url, weights_sha256)

        # 8. Clean up old models while safely preserving the current revision
        keep_revisions = config.get("keep_revisions", 5)
        storage_adapter.cleanup_old_models(keep_revisions, current_revision=current_revision)
        logger.info(f"Weight cleanup completed: kept {keep_revisions} revisions and preserved revision {current_revision}")

        # 9. Update current model
        current_model = new_model

        # 10. End training mode
        shared_state.set_training(False)
        logger.info(f"Training complete: revision={current_revision}")


def main():
    global storage_adapter
    config = parse_args()

    shared_state.config = config
    
    # Initialize storage adapter only once
    if storage_adapter is None:
        logger.info("Initializing storage adapter for the first time")
        storage_adapter = StorageAdapter(config)
    else:
        logger.warning("Storage adapter already initialized - this should not happen")
    
    # Reset if requested
    if config["reset"]:
        print("Resetting state...")
        
        # Clear weights directory
        weights_dir = "weights"
        if os.path.exists(weights_dir):
            for file in os.listdir(weights_dir):
                if file.startswith("r") and file.endswith(".pt"):
                    os.remove(os.path.join(weights_dir, file))
    
    # Start training loop in background thread
    training_thread = threading.Thread(
        target=main_training_loop,
        args=(config,),
        daemon=True
    )
    training_thread.start()
    
    # Set up authentication middleware
    app.add_middleware(
        TokenAuthMiddleware,
        token=config["auth_token"]
    )
    
    # Start web server
    uvicorn.run(
        app,
        host=config["host"],
        port=config["port"],
        log_level="warning",
        access_log=False
    )

if __name__ == "__main__":
    main()