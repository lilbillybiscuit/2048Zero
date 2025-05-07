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
    from zero.game import GameRules
    from zero.zeromodel import ZeroNetwork
    global storage_adapter

    model_config = config.get("model", {})
    k_channels = model_config.get("k_channels", 16)
    filters = model_config.get("filters", 128)
    blocks = model_config.get("blocks", 10)
    
    # Create rules to get board dimensions
    rules = GameRules()
    
    # Create model
    model = ZeroNetwork(
        rules.height,
        rules.width,
        k_channels,
        filters=filters,
        blocks=blocks
    )
    
    # Make sure storage adapter is initialized
    if storage_adapter is None:
        logger.error("Storage adapter not initialized before model initialization!")
        raise RuntimeError("Storage adapter must be initialized before model")
    
    # Save initial model using the storage adapter
    logger.info("Saving initial model (revision 0)")
    weights_path, weights_url, weights_sha256 = storage_adapter.save_model(model, 0)
    
    # Update shared state
    shared_state.update_model(0, weights_path, weights_url, weights_sha256)
    
    logger.info(f"Initialized model: revision=0, path={weights_path}, url={weights_url}")
    return model

def main_training_loop(config):
    """Main training loop"""
    logger.info("Starting main training loop")
    
    # Initialize model
    current_model = initialize_model(config)
    current_revision = 0
    
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