#!/usr/bin/env python
"""
2048-Zero Worker for Distributed Training

This script runs a worker that connects to a central server, downloads model weights,
performs self-play games, and uploads game data back to the server.
"""

import os
import sys
import json
import time
import uuid
import hashlib
import logging
import argparse
import requests
import io
import torch
import zstandard as zstd
from typing import Dict, List, Any, Optional
from datetime import datetime

from zero import GameRules, GameState, BitBoard, ZeroNetwork, ZeroPlayer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("worker.log")
    ]
)
logger = logging.getLogger(__name__)

class Worker:
    """Worker that performs self-play and uploads games to server"""
    
    def __init__(
        self,
        server_url: str,
        auth_token: str,
        batch_size: int = 8,
        device: str = None
    ):
        # Basic configuration
        self.server_url = server_url.rstrip("/")
        self.auth_token = auth_token
        self.batch_size = batch_size
        self.worker_id = str(uuid.uuid4())
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # State tracking
        self.game_buffer = []
        self.current_revision = -1  # Force initial download
        self.heartbeat_interval = 5  # Default, will be updated from server
        
        # Model and game components
        self.model = None
        self.rules = GameRules()
        
        logger.info(f"Worker {self.worker_id} initialized on {self.device}")
        logger.info(f"Server URL: {self.server_url}")
    
    def api_request(self, method: str, endpoint: str, use_base_url = True, **kwargs):
        """Make an API request to the server with automatic retries"""
        if use_base_url:
            url = f"{self.server_url}/{endpoint.lstrip('/')}"
        else:
            url = endpoint
        headers = kwargs.pop('headers', {})
        headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # For binary responses (weights download)
        is_binary = kwargs.pop('binary', False)
        
        max_retries = 5
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response = requests.request(method, url, headers=headers, **kwargs)
                response.raise_for_status()
                
                if is_binary:
                    return True, response.content
                
                return True, response.json() if response.content else {}
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed ({attempt+1}/{max_retries}): {str(e)}")
                
                # Don't retry certain client errors
                if hasattr(e, 'response') and e.response and e.response.status_code < 500 and e.response.status_code != 429:
                    error_details = e.response.json() if e.response.content else {"error": str(e)}
                    return False, error_details
                    
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    return False, {"error": f"Max retries exceeded: {str(e)}"}
        
        return False, {"error": "Request failed after retries"}
    
    def get_server_state(self):
        """Get current state from the server"""
        success, state = self.api_request("GET", "/state")
        
        if success:
            # Update heartbeat interval
            self.heartbeat_interval = state.get("heartbeat", 5)
            return state
            
        logger.error(f"Failed to get server state: {state}")
        return None
    
    def download_weights(self, weights_url: str, expected_sha256: str):
        """Download model weights from URL"""
        logger.info(f"Downloading weights from {weights_url}")
        if weights_url.startswith(self.server_url):
            endpoint = weights_url[len(self.server_url):].lstrip("/")
            print("ENDPOINT", endpoint)
            success, weights_data = self.api_request("GET", endpoint, binary=True)
        else:
            success, weights_data = self.api_request("GET", weights_url, binary=True, use_base_url=False)

        if not success:
            logger.error(f"Failed to download weights: {weights_data}")
            return None

        # Verify SHA256 hash
        sha256 = hashlib.sha256()
        sha256.update(weights_data)
        actual_sha256 = sha256.hexdigest()
        
        if actual_sha256 != expected_sha256:
            logger.error(f"SHA256 mismatch. Expected {expected_sha256}, got {actual_sha256}")
            return None
            
        logger.info(f"Successfully downloaded weights ({len(weights_data)/1024:.2f} KB)")
        return weights_data
    
    def load_model(self, weights_data: bytes, model_config: Dict[str, Any]):
        """Load model from weights data"""
        try:
            # Create model with configuration
            model = ZeroNetwork(
                self.rules.height, 
                self.rules.width,
                model_config["k_channels"],
                filters=model_config.get("filters", 128),
                blocks=model_config.get("blocks", 10)
            )
            
            # Load weights directly from memory
            buffer = io.BytesIO(weights_data)
            state_dict = torch.load(buffer, map_location="cpu")
            model.load_state_dict(state_dict)
            
            # Move to device and set to eval mode
            model = model.to(self.device).eval()
            
            logger.info(f"Model loaded successfully to {self.device}")
            return model
            
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            return None
    
    def play_game(self, model: ZeroNetwork, self_play_config: Dict[str, Any]):
        """Play a single self-play game"""
        # Create player with the provided model
        player = ZeroPlayer(
            model, 
            self.rules,
            dirichlet_alpha=self_play_config.get("dirichlet_alpha", 0.3),
            dirichlet_eps=self_play_config.get("epsilon", 0.25),
            temperature=self_play_config.get("temperature", 1.0)
        )
        
        # Initial game state
        game_state = self.rules.get_initial_state()
        moves_data = []
        turn = 0
        max_moves = self_play_config.get("max_moves_per_game", 1000)
        num_simulations = self_play_config.get("num_simulations", 800)
        
        start_time = time.time()
        
        # Play until terminal state or max moves reached
        while not self.rules.is_terminal(game_state.board) and turn < max_moves:
            # Get action from MCTS player
            action, (policy_probs, value_est) = player.play(
                game_state, 
                simulations=num_simulations
            )
            
            # Store move data
            board_str = json.dumps(game_state.board.tolist())
            moves_data.append({
                "board": board_str,
                "policy": policy_probs,
                "value": float(value_est)
            })
            
            # Apply move and update state
            new_board, gain = self.rules.apply_move(game_state.board, action)
            new_board_with_tiles = self.rules.add_random_tiles(new_board)
            new_bitboard = BitBoard.from_numpy(new_board_with_tiles)
            game_state = GameState(
                new_board_with_tiles, 
                game_state.score + gain, 
                new_bitboard
            )
            turn += 1
        
        elapsed_time = time.time() - start_time
        
        # Create game record
        final_score = int(game_state.score)
        game_id = str(uuid.uuid4())
        max_tile = self.rules.get_max_tile(game_state.board)
        
        logger.info(f"Game {game_id} finished. Score: {final_score}, Max Tile: {max_tile}, " + 
                    f"Turns: {turn}, Time: {elapsed_time:.1f}s ({elapsed_time/turn:.1f}s/move)")
        
        return {
            "game_id": game_id,
            "length": turn,
            "final_score": final_score,
            "max_tile": int(max_tile),
            "timestamp": datetime.now().isoformat(),
            "moves": moves_data
        }
    
    def upload_games(self, revision: int):
        """Upload collected games to the server"""
        if not self.game_buffer:
            return True
            
        # Prepare batch data
        batch = {
            "worker_id": self.worker_id,
            "revision": revision,
            "games": self.game_buffer
        }
        
        # Serialize to JSON
        batch_json = json.dumps(batch)
        
        # Calculate SHA-256 hash of the uncompressed JSON
        sha256 = hashlib.sha256()
        sha256.update(batch_json.encode())
        batch_hash = sha256.hexdigest()
        
        # Compress data with zstd
        compressed_data = zstd.compress(batch_json.encode())
        
        # Set up headers with revision, compression info, and hash
        headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "zstd",
            "X-Revision": str(revision),
            "X-Content-SHA256": batch_hash
        }
        
        logger.info(f"Uploading {len(self.game_buffer)} games for revision {revision}")
        
        # Upload to server
        success, response = self.api_request(
            "POST", 
            "/upload", 
            headers=headers,
            data=compressed_data
        )
        
        if success:
            logger.info(f"Successfully uploaded {len(self.game_buffer)} games")
            self.game_buffer = []
            return True
            
        # Handle server rejection (usually version mismatch or training mode)
        if 'detail' in response:
            logger.warning(f"Upload rejected: {response['detail']}. Clearing buffer.")
            self.game_buffer = []
            
        return False
    
    def run(self):
        """Main worker loop"""
        logger.info(f"Starting worker {self.worker_id}")
        
        while True:
            try:
                # 1. Get server state
                state = self.get_server_state()
                if not state:
                    time.sleep(self.heartbeat_interval)
                    continue
                
                # 2. Check if server is training
                if state.get("is_training", False):
                    logger.info("Server is training. Waiting...")
                    time.sleep(self.heartbeat_interval)
                    continue
                
                # 3. Check if deadline has passed
                if state.get("time_remaining_seconds", 0) <= 0:
                    # Upload any existing games before deadline passes
                    if self.game_buffer:
                        self.upload_games(self.current_revision)
                    logger.info("Collection round ended. Waiting for new deadline...")
                    time.sleep(self.heartbeat_interval)
                    continue

                # 4. Check revision - download new weights if needed
                server_revision = state.get("revision", 0)
                if server_revision != self.current_revision or self.model is None:
                    # Upload any existing games before updating
                    if self.game_buffer:
                        self.upload_games(self.current_revision)
                    
                    # Download and load new model
                    logger.info(f"Downloading weights for revision {server_revision}")
                    weights_data = self.download_weights(
                        state.get("weights_url", ""),
                        state.get("sha256", "")
                    )
                    
                    if not weights_data:
                        time.sleep(self.heartbeat_interval)
                        continue
                    
                    # Load model with new weights
                    self.model = self.load_model(weights_data, state.get("model_config", {}))
                    if not self.model:
                        time.sleep(self.heartbeat_interval)
                        continue
                    
                    self.current_revision = server_revision
                    logger.info(f"Now using model revision {server_revision}")
                
                # 5. Play a game
                game = self.play_game(self.model, state.get("self_play_args", {}))
                self.game_buffer.append(game)
                
                # 6. Upload if buffer is full
                if len(self.game_buffer) >= self.batch_size:
                    self.upload_games(self.current_revision)
                
            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
                break
                
            except Exception as e:
                logger.exception(f"Error in worker loop: {e}")
                # Continue after error with a short delay
                time.sleep(5)

def main():
    """Parse arguments and start worker"""
    parser = argparse.ArgumentParser(description="2048-Zero Worker")
    
    # Server connection settings
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", 
                        help="URL of the trainer server")
    parser.add_argument("--auth-token", type=str, default="2048-zero-token", 
                        help="Authentication token")
    
    # Worker settings
    parser.add_argument("--batch-size", type=int, default=8, 
                        help="Number of games to collect before uploading")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cpu, cuda, mps). If not specified, will auto-detect.")
    
    args = parser.parse_args()

    # if there are multiple GPUs and/or accelerator is set to cuda, set the device to the PID modulo the number of GPUs
    if args.device == "cuda" or torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            args.device = f"cuda:{os.getpid() % num_gpus}"
        else:
            args.device = "cuda"
    
    # Create and run worker
    worker = Worker(
        server_url=args.server_url,
        auth_token=args.auth_token,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print(f"Starting worker, connecting to {args.server_url}")

    worker.run()

if __name__ == "__main__":
    main()