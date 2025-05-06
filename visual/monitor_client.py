"""
Client for sending updates to the monitoring server
"""

import os
import time
import json
import threading
import requests
from typing import Dict, Any, List, Optional

class MonitorClient:
    """Client for sending updates to the AlphaZero 2048 monitoring server"""
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        """Initialize the monitor client
        
        Args:
            server_url: URL of the monitoring server
        """
        self.server_url = server_url
        self.enabled = True
        self.worker_update_interval = 2.0  # seconds between worker updates (to avoid overwhelming the server)
        self.last_worker_update = {}  # Track last update time for each worker
        
    def update_game(self, worker_id: int, board: List[List[int]], score: int, turn: int) -> bool:
        """Send a game state update to the server
        
        Args:
            worker_id: ID of the worker
            board: 2D array of tile values
            score: Current game score
            turn: Current turn number
            
        Returns:
            bool: True if update was sent, False otherwise
        """
        if not self.enabled:
            return False
        
        # Rate limit updates per worker (don't send updates too frequently)
        current_time = time.time()
        if worker_id in self.last_worker_update:
            if current_time - self.last_worker_update.get(worker_id, 0) < self.worker_update_interval:
                return False
        
        self.last_worker_update[worker_id] = current_time
        
        try:
            response = requests.post(
                f"{self.server_url}/api/update_game", 
                json={
                    "worker_id": worker_id,
                    "board": board,
                    "score": score,
                    "turn": turn
                },
                timeout=1  # Short timeout to avoid blocking training
            )
            return response.status_code == 200
        except Exception as e:
            # Silently fail to avoid affecting training
            return False
            
    def update_training_stats(self, stats: Dict[str, Any]) -> bool:
        """Send training statistics update to the server
        
        Args:
            stats: Dictionary of training statistics
            
        Returns:
            bool: True if update was sent, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            response = requests.post(
                f"{self.server_url}/api/update_stats", 
                json=stats,
                timeout=1  # Short timeout to avoid blocking training
            )
            return response.status_code == 200
        except Exception as e:
            # Silently fail to avoid affecting training
            return False
            
    def disable(self):
        """Disable the monitor client"""
        self.enabled = False
        
    def enable(self):
        """Enable the monitor client"""
        self.enabled = True


# Global monitor client instance
monitor = MonitorClient()

def update_game_state(worker_id: int, board: List[List[int]], score: int, turn: int):
    """Update game state in the monitor (safe to call from anywhere)"""
    monitor.update_game(worker_id, board, score, turn)
    
def update_training_stats(
    current_epoch: int, 
    total_epochs: int, 
    samples_collected: int = None,
    games_played: int = None,
    latest_loss: float = None,
    policy_loss: float = None,
    value_loss: float = None,
    max_tile_achieved: int = None,
    max_score: int = None
):
    """Update training statistics in the monitor (safe to call from anywhere)"""
    stats = {
        "current_epoch": current_epoch,
        "total_epochs": total_epochs
    }
    
    # Add optional stats if provided
    if samples_collected is not None:
        stats["samples_collected"] = samples_collected
    if games_played is not None:
        stats["games_played"] = games_played
    if latest_loss is not None:
        stats["latest_loss"] = latest_loss
    if policy_loss is not None:
        stats["policy_loss"] = policy_loss
    if value_loss is not None:
        stats["value_loss"] = value_loss
    if max_tile_achieved is not None:
        stats["max_tile_achieved"] = max_tile_achieved
    if max_score is not None:
        stats["max_score"] = max_score
        
    monitor.update_training_stats(stats)
    
# Function to start a monitoring server in a separate thread
def start_monitor_server(host='127.0.0.1', port=5000, open_browser=True):
    """Start a monitoring server in a separate thread
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        open_browser: Whether to open a browser window automatically
    """
    from threading import Thread
    
    def run_server():
        try:
            from visual.monitor_server import run_server
            run_server(host=host, port=port, open_browser=open_browser)
        except Exception as e:
            print(f"Error starting monitor server: {e}")
    
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait a bit for server to start
    time.sleep(1.5)
    
    # Check if server is running
    try:
        response = requests.get(f"http://{host}:{port}/api/status", timeout=1)
        if response.status_code == 200:
            print(f"Monitor server running at http://{host}:{port}")
            return True
    except:
        print("Monitor server may not have started properly")
    
    return False