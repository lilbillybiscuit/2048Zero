import threading
from datetime import datetime, timedelta, timezone

class SharedState:
    def __init__(self):
        self.lock = threading.RLock()

        # Training state
        self.is_training = False

        # Model information
        self.revision = 0
        self.weights_path = ""
        self.weights_sha256 = ""
        self.weights_url = ""

        self.deadline = datetime.now(timezone.utc) + timedelta(minutes=5)
        self.game_queue = []
        self.game_ids = set()  # To prevent duplicates

        self.config = {}

    def to_dict(self):
        """Get a snapshot of the current state"""
        with self.lock:
            now = datetime.now(timezone.utc)
            time_remaining = max(0, (self.deadline - now).total_seconds())

            return {
                "revision": self.revision,
                "is_training": self.is_training,
                "weights_url": self.weights_url,
                "sha256": self.weights_sha256,
                "deadline": self.deadline.isoformat(),
                "time_remaining_seconds": int(time_remaining),
                "current_queue_size": len(self.game_queue),
                "min_games": self.config.get("min_games", 512),
                "heartbeat": self.config.get("heartbeat", 5),
                "server_time": now.isoformat(),
                "model_config": self.config.get("model", {}),
                "self_play_args": self.config.get("self_play", {})
            }

    def add_games(self, batch):
        """Add games to the queue"""
        with self.lock:
            if self.is_training:
                return False, "Server is in training mode"

            # Check for duplicates
            for game in batch.get("games", []):
                game_id = game.get("game_id")
                if game_id and game_id in self.game_ids:
                    return False, f"Duplicate game ID: {game_id}"

                if game_id:
                    self.game_ids.add(game_id)

            # Add to queue
            self.game_queue.append(batch)
            return True, "Games added to queue"

    def drain_queue(self):
        """Drain the queue for training"""
        with self.lock:
            games = self.game_queue
            self.game_queue = []
            self.game_ids.clear()
            return games

    def set_training(self, training):
        """Set the training state"""
        with self.lock:
            self.is_training = training
            return self.is_training

    def update_model(self, revision, weights_path, weights_url, weights_sha256):
        """Update model information after training"""
        with self.lock:
            self.revision = revision
            self.weights_path = weights_path
            self.weights_url = weights_url
            self.weights_sha256 = weights_sha256
            # Set new deadline
            self.deadline = datetime.now(timezone.utc) + timedelta(
                minutes=self.config.get("training_deadline_minutes", 30))
            return self.revision
