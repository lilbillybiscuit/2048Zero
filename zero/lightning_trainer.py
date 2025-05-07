import os
import numpy as np
import torch
import json
import time
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

# Import PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader

# Import core components
from zero.game import GameRules, GameState, BitBoard
from zero.zeromodel import ZeroNetwork

logger = logging.getLogger(__name__)

class AlphaZeroDataset(Dataset):
    """Dataset for AlphaZero training samples with optimized preprocessing"""
    
    def __init__(self, samples: List[Tuple[np.ndarray, List[float], float]], k: int, board_height: int, board_width: int):
        """
        Args:
            samples: List of (board, policy, value) tuples
            k: Number of channels for one-hot encoding
            board_height: Height of the board
            board_width: Width of the board
        """
        self.samples = samples
        self.k = k
        self.board_height = board_height
        self.board_width = board_width
        
        # Pre-process data once during initialization to speed up training
        self.boards = []
        self.policies = []
        self.values = []
        
        # Vectorized conversion
        start_time = time.time()
        
        # Process samples into tensors
        for board, policy, value in samples:
            # Convert board to numpy array if it's not already
            if isinstance(board, list):
                board = np.array(board, dtype=np.int32)
            elif isinstance(board, str):
                board = np.array(json.loads(board), dtype=np.int32)
                
            self.boards.append(board)
            
            # Convert policy to numpy array if it's not already
            if isinstance(policy, list):
                policy = np.array(policy, dtype=np.float32)
            self.policies.append(policy)
            
            # Ensure value is a float
            self.values.append(float(value))
            
        self.boards = np.array(self.boards, dtype=np.int32)
        self.policies = torch.tensor(np.array(self.policies, dtype=np.float32), dtype=torch.float32)
        self.values = torch.tensor(np.array(self.values, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
        
        # Pre-compute one-hot encodings
        self.onehot_boards = self._precompute_onehot_encodings()
        
        logging.debug(f"Dataset initialization took {time.time() - start_time:.2f} seconds")
    
    def _precompute_onehot_encodings(self):
        """Pre-compute all one-hot encodings to avoid doing it at runtime"""
        batch_size = len(self.boards)
        onehot_shape = (batch_size, self.k, self.board_height, self.board_width)
        onehot_boards = torch.zeros(onehot_shape, dtype=torch.float32)
        
        # Use F.one_hot for efficient encoding
        for b, board in enumerate(self.boards):
            # Convert board to tensor
            board_tensor = torch.tensor(board, dtype=torch.long)
            
            # Fill the one-hot tensor
            for i in range(self.board_height):
                for j in range(self.board_width):
                    value = min(int(board[i, j]), self.k - 1)
                    onehot_boards[b, value, i, j] = 1.0
        
        return onehot_boards
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Fast item getter using pre-computed tensors"""
        return {
            'board_onehot': self.onehot_boards[idx],
            'policy': self.policies[idx],
            'value': self.values[idx]
        }

class AlphaZeroModel(pl.LightningModule):
    """PyTorch Lightning module for AlphaZero training"""
    
    def __init__(
        self,
        model: ZeroNetwork,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        scheduler_step_size: int = 20,
        scheduler_gamma: float = 0.5,
        momentum: float = 0.9
    ):
        """
        Args:
            model: The neural network model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            scheduler_step_size: Step size for learning rate scheduler
            scheduler_gamma: Gamma for learning rate scheduler
            momentum: Momentum for SGD optimizer
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.momentum = momentum
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning"""
        # Get data from batch (already preprocessed)
        board_onehot = batch['board_onehot']
        policy_target = batch['policy'] 
        value_target = batch['value']
        
        # Forward pass with preprocessed board
        policy_pred, value_pred = self.model(board_onehot)
        
        # Calculate policy loss (cross-entropy)
        policy_loss = -(policy_target * torch.log(policy_pred + 1e-8)).sum(dim=1).mean()
        
        # Calculate value loss (MSE)
        value_loss = torch.nn.functional.mse_loss(value_pred, value_target)
        
        # Total loss is sum of policy and value losses
        loss = policy_loss + value_loss
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_policy_loss', policy_loss, prog_bar=True)
        self.log('train_value_loss', value_loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning"""
        # Get data from batch (already preprocessed)
        board_onehot = batch['board_onehot']
        policy_target = batch['policy']
        value_target = batch['value']
        
        # Forward pass with preprocessed board
        policy_pred, value_pred = self.model(board_onehot)
        
        # Calculate policy loss (cross-entropy)
        policy_loss = -(policy_target * torch.log(policy_pred + 1e-8)).sum(dim=1).mean()
        
        # Calculate value loss (MSE)
        value_loss = torch.nn.functional.mse_loss(value_pred, value_target)
        
        # Total loss is sum of policy and value losses
        loss = policy_loss + value_loss
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_policy_loss', policy_loss, prog_bar=True)
        self.log('val_value_loss', value_loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

class LightningDistributedTrainer:
    """Training implementation for 2048-Zero using PyTorch Lightning
    
    This trainer is designed for accelerated training using PyTorch Lightning.
    """
    
    def __init__(
        self,
        model: ZeroNetwork,
        rules: GameRules,
        device: str = None,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        epochs: int = 10,
        momentum: float = 0.9,
        scheduler_step_size: int = 20,
        scheduler_gamma: float = 0.5,
        num_workers: int = 4,
        accelerator: str = "auto",
        val_split: float = 0.1  # Fraction of data to use for validation
    ):
        """Initialize the Lightning trainer
        
        Args:
            model: The neural network model to train
            rules: Game rules for 2048
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            weight_decay: Weight decay for regularization
            epochs: Number of epochs to train for each batch of data
            momentum: Momentum for SGD optimizer
            scheduler_step_size: Step size for learning rate scheduler
            scheduler_gamma: Gamma for learning rate scheduler
            num_workers: Number of worker processes for data loading
            accelerator: PyTorch Lightning accelerator ('cpu', 'gpu', 'tpu', etc.)
            val_split: Fraction of data to use for validation
        """
        self.model = model
        self.rules = rules
        self.device = device if device else "auto"
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.momentum = momentum
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.num_workers = num_workers
        self.accelerator = accelerator
        self.val_split = val_split
        
        # Create PyTorch Lightning model
        self.pl_model = AlphaZeroModel(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            momentum=momentum
        )
        
        # Create PyTorch Lightning trainer with callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath="lightning_checkpoints",
                filename="alphazero-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=3,
                mode="min"
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
                verbose=True
            )
        ]
        
        # Initialize the trainer with optimized settings
        self.trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            callbacks=callbacks,
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_model_summary=True,
            # Performance optimizations:
            num_sanity_val_steps=0,  # Skip validation sanity checks
            precision="32-true",     # Use native 32-bit precision rather than PyTorch Lightning's wrapper
            deterministic=False,     # Disable deterministic mode for speed
            benchmark=True,          # Enable CUDNN benchmarking for faster training
        )

    def parse_game_data(self, batch_data: List[Dict[str, Any]]) -> List[Tuple[np.ndarray, List[float], float]]:
        """Parse game data from worker uploads into training samples
        
        Args:
            batch_data: List of batches containing game data
            
        Returns:
            List of (board, policy, value) tuples for training
        """
        logger.info(f"Processing {sum(len(batch.get('games', [])) for batch in batch_data)} games for training")
        
        training_samples = []
        max_score = 0
        
        for batch in batch_data:
            for game in batch.get("games", []):
                max_score = max(max_score, game.get("final_score", 0))
                
                for move in game.get("moves", []):
                    try:
                        board = json.loads(move.get("board"))
                        if isinstance(board, list):
                            board = np.array(board, dtype=np.int32)
                        
                        policy = move.get("policy")
                        if isinstance(policy, list):
                            policy = np.array(policy, dtype=np.float32)
                            
                        value = move.get("value")
                        if not isinstance(value, (int, float)):
                            value = float(value)
                            
                        training_samples.append((board, policy, value))
                    except Exception as e:
                        logger.warning(f"Error processing move: {e}")
        
        logger.info(f"Extracted {len(training_samples)} training samples. Max score: {max_score}")
        return training_samples

    def prepare_data(self, samples: List[Tuple[np.ndarray, List[float], float]]):
        """Prepare data for PyTorch Lightning training
        
        Args:
            samples: List of (board, policy, value) tuples
            
        Returns:
            Dictionary with dataloaders
        """
        # Shuffle the samples
        random_indices = np.random.permutation(len(samples))
        shuffled_samples = [samples[i] for i in random_indices]
        
        # Split into training and validation sets
        val_size = int(len(shuffled_samples) * self.val_split)
        train_samples = shuffled_samples[val_size:]
        val_samples = shuffled_samples[:val_size]
        
        # Create datasets with board dimensions from the model
        train_dataset = AlphaZeroDataset(
            train_samples, 
            self.model.k, 
            self.model.n, 
            self.model.m
        )
        
        val_dataset = AlphaZeroDataset(
            val_samples, 
            self.model.k, 
            self.model.n, 
            self.model.m
        ) if val_samples else None
        
        # Create dataloaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(self.num_workers, 2),  # Reduce num_workers to decrease startup time
            pin_memory=True if self.device == "cuda" else False,
            persistent_workers=self.num_workers > 0,  # Keep workers alive between epochs
            prefetch_factor=2 if self.num_workers > 0 else None,  # Prefetch 2 batches per worker
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size*2,  # Use larger batch size for validation
            shuffle=False,
            num_workers=min(self.num_workers, 1),  # Use fewer workers for validation
            pin_memory=True if self.device == "cuda" else False,
            persistent_workers=self.num_workers > 0,  # Keep workers alive between epochs
        ) if val_dataset else None
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader
        }

    def train_on_batch(self, samples: List[Tuple[np.ndarray, List[float], float]]) -> Dict[str, float]:
        """Train the model using PyTorch Lightning
        
        Args:
            samples: List of (board, policy, value) tuples
            
        Returns:
            Dictionary with training metrics
        """
        if not samples:
            logger.warning("No samples provided for training")
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "samples": 0
            }
        
        logger.info(f"Training on {len(samples)} samples with PyTorch Lightning")
        
        # Prepare data
        data = self.prepare_data(samples)
        
        # Track time to measure Lightning startup latency
        train_start_time = time.time()
        logger.info("Starting Lightning training process...")
        
        # Train the model with optimized settings
        self.trainer.fit(
            self.pl_model,
            train_dataloaders=data['train_loader'],
            val_dataloaders=data['val_loader']
        )
        
        train_duration = time.time() - train_start_time
        logger.info(f"Lightning training complete in {train_duration:.2f} seconds")
        
        # Get training metrics
        metrics = {
            "loss": self.trainer.callback_metrics.get("train_loss", 0.0).item(),
            "policy_loss": self.trainer.callback_metrics.get("train_policy_loss", 0.0).item(),
            "value_loss": self.trainer.callback_metrics.get("train_value_loss", 0.0).item(),
            "val_loss": self.trainer.callback_metrics.get("val_loss", 0.0).item() if data['val_loader'] else 0.0,
            "samples": len(samples),
            "learning_rate": self.learning_rate * (self.scheduler_gamma ** (self.trainer.current_epoch // self.scheduler_step_size))
        }
        
        return metrics
            
    def train_on_game_data(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process game data and train the model using PyTorch Lightning
        
        Args:
            batch_data: List of batches containing games from workers
            
        Returns:
            Dictionary with training metrics and updated model
        """
        # Parse game data into training samples
        samples = self.parse_game_data(batch_data)
        
        # Train on samples
        metrics = self.train_on_batch(samples)
        
        # Return metrics and updated model
        return {
            "metrics": metrics,
            "model": self.model  # The model is updated in-place
        }