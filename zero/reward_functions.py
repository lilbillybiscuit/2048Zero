"""
Reward functions for 2048 Zero training

This module provides reward functions for both standard and parallel training.
"""

import math
import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple
from zero.game import GameState

# Global running maximum score for dynamic scaling
_global_max_score = 50000  # Default starting value

# Module-level reward functions that are pickle-friendly for multiprocessing
def score_reward_func(state: GameState, stats: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score-based reward function that's pickle-friendly for multiprocessing
    
    Args:
        state: Final game state
        stats: Game statistics dictionary
        
    Returns:
        tuple: (reward_value, reward_name)
    """
    score = stats['score']
    # Normalize score to [-1, 1] range using log scale
    z = min(max((math.log(score + 100) / math.log(50000 + 100)) * 2 - 1, -1.0), 1.0)
    return z, "score"

def dynamic_score_reward_func(state: GameState, stats: Dict[str, Any]) -> Tuple[float, str]:
    """
    Dynamic score-based reward function that adapts to the highest observed score
    
    Args:
        state: Final game state
        stats: Game statistics dictionary
        
    Returns:
        tuple: (reward_value, reward_name)
    """
    global _global_max_score
    score = stats['score']
    
    # Update the global maximum score if this score is higher
    if score > _global_max_score:
        # Use a smoothing factor to avoid sudden large changes
        _global_max_score = max(score, _global_max_score * 0.9 + score * 0.1)
    
    # Normalize score to [-1, 1] range using log scale with dynamic maximum
    z = min(max((math.log(score + 100) / math.log(_global_max_score + 100)) * 2 - 1, -1.0), 1.0)
    return z, "dynamic_score"

def max_tile_reward_func(state: GameState, stats: Dict[str, Any]) -> Tuple[float, str]:
    """
    Max tile reward function that's pickle-friendly for multiprocessing
    
    Args:
        state: Final game state
        stats: Game statistics dictionary
        
    Returns:
        tuple: (reward_value, reward_name)
    """
    max_tile = stats['max_tile']
    # Scale based on log2 of max tile / 11 (11 = log2(2048))
    z = math.log2(int(max_tile) + 1) / 11.0 * 2 - 1
    return z, "max_tile"

def hybrid_reward_func(state: GameState, stats: Dict[str, Any]) -> Tuple[float, str]:
    """
    Hybrid reward function that's pickle-friendly for multiprocessing
    
    Args:
        state: Final game state
        stats: Game statistics dictionary
        
    Returns:
        tuple: (reward_value, reward_name)
    """
    # Get individual rewards
    score_val, _ = score_reward_func(state, stats)
    tile_val, _ = max_tile_reward_func(state, stats)
    
    # Mix with weights
    score_weight = 0.7
    tile_weight = 0.3
    
    # Combine rewards
    z = score_weight * score_val + tile_weight * tile_val
    
    return z, "hybrid"

# Default reward function for parallel training
default_reward_func = score_reward_func

# Dynamic reward function for adaptive scaling
dynamic_reward_func = dynamic_score_reward_func

# Object-oriented reward class implementations
class RewardFunction:
    """Base class for reward functions in 2048 Zero training"""

    def __init__(self, name: str):
        """Initialize reward function
        
        Args:
            name: Name of the reward function (for logging)
        """
        self.name = name
    
    def __call__(self, game_stats: Dict[str, Any]) -> float:
        """Calculate reward value from game statistics
        
        Args:
            game_stats: Dictionary of game statistics including 'score', 'max_tile', etc.
            
        Returns:
            float: Reward value in range [-1, 1]
        """
        raise NotImplementedError("Reward function must be implemented")
    
    def get_key_metric(self) -> str:
        """Get the key metric name used for this reward function
        
        Returns:
            str: Key metric name (e.g., 'score', 'max_tile')
        """
        raise NotImplementedError("Key metric must be specified")
    
    def __str__(self) -> str:
        return f"RewardFunction({self.name})"
    
    def to_func(self) -> Callable[[GameState, Dict[str, Any]], Tuple[float, str]]:
        """
        Convert this reward function to a function format for multiprocessing
        
        Returns:
            Function that takes state and stats, returning (reward_value, reward_name)
        """
        raise NotImplementedError("Conversion to function must be implemented")


class MaxTileReward(RewardFunction):
    """Reward based on maximum tile achieved in the game"""
    
    def __init__(self):
        super().__init__("max_tile")
    
    def __call__(self, game_stats: Dict[str, Any]) -> float:
        """Calculate reward based on max tile
        
        Args:
            game_stats: Game statistics
            
        Returns:
            float: Normalized reward in range [-1, 1]
        """
        max_tile = game_stats['max_tile']
        # Scale based on log2 of max tile / 11 (11 = log2(2048))
        # This maps 2 -> -0.82, 2048 -> 1.0
        return math.log2(int(max_tile) + 1) / 11.0 * 2 - 1
    
    def get_key_metric(self) -> str:
        return "max_tile"
    
    def to_func(self) -> Callable[[GameState, Dict[str, Any]], Tuple[float, str]]:
        """Convert to function format for multiprocessing"""
        return max_tile_reward_func


class ScoreReward(RewardFunction):
    """Reward based on score achieved in the game"""
    
    def __init__(self, max_score: int = 50000):
        """Initialize score-based reward
        
        Args:
            max_score: Maximum expected score for normalization
        """
        super().__init__("score")
        self.max_score = max_score
    
    def __call__(self, game_stats: Dict[str, Any]) -> float:
        """Calculate reward based on score
        
        Args:
            game_stats: Game statistics
            
        Returns:
            float: Normalized reward in range [-1, 1]
        """
        score = game_stats['score']
        # Normalize score to [-1, 1] range
        # We use a log scale to prevent very large scores from dominating
        normalized = math.log(score + 100) / math.log(self.max_score + 100) * 2 - 1
        return min(max(normalized, -1.0), 1.0)  # Clamp to [-1, 1]
    
    def get_key_metric(self) -> str:
        return "score"
    
    def to_func(self) -> Callable[[GameState, Dict[str, Any]], Tuple[float, str]]:
        """Convert to function format for multiprocessing"""
        return score_reward_func
        
        
class DynamicScoreReward(RewardFunction):
    """Reward based on score with dynamic maximum value scaling"""
    
    def __init__(self, initial_max_score: int = 50000):
        """Initialize dynamic score-based reward
        
        Args:
            initial_max_score: Initial maximum score for normalization
        """
        super().__init__("dynamic_score")
        self.initial_max_score = initial_max_score
        # Set the global max score to the initial value if not already higher
        global _global_max_score
        _global_max_score = max(_global_max_score, initial_max_score)
    
    def __call__(self, game_stats: Dict[str, Any]) -> float:
        """Calculate reward based on score with dynamic scaling
        
        Args:
            game_stats: Game statistics
            
        Returns:
            float: Normalized reward in range [-1, 1]
        """
        global _global_max_score
        score = game_stats['score']
        
        # Update the global maximum score if this score is higher
        if score > _global_max_score:
            # Use a smoothing factor to avoid sudden large changes
            _global_max_score = max(score, _global_max_score * 0.9 + score * 0.1)
        
        # Normalize score to [-1, 1] range using log scale with dynamic maximum
        normalized = math.log(score + 100) / math.log(_global_max_score + 100) * 2 - 1
        return min(max(normalized, -1.0), 1.0)  # Clamp to [-1, 1]
    
    def get_key_metric(self) -> str:
        return "score"
    
    def to_func(self) -> Callable[[GameState, Dict[str, Any]], Tuple[float, str]]:
        """Convert to function format for multiprocessing"""
        return dynamic_score_reward_func


class HybridReward(RewardFunction):
    """Hybrid reward function that combines multiple rewards"""
    
    def __init__(self, 
                reward_functions: Dict[str, RewardFunction], 
                weights: Dict[str, float],
                primary_metric: str = "score"):
        """Initialize hybrid reward
        
        Args:
            reward_functions: Dictionary of reward functions
            weights: Dictionary of weights for each reward function
            primary_metric: Primary metric for logging/evaluation
        """
        super().__init__("hybrid")
        self.reward_functions = reward_functions
        self.weights = weights
        self.primary_metric = primary_metric
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            self.weights = {k: v / total_weight for k, v in weights.items()}
    
    def __call__(self, game_stats: Dict[str, Any]) -> float:
        """Calculate combined reward
        
        Args:
            game_stats: Game statistics
            
        Returns:
            float: Combined reward in range [-1, 1]
        """
        reward = 0.0
        for name, func in self.reward_functions.items():
            if name in self.weights:
                reward += func(game_stats) * self.weights[name]
        return reward
    
    def get_key_metric(self) -> str:
        return self.primary_metric
    
    def to_func(self) -> Callable[[GameState, Dict[str, Any]], Tuple[float, str]]:
        """Convert to function format for multiprocessing"""
        return hybrid_reward_func


# Default reward function factory
def get_reward_function(reward_type: str = "score", **kwargs) -> RewardFunction:
    """Get reward function by type
    
    Args:
        reward_type: Type of reward function ('score', 'max_tile', 'dynamic_score', or 'hybrid')
        **kwargs: Additional arguments for the reward function
        
    Returns:
        RewardFunction instance
    """
    if reward_type == "score":
        max_score = kwargs.get("max_score", 50000)
        return ScoreReward(max_score=max_score)
    elif reward_type == "dynamic_score":
        initial_max_score = kwargs.get("initial_max_score", 50000)
        return DynamicScoreReward(initial_max_score=initial_max_score)
    elif reward_type == "max_tile":
        return MaxTileReward()
    elif reward_type == "hybrid":
        # Create hybrid reward with custom settings
        reward_functions = {}
        weights = {}
        
        # Score component (static or dynamic)
        use_dynamic = kwargs.get("use_dynamic_score", False)
        if "score_weight" in kwargs and kwargs["score_weight"] > 0:
            if use_dynamic:
                initial_max_score = kwargs.get("initial_max_score", 50000)
                reward_functions["score"] = DynamicScoreReward(initial_max_score=initial_max_score)
            else:
                max_score = kwargs.get("max_score", 50000)
                reward_functions["score"] = ScoreReward(max_score=max_score)
            weights["score"] = kwargs["score_weight"]
        
        # Max tile component
        if "max_tile_weight" in kwargs and kwargs["max_tile_weight"] > 0:
            reward_functions["max_tile"] = MaxTileReward()
            weights["max_tile"] = kwargs["max_tile_weight"]
        
        primary_metric = kwargs.get("primary_metric", "score")
        
        return HybridReward(reward_functions, weights, primary_metric)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


# Helper function to convert a reward function instance to a multiprocessing-compatible function
def get_reward_func(reward_type: str = "score") -> Callable[[GameState, Dict[str, Any]], Tuple[float, str]]:
    """
    Get a multiprocessing-compatible reward function by type
    
    Args:
        reward_type: Type of reward function ('score', 'dynamic_score', 'max_tile', or 'hybrid')
        
    Returns:
        Function with signature (state, stats) -> (reward_value, reward_name)
    """
    if reward_type == "score":
        return score_reward_func
    elif reward_type == "dynamic_score":
        return dynamic_score_reward_func
    elif reward_type == "max_tile":
        return max_tile_reward_func
    elif reward_type == "hybrid":
        return hybrid_reward_func
    elif reward_type == "default" or reward_type is None:
        return default_reward_func
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")