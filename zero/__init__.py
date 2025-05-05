# Zero library for 2048 training
from .game import GameRules, GameState, BitBoard
from .zero2048 import ZeroMCTSNode, ZeroPlayer, ZeroTrainer
from .reward_functions import get_reward_function, RewardFunction, ScoreReward, MaxTileReward, HybridReward
from .zeromonitoring import ZeroMonitor, EpochStats, print_epoch_summary

try:
    from .zeromodel import ZeroNetwork
    has_zeromodel = True
except ImportError:
    has_zeromodel = False

__all__ = [
    "GameRules",
    "GameState",
    "BitBoard",

    "ZeroMCTSNode",
    "ZeroPlayer",
    "ZeroTrainer",

    "get_reward_function",
    "RewardFunction",
    "ScoreReward", 
    "MaxTileReward",
    "HybridReward",

    "ZeroMonitor",
    "EpochStats",
    "print_epoch_summary"
]

if has_zeromodel:
    __all__.append("ZeroNetwork")