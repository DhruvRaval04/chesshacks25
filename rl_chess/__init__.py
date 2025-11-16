"""
RL chess utilities package.

This module exposes key helpers so training scripts can import from
`rl_chess` without referencing submodules directly.
"""

from .chess_env import ChessEnv, RewardConfig
from . import opponents
from .position_eval import (
    get_chess_evaluation,
    evaluation_delta_reward,
    evaluation_delta_with_hippo_reward,
    hippo_position_score,
    shutdown_engine,
)

__all__ = [
    "ChessEnv",
    "RewardConfig",
    "opponents",
    "get_chess_evaluation",
    "evaluation_delta_reward",
    "evaluation_delta_with_hippo_reward",
    "hippo_position_score",
    "shutdown_engine",
]
