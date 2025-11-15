"""
Gym-compatible chess environment built on top of python-chess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# python-chess exposes results as PGN-style strings (e.g., "1-0"), so we
# use Optional[str] for type hints instead of the non-existent chess.Result.

from .move_encoding import (
    MAX_MOVES,
    index_to_move,
    move_to_index,
    random_legal_move,
)
from . import opponents

RewardFn = Callable[[chess.Board, Optional[str]], float]
OpponentPolicy = Callable[[chess.Board], chess.Move]


@dataclass
class RewardConfig:
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    illegal_move: float = -1.0


class ChessEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        agent_color: chess.Color = chess.WHITE,
        opponent_policy: Optional[OpponentPolicy] = None,
        reward_config: RewardConfig = RewardConfig(),
        max_moves: int = 128,
    ) -> None:
        super().__init__()
        self.agent_color = agent_color
        self.opponent_policy = (
            opponent_policy or opponents.greedy_material_policy
        )
        self.reward_config = reward_config
        self.max_moves = max_moves

        # Board plus metadata: 12 planes (piece type * color) + scalar feats.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(773,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(MAX_MOVES)

        self.board = chess.Board()
        self._ply_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        super().reset(seed=seed)
        self.board.reset()
        self._ply_count = 0
        if self.agent_color == chess.BLACK:
            # Opponent starts
            self._opponent_move()
        observation = self._get_obs()
        info = {"legal_moves_mask": self._legal_moves_mask()}
        return observation, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action index"
        info: Dict = {}

        move = index_to_move(action)
        if move not in self.board.legal_moves:
            # Illegal move ends the episode.
            info["reason"] = "illegal_move"
            observation = self._get_obs()
            info["legal_moves_mask"] = self._legal_moves_mask()
            return (
                observation,
                self.reward_config.illegal_move,
                True,
                False,
                info,
            )

        self.board.push(move)
        self._ply_count += 1

        terminated, reward = self._terminal_reward()
        if terminated:
            observation = self._get_obs()
            info["legal_moves_mask"] = self._legal_moves_mask()
            return observation, reward, True, False, info

        self._opponent_move()
        terminated, reward = self._terminal_reward()

        truncated = self._ply_count >= self.max_moves
        observation = self._get_obs()
        info["legal_moves_mask"] = self._legal_moves_mask()
        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        if mode == "human":
            print(self.board)
        elif mode == "ansi":
            return str(self.board)
        else:
            raise NotImplementedError(f"Unsupported render mode {mode}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _opponent_move(self):
        if self.board.is_game_over():
            return
        move = self.opponent_policy(self.board)
        if move not in self.board.legal_moves:
            # Force a random legal move to keep the episode valid.
            move = random_legal_move(self.board)
        self.board.push(move)
        self._ply_count += 1

    def _terminal_reward(self) -> Tuple[bool, float]:
        if not self.board.is_game_over():
            return False, 0.0
        result = self.board.result(claim_draw=True)
        if result == "1-0":
            reward = (
                self.reward_config.win
                if self.agent_color == chess.WHITE
                else self.reward_config.loss
            )
            return True, reward
        if result == "0-1":
            reward = (
                self.reward_config.win
                if self.agent_color == chess.BLACK
                else self.reward_config.loss
            )
            return True, reward
        return True, self.reward_config.draw

    def _legal_moves_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        for move in self.board.legal_moves:
            mask[move_to_index(move)] = 1.0
        return mask

    def _get_obs(self) -> np.ndarray:
        planes = []
        for color in (chess.WHITE, chess.BLACK):
            for piece_type in (
                chess.PAWN,
                chess.KNIGHT,
                chess.BISHOP,
                chess.ROOK,
                chess.QUEEN,
                chess.KING,
            ):
                bitboard = self.board.pieces(piece_type, color)
                plane = np.zeros(64, dtype=np.float32)
                for square in chess.SquareSet(bitboard):
                    plane[square] = 1.0
                planes.append(plane)
        board_tensor = np.stack(planes, axis=0).astype(
            np.float32
        )  # Shape (12, 64)
        board_tensor = board_tensor.reshape(-1)

        extra = np.array(
            [
                1.0
                if self.board.turn == chess.WHITE
                else 0.0,
                1.0
                if self.board.has_kingside_castling_rights(
                    chess.WHITE
                )
                else 0.0,
                1.0
                if self.board.has_queenside_castling_rights(
                    chess.WHITE
                )
                else 0.0,
                1.0
                if self.board.has_kingside_castling_rights(
                    chess.BLACK
                )
                else 0.0,
                1.0
                if self.board.has_queenside_castling_rights(
                    chess.BLACK
                )
                else 0.0,
                min(self.board.halfmove_clock / 100.0, 1.0),
                min(self.board.fullmove_number / 200.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([board_tensor, extra])
