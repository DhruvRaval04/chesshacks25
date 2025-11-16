"""
Gym-compatible chess environment built on top of python-chess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import chess
import numpy as np
import gymnasium as gym
from gym import spaces

# python-chess exposes results as PGN-style strings (e.g., "1-0"), so we
# use Optional[str] for type hints instead of the non-existent chess.Result.

from .move_encoding import (
    MAX_MOVES,
    index_to_move,
    move_to_index,
    random_legal_move,
)
from . import opponents
from .position_eval import evaluation_delta_reward

RewardFn = Callable[
    [chess.Board, chess.Board, chess.Color, Optional[str]],
    float,
]
OpponentPolicy = Callable[[chess.Board], chess.Move]


@dataclass
class RewardConfig:
    win: float = 20.0
    loss: float = -20.0
    draw: float = 0.0
    illegal_move: float = -20.0
    repetition_penalty: float = -2.0
    reward_fn: Optional[RewardFn] = evaluation_delta_reward


class ChessEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        agent_color: Optional[chess.Color] = None,
        opponent_policy: Optional[OpponentPolicy] = None,
        reward_config: RewardConfig = RewardConfig(),
        max_moves: int = 128,
    ) -> None:
        super().__init__()
        self._randomize_agent_color = agent_color is None
        if agent_color is None:
            self.agent_color = self._sample_random_color()
        else:
            self.agent_color = agent_color
        self.opponent_policy = (
            opponent_policy or opponents.random_policy
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
        if self._randomize_agent_color:
            self.agent_color = self._sample_random_color()
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

        board_before = self.board.copy(stack=False)
        self.board.push(move)
        self._ply_count += 1
        board_after_agent = self.board.copy(stack=False)
        penalty, penalty_reasons = self._repetition_penalty(board_after_agent)
        if penalty_reasons:
            info["repetition_penalty"] = {
                "amount": penalty,
                "reasons": penalty_reasons,
            }

        result = self._current_result()
        if result is not None:
            reward = self._compute_reward(
                board_before,
                board_after_agent,
                self.agent_color,
                result,
            )
            reward += penalty
            observation = self._get_obs()
            info["legal_moves_mask"] = self._legal_moves_mask()
            return observation, reward, True, False, info

        self._opponent_move()
        result = self._current_result()
        terminated = result is not None
        truncated = self._ply_count >= self.max_moves
        reward = self._compute_reward(
            board_before,
            board_after_agent,
            self.agent_color,
            result if terminated else None,
        )
        reward += penalty
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

    def _current_result(self) -> Optional[str]:
        if not self.board.is_game_over():
            return None
        return self.board.result(claim_draw=True)

    def _result_reward(self, result: Optional[str]) -> float:
        if result is None:
            return 0.0
        if result == "1-0":
            return (
                self.reward_config.win
                if self.agent_color == chess.WHITE
                else self.reward_config.loss
            )
        if result == "0-1":
            return (
                self.reward_config.win
                if self.agent_color == chess.BLACK
                else self.reward_config.loss
            )
        return self.reward_config.draw

    def _compute_reward(
        self,
        board_before: chess.Board,
        board_after_agent: chess.Board,
        agent_color: chess.Color,
        result: Optional[str],
    ) -> float:
        dense_reward = 0.0
        if self.reward_config.reward_fn is not None:
            dense_reward = self.reward_config.reward_fn(
                board_before,
                board_after_agent,
                agent_color,
                result,
            )
        terminal_bonus = self._result_reward(result)
        return dense_reward + terminal_bonus

    def _legal_moves_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        for move in self.board.legal_moves:
            mask[move_to_index(move)] = 1.0
        return mask

    def _repetition_penalty(
        self, board_after_agent: chess.Board
    ) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        if self.reward_config.repetition_penalty == 0.0:
            return 0.0, reasons
        if board_after_agent.can_claim_threefold_repetition():
            reasons.append("threefold_repetition")
        if board_after_agent.can_claim_fifty_moves():
            reasons.append("fifty_moves")
        if not reasons:
            return 0.0, reasons
        return self.reward_config.repetition_penalty, reasons

    def _sample_random_color(self) -> chess.Color:
        return chess.WHITE if np.random.rand() < 0.5 else chess.BLACK

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
