"""
Reference opponent policies for `ChessEnv`.

You can pass any of these callables to `ChessEnv(opponent_policy=...)`
or use them as templates for more advanced heuristics/engines.
"""

from __future__ import annotations

from typing import Iterable

import chess

from .move_encoding import random_legal_move

# Simple material weights keyed by python-chess piece type constants.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 200,  # large number to prioritize checkmates.
}


def random_policy(board: chess.Board) -> chess.Move:
    """Uniform random over legal moves."""
    return random_legal_move(board)


def _capture_value(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move):
        return 0
    captured = board.piece_at(move.to_square)
    if captured is None:
        return 0
    return PIECE_VALUES.get(captured.piece_type, 0)


def _promotion_bonus(move: chess.Move) -> int:
    if move.promotion is None:
        return 0
    return PIECE_VALUES.get(move.promotion, 0)


def _post_move_context(board: chess.Board, move: chess.Move) -> tuple[bool, float]:
    board.push(move)
    is_mate = board.is_checkmate()
    mobility = board.legal_moves.count()
    board.pop()
    return is_mate, mobility / 100.0


def greedy_material_policy(board: chess.Board) -> chess.Move:
    """
    Heuristic opponent that prefers:
    1. Checkmates/checks
    2. Highest-value captures
    3. Promotions / mobility
    Falls back to random if multiple moves tie.
    """

    def score(move: chess.Move) -> float:
        base = _capture_value(board, move) + _promotion_bonus(move)
        is_mate, mobility = _post_move_context(board, move)
        if is_mate:
            return float("inf")
        return base + mobility

    legal: Iterable[chess.Move] = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves available.")
    best = max(legal, key=score)
    return best

