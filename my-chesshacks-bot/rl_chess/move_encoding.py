"""
Helpers for mapping between integer action ids and python-chess Move objects.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Dict, List

import chess

PROMOTION_PIECES = ("q", "r", "b", "n")


@lru_cache(maxsize=1)
def all_moves() -> List[chess.Move]:
    """Return a deterministic list of pseudo-legal moves accepted by python-chess."""
    moves = []
    squares = [chess.square_name(sq) for sq in chess.SQUARES]
    for from_sq in squares:
        for to_sq in squares:
            if from_sq == to_sq:
                continue
            uci = f"{from_sq}{to_sq}"
            try:
                moves.append(chess.Move.from_uci(uci))
            except ValueError:
                continue
            for promo in PROMOTION_PIECES:
                try:
                    moves.append(chess.Move.from_uci(f"{uci}{promo}"))
                except ValueError:
                    continue
    # Sort to ensure deterministic ordering once duplicates removed
    unique_moves = sorted({move.uci(): move for move in moves}.items())
    return [move for _, move in unique_moves]


@lru_cache(maxsize=1)
def move_lookup() -> Dict[chess.Move, int]:
    """Map Move -> index."""
    return {move: idx for idx, move in enumerate(all_moves())}


def move_to_index(move: chess.Move) -> int:
    return move_lookup()[move]


def index_to_move(index: int) -> chess.Move:
    return all_moves()[index]


def random_legal_move(board: chess.Board) -> chess.Move:
    """Sample a legal move uniformly from the board."""
    return random.choice(list(board.legal_moves))


MAX_MOVES = len(all_moves())

