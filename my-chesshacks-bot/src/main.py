from __future__ import annotations

from pathlib import Path
import os
import random

import chess
import torch
from chess import Move

from rl_chess.models import PolicyValueNet
from rl_chess.move_encoding import MAX_MOVES, move_to_index
from .utils import chess_manager, GameContext


MODEL_ID = os.getenv("CHESSHACKS_MODEL_ID", "draval/chesshacks")
CACHE_DIR = Path(
    os.getenv("CHESSHACKS_MODEL_CACHE", "./.model_cache")
).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_policy: PolicyValueNet | None = None


def _load_policy() -> PolicyValueNet | None:
    global _policy
    if _policy is not None:
        return _policy

    if not MODEL_ID:
        print(
            "[policy] MODEL_ID environment variable is empty; "
            "skipping load."
        )
        return None

    try:
        model = (
            PolicyValueNet.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
            .to(DEVICE)
            .eval()
        )
        _policy = model
        print(f"[policy] Loaded '{MODEL_ID}' onto {DEVICE}.")
    except Exception as exc:  # noqa: BLE001
        print(f"[policy] Failed to load '{MODEL_ID}': {exc}")

    return _policy


# Load once when the module imports so the first move is fast.
_load_policy()


PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def _board_to_feature_tensor(board: chess.Board) -> torch.Tensor:
    features: list[float] = []
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in PIECE_TYPES:
            plane = [0.0] * 64
            for square in chess.SquareSet(board.pieces(piece_type, color)):
                plane[square] = 1.0
            features.extend(plane)

    features.extend(
        [
            1.0 if board.turn == chess.WHITE else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
            min(board.halfmove_clock / 100.0, 1.0),
            min(board.fullmove_number / 200.0, 1.0),
        ]
    )

    tensor = torch.tensor(features, dtype=torch.float32, device=DEVICE)
    return tensor.unsqueeze(0)


def _legal_moves_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(MAX_MOVES, dtype=torch.float32, device=DEVICE)
    for move in board.legal_moves:
        mask[move_to_index(move)] = 1.0
    return mask.unsqueeze(0)


def _normalize_probabilities(probs: dict[Move, float]) -> dict[Move, float]:
    total = sum(probs.values())
    if total <= 0:
        uniform = 1.0 / len(probs) if probs else 0.0
        return {move: uniform for move in probs}
    return {move: prob / total for move, prob in probs.items()}


# def _random_move(ctx: GameContext, legal_moves: list[Move]) -> Move:
#     move_probs = _normalize_probabilities({move: 1.0 for move in legal_moves})
#     ctx.logProbabilities(move_probs)
#     return random.choice(legal_moves)


def _model_move(ctx: GameContext, legal_moves: list[Move]) -> Move:
    model = _load_policy()

    obs = _board_to_feature_tensor(ctx.board)
    mask = _legal_moves_mask(ctx.board)

    with torch.no_grad():
        logits, _ = model(obs, mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()

    move_probs: dict[Move, float] = {}
    for move in legal_moves:
        move_idx = move_to_index(move)
        move_probs[move] = float(probs[move_idx].item())

    move_probs = _normalize_probabilities(move_probs)
    ctx.logProbabilities(move_probs)

    best_move = max(move_probs.items(), key=lambda item: item[1])[0]
    return best_move


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    legal_moves = list(ctx.board.legal_moves)
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available.")

    try:
        return _model_move(ctx, legal_moves)
    except Exception as exc:  # noqa: BLE001
        print(
            "[policy] Falling back to random move due to inference error:"
            f" {exc}"
        )


@chess_manager.reset
def reset_func(ctx: GameContext):
    # Warm the policy on reset so the next move is ready.
    _load_policy()
