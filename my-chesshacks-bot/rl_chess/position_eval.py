import atexit
import os
from typing import Optional

import chess
import chess.engine

DEFAULT_STOCKFISH_PATHS = [
    os.environ.get("STOCKFISH_PATH"),
    "/opt/homebrew/Cellar/stockfish/17.1/bin/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/games/stockfish",
]
DEFAULT_ANALYSIS_DEPTH = 12
MATE_SCORE = 1000
POSITIVE_EVAL_MULTIPLIER = 1.5
EVAL_CLAMP = 10.0

_ENGINE: Optional[chess.engine.SimpleEngine] = None


def _resolve_engine_path() -> str:
    for candidate in DEFAULT_STOCKFISH_PATHS:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Stockfish binary not found. Install via `brew install stockfish` "
        "or set STOCKFISH_PATH to the executable."
    )


def _close_engine():
    global _ENGINE
    if _ENGINE is not None:
        try:
            _ENGINE.close()
        except chess.engine.EngineTerminatedError:
            pass
        _ENGINE = None


atexit.register(_close_engine)


def shutdown_engine():
    """Public helper to stop the cached Stockfish process."""
    _close_engine()


def _get_engine() -> chess.engine.SimpleEngine:
    global _ENGINE
    if _ENGINE is None:
        engine_path = _resolve_engine_path()
        _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    return _ENGINE


def get_chess_evaluation(fen_string: str, depth: int = DEFAULT_ANALYSIS_DEPTH):
    """
    Analyzes a chess position using a local Stockfish engine.

    Args:
        fen_string (str): FEN string representing the board state.
        depth (int): Search depth for the engine.

    Returns:
        float: Evaluation in pawns from White's POV (negative favors Black),
               clipped to [-10, 10].
    """
    board = chess.Board(fen_string)
    engine = _get_engine()
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
    except chess.engine.EngineTerminatedError:
        _close_engine()
        engine = _get_engine()
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
    except chess.engine.EngineError as exc:
        print(f"[stockfish] engine failure: {exc}")
        return 0.0

    score = info["score"].pov(chess.WHITE).score(mate_score=MATE_SCORE)
    if score is None:
        return 0.0
    pawns = score / 100.0  # convert centipawns
    if pawns > 0:
        pawns *= POSITIVE_EVAL_MULTIPLIER
    pawns = max(-EVAL_CLAMP, min(EVAL_CLAMP, pawns))
    return pawns


def evaluation_delta_reward(
    board_before: chess.Board,
    board_after: chess.Board,
    agent_color: chess.Color,
    _: Optional[str] = None,
) -> float:
    """
    Computes a dense reward based on the difference in engine evaluations
    before and after the agent's move.

    Args:
        board_before (chess.Board): Board state prior to the agent's move.
        board_after (chess.Board): Board right after the agent's move.
        _ (Optional[str]): Placeholder for compatibility with the reward_fn
            signature.

    Returns:
        float: The evaluation delta (after - before). Returns 0.0 if either
        evaluation request fails.
    """
    initial_eval = get_chess_evaluation(board_before.fen())
    current_eval = get_chess_evaluation(board_after.fen())
    if initial_eval is None or current_eval is None:
        return 0.0
    delta = current_eval - initial_eval
    if agent_color == chess.BLACK:
        delta = -delta
    return delta
