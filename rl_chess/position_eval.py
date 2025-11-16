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
# Post-analysis shaping for PPO dense rewards.
EVAL_DELTA_CLAMP = 3.0
EVAL_DELTA_SCALE = 4.0

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


def hippo_position_score(board: chess.Board, agent_color: chess.Color) -> float:
    """
    Evaluates how well the position matches hippo defense characteristics.
    
    The hippo defense is a solid, passive setup where:
    - For Black: pawns on e6, d6, g6 (or e7, d7, g7)
    - For White: pawns on e3, d3, g3 (or e2, d2, g2)
    - Pieces are developed behind the pawns
    - The structure is compact and defensive
    
    Args:
        board: The chess board position
        agent_color: The color of the agent
        
    Returns:
        float: Score from 0.0 to 1.0 indicating how well the position matches
               hippo characteristics. Higher is better.
    """
    score = 0.0
    max_score = 0.0
    
    # Define hippo squares based on color
    if agent_color == chess.BLACK:
        # Black hippo: pawns on e6, d6, g6 (or e7, d7, g7)
        ideal_pawn_squares = [
            chess.E6, chess.D6, chess.G6,  # Preferred
            chess.E7, chess.D7, chess.G7,  # Early game alternative
        ]
        # Pieces should be behind pawns (ranks 7-8)
        piece_ranks = [6, 7]
        # Central pawns (d, e) are most important
        central_pawns = [chess.D6, chess.E6, chess.D7, chess.E7]
    else:
        # White hippo: pawns on e3, d3, g3 (or e2, d2, g2)
        ideal_pawn_squares = [
            chess.E3, chess.D3, chess.G3,  # Preferred
            chess.E2, chess.D2, chess.G2,  # Early game alternative
        ]
        # Pieces should be behind pawns (ranks 1-2)
        piece_ranks = [1, 2]
        # Central pawns (d, e) are most important
        central_pawns = [chess.D3, chess.E3, chess.D2, chess.E2]
    
    # Check for hippo pawn structure
    pawns = board.pieces(chess.PAWN, agent_color)
    hippo_pawns = 0
    central_hippo_pawns = 0
    
    for square in ideal_pawn_squares:
        if square in pawns:
            hippo_pawns += 1
            if square in central_pawns:
                central_hippo_pawns += 1
    
    # Reward having the key pawns in hippo positions
    # Central pawns are more important
    if central_hippo_pawns >= 2:
        score += 0.4  # Both central pawns in position
    elif central_hippo_pawns == 1:
        score += 0.2  # One central pawn in position
    
    if hippo_pawns >= 3:
        score += 0.3  # All three key pawns in hippo positions
    elif hippo_pawns == 2:
        score += 0.15  # Two of three key pawns
    
    max_score += 0.7
    
    # Reward having pieces behind the pawns (not too advanced)
    piece_types = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    pieces_behind = 0
    total_pieces = 0
    
    for piece_type in piece_types:
        pieces = board.pieces(piece_type, agent_color)
        for square in pieces:
            total_pieces += 1
            rank = chess.square_rank(square)
            if rank in piece_ranks:
                pieces_behind += 1
    
    if total_pieces > 0:
        behind_ratio = pieces_behind / total_pieces
        score += behind_ratio * 0.2  # Reward keeping pieces back
        max_score += 0.2
    
    # Reward compact structure (pawns not too far apart)
    # Check if pawns are in a compact formation
    if hippo_pawns >= 2:
        # If we have hippo pawns, check if they're not isolated
        pawn_squares = [sq for sq in ideal_pawn_squares if sq in pawns]
        if len(pawn_squares) >= 2:
            # Check if pawns are connected or close
            files = [chess.square_file(sq) for sq in pawn_squares]
            file_spread = max(files) - min(files)
            if file_spread <= 3:  # Pawns within 3 files of each other
                score += 0.1
                max_score += 0.1
    
    # Normalize to 0-1 range
    if max_score > 0:
        return min(1.0, score / max_score)
    return 0.0


def evaluation_delta_reward(
    board_before: chess.Board,
    board_after: chess.Board,
    agent_color: chess.Color,
    _: Optional[str] = None,
    *,
    scale: float = EVAL_DELTA_SCALE,
    delta_clamp: float = EVAL_DELTA_CLAMP,
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
        float: The evaluation delta (after - before), clipped to
        ``[-delta_clamp, delta_clamp]`` and scaled by ``scale``.
        Returns 0.0 if either evaluation request fails.
    """
    initial_eval = get_chess_evaluation(board_before.fen())
    current_eval = get_chess_evaluation(board_after.fen())
    if initial_eval is None or current_eval is None:
        return 0.0
    delta = current_eval - initial_eval
    if agent_color == chess.BLACK:
        delta = -delta
    delta = max(-delta_clamp, min(delta_clamp, delta))
    return delta * scale


def evaluation_delta_with_hippo_reward(
    board_before: chess.Board,
    board_after: chess.Board,
    agent_color: chess.Color,
    _: Optional[str] = None,
    hippo_bonus_weight: float = 0.3,
) -> float:
    """
    Computes a reward combining evaluation delta with hippo defense bonuses.
    
    This reward function encourages the agent to:
    1. Improve position evaluation (standard reward)
    2. Maintain hippo-style defensive structures (style bonus)
    
    Args:
        board_before: Board state prior to the agent's move.
        board_after: Board right after the agent's move.
        agent_color: The color of the agent.
        _: Placeholder for compatibility.
        hippo_bonus_weight: Weight for hippo bonus (0.0 to 1.0).
                           Higher values emphasize hippo style more.
    
    Returns:
        float: Combined reward (evaluation delta + hippo bonus).
    """
    # Standard evaluation delta reward
    eval_reward = evaluation_delta_reward(board_before, board_after, agent_color, _)
    
    # Hippo position bonus
    hippo_score_after = hippo_position_score(board_after, agent_color)
    hippo_score_before = hippo_position_score(board_before, agent_color)
    
    # Reward improving or maintaining hippo structure
    hippo_delta = hippo_score_after - hippo_score_before
    hippo_bonus = hippo_delta * hippo_bonus_weight * 2.0  # Scale to be meaningful
    
    # Also give a small bonus for maintaining a good hippo structure
    if hippo_score_after > 0.5:
        hippo_bonus += 0.1 * hippo_bonus_weight * hippo_score_after
    
    return eval_reward + hippo_bonus
