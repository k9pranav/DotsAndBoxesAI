from __future__ import annotations

import random
from typing import List, Optional

from engine import DotsAndBoxesState, Edge
from agents.base import AgentDecision


class GreedyAgent:
    """
    Greedy baseline:
    1. Prefer moves that immediately complete the maximum number of boxes.
    2. If no scoring move exists, pick randomly among remaining legal moves.
    """

    def __init__(self, name: str = "GreedyAgent", seed: Optional[int] = None) -> None:
        self.name = name
        self._rng = random.Random(seed)

    def select_move(self, state: DotsAndBoxesState) -> AgentDecision:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")

        best_moves: List[Edge] = []
        best_score = -1

        for move in legal_moves:
            next_state = state.next_state(move)
            gained_boxes = (
                next_state.scores[state.current_player] - state.scores[state.current_player]
            )

            if gained_boxes > best_score:
                best_score = gained_boxes
                best_moves = [move]
            elif gained_boxes == best_score:
                best_moves.append(move)

        chosen = self._rng.choice(best_moves)

        if best_score > 0:
            return AgentDecision(
                move=chosen,
                evaluation=float(best_score),
                reason=f"immediate_capture_{best_score}",
            )

        return AgentDecision(
            move=chosen,
            evaluation=0.0,
            reason="no_immediate_capture",
        )