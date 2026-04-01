from __future__ import annotations

import random
from typing import Optional

from engine import DotsAndBoxesState
from agents.base import AgentDecision


class RandomAgent:
    def __init__(self, name: str = "RandomAgent", seed: Optional[int] = None) -> None:
        self.name = name
        self._rng = random.Random(seed)

    def select_move(self, state: DotsAndBoxesState) -> AgentDecision:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")
        move = self._rng.choice(legal_moves)
        return AgentDecision(move=move, reason="random_legal_move")