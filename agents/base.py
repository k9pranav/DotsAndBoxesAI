from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from engine import DotsAndBoxesState, Edge


@dataclass(frozen=True)
class AgentDecision:
    move: Edge
    evaluation: Optional[float] = None
    reason: Optional[str] = None


class Agent(Protocol):
    name: str

    def select_move(self, state: DotsAndBoxesState) -> AgentDecision:
        ...