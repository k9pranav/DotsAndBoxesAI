from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from engine import DotsAndBoxesState, Edge
from agents.base import AgentDecision


@dataclass(frozen=True)
class MinimaxStats:
    nodes_expanded: int
    depth_limit: int
    best_value: float


class MinimaxAgent:
    """
    Plain depth-limited minimax for Dots and Boxes.

    Important:
    - This does NOT alternate max/min by depth parity.
    - Instead, after each move it checks child.current_player.
    - If the acting player completed a box, they keep the turn,
      so the recursion may remain on a maximizing node or a minimizing node.

    Evaluation (first version):
        score(root_player) - score(other_player)
    """

    def __init__(self, depth: int, name: str = "MinimaxAgent", seed: Optional[int] = None) -> None:
        if depth <= 0:
            raise ValueError("depth must be a positive integer")

        self.name = name
        self.depth = depth
        self._rng = random.Random(seed)

        # Updated on each select_move call
        self.last_stats: Optional[MinimaxStats] = None

    def select_move(self, state: DotsAndBoxesState) -> AgentDecision:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")

        root_player = state.current_player
        nodes_expanded = 0

        best_value = -math.inf
        best_moves: List[Edge] = []

        for move in legal_moves:
            child = state.next_state(move)
            value, child_nodes = self._minimax(
                state=child,
                depth=self.depth - 1,
                root_player=root_player,
            )
            nodes_expanded += child_nodes

            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)

        chosen_move = self._rng.choice(best_moves)

        # Count the root as expanded too
        nodes_expanded += 1
        self.last_stats = MinimaxStats(
            nodes_expanded=nodes_expanded,
            depth_limit=self.depth,
            best_value=best_value,
        )

        return AgentDecision(
            move=chosen_move,
            evaluation=float(best_value),
            reason=f"minimax_depth_{self.depth}",
        )

    def _minimax(self, state: DotsAndBoxesState, depth: int, root_player: int) -> tuple[float, int]:
        """
        Returns:
            (value, nodes_expanded_in_this_subtree)
        """
        # Count this node
        nodes_expanded = 1

        if state.is_terminal() or depth == 0:
            return float(self._evaluate(state, root_player)), nodes_expanded

        legal_moves = state.legal_moves()
        if not legal_moves:
            # Should not normally happen before terminal, but safe fallback
            return float(self._evaluate(state, root_player)), nodes_expanded

        if state.current_player == root_player:
            best_value = -math.inf
            for move in legal_moves:
                child = state.next_state(move)
                value, child_nodes = self._minimax(
                    state=child,
                    depth=depth - 1,
                    root_player=root_player,
                )
                nodes_expanded += child_nodes
                if value > best_value:
                    best_value = value
            return best_value, nodes_expanded

        best_value = math.inf
        for move in legal_moves:
            child = state.next_state(move)
            value, child_nodes = self._minimax(
                state=child,
                depth=depth - 1,
                root_player=root_player,
            )
            nodes_expanded += child_nodes
            if value < best_value:
                best_value = value
        return best_value, nodes_expanded

    def _evaluate(self, state: DotsAndBoxesState, root_player: int) -> int:
        """
        First version heuristic:
            score(root_player) - score(opponent)
        """
        return state.scores[root_player] - state.scores[1 - root_player]