from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from engine import DotsAndBoxesState, Edge
from agents.base import AgentDecision


@dataclass(frozen=True)
class AlphaBetaStats:
    nodes_expanded: int
    prunes: int
    depth_limit: int
    best_value: float


class AlphaBetaAgent:
    """
    Depth-limited alpha-beta pruning for Dots and Boxes.

    Important:
    - Does NOT alternate max/min by depth parity.
    - Uses child.current_player implicitly through recursion:
      if the same player keeps the turn after scoring, the next node may
      still be maximizing (or minimizing).
    - Evaluation is from a fixed root_player perspective.

    Optional move ordering:
    - immediate box-taking moves first
    - then remaining moves
    """

    def __init__(
        self,
        depth: int,
        name: str = "AlphaBetaAgent",
        seed: Optional[int] = None,
        use_move_ordering: bool = True,
    ) -> None:
        if depth <= 0:
            raise ValueError("depth must be a positive integer")

        self.name = name
        self.depth = depth
        self.use_move_ordering = use_move_ordering
        self._rng = random.Random(seed)

        self.last_stats: Optional[AlphaBetaStats] = None

    def select_move(self, state: DotsAndBoxesState) -> AgentDecision:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")

        root_player = state.current_player
        nodes_expanded = 1  # count root
        prunes = 0

        alpha = -math.inf
        beta = math.inf

        ordered_moves = self._ordered_moves(state, legal_moves)

        best_value = -math.inf
        best_moves: List[Edge] = []

        for move in ordered_moves:
            child = state.next_state(move)
            value, child_nodes, child_prunes = self._alphabeta(
                state=child,
                depth=self.depth - 1,
                root_player=root_player,
                alpha=alpha,
                beta=beta,
            )
            nodes_expanded += child_nodes
            prunes += child_prunes

            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)

            alpha = max(alpha, best_value)

        chosen_move = self._rng.choice(best_moves)

        self.last_stats = AlphaBetaStats(
            nodes_expanded=nodes_expanded,
            prunes=prunes,
            depth_limit=self.depth,
            best_value=best_value,
        )

        return AgentDecision(
            move=chosen_move,
            evaluation=float(best_value),
            reason=f"alphabeta_depth_{self.depth}",
        )

    def _alphabeta(
        self,
        state: DotsAndBoxesState,
        depth: int,
        root_player: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, int, int]:
        """
        Returns:
            (value, nodes_expanded_in_subtree, prunes_in_subtree)
        """
        nodes_expanded = 1
        prunes = 0

        if state.is_terminal() or depth == 0:
            return float(self._evaluate(state, root_player)), nodes_expanded, prunes

        legal_moves = state.legal_moves()
        if not legal_moves:
            return float(self._evaluate(state, root_player)), nodes_expanded, prunes

        ordered_moves = self._ordered_moves(state, legal_moves)

        if state.current_player == root_player:
            value = -math.inf
            for move in ordered_moves:
                child = state.next_state(move)
                child_value, child_nodes, child_prunes = self._alphabeta(
                    state=child,
                    depth=depth - 1,
                    root_player=root_player,
                    alpha=alpha,
                    beta=beta,
                )
                nodes_expanded += child_nodes
                prunes += child_prunes

                value = max(value, child_value)
                alpha = max(alpha, value)

                if alpha >= beta:
                    prunes += 1
                    break

            return value, nodes_expanded, prunes

        value = math.inf
        for move in ordered_moves:
            child = state.next_state(move)
            child_value, child_nodes, child_prunes = self._alphabeta(
                state=child,
                depth=depth - 1,
                root_player=root_player,
                alpha=alpha,
                beta=beta,
            )
            nodes_expanded += child_nodes
            prunes += child_prunes

            value = min(value, child_value)
            beta = min(beta, value)

            if alpha >= beta:
                prunes += 1
                break

        return value, nodes_expanded, prunes

    def _evaluate(self, state: DotsAndBoxesState, root_player: int) -> int:
        return state.scores[root_player] - state.scores[1 - root_player]

    def _ordered_moves(
        self,
        state: DotsAndBoxesState,
        legal_moves: List[Edge],
    ) -> List[Edge]:
        if not self.use_move_ordering or len(legal_moves) <= 1:
            return legal_moves

        scoring_moves: List[Edge] = []
        other_moves: List[Edge] = []

        current_player = state.current_player
        current_score = state.scores[current_player]

        for move in legal_moves:
            child = state.next_state(move)
            gained = child.scores[current_player] - current_score
            if gained > 0:
                scoring_moves.append(move)
            else:
                other_moves.append(move)

        # Mild randomness only within buckets to avoid deterministic tie artifacts.
        self._rng.shuffle(scoring_moves)
        self._rng.shuffle(other_moves)

        return scoring_moves + other_moves