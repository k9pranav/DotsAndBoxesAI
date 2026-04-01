from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from engine import DotsAndBoxesState, Edge
from agents.base import AgentDecision
from agents.box_gnn_value_agent import BoxGNNValueAgent


@dataclass(frozen=True)
class AlphaBetaBoxGNNStats:
    depth: int
    nodes_expanded: int
    prunes: int
    best_value: float


class AlphaBetaBoxGNNValueAgent:
    """
    Shallow alpha-beta search with BoxGNN leaf evaluation.

    This is usually much stronger than one-ply value selection because
    search catches immediate tactical traps while the GNN evaluates leaves.
    """

    def __init__(
        self,
        checkpoint_path: str,
        depth: int = 2,
        name: str = "AlphaBetaBoxGNNValueAgent",
        device: Optional[str] = None,
    ) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.name = name
        self.depth = depth
        self.evaluator = BoxGNNValueAgent(
            checkpoint_path=checkpoint_path,
            name=f"{name}_LeafEvaluator",
            device=device,
        )
        self.last_stats: Optional[AlphaBetaBoxGNNStats] = None

    def select_move(self, state: DotsAndBoxesState) -> AgentDecision:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")

        root_player = state.current_player
        nodes_expanded = 1
        prunes = 0

        alpha = -math.inf
        beta = math.inf

        ordered_moves = self._ordered_moves(state)

        best_value = -math.inf
        best_move: Optional[Edge] = None

        for move in ordered_moves:
            child = state.next_state(move)
            value, child_nodes, child_prunes = self._alphabeta(
                child,
                depth=self.depth - 1,
                root_player=root_player,
                alpha=alpha,
                beta=beta,
            )
            nodes_expanded += child_nodes
            prunes += child_prunes

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)

        if best_move is None:
            raise RuntimeError("Failed to select a move.")

        self.last_stats = AlphaBetaBoxGNNStats(
            depth=self.depth,
            nodes_expanded=nodes_expanded,
            prunes=prunes,
            best_value=float(best_value),
        )

        return AgentDecision(
            move=best_move,
            evaluation=float(best_value),
            reason=f"alphabeta_box_gnn_depth_{self.depth}",
        )

    def _alphabeta(
        self,
        state: DotsAndBoxesState,
        depth: int,
        root_player: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, int, int]:
        nodes_expanded = 1
        prunes = 0

        if state.is_terminal() or depth == 0:
            value = self._evaluate_leaf(state, root_player)
            return value, nodes_expanded, prunes

        legal_moves = state.legal_moves()
        if not legal_moves:
            value = self._evaluate_leaf(state, root_player)
            return value, nodes_expanded, prunes

        ordered_moves = self._ordered_moves(state)

        if state.current_player == root_player:
            value = -math.inf
            for move in ordered_moves:
                child = state.next_state(move)
                child_value, child_nodes, child_prunes = self._alphabeta(
                    child, depth - 1, root_player, alpha, beta
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
                child, depth - 1, root_player, alpha, beta
            )
            nodes_expanded += child_nodes
            prunes += child_prunes

            value = min(value, child_value)
            beta = min(beta, value)
            if alpha >= beta:
                prunes += 1
                break
        return value, nodes_expanded, prunes

    def _evaluate_leaf(self, state: DotsAndBoxesState, root_player: int) -> float:
        # reuse the BoxGNNValueAgent's perspective handling
        return self.evaluator._predict_root_perspective_value(state, root_player)

    def _ordered_moves(self, state: DotsAndBoxesState) -> list[Edge]:
        """
        Cheap move ordering:
        - immediate scoring moves first
        - then others
        """
        scoring = []
        other = []

        current_player = state.current_player
        current_score = state.scores[current_player]

        for move in state.legal_moves():
            child = state.next_state(move)
            gain = child.scores[current_player] - current_score
            if gain > 0:
                scoring.append(move)
            else:
                other.append(move)

        return scoring + other