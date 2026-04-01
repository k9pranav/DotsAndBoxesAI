from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from engine import DotsAndBoxesState, Edge
from agents.base import AgentDecision


@dataclass(frozen=True)
class MCTSStats:
    simulations: int
    nodes_created: int
    best_move_visits: int
    best_move_value: float


class _MCTSNode:
    def __init__(
        self,
        state: DotsAndBoxesState,
        parent: Optional["_MCTSNode"] = None,
        move_from_parent: Optional[Edge] = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.move_from_parent = move_from_parent

        self.children: dict[Edge, _MCTSNode] = {}
        self.untried_moves: list[Edge] = state.legal_moves()

        self.visits: int = 0
        self.total_value: float = 0.0

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


class MCTSAgent:
    """
    Basic Monte Carlo Tree Search for Dots and Boxes.

    Design choices in this first version:
    - Fixed number of simulations
    - UCT tree policy
    - Random rollout policy by default
    - Value is measured from the root player's perspective
    - Final reward = final score difference for root player

    Important:
    - Uses the engine's current_player directly
    - No assumption that turns alternate every move
    """

    def __init__(
        self,
        num_simulations: int = 500,
        exploration_constant: float = math.sqrt(2.0),
        name: str = "MCTSAgent",
        seed: Optional[int] = None,
        rollout_policy: str = "random",
    ) -> None:
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if exploration_constant <= 0:
            raise ValueError("exploration_constant must be positive")
        if rollout_policy not in {"random", "greedy"}:
            raise ValueError("rollout_policy must be 'random' or 'greedy'")

        self.name = name
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.rollout_policy = rollout_policy
        self._rng = random.Random(seed)

        self.last_stats: Optional[MCTSStats] = None

    def select_move(self, state: DotsAndBoxesState) -> AgentDecision:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")

        root_player = state.current_player
        root = _MCTSNode(state=state.clone())
        nodes_created = 1

        for _ in range(self.num_simulations):
            node = root

            # 1. Selection
            while (
                not node.is_terminal()
                and node.is_fully_expanded()
                and len(node.children) > 0
            ):
                node = self._select_child(node)

            # 2. Expansion
            if not node.is_terminal() and node.untried_moves:
                move = self._rng.choice(node.untried_moves)
                node.untried_moves.remove(move)

                child_state = node.state.next_state(move)
                child = _MCTSNode(
                    state=child_state,
                    parent=node,
                    move_from_parent=move,
                )
                node.children[move] = child
                node = child
                nodes_created += 1

            # 3. Simulation / rollout
            reward = self._rollout(node.state.clone(), root_player)

            # 4. Backpropagation
            self._backpropagate(node, reward)

        # Choose move with highest visit count
        best_move = None
        best_child = None

        for move, child in root.children.items():
            if best_child is None or child.visits > best_child.visits:
                best_move = move
                best_child = child

        # Fallback safety
        if best_move is None or best_child is None:
            best_move = self._rng.choice(legal_moves)
            best_value = 0.0
            best_visits = 0
        else:
            best_value = best_child.mean_value()
            best_visits = best_child.visits

        self.last_stats = MCTSStats(
            simulations=self.num_simulations,
            nodes_created=nodes_created,
            best_move_visits=best_visits,
            best_move_value=best_value,
        )

        return AgentDecision(
            move=best_move,
            evaluation=best_value,
            reason=f"mcts_{self.num_simulations}",
        )

    def _select_child(self, node: _MCTSNode) -> _MCTSNode:
        assert node.visits > 0, "UCT selection requires parent visits > 0"

        best_score = -math.inf
        best_child = None

        for child in node.children.values():
            if child.visits == 0:
                uct_score = math.inf
            else:
                exploitation = child.total_value / child.visits
                exploration = self.exploration_constant * math.sqrt(
                    math.log(node.visits) / child.visits
                )
                uct_score = exploitation + exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        if best_child is None:
            raise RuntimeError("No child found during UCT selection")

        return best_child

    def _rollout(self, state: DotsAndBoxesState, root_player: int) -> float:
        while not state.is_terminal():
            move = self._choose_rollout_move(state)
            state.apply_move(move)

        return float(self._terminal_reward(state, root_player))

    def _choose_rollout_move(self, state: DotsAndBoxesState) -> Edge:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available during rollout")

        if self.rollout_policy == "random":
            return self._rng.choice(legal_moves)

        # Greedy rollout: prefer immediate captures
        current_player = state.current_player
        current_score = state.scores[current_player]

        best_moves: list[Edge] = []
        best_gain = -1

        for move in legal_moves:
            child = state.next_state(move)
            gain = child.scores[current_player] - current_score

            if gain > best_gain:
                best_gain = gain
                best_moves = [move]
            elif gain == best_gain:
                best_moves.append(move)

        return self._rng.choice(best_moves)

    def _terminal_reward(self, terminal_state: DotsAndBoxesState, root_player: int) -> int:
        return (
            terminal_state.scores[root_player]
            - terminal_state.scores[1 - root_player]
        )

    def _backpropagate(self, node: _MCTSNode, reward: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += reward
            current = current.parent