from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from engine import DotsAndBoxesState, Edge, MoveResult
from agents import Agent, AgentDecision


@dataclass(frozen=True)
class MoveRecord:
    move_number: int
    acting_player: int
    agent_name: str
    move: Edge
    completed_boxes: List[tuple[int, int]]
    scores_after: tuple[int, int]
    next_player: int
    move_time_seconds: float
    decision_reason: Optional[str] = None
    decision_evaluation: Optional[float] = None
    game_over: bool = False


@dataclass
class GameSummary:
    final_state: DotsAndBoxesState
    winner: Optional[int]
    result_string: str
    move_history: List[MoveRecord] = field(default_factory=list)

    @property
    def final_scores(self) -> tuple[int, int]:
        return (self.final_state.scores[0], self.final_state.scores[1])


class GameRunner:
    """
    Orchestrates a single Dots and Boxes game between two controllers.

    player0 controls state.current_player == 0
    player1 controls state.current_player == 1
    """

    def __init__(
        self,
        initial_state: DotsAndBoxesState,
        player0: Agent,
        player1: Agent,
    ) -> None:
        self.state = initial_state
        self.player0 = player0
        self.player1 = player1
        self.move_history: List[MoveRecord] = []
        self.move_counter = 0

    def current_agent(self) -> Agent:
        return self.player0 if self.state.current_player == 0 else self.player1

    def step(self) -> MoveRecord:
        if self.state.is_terminal():
            raise ValueError("Game is already over.")

        agent = self.current_agent()
        acting_player = self.state.current_player

        start = time.perf_counter()
        decision: AgentDecision = agent.select_move(self.state.clone())
        elapsed = time.perf_counter() - start

        if not self.state.is_legal_move(decision.move):
            raise ValueError(
                f"{agent.name} returned illegal move {decision.move} "
                f"for player {acting_player}."
            )

        result: MoveResult = self.state.apply_move(decision.move)
        self.move_counter += 1

        record = MoveRecord(
            move_number=self.move_counter,
            acting_player=acting_player,
            agent_name=agent.name,
            move=decision.move,
            completed_boxes=result.completed_boxes,
            scores_after=(self.state.scores[0], self.state.scores[1]),
            next_player=result.next_player,
            move_time_seconds=elapsed,
            decision_reason=decision.reason,
            decision_evaluation=decision.evaluation,
            game_over=result.game_over,
        )
        self.move_history.append(record)
        return record

    def play(self, verbose: bool = False) -> GameSummary:
        while not self.state.is_terminal():
            record = self.step()
            if verbose:
                print(
                    f"[Move {record.move_number}] "
                    f"Player {record.acting_player} ({record.agent_name}) -> {record.move} | "
                    f"boxes={record.completed_boxes} | "
                    f"scores={record.scores_after} | "
                    f"next={record.next_player}"
                )
                print(self.state)
                print("-" * 60)

        return GameSummary(
            final_state=self.state,
            winner=self.state.winner(),
            result_string=self.state.result_string(),
            move_history=self.move_history[:],
        )