from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Optional

from engine import DotsAndBoxesState
from runner import GameRunner
from agents import (
    RandomAgent,
    GreedyAgent,
    MinimaxAgent,
    AlphaBetaAgent,
    MCTSAgent,
    NNPolicyAgent,
    NNValueAgent,
    BoxGNNValueAgent,
    AlphaBetaBoxGNNValueAgent,

)

@dataclass
class BenchmarkResult:
    matchup_name: str
    rows: int
    cols: int
    n_games: int
    player0_wins: int
    player1_wins: int
    draws: int
    avg_score_p0: float
    avg_score_p1: float
    avg_score_diff_p0_minus_p1:float
    avg_total_moves:float
    avg_move_time_seconds: float

def run_match_batch(
        rows: int,
        cols: int,
        n_games: int,
        player0_factory: Callable[[int], object],
        player1_factory: Callable[[int], object],
        matchup_name: str,
        verbose: bool = False
) -> BenchmarkResult:
    player0_wins = 0
    player1_wins = 0
    draws = 0

    score_p0_list = []
    score_p1_list = []
    move_counts = []
    move_times = []

    for game_idx in range(n_games):
        state = DotsAndBoxesState(rows=rows, cols=cols)
        player0 = player0_factory(game_idx)
        player1 = player1_factory(game_idx)

        runner = GameRunner(
            initial_state=state,
            player0=player0,
            player1=player1,
        )
        summary = runner.play(verbose=False)

        p0_score, p1_score = summary.final_scores
        score_p0_list.append(p0_score)
        score_p1_list.append(p1_score)
        move_counts.append(len(summary.move_history))

        for rec in summary.move_history:
            move_times.append(rec.move_time_seconds)

        if summary.winner == 0:
            player0_wins += 1
        elif summary.winner == 1:
            player1_wins += 1
        else:
            draws += 1

        if verbose:
            print(
                f"[Game {game_idx + 1}/{n_games}] "
                f"winner={summary.winner}, scores={summary.final_scores}, "
                f"moves={len(summary.move_history)}"
            )

    return BenchmarkResult(
        matchup_name=matchup_name,
        rows=rows,
        cols=cols,
        n_games=n_games,
        player0_wins=player0_wins,
        player1_wins=player1_wins,
        draws=draws,
        avg_score_p0=statistics.mean(score_p0_list),
        avg_score_p1=statistics.mean(score_p1_list),
        avg_score_diff_p0_minus_p1=statistics.mean(
            [a - b for a, b in zip(score_p0_list, score_p1_list)]
        ),
        avg_total_moves=statistics.mean(move_counts),
        avg_move_time_seconds=statistics.mean(move_times) if move_times else 0.0,
    )

def print_result(result: BenchmarkResult) -> None:
    print("=" * 72)
    print(f"Matchup: {result.matchup_name}")
    print(f"Board: {result.rows}x{result.cols}")
    print(f"Games: {result.n_games}")
    print(f"Player 0 wins: {result.player0_wins}")
    print(f"Player 1 wins: {result.player1_wins}")
    print(f"Draws: {result.draws}")
    print(f"Avg score P0: {result.avg_score_p0:.3f}")
    print(f"Avg score P1: {result.avg_score_p1:.3f}")
    print(f"Avg score diff (P0-P1): {result.avg_score_diff_p0_minus_p1:.3f}")
    print(f"Avg total moves: {result.avg_total_moves:.3f}")
    print(f"Avg move time (seconds): {result.avg_move_time_seconds:.6f}")
    print("=" * 72)
    print()


def main() -> None:
    configs = [
    (
    "AB-BoxGNN(d2, 5x5 ckpt) vs Random [5x5]",
    3,
    3,
    5,
    lambda seed: AlphaBetaBoxGNNValueAgent(
        checkpoint_path="checkpoints/box_gnn_value_5x5_searchvalue.pt",
        depth=2,
        name="AB_BoxGNN_d2_5x5ckpt",
    ),
    lambda seed: RandomAgent(name="Random", seed=1000 + seed),
),
(
    "AB-BoxGNN(d2, 5x5 ckpt) vs Greedy [5x5]",
    3,
    3,
    5,
    lambda seed: AlphaBetaBoxGNNValueAgent(
        checkpoint_path="checkpoints/box_gnn_value_5x5_searchvalue.pt",
        depth=2,
        name="AB_BoxGNN_d2_5x5ckpt",
    ),
    lambda seed: GreedyAgent(name="Greedy", seed=2000 + seed),
),
(
    "AlphaBeta(d=2) vs AB-BoxGNN(d2, 5x5 ckpt) [5x5]",
    3,
    3,
    20,
    lambda seed: AlphaBetaAgent(depth=2 , seed=3000 + seed),
    lambda seed: AlphaBetaBoxGNNValueAgent(
        checkpoint_path="checkpoints/box_gnn_value_5x5_searchvalue.pt",
        depth=2,
        name="AB_BoxGNN_d2_5x5ckpt",
    ),
),
]

    for matchup_name, rows, cols, n_games, p0_factory, p1_factory in configs:
        result = run_match_batch(
            rows=rows,
            cols=cols, 
            n_games=n_games,
            player0_factory=p0_factory,
            player1_factory=p1_factory,
            matchup_name=matchup_name,
            verbose=False,
        )
        print_result(result)

if __name__ == "__main__":
    main()


        



    