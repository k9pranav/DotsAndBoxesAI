from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any

from engine import DotsAndBoxesState
from agents import RandomAgent, GreedyAgent, MinimaxAgent, AlphaBetaAgent, MCTSAgent
from ml.state_transforms import serialize_state


# ------------------------------------------------------------
# Teacher / rollout agents
# ------------------------------------------------------------

def make_rollout_agent(bot_name: str, seed: int):
    bot_name = bot_name.lower()

    if bot_name == "random":
        return RandomAgent(name="RandomBot", seed=seed)
    if bot_name == "greedy":
        return GreedyAgent(name="GreedyBot", seed=seed)
    if bot_name == "minimax_d2":
        return MinimaxAgent(depth=2, name="Minimax_d2", seed=seed)
    if bot_name == "minimax_d3":
        return MinimaxAgent(depth=3, name="Minimax_d3", seed=seed)
    if bot_name == "alphabeta_d2":
        return AlphaBetaAgent(depth=2, name="AlphaBeta_d2", seed=seed)
    if bot_name == "alphabeta_d3":
        return AlphaBetaAgent(depth=3, name="AlphaBeta_d3", seed=seed)
    if bot_name == "alphabeta_d4":
        return AlphaBetaAgent(depth=4, name="AlphaBeta_d4", seed=seed)
    if bot_name == "mcts_200":
        return MCTSAgent(
            num_simulations=200,
            rollout_policy="greedy",
            name="MCTS_200",
            seed=seed,
        )
    if bot_name == "mcts_500":
        return MCTSAgent(
            num_simulations=500,
            rollout_policy="greedy",
            name="MCTS_500",
            seed=seed,
        )

    raise ValueError(f"Unknown rollout bot: {bot_name}")


def make_value_teacher(bot_name: str, seed: int):
    """
    The teacher used to assign search-value labels.
    Must provide `select_move(state)` with `decision.evaluation`.
    """
    bot_name = bot_name.lower()

    if bot_name == "minimax_d2":
        return MinimaxAgent(depth=2, name="Minimax_d2_Teacher", seed=seed)
    if bot_name == "minimax_d3":
        return MinimaxAgent(depth=3, name="Minimax_d3_Teacher", seed=seed)
    if bot_name == "alphabeta_d2":
        return AlphaBetaAgent(depth=2, name="AlphaBeta_d2_Teacher", seed=seed)
    if bot_name == "alphabeta_d3":
        return AlphaBetaAgent(depth=3, name="AlphaBeta_d3_Teacher", seed=seed)
    if bot_name == "alphabeta_d4":
        return AlphaBetaAgent(depth=4, name="AlphaBeta_d4_Teacher", seed=seed)
    if bot_name == "mcts_200":
        return MCTSAgent(
            num_simulations=200,
            rollout_policy="greedy",
            name="MCTS_200_Teacher",
            seed=seed,
        )
    if bot_name == "mcts_500":
        return MCTSAgent(
            num_simulations=500,
            rollout_policy="greedy",
            name="MCTS_500_Teacher",
            seed=seed,
        )

    raise ValueError(f"Unknown value teacher: {bot_name}")


# ------------------------------------------------------------
# Data collection
# ------------------------------------------------------------

def collect_examples_from_game(
    rows: int,
    cols: int,
    player0_name: str,
    player1_name: str,
    value_teacher_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Play a game using player0/player1 rollout agents.
    At each visited state, assign target_value from a stronger teacher's
    search evaluation of that state.
    """
    player0 = make_rollout_agent(player0_name, seed=seed)
    player1 = make_rollout_agent(player1_name, seed=seed + 10_000)
    teacher = make_value_teacher(value_teacher_name, seed=seed + 20_000)

    state = DotsAndBoxesState(rows=rows, cols=cols)
    examples: list[dict[str, Any]] = []

    while not state.is_terminal():
        state_before = state.clone()

        teacher_decision = teacher.select_move(state_before.clone())
        target_value = float(teacher_decision.evaluation)

        examples.append(
            {
                "state": serialize_state(state_before),
                "current_player": state_before.current_player,
                "target_value": target_value,
            }
        )

        agent = player0 if state.current_player == 0 else player1
        decision = agent.select_move(state.clone())
        state.apply_move(decision.move)

    return examples


def _worker(task: tuple[int, int, str, str, str, int]) -> list[dict[str, Any]]:
    rows, cols, player0_name, player1_name, value_teacher_name, seed = task
    return collect_examples_from_game(
        rows=rows,
        cols=cols,
        player0_name=player0_name,
        player1_name=player1_name,
        value_teacher_name=value_teacher_name,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)

    # rollout policy distribution for encountered states
    parser.add_argument("--player0", type=str, required=True)
    parser.add_argument("--player1", type=str, required=True)

    # stronger evaluator used for the label
    parser.add_argument("--value-teacher", type=str, required=True)

    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--processes", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            args.rows,
            args.cols,
            args.player0,
            args.player1,
            args.value_teacher,
            rng.randint(0, 10**9),
        )
        for _ in range(args.games)
    ]

    total_examples = 0

    with out_path.open("w", encoding="utf-8") as f:
        with mp.Pool(processes=args.processes) as pool:
            for game_idx, game_examples in enumerate(pool.imap_unordered(_worker, tasks), start=1):
                for ex in game_examples:
                    f.write(json.dumps(ex) + "\n")
                    total_examples += 1

                if game_idx % 10 == 0:
                    print(
                        f"[{game_idx}/{args.games}] "
                        f"written examples so far: {total_examples}"
                    )

    print(f"Saved {total_examples} examples to {out_path}")


if __name__ == "__main__":
    main()