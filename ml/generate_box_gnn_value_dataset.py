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


def make_agent(bot_name: str, seed: int):
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
    if bot_name == "mcts_200":
        return MCTSAgent(num_simulations=200, rollout_policy="greedy", name="MCTS_200", seed=seed)
    if bot_name == "mcts_500":
        return MCTSAgent(num_simulations=500, rollout_policy="greedy", name="MCTS_500", seed=seed)

    raise ValueError(f"Unknown bot: {bot_name}")


def collect_examples_from_game(
    rows: int,
    cols: int,
    player0_name: str,
    player1_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    player0 = make_agent(player0_name, seed=seed)
    player1 = make_agent(player1_name, seed=seed + 10_000)

    state = DotsAndBoxesState(rows=rows, cols=cols)
    recorded_states: list[dict[str, Any]] = []

    while not state.is_terminal():
        state_before = state.clone()
        recorded_states.append(
            {
                "state": serialize_state(state_before),
                "current_player": state_before.current_player,
            }
        )

        agent = player0 if state.current_player == 0 else player1
        decision = agent.select_move(state.clone())
        state.apply_move(decision.move)

    final_scores = state.scores[:]

    for item in recorded_states:
        perspective_player = int(item["current_player"])
        target_value = final_scores[perspective_player] - final_scores[1 - perspective_player]
        item["target_value"] = float(target_value)

    return recorded_states


def _worker(task: tuple[int, int, str, str, int]) -> list[dict[str, Any]]:
    rows, cols, player0_name, player1_name, seed = task
    return collect_examples_from_game(rows, cols, player0_name, player1_name, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--player0", type=str, required=True)
    parser.add_argument("--player1", type=str, required=True)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--processes", type=int, default=max(1, (mp.cpu_count() - 1)))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        (args.rows, args.cols, args.player0, args.player1, rng.randint(0, 10**9))
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