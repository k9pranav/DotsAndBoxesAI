from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from engine import DotsAndBoxesState
from agents import RandomAgent, GreedyAgent, MinimaxAgent, AlphaBetaAgent, MCTSAgent
from ml.encoding import encode_state_flat


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

    raise ValueError(f"Unknown bot: {bot_name}")


def collect_states_from_game(
    rows: int,
    cols: int,
    player0_name: str,
    player1_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Plays one full game and records all encountered states.
    After the game ends, assigns a target value to each recorded state:
        final score difference from that state's current_player perspective.
    """
    player0 = make_agent(player0_name, seed=seed)
    player1 = make_agent(player1_name, seed=seed + 10_000)

    state = DotsAndBoxesState(rows=rows, cols=cols)

    recorded_states: list[dict[str, Any]] = []

    while not state.is_terminal():
        state_before = state.clone()

        recorded_states.append(
            {
                "rows": rows,
                "cols": cols,
                "features": encode_state_flat(state_before),
                "current_player": state_before.current_player,
                "scores_before": state_before.scores[:],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--player0", type=str, required=True)
    parser.add_argument("--player1", type=str, required=True)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    all_examples: list[dict[str, Any]] = []

    for game_idx in range(args.games):
        seed = rng.randint(0, 10**9)

        game_examples = collect_states_from_game(
            rows=args.rows,
            cols=args.cols,
            player0_name=args.player0,
            player1_name=args.player1,
            seed=seed,
        )
        all_examples.extend(game_examples)

        if (game_idx + 1) % 10 == 0:
            print(
                f"[{game_idx + 1}/{args.games}] "
                f"collected examples so far: {len(all_examples)}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_examples, f)

    print(f"Saved {len(all_examples)} examples to {out_path}")


if __name__ == "__main__":
    main()