from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from engine import DotsAndBoxesState
from agents import RandomAgent, GreedyAgent, MinimaxAgent, AlphaBetaAgent, MCTSAgent
from ml.encoding import encode_state_flat, legal_move_mask


def make_teacher(bot_name: str, seed: int):
    bot_name = bot_name.lower()

    if bot_name == "greedy":
        return GreedyAgent(name="GreedyTeacher", seed=seed)
    if bot_name == "minimax_d2":
        return MinimaxAgent(depth=2, name="Minimax_d2_Teacher", seed=seed)
    if bot_name == "minimax_d3":
        return MinimaxAgent(depth=3, name="Minimax_d3_Teacher", seed=seed)
    if bot_name == "alphabeta_d2":
        return AlphaBetaAgent(depth=2, name="AlphaBeta_d2_Teacher", seed=seed)
    if bot_name == "alphabeta_d3":
        return AlphaBetaAgent(depth=3, name="AlphaBeta_d3_Teacher", seed=seed)
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
    if bot_name == "random":
        return RandomAgent(name="RandomTeacher", seed=seed)

    raise ValueError(f"Unknown teacher bot: {bot_name}")


def maybe_make_opponent(bot_name: str, seed: int):
    bot_name = bot_name.lower()

    if bot_name == "random":
        return RandomAgent(name="RandomOpponent", seed=seed)
    if bot_name == "greedy":
        return GreedyAgent(name="GreedyOpponent", seed=seed)
    if bot_name == "minimax_d2":
        return MinimaxAgent(depth=2, name="Minimax_d2_Opponent", seed=seed)
    if bot_name == "alphabeta_d2":
        return AlphaBetaAgent(depth=2, name="AlphaBeta_d2_Opponent", seed=seed)
    if bot_name == "mcts_200":
        return MCTSAgent(
            num_simulations=200,
            rollout_policy="greedy",
            name="MCTS_200_Opponent",
            seed=seed,
        )

    raise ValueError(f"Unknown opponent bot: {bot_name}")


def collect_examples_from_game(
    rows: int,
    cols: int,
    teacher_name: str,
    opponent_name: str,
    seed: int,
    teacher_as_player: int,
) -> list[dict[str, Any]]:
    """
    Plays one full game and collects training examples only from turns
    where the teacher is the acting player.
    """
    teacher = make_teacher(teacher_name, seed=seed)
    opponent = maybe_make_opponent(opponent_name, seed=seed + 10_000)

    state = DotsAndBoxesState(rows=rows, cols=cols)
    state.current_player = 0

    examples: list[dict[str, Any]] = []

    while not state.is_terminal():
        acting_player = state.current_player

        if acting_player == teacher_as_player:
            agent = teacher
        else:
            agent = opponent

        state_before = state.clone()
        decision = agent.select_move(state.clone())
        move = decision.move
        action_index = state.edge_to_index(move)

        if acting_player == teacher_as_player:
            examples.append(
                {
                    "rows": rows,
                    "cols": cols,
                    "teacher": teacher_name,
                    "opponent": opponent_name,
                    "features": encode_state_flat(state_before),
                    "legal_mask": legal_move_mask(state_before),
                    "action_index": action_index,
                    "current_player": state_before.current_player,
                    "scores": state_before.scores[:],
                }
            )

        state.apply_move(move)

    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--opponent", type=str, default="random")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    all_examples: list[dict[str, Any]] = []

    for game_idx in range(args.games):
        # Alternate teacher seat for diversity
        teacher_as_player = game_idx % 2
        seed = rng.randint(0, 10**9)

        game_examples = collect_examples_from_game(
            rows=args.rows,
            cols=args.cols,
            teacher_name=args.teacher,
            opponent_name=args.opponent,
            seed=seed,
            teacher_as_player=teacher_as_player,
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