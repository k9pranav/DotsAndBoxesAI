from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from engine import DotsAndBoxesState
from runner import GameRunner
from agents import (
    RandomAgent,
    GreedyAgent,
    AlphaBetaAgent,
    MCTSAgent,
    BoxGNNValueAgent,
)
from ml.box_gnn_encoding import encode_box_graph
from ml.box_gnn_value_model import BoxGNNValueConfig, BoxGNNValueModel


# ============================================================
# Device helper
# ============================================================

def pick_device(explicit: Optional[str] = None) -> str:
    if explicit is not None:
        return explicit
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# Teacher / anchor / self-play agents
# ============================================================

def make_anchor_agent(spec: str, seed: int):
    spec = spec.lower()

    if spec == "random":
        return RandomAgent(name="Random", seed=seed)
    if spec == "greedy":
        return GreedyAgent(name="Greedy", seed=seed)
    if spec == "alphabeta_d2":
        return AlphaBetaAgent(depth=2, name="AlphaBeta_d2", seed=seed)
    if spec == "alphabeta_d3":
        return AlphaBetaAgent(depth=3, name="AlphaBeta_d3", seed=seed)
    if spec == "mcts_200":
        return MCTSAgent(
            num_simulations=200,
            rollout_policy="greedy",
            name="MCTS_200",
            seed=seed,
        )
    if spec == "mcts_500":
        return MCTSAgent(
            num_simulations=500,
            rollout_policy="greedy",
            name="MCTS_500",
            seed=seed,
        )

    raise ValueError(f"Unknown anchor agent spec: {spec}")


def make_box_gnn_agent(checkpoint_path: str, name: str, device: Optional[str] = None):
    return BoxGNNValueAgent(
        checkpoint_path=checkpoint_path,
        name=name,
        device=device,
    )


# ============================================================
# Dataset format
# ============================================================

def state_to_example(state_before: DotsAndBoxesState) -> dict[str, Any]:
    graph = encode_box_graph(state_before)
    return {
        "rows": state_before.rows,
        "cols": state_before.cols,
        "node_features": graph["node_features"].tolist(),
        "edge_index": graph["edge_index"].tolist(),
        "global_features": graph["global_features"].tolist(),
        "current_player": state_before.current_player,
        # target_value gets filled in at game end
    }


def finalize_examples_with_outcome(
    recorded_states: list[dict[str, Any]],
    final_scores: list[int],
) -> list[dict[str, Any]]:
    for item in recorded_states:
        perspective_player = int(item["current_player"])
        target_value = final_scores[perspective_player] - final_scores[1 - perspective_player]
        item["target_value"] = float(target_value)
    return recorded_states


# ============================================================
# Game data generation
# ============================================================

def collect_examples_from_match(
    rows: int,
    cols: int,
    player0,
    player1,
) -> list[dict[str, Any]]:
    state = DotsAndBoxesState(rows=rows, cols=cols)
    examples: list[dict[str, Any]] = []

    while not state.is_terminal():
        state_before = state.clone()
        examples.append(state_to_example(state_before))

        agent = player0 if state.current_player == 0 else player1
        decision = agent.select_move(state.clone())
        state.apply_move(decision.move)

    return finalize_examples_with_outcome(examples, state.scores[:])


def generate_selfplay_examples(
    rows: int,
    cols: int,
    checkpoint_path: str,
    num_games: int,
    seed: int,
    device: Optional[str] = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_examples: list[dict[str, Any]] = []

    for game_idx in range(num_games):
        agent_a = make_box_gnn_agent(
            checkpoint_path=checkpoint_path,
            name="SelfPlayA",
            device=device,
        )
        agent_b = make_box_gnn_agent(
            checkpoint_path=checkpoint_path,
            name="SelfPlayB",
            device=device,
        )

        # Seat-swap symmetry through alternating assignment
        if game_idx % 2 == 0:
            p0, p1 = agent_a, agent_b
        else:
            p0, p1 = agent_b, agent_a

        examples = collect_examples_from_match(rows, cols, p0, p1)
        all_examples.extend(examples)

        if (game_idx + 1) % 10 == 0:
            print(
                f"[self-play {game_idx + 1}/{num_games}] "
                f"examples so far: {len(all_examples)}"
            )

        _ = rng.randint(0, 10**9)  # consume RNG for future extensibility

    return all_examples


def generate_anchor_examples(
    rows: int,
    cols: int,
    checkpoint_path: str,
    opponent_spec: str,
    num_games: int,
    seed: int,
    device: Optional[str] = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_examples: list[dict[str, Any]] = []

    for game_idx in range(num_games):
        model_agent = make_box_gnn_agent(
            checkpoint_path=checkpoint_path,
            name="Model",
            device=device,
        )
        opp_seed = rng.randint(0, 10**9)
        opponent = make_anchor_agent(opponent_spec, seed=opp_seed)

        if game_idx % 2 == 0:
            p0, p1 = model_agent, opponent
        else:
            p0, p1 = opponent, model_agent

        examples = collect_examples_from_match(rows, cols, p0, p1)
        all_examples.extend(examples)

        if (game_idx + 1) % 10 == 0:
            print(
                f"[anchor={opponent_spec} {game_idx + 1}/{num_games}] "
                f"examples so far: {len(all_examples)}"
            )

    return all_examples


# ============================================================
# Training code (same idea as your train_box_gnn_value.py)
# ============================================================

class BoxGNNValueDataset(Dataset):
    def __init__(self, data: list[dict]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        node_features = torch.tensor(item["node_features"], dtype=torch.float32)
        edge_index = torch.tensor(item["edge_index"], dtype=torch.long)
        global_features = torch.tensor(item["global_features"], dtype=torch.float32)
        target_value = torch.tensor([item["target_value"]], dtype=torch.float32)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "global_features": global_features,
            "target_value": target_value,
        }


def collate_graph_batch(batch: list[dict]) -> dict:
    node_features_list = []
    edge_index_list = []
    global_features_list = []
    target_values_list = []
    batch_index_list = []

    node_offset = 0

    for graph_id, item in enumerate(batch):
        node_features = item["node_features"]
        edge_index = item["edge_index"]
        global_features = item["global_features"]
        target_value = item["target_value"]

        num_nodes = node_features.shape[0]

        node_features_list.append(node_features)
        global_features_list.append(global_features)
        target_values_list.append(target_value)
        batch_index_list.append(
            torch.full((num_nodes,), graph_id, dtype=torch.long)
        )

        if edge_index.numel() > 0:
            edge_index_list.append(edge_index + node_offset)

        node_offset += num_nodes

    node_features = torch.cat(node_features_list, dim=0)
    global_features = torch.stack(global_features_list, dim=0)
    target_values = torch.stack(target_values_list, dim=0)
    batch_index = torch.cat(batch_index_list, dim=0)

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "batch_index": batch_index,
        "global_features": global_features,
        "target_values": target_values,
    }


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            node_features = batch["node_features"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_index = batch["batch_index"].to(device)
            global_features = batch["global_features"].to(device)
            target_values = batch["target_values"].to(device)

            preds = model(
                node_features=node_features,
                edge_index=edge_index,
                batch_index=batch_index,
                global_features=global_features,
            )
            loss = criterion(preds, target_values)

            batch_size = target_values.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

    return total_loss / total_items if total_items > 0 else 0.0


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        node_features = batch["node_features"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_index = batch["batch_index"].to(device)
        global_features = batch["global_features"].to(device)
        target_values = batch["target_values"].to(device)

        preds = model(
            node_features=node_features,
            edge_index=edge_index,
            batch_index=batch_index,
            global_features=global_features,
        )
        loss = criterion(preds, target_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = target_values.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / total_items if total_items > 0 else 0.0


def train_candidate_checkpoint(
    data: list[dict],
    out_path: str,
    hidden_dim: int,
    num_message_passing_layers: int,
    readout_hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    train_split: float,
    seed: int,
    device: str,
) -> float:
    random.seed(seed)
    torch.manual_seed(seed)

    dataset = BoxGNNValueDataset(data)
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graph_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
    )

    config = BoxGNNValueConfig(
        node_feature_dim=10,
        global_feature_dim=7,
        hidden_dim=hidden_dim,
        num_message_passing_layers=num_message_passing_layers,
        readout_hidden_dim=readout_hidden_dim,
    )
    model = BoxGNNValueModel(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:02d} | train_mse={train_loss:.4f} | val_mse={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if best_state_dict is None:
        best_state_dict = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "node_feature_dim": 10,
            "global_feature_dim": 7,
            "hidden_dim": hidden_dim,
            "num_message_passing_layers": num_message_passing_layers,
            "readout_hidden_dim": readout_hidden_dim,
            "best_val_loss": best_val_loss,
        },
        out_path,
    )

    print(f"Saved candidate checkpoint to {out_path}")
    print(f"Best validation MSE: {best_val_loss:.4f}")
    return best_val_loss


# ============================================================
# Evaluation / gating
# ============================================================

@dataclass
class HeadToHeadResult:
    candidate_wins: int
    incumbent_wins: int
    draws: int
    avg_score_diff_candidate_minus_incumbent: float


def evaluate_candidate_vs_incumbent(
    rows: int,
    cols: int,
    candidate_ckpt: str,
    incumbent_ckpt: str,
    num_games: int,
    seed: int,
    device: Optional[str] = None,
) -> HeadToHeadResult:
    rng = random.Random(seed)

    candidate_wins = 0
    incumbent_wins = 0
    draws = 0
    score_diffs = []

    for game_idx in range(num_games):
        cand = make_box_gnn_agent(candidate_ckpt, "Candidate", device=device)
        inc = make_box_gnn_agent(incumbent_ckpt, "Incumbent", device=device)

        state = DotsAndBoxesState(rows=rows, cols=cols)

        # Seat swapping
        if game_idx % 2 == 0:
            player0, player1 = cand, inc
            candidate_side = 0
        else:
            player0, player1 = inc, cand
            candidate_side = 1

        runner = GameRunner(
            initial_state=state,
            player0=player0,
            player1=player1,
        )
        summary = runner.play(verbose=False)

        cand_score = summary.final_scores[candidate_side]
        inc_score = summary.final_scores[1 - candidate_side]
        score_diffs.append(cand_score - inc_score)

        if cand_score > inc_score:
            candidate_wins += 1
        elif inc_score > cand_score:
            incumbent_wins += 1
        else:
            draws += 1

        _ = rng.randint(0, 10**9)

    avg_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0.0

    return HeadToHeadResult(
        candidate_wins=candidate_wins,
        incumbent_wins=incumbent_wins,
        draws=draws,
        avg_score_diff_candidate_minus_incumbent=avg_diff,
    )


def promote_candidate(
    result: HeadToHeadResult,
    promotion_threshold: float,
) -> bool:
    decisive_games = result.candidate_wins + result.incumbent_wins
    if decisive_games == 0:
        return False

    candidate_win_rate = result.candidate_wins / decisive_games
    return candidate_win_rate >= promotion_threshold


# ============================================================
# Orchestration
# ============================================================

def save_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def main() -> None:
    parser = argparse.ArgumentParser()

    # Core loop
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--initial-checkpoint", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default="selfplay_box_gnn_runs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)

    # Mixed data generation
    parser.add_argument("--selfplay-games", type=int, default=200)
    parser.add_argument("--anchor-greedy-games", type=int, default=50)
    parser.add_argument("--anchor-abd2-games", type=int, default=50)
    parser.add_argument("--anchor-random-games", type=int, default=0)
    parser.add_argument("--anchor-mcts200-games", type=int, default=0)

    # Training
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-message-passing-layers", type=int, default=3)
    parser.add_argument("--readout-hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.9)

    # Gating
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)

    args = parser.parse_args()

    device = pick_device(args.device)
    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    data_dir = out_dir / "datasets"
    logs_dir = out_dir / "logs"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Copy initial checkpoint into the run folder as v0
    incumbent_ckpt = ckpt_dir / "box_gnn_v0.pt"
    if not incumbent_ckpt.exists():
        shutil.copyfile(args.initial_checkpoint, incumbent_ckpt)

    print(f"Using device: {device}")
    print(f"Incumbent starts at: {incumbent_ckpt}")

    for i in range(1, args.iterations + 1):
        print("=" * 80)
        print(f"ITERATION {i}")
        print("=" * 80)

        iter_seed = rng.randint(0, 10**9)

        # ----------------------------------------------------
        # Generate mixed dataset
        # ----------------------------------------------------
        mixed_examples: list[dict[str, Any]] = []

        print("\nGenerating self-play examples...")
        mixed_examples.extend(
            generate_selfplay_examples(
                rows=args.rows,
                cols=args.cols,
                checkpoint_path=str(incumbent_ckpt),
                num_games=args.selfplay_games,
                seed=iter_seed,
                device=device,
            )
        )

        if args.anchor_greedy_games > 0:
            print("\nGenerating anchor examples vs Greedy...")
            mixed_examples.extend(
                generate_anchor_examples(
                    rows=args.rows,
                    cols=args.cols,
                    checkpoint_path=str(incumbent_ckpt),
                    opponent_spec="greedy",
                    num_games=args.anchor_greedy_games,
                    seed=iter_seed + 1,
                    device=device,
                )
            )

        if args.anchor_abd2_games > 0:
            print("\nGenerating anchor examples vs AlphaBeta_d2...")
            mixed_examples.extend(
                generate_anchor_examples(
                    rows=args.rows,
                    cols=args.cols,
                    checkpoint_path=str(incumbent_ckpt),
                    opponent_spec="alphabeta_d2",
                    num_games=args.anchor_abd2_games,
                    seed=iter_seed + 2,
                    device=device,
                )
            )

        if args.anchor_random_games > 0:
            print("\nGenerating anchor examples vs Random...")
            mixed_examples.extend(
                generate_anchor_examples(
                    rows=args.rows,
                    cols=args.cols,
                    checkpoint_path=str(incumbent_ckpt),
                    opponent_spec="random",
                    num_games=args.anchor_random_games,
                    seed=iter_seed + 3,
                    device=device,
                )
            )

        if args.anchor_mcts200_games > 0:
            print("\nGenerating anchor examples vs MCTS_200...")
            mixed_examples.extend(
                generate_anchor_examples(
                    rows=args.rows,
                    cols=args.cols,
                    checkpoint_path=str(incumbent_ckpt),
                    opponent_spec="mcts_200",
                    num_games=args.anchor_mcts200_games,
                    seed=iter_seed + 4,
                    device=device,
                )
            )

        dataset_path = data_dir / f"mixed_iter_{i}.json"
        save_json(mixed_examples, dataset_path)
        print(f"\nSaved mixed dataset with {len(mixed_examples)} examples to {dataset_path}")

        # ----------------------------------------------------
        # Train candidate
        # ----------------------------------------------------
        candidate_ckpt = ckpt_dir / f"box_gnn_candidate_iter_{i}.pt"
        print("\nTraining candidate checkpoint...")

        best_val_loss = train_candidate_checkpoint(
            data=mixed_examples,
            out_path=str(candidate_ckpt),
            hidden_dim=args.hidden_dim,
            num_message_passing_layers=args.num_message_passing_layers,
            readout_hidden_dim=args.readout_hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_split=args.train_split,
            seed=iter_seed + 10,
            device=device,
        )

        # ----------------------------------------------------
        # Gate candidate vs incumbent
        # ----------------------------------------------------
        print("\nEvaluating candidate vs incumbent...")
        result = evaluate_candidate_vs_incumbent(
            rows=args.rows,
            cols=args.cols,
            candidate_ckpt=str(candidate_ckpt),
            incumbent_ckpt=str(incumbent_ckpt),
            num_games=args.eval_games,
            seed=iter_seed + 20,
            device=device,
        )

        print(
            f"Candidate wins: {result.candidate_wins} | "
            f"Incumbent wins: {result.incumbent_wins} | "
            f"Draws: {result.draws} | "
            f"Avg score diff (cand-inc): {result.avg_score_diff_candidate_minus_incumbent:.3f}"
        )

        accepted = promote_candidate(
            result=result,
            promotion_threshold=args.promotion_threshold,
        )

        iter_log = {
            "iteration": i,
            "dataset_size": len(mixed_examples),
            "best_val_loss": best_val_loss,
            "candidate_wins": result.candidate_wins,
            "incumbent_wins": result.incumbent_wins,
            "draws": result.draws,
            "avg_score_diff_candidate_minus_incumbent": result.avg_score_diff_candidate_minus_incumbent,
            "promotion_threshold": args.promotion_threshold,
            "accepted": accepted,
        }

        log_path = logs_dir / f"iter_{i}.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(iter_log, f, indent=2)

        if accepted:
            new_incumbent = ckpt_dir / f"box_gnn_v{i}.pt"
            shutil.copyfile(candidate_ckpt, new_incumbent)
            incumbent_ckpt = new_incumbent
            print(f"\nPROMOTED candidate -> {incumbent_ckpt}")
        else:
            print("\nCandidate rejected; incumbent remains unchanged.")

    print("\nDone.")
    print(f"Final incumbent checkpoint: {incumbent_ckpt}")


if __name__ == "__main__":
    main()