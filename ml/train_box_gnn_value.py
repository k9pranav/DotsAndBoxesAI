from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from ml.box_gnn_encoding import encode_box_graph
from ml.box_gnn_value_model import BoxGNNValueConfig, BoxGNNValueModel
from ml.state_transforms import deserialize_state, random_symmetry_transform


def pick_device(explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BoxGNNValueIterableDataset(IterableDataset):
    def __init__(
        self,
        jsonl_path: str,
        split: str,
        val_fraction: float = 0.1,
        seed: int = 123,
        augment: bool = False,
    ) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.jsonl_path = jsonl_path
        self.split = split
        self.val_fraction = val_fraction
        self.seed = seed
        self.augment = augment

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers

        rng = random.Random(self.seed + 1000 * worker_id)

        stride = max(1, round(1.0 / self.val_fraction))

        with Path(self.jsonl_path).open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                # shard lines across workers
                if line_idx % num_workers != worker_id:
                    continue

                is_val = (line_idx % stride == 0)

                if self.split == "train" and is_val:
                    continue
                if self.split == "val" and not is_val:
                    continue

                item = json.loads(line)
                state = deserialize_state(item["state"])

                if self.augment and self.split == "train":
                    state = random_symmetry_transform(state, rng)

                graph = encode_box_graph(state)

                yield {
                    "node_features": graph["node_features"],
                    "edge_index": graph["edge_index"],
                    "global_features": graph["global_features"],
                    "target_value": torch.tensor([item["target_value"]], dtype=torch.float32),
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
        batch_index_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))

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
        preds = model(
            node_features=batch["node_features"].to(device),
            edge_index=batch["edge_index"].to(device),
            batch_index=batch["batch_index"].to(device),
            global_features=batch["global_features"].to(device),
        )
        loss = criterion(preds, batch["target_values"].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = batch["target_values"].shape[0]
        total_loss += float(loss.item()) * bs
        total_items += bs

    return total_loss / total_items if total_items > 0 else 0.0


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            preds = model(
                node_features=batch["node_features"].to(device),
                edge_index=batch["edge_index"].to(device),
                batch_index=batch["batch_index"].to(device),
                global_features=batch["global_features"].to(device),
            )
            loss = criterion(preds, batch["target_values"].to(device))

            bs = batch["target_values"].shape[0]
            total_loss += float(loss.item()) * bs
            total_items += bs

    return total_loss / total_items if total_items > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-message-passing-layers", type=int, default=3)
    parser.add_argument("--readout-hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds = BoxGNNValueIterableDataset(
        jsonl_path=args.data,
        split="train",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=args.augment,
    )
    val_ds = BoxGNNValueIterableDataset(
        jsonl_path=args.data,
        split="val",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=collate_graph_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=collate_graph_batch,
        num_workers=args.num_workers,
    )

    config = BoxGNNValueConfig(
        node_feature_dim=26,
        global_feature_dim=8,
        hidden_dim=args.hidden_dim,
        num_message_passing_layers=args.num_message_passing_layers,
        readout_hidden_dim=args.readout_hidden_dim,
    )
    model = BoxGNNValueModel(config)

    device = pick_device(args.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | train_mse={train_loss:.4f} | val_mse={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state_dict if best_state_dict is not None else model.state_dict(),
            "node_feature_dim": 26,
            "global_feature_dim": 8,
            "hidden_dim": args.hidden_dim,
            "num_message_passing_layers": args.num_message_passing_layers,
            "readout_hidden_dim": args.readout_hidden_dim,
            "best_val_loss": best_val_loss,
        },
        out_path,
    )

    print(f"Saved checkpoint to {out_path}")
    print(f"Best validation MSE: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()