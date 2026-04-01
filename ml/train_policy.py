from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from ml import PolicyMLP, PolicyModelConfig


class PolicyDataset(Dataset):
    def __init__(self, data: list[dict]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        x = torch.tensor(item["features"], dtype=torch.float32)
        y = torch.tensor(item["action_index"], dtype=torch.long)
        legal_mask = torch.tensor(item["legal_mask"], dtype=torch.float32)
        return x, y, legal_mask


def compute_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, legal_mask in loader:
            x = x.to(device)
            y = y.to(device)
            legal_mask = legal_mask.to(device)

            logits = model(x)

            # Mask illegal actions during evaluation
            masked_logits = logits.masked_fill(legal_mask == 0, float("-inf"))
            preds = torch.argmax(masked_logits, dim=1)

            correct += int((preds == y).sum().item())
            total += y.shape[0]

    return correct / total if total > 0 else 0.0


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

    for x, y, legal_mask in loader:
        x = x.to(device)
        y = y.to(device)
        legal_mask = legal_mask.to(device)

        logits = model(x)

        # Ensure the model is not rewarded for illegal actions
        masked_logits = logits.masked_fill(legal_mask == 0, float("-inf"))

        loss = criterion(masked_logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = y.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / total_items if total_items > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data)
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) == 0:
        raise ValueError("Dataset is empty.")

    rows = int(data[0]["rows"])
    cols = int(data[0]["cols"])

    for item in data:
        if int(item["rows"]) != rows or int(item["cols"]) != cols:
            raise ValueError("All examples in one dataset must have the same rows/cols.")

    dataset = PolicyDataset(data)

    n_total = len(dataset)
    n_train = int(n_total * args.train_split)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    config = PolicyModelConfig(
        rows=rows,
        cols=cols,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    )
    model = PolicyMLP(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        train_acc = compute_accuracy(model, train_loader, device=device)
        val_acc = compute_accuracy(model, val_loader, device=device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if best_state_dict is None:
        best_state_dict = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "rows": rows,
            "cols": cols,
            "hidden_dim": args.hidden_dim,
            "num_hidden_layers": args.num_hidden_layers,
            "best_val_acc": best_val_acc,
        },
        out_path,
    )

    print(f"Saved checkpoint to {out_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()