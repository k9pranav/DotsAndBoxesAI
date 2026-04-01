from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from engine import DotsAndBoxesState, Edge
from agents.base import AgentDecision
from ml import PolicyMLP, PolicyModelConfig


@dataclass(frozen=True)
class NNPolicyStats:
    checkpoint_path: str
    chosen_action_index: int
    chosen_logit: float


class NNPolicyAgent:
    """
    Inference-only neural policy agent.

    Expected workflow:
    1. Train a PolicyMLP offline for a fixed board size.
    2. Save checkpoint with:
         {
           "model_state_dict": ...,
           "rows": int,
           "cols": int,
           "hidden_dim": int,
           "num_hidden_layers": int,
         }
    3. Load that checkpoint here.
    4. For each state:
         - encode state into feature vector
         - run model to get logits over all edges
         - mask illegal moves
         - choose highest-logit legal move

    This first version is deterministic argmax after masking.
    """

    def __init__(
        self,
        checkpoint_path: str,
        name: str = "NNPolicyAgent",
        device: Optional[str] = None,
    ) -> None:
        self.name = name
        self.checkpoint_path = checkpoint_path

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        rows = int(checkpoint["rows"])
        cols = int(checkpoint["cols"])
        hidden_dim = int(checkpoint.get("hidden_dim", 128))
        num_hidden_layers = int(checkpoint.get("num_hidden_layers", 2))

        self.config = PolicyModelConfig(
            rows=rows,
            cols=cols,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

        self.model = PolicyMLP(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.last_stats: Optional[NNPolicyStats] = None

    def select_move(self, state: DotsAndBoxesState) -> AgentDecision:
        if state.rows != self.config.rows or state.cols != self.config.cols:
            raise ValueError(
                f"Checkpoint expects board {self.config.rows}x{self.config.cols}, "
                f"but got {state.rows}x{state.cols}."
            )

        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")

        x = self._encode_state(state).to(self.device)

        with torch.no_grad():
            logits = self.model(x).squeeze(0)  # shape: [num_actions]

        masked_logits = logits.clone()
        legal_indices = {state.edge_to_index(move) for move in legal_moves}

        for idx in range(state.total_edges()):
            if idx not in legal_indices:
                masked_logits[idx] = float("-inf")

        chosen_idx = int(torch.argmax(masked_logits).item())
        chosen_move = state.index_to_edge(chosen_idx)
        chosen_logit = float(masked_logits[chosen_idx].item())

        self.last_stats = NNPolicyStats(
            checkpoint_path=str(Path(self.checkpoint_path)),
            chosen_action_index=chosen_idx,
            chosen_logit=chosen_logit,
        )

        return AgentDecision(
            move=chosen_move,
            evaluation=chosen_logit,
            reason="nn_policy_argmax",
        )

    def _encode_state(self, state: DotsAndBoxesState) -> torch.Tensor:
        """
        Feature layout:
          [flattened h_edges,
           flattened v_edges,
           flattened box_owners_encoded,
           current_player_bit]

        box owner encoding:
          -1 -> unclaimed
           0 -> player 0
           1 -> player 1
        """
        h_flat = [float(int(cell)) for row in state.h_edges for cell in row]
        v_flat = [float(int(cell)) for row in state.v_edges for cell in row]
        b_flat = [
            float(-1 if owner is None else owner)
            for row in state.box_owners
            for owner in row
        ]
        current_player = [float(state.current_player)]

        features = h_flat + v_flat + b_flat + current_player
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, input_dim]
        return x