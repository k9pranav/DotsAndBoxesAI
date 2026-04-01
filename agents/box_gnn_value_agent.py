from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from engine import DotsAndBoxesState, Edge
from agents.base import AgentDecision
from ml.box_gnn_encoding import encode_box_graph
from ml.box_gnn_value_model import BoxGNNValueConfig, BoxGNNValueModel


@dataclass(frozen=True)
class BoxGNNValueStats:
    checkpoint_path: str
    chosen_move: Edge
    chosen_value: float


class BoxGNNValueAgent:
    def __init__(
        self,
        checkpoint_path: str,
        name: str = "BoxGNNValueAgent",
        device: Optional[str] = None,
    ) -> None:
        self.name = name
        self.checkpoint_path = checkpoint_path

        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.config = BoxGNNValueConfig(
            node_feature_dim=int(checkpoint.get("node_feature_dim", 26)),
            global_feature_dim=int(checkpoint.get("global_feature_dim", 8)),
            hidden_dim=int(checkpoint.get("hidden_dim", 128)),
            num_message_passing_layers=int(checkpoint.get("num_message_passing_layers", 3)),
            readout_hidden_dim=int(checkpoint.get("readout_hidden_dim", 128)),
        )

        self.model = BoxGNNValueModel(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.last_stats: Optional[BoxGNNValueStats] = None

    def select_move(self, state: DotsAndBoxesState) -> AgentDecision:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")

        root_player = state.current_player
        best_move: Optional[Edge] = None
        best_value = float("-inf")

        for move in legal_moves:
            child = state.next_state(move)
            value = self._predict_root_perspective_value(child, root_player)

            if value > best_value:
                best_value = value
                best_move = move

        if best_move is None:
            raise RuntimeError("Failed to select a move.")

        self.last_stats = BoxGNNValueStats(
            checkpoint_path=str(Path(self.checkpoint_path)),
            chosen_move=best_move,
            chosen_value=float(best_value),
        )

        return AgentDecision(
            move=best_move,
            evaluation=float(best_value),
            reason="box_gnn_value_one_ply",
        )

    def _predict_root_perspective_value(
        self,
        state: DotsAndBoxesState,
        root_player: int,
    ) -> float:
        graph = encode_box_graph(state)

        node_features = graph["node_features"].to(self.device)
        edge_index = graph["edge_index"].to(self.device)
        global_features = graph["global_features"].to(self.device).unsqueeze(0)

        batch_index = torch.zeros(
            node_features.shape[0],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            value = float(
                self.model(
                    node_features=node_features,
                    edge_index=edge_index,
                    batch_index=batch_index,
                    global_features=global_features,
                ).squeeze(0).item()
            )

        if state.current_player != root_player:
            value = -value

        return value