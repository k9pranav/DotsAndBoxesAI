from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from engine import DotsAndBoxesState, Edge
from agents.base import AgentDecision
from ml import ValueMLP, ValueModelConfig
from ml.encoding import encode_state_flat


@dataclass(frozen=True)
class NNValueStats:
    checkpoint_path: str
    chosen_move: Edge
    chosen_value: float


class NNValueAgent:
    """
    Inference-time value-network agent.

    Strategy:
      - For each legal move:
          child = state.next_state(move)
          predict value(child)
      - The value model is trained from the perspective of child.current_player
      - Convert that prediction into root-player perspective:
          if child.current_player != root_player:
              value = -value
      - Choose move with maximum root-player value
    """

    def __init__(
        self,
        checkpoint_path: str,
        name: str = "NNValueAgent",
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

        self.config = ValueModelConfig(
            rows=rows,
            cols=cols,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

        self.model = ValueMLP(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.last_stats: Optional[NNValueStats] = None

    def select_move(self, state: DotsAndBoxesState) -> AgentDecision:
        if state.rows != self.config.rows or state.cols != self.config.cols:
            raise ValueError(
                f"Checkpoint expects board {self.config.rows}x{self.config.cols}, "
                f"but got {state.rows}x{state.cols}."
            )

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

        self.last_stats = NNValueStats(
            checkpoint_path=str(Path(self.checkpoint_path)),
            chosen_move=best_move,
            chosen_value=float(best_value),
        )

        return AgentDecision(
            move=best_move,
            evaluation=float(best_value),
            reason="nn_value_one_ply",
        )

    def _predict_root_perspective_value(
        self,
        state: DotsAndBoxesState,
        root_player: int,
    ) -> float:
        x = self._encode_state(state).to(self.device)

        with torch.no_grad():
            value = float(self.model(x).squeeze(0).item())

        # The model predicts from the perspective of state.current_player.
        # Convert to root-player perspective if needed.
        if state.current_player != root_player:
            value = -value

        return value

    def _encode_state(self, state: DotsAndBoxesState) -> torch.Tensor:
        features = encode_state_flat(state)
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)