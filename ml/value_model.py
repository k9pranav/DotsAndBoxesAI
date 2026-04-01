from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ValueModelConfig:
    rows: int
    cols: int
    hidden_dim: int = 128
    num_hidden_layers: int = 2

    @property
    def input_dim(self) -> int:
        # h_edges + v_edges + box_owners + current_player_bit
        num_h = (self.rows + 1) * self.cols
        num_v = self.rows * (self.cols + 1)
        num_b = self.rows * self.cols
        return num_h + num_v + num_b + 1


class ValueMLP(nn.Module):
    """
    Simple MLP value network for a fixed board size.

    Input:
        flattened feature vector of length config.input_dim

    Output:
        a single scalar value
    """

    def __init__(self, config: ValueModelConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.input_dim

        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = config.hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape: [batch, 1]
        return self.net(x)