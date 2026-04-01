from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PolicyModelConfig:
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

    @property
    def output_dim(self) -> int:
        return (self.rows + 1) * self.cols + self.rows * (self.cols + 1)


class PolicyMLP(nn.Module):
    """
    Simple MLP policy network for a fixed board size.

    Input:
        flattened feature vector of length config.input_dim

    Output:
        logits over all possible edges of length config.output_dim
    """

    def __init__(self, config: PolicyModelConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.input_dim

        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = config.hidden_dim

        layers.append(nn.Linear(in_dim, config.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)