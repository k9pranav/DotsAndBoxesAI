from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BoxGNNValueConfig:
    node_feature_dim: int = 26
    global_feature_dim: int = 8
    hidden_dim: int = 128
    num_message_passing_layers: int = 3
    readout_hidden_dim: int = 128


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            neighbor_mean = torch.zeros_like(h)
            return self.update(torch.cat([h, neighbor_mean], dim=1))

        src = edge_index[0]
        dst = edge_index[1]

        neighbor_sum = torch.zeros_like(h)
        neighbor_count = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)

        neighbor_sum.index_add_(0, dst, h[src])

        ones = torch.ones((dst.shape[0], 1), dtype=h.dtype, device=h.device)
        neighbor_count.index_add_(0, dst, ones)

        neighbor_mean = neighbor_sum / neighbor_count.clamp(min=1.0)
        return self.update(torch.cat([h, neighbor_mean], dim=1))


class BoxGNNValueModel(nn.Module):
    """
    Bipartite edge+box GNN value model.
    Uses mean pooling over all nodes.
    """

    def __init__(self, config: BoxGNNValueConfig) -> None:
        super().__init__()
        self.config = config

        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            [MessagePassingLayer(config.hidden_dim) for _ in range(config.num_message_passing_layers)]
        )

        readout_in_dim = config.hidden_dim + config.global_feature_dim

        self.readout = nn.Sequential(
            nn.Linear(readout_in_dim, config.readout_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.readout_hidden_dim, config.readout_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.readout_hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_encoder(node_features)

        for layer in self.layers:
            h = layer(h, edge_index)

        graph_mean = self._global_mean_pool(h, batch_index, global_features.shape[0])
        graph_repr = torch.cat([graph_mean, global_features], dim=1)
        return self.readout(graph_repr)

    def _global_mean_pool(
        self, h: torch.Tensor, batch_index: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        hidden_dim = h.shape[1]
        pooled = torch.zeros((batch_size, hidden_dim), dtype=h.dtype, device=h.device)
        counts = torch.zeros((batch_size, 1), dtype=h.dtype, device=h.device)

        pooled.index_add_(0, batch_index, h)
        ones = torch.ones((h.shape[0], 1), dtype=h.dtype, device=h.device)
        counts.index_add_(0, batch_index, ones)

        return pooled / counts.clamp(min=1.0)