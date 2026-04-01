from __future__ import annotations

from typing import Any

import torch

from engine import DotsAndBoxesState, Edge


def _edge_to_box_adjacency_info(state: DotsAndBoxesState, edge: Edge) -> list[tuple[int, int]]:
    return state._adjacent_boxes_for_edge(edge)


def _box_index(rows: int, cols: int, r: int, c: int) -> int:
    return r * cols + c


def _is_boundary_edge(state: DotsAndBoxesState, edge: Edge) -> float:
    orientation, r, c = edge
    if orientation == "h":
        return float(r == 0 or r == state.rows)
    return float(c == 0 or c == state.cols)


def _edge_norm_position(state: DotsAndBoxesState, edge: Edge) -> tuple[float, float]:
    orientation, r, c = edge
    if orientation == "h":
        row_norm = 0.0 if state.rows == 0 else r / state.rows
        col_norm = 0.0 if state.cols <= 1 else c / (state.cols - 1)
    else:
        row_norm = 0.0 if state.rows <= 1 else r / (state.rows - 1)
        col_norm = 0.0 if state.cols == 0 else c / state.cols
    return row_norm, col_norm


def _box_position_flags(rows: int, cols: int, r: int, c: int) -> tuple[float, float, float]:
    is_top_or_bottom = (r == 0 or r == rows - 1)
    is_left_or_right = (c == 0 or c == cols - 1)
    is_corner = is_top_or_bottom and is_left_or_right
    is_boundary = (is_top_or_bottom or is_left_or_right) and not is_corner
    is_interior = not is_corner and not is_boundary
    return float(is_corner), float(is_boundary), float(is_interior)


def _count_neighbor_boxes_with_side_count(state: DotsAndBoxesState, box_r: int, box_c: int, k: int) -> int:
    count = 0
    for nr, nc in ((box_r - 1, box_c), (box_r + 1, box_c), (box_r, box_c - 1), (box_r, box_c + 1)):
        if 0 <= nr < state.rows and 0 <= nc < state.cols:
            if state.count_box_sides(nr, nc) == k:
                count += 1
    return count


def encode_box_graph(state: DotsAndBoxesState) -> dict[str, Any]:
    """
    Bipartite graph:
      - edge nodes first
      - box nodes second

    Edge-node features:
      [is_edge_node,
       is_box_node,
       drawn,
       legal,
       is_horizontal,
       is_vertical,
       boundary_edge,
       row_norm,
       col_norm,
       adj1_exists,
       adj1_sides_0,
       adj1_sides_1,
       adj1_sides_2,
       adj1_sides_3,
       adj1_sides_4,
       adj1_completable_if_played,
       adj2_exists,
       adj2_sides_0,
       adj2_sides_1,
       adj2_sides_2,
       adj2_sides_3,
       adj2_sides_4,
       adj2_completable_if_played,
       current_player,
       score_diff_current_norm,
       boxes_remaining_norm]

    Box-node features:
      [is_edge_node,
       is_box_node,
       owner_unclaimed,
       owner_p0,
       owner_p1,
       sides_0,
       sides_1,
       sides_2,
       sides_3,
       sides_4,
       completable_now,
       is_corner,
       is_boundary,
       is_interior,
       row_norm,
       col_norm,
       neigh_count_sides_2_norm,
       neigh_count_sides_3_norm,
       current_player,
       score_diff_current_norm,
       boxes_remaining_norm,
       zeros... padded to edge feature dim]
    """
    rows, cols = state.rows, state.cols
    total_boxes = rows * cols

    current_player = state.current_player
    score_diff_current = state.scores[current_player] - state.scores[1 - current_player]
    score_diff_current_norm = float(score_diff_current / max(total_boxes, 1))
    boxes_remaining_norm = float((total_boxes - (state.scores[0] + state.scores[1])) / max(total_boxes, 1))

    all_edge_nodes = list(state.index_to_edge(i) for i in range(state.total_edges()))
    num_edge_nodes = len(all_edge_nodes)

    edge_node_features = []
    incidence_edges = []

    for edge_idx, edge in enumerate(all_edge_nodes):
        drawn = float(state.is_edge_drawn(edge))
        legal = float(state.is_legal_move(edge))
        orientation, r, c = edge
        is_horizontal = float(orientation == "h")
        is_vertical = float(orientation == "v")
        boundary_edge = _is_boundary_edge(state, edge)
        row_norm, col_norm = _edge_norm_position(state, edge)

        adj_boxes = _edge_to_box_adjacency_info(state, edge)

        def adj_box_feats(pos: int) -> list[float]:
            if pos >= len(adj_boxes):
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            br, bc = adj_boxes[pos]
            sides = state.count_box_sides(br, bc)
            one_hot = [0.0] * 5
            one_hot[sides] = 1.0
            completable_if_played = float(
                state.box_owners[br][bc] is None and sides == 3
            )
            return [1.0, *one_hot, completable_if_played]

        adj1 = adj_box_feats(0)
        adj2 = adj_box_feats(1)

        feat = [
            1.0, 0.0,
            drawn,
            legal,
            is_horizontal,
            is_vertical,
            boundary_edge,
            row_norm,
            col_norm,
            *adj1,
            *adj2,
            float(current_player),
            score_diff_current_norm,
            boxes_remaining_norm,
        ]
        edge_node_features.append(feat)

        for br, bc in adj_boxes:
            box_global_idx = num_edge_nodes + _box_index(rows, cols, br, bc)
            incidence_edges.append((edge_idx, box_global_idx))
            incidence_edges.append((box_global_idx, edge_idx))

    edge_feature_dim = len(edge_node_features[0]) if edge_node_features else 26

    box_node_features = []
    for r in range(rows):
        for c in range(cols):
            owner = state.box_owners[r][c]
            owner_unclaimed = 1.0 if owner is None else 0.0
            owner_p0 = 1.0 if owner == 0 else 0.0
            owner_p1 = 1.0 if owner == 1 else 0.0

            sides = state.count_box_sides(r, c)
            sides_one_hot = [0.0] * 5
            sides_one_hot[sides] = 1.0
            completable_now = float(owner is None and sides == 3)

            is_corner, is_boundary, is_interior = _box_position_flags(rows, cols, r, c)
            row_norm = 0.0 if rows == 1 else r / (rows - 1)
            col_norm = 0.0 if cols == 1 else c / (cols - 1)

            neigh2 = _count_neighbor_boxes_with_side_count(state, r, c, 2) / 4.0
            neigh3 = _count_neighbor_boxes_with_side_count(state, r, c, 3) / 4.0

            feat = [
                0.0, 1.0,
                owner_unclaimed,
                owner_p0,
                owner_p1,
                *sides_one_hot,
                completable_now,
                is_corner,
                is_boundary,
                is_interior,
                row_norm,
                col_norm,
                float(neigh2),
                float(neigh3),
                float(current_player),
                score_diff_current_norm,
                boxes_remaining_norm,
            ]

            if len(feat) < edge_feature_dim:
                feat = feat + [0.0] * (edge_feature_dim - len(feat))
            box_node_features.append(feat)

    node_features = torch.tensor(edge_node_features + box_node_features, dtype=torch.float32)

    if incidence_edges:
        edge_index = torch.tensor(incidence_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    global_features = torch.tensor(
        [
            float(current_player),
            float(state.scores[0] / max(total_boxes, 1)),
            float(state.scores[1] / max(total_boxes, 1)),
            score_diff_current_norm,
            float((state.total_edges() - state.drawn_edge_count()) / state.total_edges()),
            boxes_remaining_norm,
            float(rows / 5.0),
            float(cols / 5.0),
        ],
        dtype=torch.float32,
    )

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "global_features": global_features,
        "num_edge_nodes": num_edge_nodes,
        "num_box_nodes": rows * cols,
    }