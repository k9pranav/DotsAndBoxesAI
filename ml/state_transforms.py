from __future__ import annotations

import random
from typing import Any

from engine import DotsAndBoxesState


def serialize_state(state: DotsAndBoxesState) -> dict[str, Any]:
    return {
        "rows": state.rows,
        "cols": state.cols,
        "current_player": state.current_player,
        "scores": state.scores[:],
        "h_edges": [[bool(x) for x in row] for row in state.h_edges],
        "v_edges": [[bool(x) for x in row] for row in state.v_edges],
        "box_owners": [[owner for owner in row] for row in state.box_owners],
    }


def deserialize_state(obj: dict[str, Any]) -> DotsAndBoxesState:
    state = DotsAndBoxesState(
        rows=int(obj["rows"]),
        cols=int(obj["cols"]),
        current_player=int(obj["current_player"]),
    )
    state.scores = [int(obj["scores"][0]), int(obj["scores"][1])]
    state.h_edges = [[bool(x) for x in row] for row in obj["h_edges"]]
    state.v_edges = [[bool(x) for x in row] for row in obj["v_edges"]]
    state.box_owners = [[owner for owner in row] for row in obj["box_owners"]]
    return state


def hflip_state(state: DotsAndBoxesState) -> DotsAndBoxesState:
    out = DotsAndBoxesState(state.rows, state.cols, state.current_player)
    out.scores = state.scores[:]

    for r in range(state.rows + 1):
        for c in range(state.cols):
            out.h_edges[r][state.cols - 1 - c] = state.h_edges[r][c]

    for r in range(state.rows):
        for c in range(state.cols + 1):
            out.v_edges[r][state.cols - c] = state.v_edges[r][c]

    for r in range(state.rows):
        for c in range(state.cols):
            out.box_owners[r][state.cols - 1 - c] = state.box_owners[r][c]

    return out


def vflip_state(state: DotsAndBoxesState) -> DotsAndBoxesState:
    out = DotsAndBoxesState(state.rows, state.cols, state.current_player)
    out.scores = state.scores[:]

    for r in range(state.rows + 1):
        for c in range(state.cols):
            out.h_edges[state.rows - r][c] = state.h_edges[r][c]

    for r in range(state.rows):
        for c in range(state.cols + 1):
            out.v_edges[state.rows - 1 - r][c] = state.v_edges[r][c]

    for r in range(state.rows):
        for c in range(state.cols):
            out.box_owners[state.rows - 1 - r][c] = state.box_owners[r][c]

    return out


def rotate180_state(state: DotsAndBoxesState) -> DotsAndBoxesState:
    return vflip_state(hflip_state(state))


def rotate90_cw_state(state: DotsAndBoxesState) -> DotsAndBoxesState:
    if state.rows != state.cols:
        raise ValueError("90-degree rotation is only supported for square boards.")
    n = state.rows

    out = DotsAndBoxesState(n, n, state.current_player)
    out.scores = state.scores[:]

    # Old horizontal -> new vertical
    for r in range(n + 1):
        for c in range(n):
            out.v_edges[c][n - r] = state.h_edges[r][c]

    # Old vertical -> new horizontal
    for r in range(n):
        for c in range(n + 1):
            out.h_edges[c][n - 1 - r] = state.v_edges[r][c]

    for r in range(n):
        for c in range(n):
            out.box_owners[c][n - 1 - r] = state.box_owners[r][c]

    return out


def rotate270_cw_state(state: DotsAndBoxesState) -> DotsAndBoxesState:
    return rotate90_cw_state(rotate180_state(state))


def random_symmetry_transform(
    state: DotsAndBoxesState,
    rng: random.Random,
) -> DotsAndBoxesState:
    transforms = [
        lambda s: s.clone(),
        hflip_state,
        vflip_state,
        rotate180_state,
    ]

    if state.rows == state.cols:
        transforms.extend([rotate90_cw_state, rotate270_cw_state])

    fn = rng.choice(transforms)
    return fn(state)