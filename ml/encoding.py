from __future__ import annotations

from typing import List

from engine import DotsAndBoxesState


def encode_state_flat(state: DotsAndBoxesState) -> List[float]:
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

    return h_flat + v_flat + b_flat + current_player


def legal_move_mask(state: DotsAndBoxesState) -> List[int]:
    """
    Returns a binary mask over all actions:
      1 = legal
      0 = illegal
    """
    mask = [0 for _ in range(state.total_edges())]
    for move in state.legal_moves():
        idx = state.edge_to_index(move)
        mask[idx] = 1
    return mask