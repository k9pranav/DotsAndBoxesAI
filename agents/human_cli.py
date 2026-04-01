from __future__ import annotations

from engine import DotsAndBoxesState, Edge
from agents.base import AgentDecision


class HumanCLIPlayer:
    """
    Simple CLI human controller.

    Expected input format:
        h 0 1
        v 2 0

    Where:
      - h = horizontal edge
      - v = vertical edge
      - row and col are integers
    """

    def __init__(self, name: str = "HumanCLI") -> None:
        self.name = name

    def select_move(self, state: DotsAndBoxesState) -> AgentDecision:
        while True:
            raw = input(
                f"Player {state.current_player}, enter move as [h/v row col]: "
            ).strip()

            parts = raw.split()
            if len(parts) != 3:
                print("Invalid format. Example: h 0 1")
                continue

            orientation = parts[0].lower()

            if orientation not in ("h", "v"):
                print("First value must be 'h' or 'v'.")
                continue

            try:
                r = int(parts[1])
                c = int(parts[2])
            except ValueError:
                print("Row and col must be integers.")
                continue

            move: Edge = (orientation, r, c)

            if not state.is_valid_edge(move):
                print(f"Invalid edge coordinates: {move}")
                continue

            if not state.is_legal_move(move):
                print(f"Illegal move (already drawn): {move}")
                continue

            return AgentDecision(move=move, reason="human_input")