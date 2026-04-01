from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

#The Edge struct. "h" for horizontal, "v" for vertical, the 2 ints to signify the location
Edge = Tuple[str, int, int] 

#Keeps track of a move 
@dataclass(frozen=True)
class MoveResult:
    move: Edge
    player: int
    completed_boxes: List[Tuple[int, int]]
    next_player: int
    game_over: bool

@dataclass
class DotsAndBoxesState:
    """
    Resusable Dots and Boxes engine for a board with rows x cols boxes
    IMPORTANT NOTe: rows x cols mean BOXES, not dots

    Example:
        rows = 2, cols = 2 -> 2x2
        rows=3, cols=3 -> 3x3

    Players are represented as 0 and 1
    """

    rows: int
    cols: int
    current_player: int = 0

    #For Edges -> 
    
    #Horizontal Edges; 0 is empty, 1 is drawmn
    h_edges: List[List[bool]] = field(init=False)

    #Vertical Edges; 0 is empty, 1 is drawmn
    v_edges: List[List[bool]] = field(init=False)

    #Box Owners: {None, 0, 1} -> 0, 1 here represent the players
    box_owners: List[List[Optional[int]]] = field(init=False)

    #Score of the player, indexed by 0, 1 ie player int
    scores: List[int] = field(init=False)

    def __post_init__(self) ->None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Rows and Cols must be positive numbers!")
        
        self.h_edges = [[False for _ in range(self.cols)] for _ in range(self.rows + 1)]
        self.v_edges = [[False for _ in range(self.cols + 1)] for _ in range(self.rows)]
        self.box_owners = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.scores = [0, 0]
    
    #State Helpers
    def clone(self) -> "DotsAndBoxesState":
        new_state = DotsAndBoxesState(self.rows, self.cols, self.current_player)
        new_state.h_edges = [row[:] for row in self.h_edges]
        new_state.v_edges = [row[:] for row in self.v_edges]
        new_state.box_owners = [row[:] for row in self.box_owners]
        new_state.scores = self.scores[:]
        return new_state

    def reset(self) -> None:
        self.h_edges = [[False for _ in range(self.cols)] for _ in range(self.rows + 1)]
        self.v_edges = [[False for _ in range(self.cols + 1)] for _ in range(self.rows)]
        self.box_owners = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.scores = [0, 0]
        self.current_player = 0
    
    #Edge Validation And Access

    def is_valid_edge(self, edge: Edge) -> bool:
        orientation, r, c = edge
        if orientation == 'h':
            return 0 <= r <= self.rows and 0 <= c < self.cols
        if orientation == 'v':
            return 0 <= r < self.rows and 0 <= c <= self.cols
        return False
    
    def is_edge_drawn(self, edge:Edge) -> bool:
        if not self.is_valid_edge(edge):
            raise ValueError(f"Invalid edge: {edge}")
        orientation, r, c = edge
        if orientation == 'h':
            return self.h_edges[r][c]
        else:
            return self.v_edges[r][c]
    
    def is_legal_move(self, edge:Edge) -> bool:
        return self.is_valid_edge(edge) and not self.is_edge_drawn(edge)

    def legal_moves(self) -> List[Edge]:
        moves: List[Edge] = []
        for r in range(self.rows + 1):
            for c in range(self.cols):
                if not self.h_edges[r][c]:
                    moves.append(('h', r, c))
        for r in range(self.rows):
            for c in range(self.cols + 1):
                if not self.v_edges[r][c]:
                    moves.append(('v', r, c))
        return moves
    
    def num_legal_moves(self) -> int:
        return len(self.legal_moves())
    
    def total_edges(self) -> int:
        return (self.rows + 1) * self.cols + self.rows * (self.cols + 1)
    
    def drawn_edge_count(self) -> int:
        h_count = sum(int(cell) for row in self.h_edges for cell in row) 
        v_count = sum(int(cell) for row in self.v_edges for cell in row)
        return h_count + v_count

    #Box logic; private functions

    #Again, note that rows and cols in the class Represent BOXES!

    def _box_is_complete(self, box_r: int, box_c: int) -> bool:
        
        return (
            self.h_edges[box_r][box_c]
            and self.h_edges[box_r + 1][box_c]
            and self.v_edges[box_r][box_c]
            and self.v_edges[box_r][box_c + 1]
        )
    
    #If i draw this line, what boxes could it potentially close?

    def _adjacent_boxes_for_edge(self, edge:Edge) -> List[Tuple[int, int]]:
        orientation, r, c = edge
        boxes: List[Tuple[int, int]] = [] #List of boxes that can potentially be closed

        if orientation == 'h':
            #Box above edge
            if r - 1 >= 0:
                boxes.append((r - 1, c))
            # Box below the edge
            if r < self.rows:
                boxes.append((r, c))
        else: #orientation is v
            # Box to the left of the edge
            if c - 1 >= 0:
                boxes.append((r, c - 1))
            # Box to the right of the edge
            if c < self.cols:
                boxes.append((r, c))

        return boxes
    
    #For that specific box (row, col), how many of its 4 walls are currently drawn
    def count_box_sides(self, box_r: int, box_c: int) -> int:
        if not (0 <= box_r < self.rows and 0 <= box_c < self.cols):
            raise ValueError(f"Invalid box coordinate: {(box_r, box_c)}")
        return int(self.h_edges[box_r][box_c]) + int(self.h_edges[box_r + 1][box_c]) + int(self.v_edges[box_r][box_c]) + int(self.v_edges[box_r][box_c + 1])

    #Scans the board and finds boxes with exactly n sides filled
    #Would be helpful for AI implementation
    def boxes_with_n_sides(self, n: int) -> List[Tuple[int, int]]:
        if not (0 <= n <= 4):
            raise ValueError("n must be between 0 and 4")
        
        found: List[Tuple[int, int]] = []

        for r in range(self.rows):
            for c in range(self.cols):
                if self.count_box_sides(r, c) == n:
                    found.append((r, c))
        return found
    
    #Applying Moves

    #Returns a move result; DOES NOT CHANGE THE DotsAndBoxesState
    def apply_move(self, edge: Edge) -> MoveResult:
        if not self.is_valid_edge(edge):
            raise ValueError(f"Invalid edge: {edge}")
        if self.is_edge_drawn(edge):
            raise ValueError(f"Illegal move; edge already drawn: {edge}")
        if self.is_terminal():
            raise ValueError("Cannot apply move to a finished game")

        acting_player = self.current_player
        orientation, r, c = edge

        if orientation == 'h':
            self.h_edges[r][c] = True
        else:
            self.v_edges[r][c] = True

        completed_boxes: List[Tuple[int, int]] = []
        for box_r, box_c in self._adjacent_boxes_for_edge(edge):
            if self.box_owners[box_r][box_c] is None and self._box_is_complete(box_r, box_c):
                self.box_owners[box_r][box_c] = acting_player
                self.scores[acting_player] += 1
                completed_boxes.append((box_r, box_c))

        if not completed_boxes:
            self.current_player = 1 - self.current_player

        return MoveResult(
            move=edge,
            player=acting_player,
            completed_boxes=completed_boxes,
            next_player=self.current_player,
            game_over=self.is_terminal(),
        )
    
    #
    def next_state(self, edge: Edge) -> "DotsAndBoxesState":
        new_state = self.clone()
        new_state.apply_move(edge)
        return new_state
    
    #GameOver/Winner logic

    #NO more edges left!
    def is_terminal(self) -> bool:
        return self.drawn_edge_count() == self.total_edges()
    
    def winner(self) -> Optional[int]:
        if not self.is_terminal():
            return None
        if self.scores[0] > self.scores[1]:
            return 0
        if self.scores[1] > self.scores[0]:
            return 1
        return None

    def result_string(self) -> str:
        if not self.is_terminal():
            return "in_progress"
        w = self.winner()
        if w is None:
            return "draw"
        return f"player_{w}_wins"
    
    #Potential Useful utilities for AI training/debugging
    def score_difference(self, perspective_player: int = 0) -> int:
        if perspective_player not in (0, 1):
            raise ValueError("perspective_player must be 0 or 1")
        return self.scores[perspective_player] - self.scores[1 - perspective_player]

    # Maps and edge to an integer -> NN and Q-Learning Tables can not easily output a tuple; prefer single ints
    # Treats all horizontal edges as the first block of numbers and all vertical edges as the second block.
    def edge_to_index(self, edge: Edge) -> int:
        
        if not self.is_valid_edge(edge):
            raise ValueError(f"Invalid edge: {edge}")
        orientation, r, c = edge
        num_h = (self.rows + 1) * self.cols
        if orientation == 'h':
            return r * self.cols + c
        return num_h + r * (self.cols + 1) + c
    
    #Inverse of edge_to_index
    def index_to_edge(self, idx: int) -> Edge:
        num_h = (self.rows + 1) * self.cols
        total = self.total_edges()
        if not (0 <= idx < total):
            raise ValueError(f"Invalid edge index: {idx}")
        if idx < num_h:
            r, c = divmod(idx, self.cols)
            return ('h', r, c)
        idx -= num_h
        r, c = divmod(idx, self.cols + 1)
        return ('v', r, c)
    
    #Converts the game state to a 3D list
    #Layer 0: A map of all horizontal lines.
    #Layer 1: A map of all vertical lines.
    #Layer 2: A map of who owns which box.
    def observation_tensor(self) -> List[List[List[int]]]:
        h = [[int(x) for x in row] for row in self.h_edges]
        v = [[int(x) for x in row] for row in self.v_edges]
        b = [[(-1 if owner is None else owner) for owner in row] for row in self.box_owners]
        return [h, v, b]
    
    #ASCI Rendereres
    def to_ascii(self) -> str:
        """
        Produces a human-readable ASCII rendering.

        Example symbols:
            .   = dot
            --- = drawn horizontal edge
            |   = drawn vertical edge
            A/B = claimed box owner
        """
        lines: List[str] = []

        for r in range(self.rows + 1):
            # Dot row with horizontal edges
            top_parts: List[str] = []
            for c in range(self.cols):
                top_parts.append('.')
                top_parts.append('---' if self.h_edges[r][c] else '   ')
            top_parts.append('.')
            lines.append(''.join(top_parts))

            if r < self.rows:
                mid_parts: List[str] = []
                for c in range(self.cols):
                    mid_parts.append('|' if self.v_edges[r][c] else ' ')
                    owner = self.box_owners[r][c]
                    if owner is None:
                        mid_parts.append('   ')
                    else:
                        mid_parts.append(f' {"A" if owner == 0 else "B"} ')
                mid_parts.append('|' if self.v_edges[r][self.cols] else ' ')
                lines.append(''.join(mid_parts))

        return '\n'.join(lines)
    

    #For the print statements
    def __str__(self) -> str:
        status = (
            f"DotsAndBoxesState(rows={self.rows}, cols={self.cols}, "
            f"current_player={self.current_player}, scores={self.scores}, "
            f"result={self.result_string()})"
        )
        return status + "\n" + self.to_ascii()
    
def initial_state(rows: int, cols: int) -> DotsAndBoxesState:
    return DotsAndBoxesState(rows=rows, cols=cols)


def other_player(player: int) -> int:
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    return 1 - player


def all_edges(rows: int, cols: int) -> Iterable[Edge]:
    for r in range(rows + 1):
        for c in range(cols):
            yield ('h', r, c)
    for r in range(rows):
        for c in range(cols + 1):
            yield ('v', r, c)


if __name__ == "__main__":
    # Minimal smoke demo
    state = DotsAndBoxesState(2, 2)
    print("Initial state:")
    print(state)
    print()

    demo_moves = [
        ('v', 0, 0),
        ('v', 1, 2)
    ]

    for mv in demo_moves:
        result = state.apply_move(mv)
        print(f"Applied move: {mv}")
        print(f"Completed boxes: {result.completed_boxes}")
        print(f"Next player: {result.next_player}")
        print(state)
        print('-' * 40)



    



    





