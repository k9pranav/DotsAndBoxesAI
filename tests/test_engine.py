from engine import DotsAndBoxesState


def test_initial_legal_move_count_2x2() -> None:
    state = DotsAndBoxesState(2, 2)
    assert state.total_edges() == 12
    assert state.num_legal_moves() == 12
    assert len(state.legal_moves()) == 12
    assert not state.is_terminal()
    assert state.scores == [0, 0]
    assert state.current_player == 0


def test_invalid_edge_detection() -> None:
    state = DotsAndBoxesState(2, 2)

    assert not state.is_valid_edge(("x", 0, 0))
    assert not state.is_valid_edge(("h", -1, 0))
    assert not state.is_valid_edge(("h", 3, 0))
    assert not state.is_valid_edge(("h", 0, 2))
    assert not state.is_valid_edge(("v", 2, 0))
    assert not state.is_valid_edge(("v", 0, 3))


def test_apply_non_scoring_move_switches_turn() -> None:
    state = DotsAndBoxesState(2, 2)

    result = state.apply_move(("h", 0, 0))

    assert result.completed_boxes == []
    assert result.player == 0
    assert result.next_player == 1
    assert state.current_player == 1
    assert state.scores == [0, 0]


def test_single_box_completion_scores_and_keeps_turn() -> None:
    state = DotsAndBoxesState(2, 2)

    state.apply_move(("h", 0, 0))  # p0
    state.apply_move(("h", 1, 0))  # p1
    state.apply_move(("v", 0, 0))  # p0

    # no box yet; turn should be p1
    assert state.current_player == 1

    result = state.apply_move(("v", 0, 1))  # p1 completes top-left box

    assert result.completed_boxes == [(0, 0)]
    assert result.player == 1
    assert result.next_player == 1
    assert state.current_player == 1
    assert state.box_owners[0][0] == 1
    assert state.scores == [0, 1]


def test_double_box_completion_in_one_move() -> None:
    state = DotsAndBoxesState(2, 1)  # two stacked boxes, one shared middle horizontal

    # Fill all except the shared middle horizontal edge h(1,0)
    state.apply_move(("h", 0, 0))  # p0
    state.apply_move(("v", 0, 0))  # p1
    state.apply_move(("v", 0, 1))  # p0
    state.apply_move(("h", 2, 0))  # p1
    state.apply_move(("v", 1, 0))  # p0
    state.apply_move(("v", 1, 1))  # p1

    # Now p0 to move; h(1,0) completes both boxes
    assert state.current_player == 0
    result = state.apply_move(("h", 1, 0))

    assert sorted(result.completed_boxes) == [(0, 0), (1, 0)]
    assert result.player == 0
    assert result.next_player == 0
    assert state.scores == [2, 0]
    assert state.box_owners[0][0] == 0
    assert state.box_owners[1][0] == 0


def test_repeated_edge_is_illegal() -> None:
    state = DotsAndBoxesState(2, 2)
    state.apply_move(("h", 0, 0))

    try:
        state.apply_move(("h", 0, 0))
        assert False, "Expected ValueError for repeated edge"
    except ValueError:
        pass


def test_next_state_does_not_mutate_original() -> None:
    state = DotsAndBoxesState(2, 2)
    new_state = state.next_state(("h", 0, 0))

    assert state.is_edge_drawn(("h", 0, 0)) is False
    assert new_state.is_edge_drawn(("h", 0, 0)) is True
    assert state.current_player == 0
    assert new_state.current_player == 1


def test_clone_is_independent() -> None:
    state = DotsAndBoxesState(2, 2)
    clone = state.clone()

    clone.apply_move(("h", 0, 0))

    assert not state.is_edge_drawn(("h", 0, 0))
    assert clone.is_edge_drawn(("h", 0, 0))


def test_terminal_and_winner() -> None:
    state = DotsAndBoxesState(1, 1)

    # 4 edges total
    state.apply_move(("h", 0, 0))  # p0
    state.apply_move(("v", 0, 0))  # p1
    state.apply_move(("h", 1, 0))  # p0
    state.apply_move(("v", 0, 1))  # p1 gets box

    assert state.is_terminal()
    assert state.winner() == 1
    assert state.result_string() == "player_1_wins"
    assert state.scores == [0, 1]


def test_draw_result_string() -> None:
    state = DotsAndBoxesState(1, 2)

    # Build a 1-2 tie. There are 7 edges.
    state.apply_move(("h", 0, 0))  # p0
    state.apply_move(("h", 0, 1))  # p1
    state.apply_move(("v", 0, 0))  # p0
    state.apply_move(("v", 0, 2))  # p1
    state.apply_move(("h", 1, 0))  # p0
    state.apply_move(("h", 1, 1))  # p1

    # p0 plays middle vertical and completes both? No, this would award both.
    # So instead create a direct tied final state manually:
    tied = DotsAndBoxesState(1, 2)
    tied.h_edges = [[True, True], [True, True]]
    tied.v_edges = [[True, True, True]]
    tied.box_owners = [[0, 1]]
    tied.scores = [1, 1]
    tied.current_player = 0

    assert tied.is_terminal()
    assert tied.winner() is None
    assert tied.result_string() == "draw"


def test_edge_index_roundtrip() -> None:
    state = DotsAndBoxesState(3, 3)

    for move in state.legal_moves():
        idx = state.edge_to_index(move)
        recovered = state.index_to_edge(idx)
        assert recovered == move


def test_boxes_with_n_sides() -> None:
    state = DotsAndBoxesState(2, 2)

    state.apply_move(("h", 0, 0))
    state.apply_move(("v", 0, 0))
    state.apply_move(("h", 1, 0))

    assert (0, 0) in state.boxes_with_n_sides(3)