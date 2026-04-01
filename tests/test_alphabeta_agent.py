from engine import DotsAndBoxesState
from agents import AlphaBetaAgent


def test_alphabeta_returns_legal_move() -> None:
    state = DotsAndBoxesState(2, 2)
    agent = AlphaBetaAgent(depth=2, seed=123)

    decision = agent.select_move(state)

    assert state.is_legal_move(decision.move)
    assert decision.reason == "alphabeta_depth_2"
    assert agent.last_stats is not None
    assert agent.last_stats.depth_limit == 2
    assert agent.last_stats.nodes_expanded >= 1
    assert agent.last_stats.prunes >= 0


def test_alphabeta_takes_available_box() -> None:
    state = DotsAndBoxesState(2, 2)
    agent = AlphaBetaAgent(depth=2, seed=123)

    state.apply_move(("h", 0, 0))  # p0
    state.apply_move(("h", 1, 0))  # p1
    state.apply_move(("v", 0, 0))  # p0

    decision = agent.select_move(state)

    assert decision.move == ("v", 0, 1)


def test_alphabeta_does_not_mutate_input_state() -> None:
    state = DotsAndBoxesState(2, 2)
    before = state.clone()
    agent = AlphaBetaAgent(depth=2, seed=123)

    _ = agent.select_move(state)

    assert state.h_edges == before.h_edges
    assert state.v_edges == before.v_edges
    assert state.box_owners == before.box_owners
    assert state.scores == before.scores
    assert state.current_player == before.current_player