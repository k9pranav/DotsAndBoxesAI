from engine import DotsAndBoxesState
from agents import RandomAgent, GreedyAgent


def test_random_agent_returns_legal_move() -> None:
    state = DotsAndBoxesState(2, 2)
    agent = RandomAgent(seed=123)

    decision = agent.select_move(state)

    assert state.is_legal_move(decision.move)
    assert decision.reason == "random_legal_move"


def test_greedy_agent_takes_available_box() -> None:
    state = DotsAndBoxesState(2, 2)
    agent = GreedyAgent(seed=123)

    state.apply_move(("h", 0, 0))  # p0
    state.apply_move(("h", 1, 0))  # p1
    state.apply_move(("v", 0, 0))  # p0

    # Now p1 can complete (0,0) with v(0,1)
    decision = agent.select_move(state)

    assert decision.move == ("v", 0, 1)
    assert decision.reason == "immediate_capture_1"


def test_greedy_agent_returns_legal_non_scoring_move_when_no_capture_exists() -> None:
    state = DotsAndBoxesState(2, 2)
    agent = GreedyAgent(seed=123)

    decision = agent.select_move(state)

    assert state.is_legal_move(decision.move)
    assert decision.reason == "no_immediate_capture"


def test_agents_do_not_mutate_input_state() -> None:
    state = DotsAndBoxesState(2, 2)
    random_agent = RandomAgent(seed=1)
    greedy_agent = GreedyAgent(seed=1)

    before_random = state.clone()
    _ = random_agent.select_move(state)
    assert state.h_edges == before_random.h_edges
    assert state.v_edges == before_random.v_edges
    assert state.box_owners == before_random.box_owners
    assert state.scores == before_random.scores
    assert state.current_player == before_random.current_player

    before_greedy = state.clone()
    _ = greedy_agent.select_move(state)
    assert state.h_edges == before_greedy.h_edges
    assert state.v_edges == before_greedy.v_edges
    assert state.box_owners == before_greedy.box_owners
    assert state.scores == before_greedy.scores
    assert state.current_player == before_greedy.current_player