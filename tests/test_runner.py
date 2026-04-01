from engine import DotsAndBoxesState
from runner import GameRunner
from agents.base import AgentDecision


class FixedAgent:
    def __init__(self, moves, name="FixedAgent"):
        self.moves = list(moves)
        self.name = name

    def select_move(self, state):
        if not self.moves:
            raise ValueError("No scripted moves left.")
        return AgentDecision(move=self.moves.pop(0), reason="scripted")


class IllegalMoveAgent:
    def __init__(self, move, name="IllegalMoveAgent"):
        self.move = move
        self.name = name

    def select_move(self, state):
        return AgentDecision(move=self.move, reason="illegal_test")


def test_runner_step_records_move() -> None:
    state = DotsAndBoxesState(2, 2)
    p0 = FixedAgent([("h", 0, 0)], name="P0")
    p1 = FixedAgent([("h", 0, 1)], name="P1")

    runner = GameRunner(state, p0, p1)
    record = runner.step()

    assert record.move_number == 1
    assert record.acting_player == 0
    assert record.agent_name == "P0"
    assert record.move == ("h", 0, 0)
    assert record.completed_boxes == []
    assert record.scores_after == (0, 0)
    assert record.next_player == 1
    assert len(runner.move_history) == 1


def test_runner_uses_correct_agent_by_turn() -> None:
    state = DotsAndBoxesState(2, 2)
    p0 = FixedAgent([("h", 0, 0)], name="P0")
    p1 = FixedAgent([("h", 0, 1)], name="P1")

    runner = GameRunner(state, p0, p1)
    r1 = runner.step()
    r2 = runner.step()

    assert r1.agent_name == "P0"
    assert r2.agent_name == "P1"


def test_runner_illegal_agent_move_raises() -> None:
    state = DotsAndBoxesState(2, 2)
    p0 = IllegalMoveAgent(("x", 99, 99), name="BadAgent")
    p1 = FixedAgent([("h", 0, 0)], name="P1")

    runner = GameRunner(state, p0, p1)

    try:
        runner.step()
        assert False, "Expected ValueError for illegal agent move"
    except ValueError:
        pass


def test_runner_play_finishes_game() -> None:
    state = DotsAndBoxesState(1, 1)

    # Script a full game
    p0 = FixedAgent([("h", 0, 0), ("h", 1, 0)], name="P0")
    p1 = FixedAgent([("v", 0, 0), ("v", 0, 1)], name="P1")

    runner = GameRunner(state, p0, p1)
    summary = runner.play(verbose=False)

    assert summary.final_state.is_terminal()
    assert len(summary.move_history) == 4
    assert summary.final_scores == (0, 1)
    assert summary.winner == 1
    assert summary.result_string == "player_1_wins"


def test_runner_keeps_same_player_after_scoring_move() -> None:
    state = DotsAndBoxesState(2, 2)

    # p0: h00
    # p1: h10
    # p0: v00
    # p1: v01 completes box and should keep turn
    p0 = FixedAgent([("h", 0, 0), ("v", 0, 0)], name="P0")
    p1 = FixedAgent([("h", 1, 0), ("v", 0, 1)], name="P1")

    runner = GameRunner(state, p0, p1)

    runner.step()  # p0
    runner.step()  # p1
    runner.step()  # p0
    record = runner.step()  # p1 scores

    assert record.acting_player == 1
    assert record.completed_boxes == [(0, 0)]
    assert record.next_player == 1
    assert runner.state.current_player == 1