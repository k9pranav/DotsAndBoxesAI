from engine import DotsAndBoxesState
from runner import GameRunner
from agents import MinimaxAgent, GreedyAgent


def main() -> None:
    rows, cols = 3, 3

    state = DotsAndBoxesState(rows=rows, cols=cols)

    player0 = GreedyAgent(name="GreedyBot", seed=42)
    player1 = MinimaxAgent(depth=3, name="minimax_d3", seed=42)

    runner = GameRunner(
        initial_state=state,
        player0=player0,
        player1=player1,
    )

    print(f"Dots and Boxes AI vs AI on {rows}x{cols}")
    print("Initial board:")
    print(state)
    print()

    summary = runner.play(verbose=True)

    print("\nGame over.")
    print(f"Result: {summary.result_string}")
    print(f"Winner: {summary.winner}")
    print(f"Final scores: {summary.final_scores}")
    print(f"Total moves: {len(summary.move_history)}")


if __name__ == "__main__":
    main()