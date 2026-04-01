from engine import DotsAndBoxesState
from agents import HumanCLIPlayer, GreedyAgent
from runner import GameRunner




def main() -> None:
    ##Initializing my state
    state = DotsAndBoxesState(rows=5, cols=5)

    ##Initializing agents
    player0 = HumanCLIPlayer(name="You")
    player1 = GreedyAgent(name="GreedyBot", seed=42)

    runner = GameRunner(
        initial_state=state,
        player0=player0,
        player1=player1,
    )

    print("Dots and Boxes: Human vs Random Bot")
    print("Enter moves in the format: h 0 1  or  v 1 2")
    print("h = horizontal edge, v = vertical edge")
    print()
    print("Initial board:")
    print(state)
    print()

    summary = runner.play(verbose=True)

    print("\nGame over.")
    print(f"Result: {summary.result_string}")
    print(f"Winner: {summary.winner}")
    print(f"Final scores: {summary.final_scores}")


if __name__ == "__main__":
    main()