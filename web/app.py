from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import DotsAndBoxesState
from agents import RandomAgent, GreedyAgent, MinimaxAgent, AlphaBetaAgent, AlphaBetaBoxGNNValueAgent


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Dots and Boxes MVP")

#Session State
@dataclass
class GameSession:
    game_id: str
    state: DotsAndBoxesState
    bot_name:str
    bot_agent: Any
    human_player: int
    bot_player: int
    created_at: float = field(default_factory=time.time)
    move_history: list[dict[str, Any]] = field(default_factory=list)

GAMES: dict[str, GameSession] = {}

#Request models for the web appls
class NewGameRequest(BaseModel):
    rows: int = Field(ge=2, le=5)
    cols: int = Field(ge=2, le=5)
    bot: str
    human_starts: bool = True

class MoveRequest(BaseModel):
    orientation: str
    row: int
    col: int

#Helpers
def make_bot(bot_name: str):
    bot_name = bot_name.lower()

    if bot_name == "random":
        return RandomAgent(name="RandomBot")
    if bot_name == "greedy":
        return GreedyAgent(name="GreedyBot")
    if bot_name == "minimax_d2":
        return MinimaxAgent(depth=2, name="Minimax_d2")
    if bot_name == "minimax_d3":
        return MinimaxAgent(depth=3, name="Minimax_d3")
    if bot_name == "alphabeta_d2":
        return AlphaBetaAgent(depth=2, name="AlphaBeta_d2")
    if bot_name == "alphabeta_d3":
        return AlphaBetaAgent(depth=3, name="AlphaBeta_d3")
    if bot_name == "graphical_nn":
        return AlphaBetaBoxGNNValueAgent(checkpoint_path="checkpoints/box_gnn_value_5x5_searchvalue.pt",
        depth=2, name="Graphical_nn")

    raise ValueError(f"Unknown bot: {bot_name}")


def serialize_state(session: GameSession) -> dict[str, Any]:
    '''
    Converting gameSession into JSON
    '''
    
    state = session.state

    return {
        "game_id": session.game_id,
        "rows": state.rows,
        "cols": state.cols,
        "current_player": state.current_player,
        "human_player": session.human_player,
        "bot_player": session.bot_player,
        "bot_name": session.bot_name,
        "scores": state.scores,
        "h_edges": state.h_edges,
        "v_edges": state.v_edges,
        "box_owners": state.box_owners,
        "is_terminal": state.is_terminal(),
        "winner": state.winner(),
        "result_string": state.result_string(),
        "move_history": session.move_history,
    }


def record_move(session:GameSession, player:int, move: tuple[str, int, int], completed_boxes: list[tuple[int, int]], actor_name:str) -> None:
    session.move_history.append(
        {
            "move_number":len(session.move_history) + 1,
            "player":player,
            "actor_name":actor_name,
            "move": {
                    "orientation":move[0],
                    "row":move[1],
                    "col":move[2]
                },
            "completed_boxes": [{"row": r, "col":c} for (r,c) in completed_boxes],
            "scores_after": session.state.scores[:],
            "next_player":session.state.current_player,               
        }        
    )

def autoplay_bot(session: GameSession) -> None:
    while (not session.state.is_terminal() and session.state.current_player == session.bot_player):
        
        decision = session.bot_agent.select_move(session.state.clone())
        move = decision.move
        acting_player = session.state.current_player
        result = session.state.apply_move(move)

        record_move(
            session=session,
            player=acting_player,
            move=move,
            completed_boxes=result.completed_boxes,
            actor_name=session.bot_agent.name,
        )

#API routes
@app.post("/api/new-game")
def new_game(req: NewGameRequest):
    try:
        bot_agent = make_bot(req.bot)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    state = DotsAndBoxesState(rows=req.rows, cols=req.cols)
    
    human_player = 0 if req.human_starts else 1
    bot_player = 1 - human_player

    if not req.human_starts:
        state.current_player = bot_player

    game_id = str(uuid.uuid4())
    session = GameSession(
        game_id=game_id,
        state=state,
        bot_name=req.bot,
        bot_agent=bot_agent,
        human_player=human_player,
        bot_player=bot_player
    )

    GAMES[game_id] = session

    if session.state.current_player == session.bot_player:
        autoplay_bot(session)

    return serialize_state(session)

@app.get("/api/game/{game_id}")
def get_game(game_id: str):
    session = GAMES.get(game_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return serialize_state(session)

@app.post("/api/game/{game_id}/move")
def make_move(game_id:str, req:MoveRequest):
    session = GAMES.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")

    if session.state.is_terminal():
        raise HTTPException(status_code=400, detail="Game is already over")

    if session.state.current_player != session.human_player:
        raise HTTPException(status_code=400, detail="It is not the human player's turn")

    move = (req.orientation, req.row, req.col)

    if not session.state.is_valid_edge(move):
        raise HTTPException(status_code=400, detail=f"Invalid edge: {move}")

    if not session.state.is_legal_move(move):
        raise HTTPException(status_code=400, detail=f"Illegal move: {move}")

    acting_player = session.state.current_player
    result = session.state.apply_move(move)
    record_move(
        session=session,
        player=acting_player,
        move=move,
        completed_boxes=result.completed_boxes,
        actor_name="You",
    )

    autoplay_bot(session)

    return serialize_state(session)


#Static Fronted
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
    

        
