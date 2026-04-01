let gameState = null;
let isAnimatingBot = false;

const BOT_MOVE_DELAY_MS = 400;

const boardSvg = document.getElementById("boardSvg");
const newGameBtn = document.getElementById("newGameBtn");
const rowsSelect = document.getElementById("rowsSelect");
const colsSelect = document.getElementById("colsSelect");
const botSelect = document.getElementById("botSelect");
const humanStartsCheckbox = document.getElementById("humanStartsCheckbox");

const statusText = document.getElementById("statusText");
const scoreText = document.getElementById("scoreText");
const turnText = document.getElementById("turnText");
const moveLog = document.getElementById("moveLog");

newGameBtn.addEventListener("click", async () => {
  await createNewGame();
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function createNewGame() {
  const payload = {
    rows: parseInt(rowsSelect.value, 10),
    cols: parseInt(colsSelect.value, 10),
    bot: botSelect.value,
    human_starts: humanStartsCheckbox.checked,
  };

  const response = await fetch("/api/new-game", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const err = await response.json();
    alert(err.detail || "Failed to create game");
    return;
  }

  gameState = await response.json();
  renderAll();
}

async function sendMove(orientation, row, col) {
  if (!gameState || gameState.is_terminal || isAnimatingBot) {
    return;
  }

  // Prevent double clicks while request is in flight
  isAnimatingBot = true;

  // Save the original state
  const originalState = JSON.parse(JSON.stringify(gameState));

  // 1. Show the human move instantly
  const optimisticState = JSON.parse(JSON.stringify(gameState));
  applyOptimisticHumanMove(optimisticState, orientation, row, col);
  gameState = optimisticState;
  renderAll();

  try {
    // 2. Ask backend for the real updated state
    const response = await fetch(`/api/game/${originalState.game_id}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        orientation,
        row,
        col,
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      alert(err.detail || "Move failed");

      // rollback if move failed
      gameState = originalState;
      renderAll();
      isAnimatingBot = false;
      return;
    }

    const finalState = await response.json();

    // 3. Delay only the bot reveal
    await revealBotChangesAfterDelay(optimisticState, finalState, orientation, row, col);

    isAnimatingBot = false;
    renderAll();

  } catch (err) {
    console.error(err);
    gameState = originalState;
    renderAll();
  }

  isAnimatingBot = false;
}

function applyOptimisticHumanMove(state, orientation, row, col) {
  if (orientation === "h") {
    state.h_edges[row][col] = true;
  } else {
    state.v_edges[row][col] = true;
  }

  // Usually after human move, turn goes to bot unless a box was completed.
  // We do not know completed boxes yet, so this is only a temporary visual state.
  state.current_player = state.bot_player;
}

async function revealBotChangesAfterDelay(optimisticState, finalState, humanOrientation, humanRow, humanCol) {
  // Find all moves added by the server after the optimistic human move
  const oldLen = optimisticState.move_history.length;
  const newEntries = finalState.move_history.slice(oldLen);

  // If backend includes the human move in move_history, skip it
  const botEntries = newEntries.filter((entry) => {
    const m = entry.move;
    const isHumanMove =
      m.orientation === humanOrientation &&
      m.row === humanRow &&
      m.col === humanCol &&
      entry.player === finalState.human_player;

    return !isHumanMove;
  });

  // First update state to final human consequences if needed
  // For example: completed boxes, score, extra turn handling, etc.
  // We do this by taking final state and temporarily removing bot moves visually.
  const preBotState = buildStateBeforeBotMoves(finalState, botEntries);
  gameState = preBotState;
  renderAll();

  if (botEntries.length > 0) {
    await sleep(BOT_MOVE_DELAY_MS);
  }

  // Now show bot result
  gameState = finalState;
  renderAll();
}

function buildStateBeforeBotMoves(finalState, botEntries) {
  const state = JSON.parse(JSON.stringify(finalState));

  // Remove bot moves from edge grids
  for (const entry of botEntries) {
    const m = entry.move;
    if (m.orientation === "h") {
      state.h_edges[m.row][m.col] = false;
    } else {
      state.v_edges[m.row][m.col] = false;
    }
  }

  // Rebuild box owners from move history without botEntries
  for (let r = 0; r < state.rows; r++) {
    for (let c = 0; c < state.cols; c++) {
      state.box_owners[r][c] = null;
    }
  }

  const filteredHistory = finalState.move_history.filter(
    (entry) => !botEntries.includes(entry)
  );

  for (const entry of filteredHistory) {
    for (const box of entry.completed_boxes) {
      state.box_owners[box.row][box.col] = entry.player;
    }
  }

  state.move_history = filteredHistory;

  // Recompute scores from box owners
  state.scores = [0, 0];
  for (let r = 0; r < state.rows; r++) {
    for (let c = 0; c < state.cols; c++) {
      const owner = state.box_owners[r][c];
      if (owner !== null) {
        state.scores[owner] += 1;
      }
    }
  }

  // After human move but before bot reveal, it should visually be bot's turn
  if (!state.is_terminal) {
    state.current_player = state.bot_player;
  }

  return state;
}

async function animateTransition(oldState, newState) {
  const oldHistoryLen = oldState.move_history.length;
  const newMoves = newState.move_history.slice(oldHistoryLen);

  if (newMoves.length === 0) {
    gameState = newState;
    renderAll();
    return;
  }

  isAnimatingBot = true;

  let workingState = JSON.parse(JSON.stringify(oldState));

  for (let i = 0; i < newMoves.length; i++) {
    const entry = newMoves[i];
    applyMoveToState(workingState, entry);

    gameState = JSON.parse(JSON.stringify(workingState));
    renderAll();

    const isLast = i === newMoves.length - 1;
    if (!isLast && entry.player === newState.bot_player) {
      await sleep(BOT_MOVE_DELAY_MS);
    }
  }

  gameState = newState;
  renderAll();
  isAnimatingBot = false;
}

function applyMoveToState(state, entry) {
  const move = entry.move;

  if (move.orientation === "h") {
    state.h_edges[move.row][move.col] = true;
  } else {
    state.v_edges[move.row][move.col] = true;
  }

  for (const box of entry.completed_boxes) {
    state.box_owners[box.row][box.col] = entry.player;
  }

  state.move_history.push(entry);
  state.scores = [...entry.scores_after];
  state.current_player = entry.next_player;
  state.is_terminal = entry.is_terminal_after;
  state.winner = entry.winner_after;
}

function renderAll() {
  renderStatus();
  renderMoveLog();
  renderBoard();
}

function renderStatus() {
  if (!gameState) {
    statusText.textContent = "Create a game to begin.";
    scoreText.textContent = "Scores: -";
    turnText.textContent = "Turn: -";
    return;
  }

  const humanScore = gameState.scores[gameState.human_player];
  const botScore = gameState.scores[gameState.bot_player];

  scoreText.textContent = `You: ${humanScore} | Bot: ${botScore}`;

  if (gameState.is_terminal) {
    if (gameState.winner === null) {
      statusText.textContent = "Game over: draw.";
    } else if (gameState.winner === gameState.human_player) {
      statusText.textContent = "Game over: you win.";
    } else {
      statusText.textContent = "Game over: bot wins.";
    }
    turnText.textContent = "Turn: finished";
  } else {
    const whoseTurn =
      gameState.current_player === gameState.human_player ? "You" : "Bot";
    statusText.textContent = `Bot: ${gameState.bot_name}`;
    turnText.textContent = `Turn: ${whoseTurn}`;
  }
}

function renderMoveLog() {
  moveLog.innerHTML = "";

  if (!gameState) return;

  for (const entry of gameState.move_history) {
    const div = document.createElement("div");
    div.className = "move-entry";

    const move = entry.move;
    const boxText =
      entry.completed_boxes.length > 0
        ? ` | boxes: ${entry.completed_boxes.length}`
        : "";

    div.textContent =
      `#${entry.move_number} ${entry.actor_name} -> ` +
      `${move.orientation}(${move.row},${move.col})${boxText}`;

    moveLog.appendChild(div);
  }
}

function renderBoard() {
  boardSvg.innerHTML = "";

  if (!gameState) return;

  const rows = gameState.rows;
  const cols = gameState.cols;

  const margin = 80;
  const cellSize = Math.min(
    (700 - 2 * margin) / cols,
    (700 - 2 * margin) / rows
  );

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const owner = gameState.box_owners[r][c];
      if (owner === null) continue;

      const x = margin + c * cellSize;
      const y = margin + r * cellSize;

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", x + 10);
      rect.setAttribute("y", y + 10);
      rect.setAttribute("width", cellSize - 20);
      rect.setAttribute("height", cellSize - 20);

      if (owner === gameState.human_player) {
        rect.setAttribute("class", "box-human");
      } else {
        rect.setAttribute("class", "box-bot");
      }

      boardSvg.appendChild(rect);

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", x + cellSize / 2);
      label.setAttribute("y", y + cellSize / 2);
      label.setAttribute("class", "box-label");
      label.textContent = owner === gameState.human_player ? "Y" : "B";
      boardSvg.appendChild(label);
    }
  }

  for (let r = 0; r <= rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x1 = margin + c * cellSize;
      const y1 = margin + r * cellSize;
      const x2 = margin + (c + 1) * cellSize;
      const y2 = y1;

      const drawn = gameState.h_edges[r][c];
      drawEdge({
        orientation: "h",
        row: r,
        col: c,
        x1,
        y1,
        x2,
        y2,
        drawn,
      });
    }
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c <= cols; c++) {
      const x1 = margin + c * cellSize;
      const y1 = margin + r * cellSize;
      const x2 = x1;
      const y2 = margin + (r + 1) * cellSize;

      const drawn = gameState.v_edges[r][c];
      drawEdge({
        orientation: "v",
        row: r,
        col: c,
        x1,
        y1,
        x2,
        y2,
        drawn,
      });
    }
  }

  for (let r = 0; r <= rows; r++) {
    for (let c = 0; c <= cols; c++) {
      const cx = margin + c * cellSize;
      const cy = margin + r * cellSize;

      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", cx);
      dot.setAttribute("cy", cy);
      dot.setAttribute("r", 8);
      dot.setAttribute("class", "dot");
      boardSvg.appendChild(dot);
    }
  }
}

function drawEdge({ orientation, row, col, x1, y1, x2, y2, drawn }) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);

  if (!drawn) {
    const humanTurn =
      gameState.current_player === gameState.human_player &&
      !gameState.is_terminal &&
      !isAnimatingBot;

    line.setAttribute("class", "edge-undrawn");

    if (humanTurn) {
      line.addEventListener("click", async () => {
        await sendMove(orientation, row, col);
      });
    } else {
      line.style.cursor = "default";
      line.style.opacity = "0.55";
    }
  } else {
    const ownerClass = inferEdgeOwnerClass(orientation, row, col);
    line.setAttribute("class", ownerClass);
  }

  boardSvg.appendChild(line);
}

function inferEdgeOwnerClass(orientation, row, col) {
  if (!gameState || !gameState.move_history) {
    return "edge-drawn-human";
  }

  for (let i = gameState.move_history.length - 1; i >= 0; i--) {
    const entry = gameState.move_history[i];
    const m = entry.move;
    if (
      m.orientation === orientation &&
      m.row === row &&
      m.col === col
    ) {
      return entry.player === gameState.human_player
        ? "edge-drawn-human"
        : "edge-drawn-bot";
    }
  }

  return "edge-drawn-human";
}