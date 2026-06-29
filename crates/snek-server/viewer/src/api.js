// Thin client for the snek-server /viewer API. In dev these are proxied to a
// running snek-server (see vite.config.js); in the shipped build they hit the
// same origin that serves /app.
const BASE = "/viewer";

async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      msg = (await r.json()).error || msg;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(msg);
  }
  return r.json();
}

export const listGames = () => getJson(`${BASE}/games`);
export const getGame = (id) => getJson(`${BASE}/games/${encodeURIComponent(id)}`);
export const getTree = (id, turn, sims) =>
  getJson(
    `${BASE}/games/${encodeURIComponent(id)}/tree?turn=${turn}` +
      (sims ? `&sims=${sims}` : "")
  );

// Move-index convention shared with the engine: 0=up 1=down 2=left 3=right.
export const MOVES = ["up", "down", "left", "right"];
export const MOVE_LABEL = ["Up", "Down", "Left", "Right"];
export const MOVE_ARROW = ["↑", "↓", "←", "→"];
// Pixel deltas for drawing a move arrow (screen coords: y grows downward).
export const MOVE_DELTA = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];
