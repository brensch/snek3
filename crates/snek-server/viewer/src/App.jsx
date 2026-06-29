import React, { useCallback, useEffect, useRef, useState } from "react";
import Board from "./Board.jsx";
import MovePanel from "./MovePanel.jsx";
import TreeExplorer from "./TreeExplorer.jsx";
import { getGame, getTree, listGames } from "./api.js";

export default function App() {
  const initial = parseQuery();
  const [games, setGames] = useState([]);
  const [gameId, setGameId] = useState(initial.game || null);
  const [game, setGame] = useState(null);
  const [error, setError] = useState(null);
  const [turn, setTurn] = useState(initial.turn || 0); // index into moves[]
  const [playing, setPlaying] = useState(false);
  const [tps, setTps] = useState(6); // turns per second

  // Tree-explorer state (on demand for the current turn).
  const [tree, setTree] = useState(null);
  const [treeBusy, setTreeBusy] = useState(false);
  const [treeErr, setTreeErr] = useState(null);
  const [simsOverride, setSimsOverride] = useState("");

  useEffect(() => {
    listGames().then(setGames).catch((e) => setError(String(e)));
  }, []);

  // Load the selected game.
  useEffect(() => {
    if (!gameId) return;
    setGame(null);
    setError(null);
    setTree(null);
    getGame(gameId)
      .then((g) => {
        setGame(g);
        setTurn((t) => clamp(t, 0, (g.moves?.length || 1) - 1));
      })
      .catch((e) => setError(String(e)));
  }, [gameId]);

  const moves = game?.moves || [];
  const move = moves[turn] || null;
  const last = Math.max(0, moves.length - 1);

  // Keep the URL shareable.
  useEffect(() => {
    const q = new URLSearchParams();
    if (gameId) q.set("game", gameId);
    if (turn) q.set("turn", String(turn));
    const s = q.toString();
    window.history.replaceState(null, "", s ? `?${s}` : window.location.pathname);
  }, [gameId, turn]);

  // Clear the tree whenever the frame changes (it's specific to one turn).
  useEffect(() => {
    setTree(null);
    setTreeErr(null);
  }, [turn, gameId]);

  // Autoplay.
  useEffect(() => {
    if (!playing || !moves.length) return;
    const id = setInterval(() => {
      setTurn((t) => {
        if (t >= last) {
          setPlaying(false);
          return t;
        }
        return t + 1;
      });
    }, 1000 / tps);
    return () => clearInterval(id);
  }, [playing, tps, last, moves.length]);

  const loadTree = useCallback(() => {
    if (!gameId || !move) return;
    setTreeBusy(true);
    setTreeErr(null);
    getTree(gameId, move.turn, simsOverride ? Number(simsOverride) : undefined)
      .then(setTree)
      .catch((e) => setTreeErr(String(e)))
      .finally(() => setTreeBusy(false));
  }, [gameId, move, simsOverride]);

  // Keyboard transport.
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") setTurn((t) => clamp(t + 1, 0, last));
      else if (e.key === "ArrowLeft") setTurn((t) => clamp(t - 1, 0, last));
      else if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last]);

  const board = game?.board || {};
  const search = move?.search || {};

  return (
    <div className="app">
      <header className="topbar">
        <h1>snek viewer</h1>
        <select
          className="game-select"
          value={gameId || ""}
          onChange={(e) => setGameId(e.target.value || null)}
        >
          <option value="">— pick a game —</option>
          {games.map((g) => (
            <option key={g.id} value={g.id}>
              {g.id.slice(0, 8)} · {fmtBytes(g.bytes)}
            </option>
          ))}
        </select>
        {game && (
          <span className="game-info">
            {board.width}×{board.height} · {game.finished?.state} ·{" "}
            {moves.length} turns · {game.roster?.map((r) => r.name).join(" vs ")}
          </span>
        )}
        <span className="spacer" />
        <a className="ghost" href={shareLink(gameId, turn)}>
          link
        </a>
      </header>

      {error && <div className="error">{error}</div>}
      {!gameId && !error && <div className="hint">Select a game to begin.</div>}

      {game && move && (
        <div className="main">
          <section className="left">
            <Board
              width={board.width}
              height={board.height}
              snakes={move.snakes}
              food={move.food}
              hazards={move.hazards}
              youIndex={move.you}
              chosenMove={move.chosen_move}
            />
            <div className="transport">
              <button onClick={() => setTurn((t) => clamp(t - 1, 0, last))}>‹</button>
              <button className="play" onClick={() => setPlaying((p) => !p)}>
                {playing ? "❚❚" : "►"}
              </button>
              <button onClick={() => setTurn((t) => clamp(t + 1, 0, last))}>›</button>
              <input
                className="scrub"
                type="range"
                min={0}
                max={last}
                value={turn}
                onChange={(e) => {
                  setPlaying(false);
                  setTurn(Number(e.target.value));
                }}
              />
              <span className="turnlbl">
                turn {move.turn} / {moves[last]?.turn}
              </span>
              <label className="speed">
                {tps}/s
                <input
                  type="range"
                  min={1}
                  max={20}
                  value={tps}
                  onChange={(e) => setTps(Number(e.target.value))}
                />
              </label>
            </div>

            <div className="searchstats">
              <Stat label="sims" value={search.sims_completed ?? "·"} />
              <Stat label="forward" value={search.forward_calls ?? "·"} />
              <Stat label="eval rows" value={search.eval_rows ?? "·"} />
              <Stat label="terminal" value={search.terminal_only_sims ?? "·"} />
              <Stat
                label="depth"
                value={
                  search.max_depth != null
                    ? search.max_depth
                    : tree?.tree
                      ? tree.tree.max_depth
                      : "↓ tree"
                }
              />
              <Stat
                label="search ms"
                value={move.timing ? Math.round(move.timing[1]) : "·"}
              />
              <Stat label="stop" value={search.stopped_reason || "·"} />
            </div>
          </section>

          <section className="right">
            <MovePanel move={move} roster={move.snakes} nSnakes={move.snakes.length} />
          </section>
        </div>
      )}

      {game && move && (
        <div className="treesection">
          <div className="tree-controls">
            <button className="primary" onClick={loadTree} disabled={treeBusy}>
              {treeBusy ? "replaying…" : "explore search tree"}
            </button>
            <label className="sims">
              sims
              <input
                type="number"
                placeholder={String(search.sims_completed ?? "")}
                value={simsOverride}
                onChange={(e) => setSimsOverride(e.target.value)}
              />
            </label>
            <span className="muted small">
              replays turn {move.turn} with the exact recorded sim count by default
            </span>
          </div>

          {treeErr && <div className="error">{treeErr}</div>}
          {tree && tree.model_match === false && (
            <div className="warn">
              ⚠ replay used a different model than the recording
              {tree.recorded_model_sha
                ? ` (recorded ${tree.recorded_model_sha.slice(0, 10)}, server ${tree.server_model_sha.slice(0, 10)})`
                : " (the recording predates model-hash tracking)"}
              — the tree is illustrative, not an exact reproduction.
            </div>
          )}
          {tree && (
            <div className="replay-meta">
              recorded move <b>{moveName(tree.recorded_move)}</b> · replay chose{" "}
              <b>{moveName(tree.replay_move)}</b> · {tree.sims} sims ·{" "}
              {tree.tree?.node_count} nodes
              {tree.model_match && <span className="ok"> · model match ✓</span>}
            </div>
          )}
          {tree?.tree && (
            <TreeExplorer
              data={tree}
              cPuct={game.config?.c_puct ?? 1.5}
              food={move.food}
              hazards={move.hazards}
              width={board.width}
              height={board.height}
            />
          )}
        </div>
      )}
    </div>
  );
}

const Stat = ({ label, value }) => (
  <div className="stat">
    <div className="stat-v">{value}</div>
    <div className="stat-l">{label}</div>
  </div>
);

function parseQuery() {
  const q = new URLSearchParams(window.location.search);
  return { game: q.get("game"), turn: Number(q.get("turn")) || 0 };
}
function shareLink(game, turn) {
  const q = new URLSearchParams();
  if (game) q.set("game", game);
  if (turn) q.set("turn", String(turn));
  return `?${q.toString()}`;
}
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmtBytes = (b) => (b > 1024 ? `${(b / 1024).toFixed(0)}kB` : `${b}B`);
const MOVE_LABEL = ["Up", "Down", "Left", "Right"];
const moveName = (m) => (m == null ? "·" : MOVE_LABEL[m] ?? m);
