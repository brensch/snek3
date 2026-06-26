import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";

const COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ec4899", "#a78bfa", "#f87171", "#2dd4bf", "#facc15"];

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// Live, auto-advancing replay viewer. Plays the newest recorded games back to
// back; as new generations stream in they get priority.
export default function GameViewer({ run, gamesIndex }) {
  const boardRef = useRef(null);
  const fileCache = useRef(new Map());
  const shown = useRef(new Set());

  const [current, setCurrent] = useState(null); // {key, gen, opponent, winner, frames}
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [fps, setFps] = useState(24);
  const [followLatest, setFollowLatest] = useState(true);
  const [streamed, setStreamed] = useState(0);
  const followRef = useRef(true);
  useEffect(() => { followRef.current = followLatest; }, [followLatest]);

  // Reset everything when the run changes.
  useEffect(() => {
    fileCache.current.clear();
    shown.current.clear();
    setCurrent(null);
    setFrame(0);
    setStreamed(0);
  }, [run]);

  const allGames = useMemo(() => {
    const out = [];
    for (const f of gamesIndex)
      (f.games || []).forEach((g, i) =>
        out.push({ file: f.file, gen: f.gen, idx: i, meta: g, key: f.file + "#" + i })
      );
    return out;
  }, [gamesIndex]);

  const fetchFile = useCallback(async (file) => {
    if (fileCache.current.has(file)) return fileCache.current.get(file);
    const data = await api.gameFile(run, file);
    fileCache.current.set(file, data);
    if (fileCache.current.size > 60) fileCache.current.delete(fileCache.current.keys().next().value);
    return data;
  }, [run]);

  const loadRef = useCallback(async (ref, markShown) => {
    const data = await fetchFile(ref.file);
    const g = data.games[ref.idx];
    if (!g) return;
    if (markShown && !shown.current.has(ref.key)) {
      shown.current.add(ref.key);
      setStreamed((n) => n + 1);
    }
    setCurrent({ key: ref.key, gen: ref.gen, opponent: g.opponent, winner: g.winner, frames: g.frames });
    setFrame(0);
  }, [fetchFile]);

  const advanceStream = useCallback(async () => {
    const ref = allGames.find((g) => !shown.current.has(g.key)) || allGames[0];
    if (ref) await loadRef(ref, true);
  }, [allGames, loadRef]);

  // Kick the stream when games first arrive (or after a run reset).
  useEffect(() => {
    if (!current && allGames.length) advanceStream();
  }, [allGames, current, advanceStream]);

  // Playback clock: advance one frame per tick.
  useEffect(() => {
    if (!playing || !current) return;
    const id = setInterval(
      () => setFrame((f) => Math.min(f + 1, current.frames.length - 1)),
      1000 / fps
    );
    return () => clearInterval(id);
  }, [playing, fps, current]);

  // At end of a game: advance the stream (or replay if not following).
  useEffect(() => {
    if (!current || !playing || frame < current.frames.length - 1) return;
    const t = setTimeout(() => {
      if (followRef.current) advanceStream();
      else setFrame(0);
    }, 1000 / fps);
    return () => clearTimeout(t);
  }, [frame, current, playing, fps, advanceStream]);

  // Draw the board for the current frame.
  useEffect(() => {
    const c = boardRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    const fr = current && current.frames[frame];
    if (!fr) {
      ctx.fillStyle = "#8b949e";
      ctx.fillText("waiting for games…", 20, 30);
      return;
    }
    const W = fr.width, H = fr.height, cell = Math.floor(Math.min(c.width, c.height) / Math.max(W, H));
    const px = (x) => x * cell, py = (y) => (H - 1 - y) * cell;
    ctx.fillStyle = "#0b0e13";
    ctx.fillRect(0, 0, W * cell, H * cell);
    ctx.strokeStyle = "#161b22";
    ctx.lineWidth = 1;
    for (let i = 0; i <= W; i++) { ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, H * cell); ctx.stroke(); }
    for (let j = 0; j <= H; j++) { ctx.beginPath(); ctx.moveTo(0, j * cell); ctx.lineTo(W * cell, j * cell); ctx.stroke(); }
    (fr.food || []).forEach(([x, y]) => { ctx.fillStyle = "#ef4444"; ctx.beginPath(); ctx.arc(px(x) + cell / 2, py(y) + cell / 2, cell * 0.22, 0, 7); ctx.fill(); });
    (fr.hazards || []).forEach(([x, y]) => { ctx.fillStyle = "rgba(234,179,8,0.12)"; ctx.fillRect(px(x), py(y), cell, cell); });
    fr.snakes.forEach((s, si) => {
      const col = COLORS[si % COLORS.length];
      ctx.globalAlpha = s.alive ? 1 : 0.22;
      s.body.forEach(([x, y], bi) => {
        ctx.fillStyle = col;
        const pad = cell * 0.12;
        roundRect(ctx, px(x) + pad, py(y) + pad, cell - 2 * pad, cell - 2 * pad, cell * 0.2);
        ctx.fill();
        if (bi === 0) {
          // Snake index on the head so you can tell who's who on the board.
          ctx.fillStyle = "rgba(255,255,255,0.95)";
          ctx.font = `bold ${Math.floor(cell * 0.5)}px ui-sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(si), px(x) + cell / 2, py(y) + cell / 2);
        }
      });
      ctx.globalAlpha = 1;
    });
  }, [current, frame]);

  const fr = current && current.frames[frame];
  const result = current
    ? current.winner === 0 ? ["WIN", "win"] : current.winner === 1 ? ["loss", "loss"] : current.winner === -1 ? ["draw", "draw"] : ["—", ""]
    : ["—", ""];

  return (
    <div className="card">
      <h2>Live game stream</h2>
      <div className="row">
        <span className="muted">
          {current ? `gen ${current.gen} · ${current.opponent === "net" ? "self-play" : "vs baseline"} · ${current.frames.length} turns` : "waiting for games…"}
        </span>
        <span className={"badge " + result[1]}>{result[0]}</span>
        <div className="grow" />
        <span className="muted">{streamed.toLocaleString()} streamed</span>
      </div>
      <canvas ref={boardRef} width={448} height={448} />
      <div className="controls">
        <button onClick={() => setPlaying((p) => !p)}>{playing ? "⏸ Pause" : "▶ Play"}</button>
        <button onClick={() => setFrame((f) => Math.max(0, f - 1))}>⟨</button>
        <button onClick={() => setFrame((f) => current ? Math.min(current.frames.length - 1, f + 1) : 0)}>⟩</button>
        <button onClick={advanceStream} title="next game">⏭</button>
        <input id="scrub" type="range" min={0} max={current ? current.frames.length - 1 : 0} value={frame} onChange={(e) => setFrame(+e.target.value)} />
        <span className="muted">
          {fr ? `turn ${fr.turn} · ${frame + 1}/${current.frames.length}  ${fr.snakes.map((s, i) => `S${i}:${s.alive ? "hp" + s.health : "✕"}`).join("  ")}` : "—"}
        </span>
      </div>
      {current && (
        <div className="legend">
          {current.frames[0].snakes.map((_, i) => {
            const role = i === 0 ? "our net" : current.opponent === "net" ? "our net (self-play)" : "flood-fill baseline";
            const atEnd = frame >= current.frames.length - 1;
            const dead = fr && !fr.snakes[i].alive;
            const won = current.winner === i;
            return (
              <span key={i}>
                <i className="swatch" style={{ background: COLORS[i] }} />
                snake {i} — {role}
                {won ? " · winner ✓" : atEnd && dead ? " · eliminated ✕" : ""}
              </span>
            );
          })}
        </div>
      )}
      <div className="controls">
        <label className="chk"><input type="checkbox" checked={followLatest} onChange={(e) => { setFollowLatest(e.target.checked); if (e.target.checked) advanceStream(); }} /> stream newest</label>
        <label className="chk">speed <input type="range" min={4} max={60} value={fps} onChange={(e) => setFps(+e.target.value)} /></label>
        <span className="muted">{fps} fps</span>
      </div>

      <div className="gametable">
        <table>
          <thead>
            <tr><th>gen</th><th>type</th><th>result</th><th style={{ textAlign: "right" }}>turns</th></tr>
          </thead>
          <tbody>
            {allGames.slice(0, 120).map((g) => {
              const w = g.meta.winner;
              const res = w === 0 ? ["W", "win"] : w === 1 ? ["L", "loss"] : ["D", "draw"];
              const active = current && g.key === current.key;
              return (
                <tr
                  key={g.key}
                  className={active ? "active" : ""}
                  onClick={() => { setFollowLatest(false); loadRef(g, false); }}
                >
                  <td>{g.gen}</td>
                  <td>{g.meta.opponent === "net" ? "self-play" : "vs baseline"}</td>
                  <td><span className={"badge " + res[1]}>{res[0]}</span></td>
                  <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{g.meta.num_turns}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
