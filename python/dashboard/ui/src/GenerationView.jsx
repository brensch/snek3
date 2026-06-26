import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";
import { resultOf } from "./board.js";
import MiniBoard from "./MiniBoard.jsx";

// Pick a generation (table) and show ALL its recorded games at once as a grid
// of mini-boards playing together on a shared clock.
export default function GenerationView({ run, gamesIndex, metrics }) {
  const [picked, setPicked] = useState(null); // file the user clicked (null = follow latest)
  const [genData, setGenData] = useState(null);
  const [playing, setPlaying] = useState(true);
  const [fps, setFps] = useState(18);
  const [tick, setTick] = useState(0);
  const cache = useRef(new Map());

  useEffect(() => { setPicked(null); setGenData(null); cache.current.clear(); }, [run]);

  const latestFile = gamesIndex[0] ? gamesIndex[0].file : null;
  const activeFile = picked && gamesIndex.some((f) => f.file === picked) ? picked : latestFile;

  useEffect(() => {
    if (!activeFile) return;
    let alive = true;
    const load = async () => {
      let data = cache.current.get(activeFile);
      if (!data) {
        data = await api.gameFile(run, activeFile);
        cache.current.set(activeFile, data);
      }
      if (alive) setGenData(data);
    };
    load();
    return () => { alive = false; };
  }, [run, activeFile]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setTick((t) => t + 1), 1000 / fps);
    return () => clearInterval(id);
  }, [playing, fps]);

  const winRateByGen = useMemo(() => {
    const m = new Map();
    for (const x of metrics) if (x.win_rate != null) m.set(x.gen, x.win_rate);
    return m;
  }, [metrics]);

  const summarize = (games) => {
    let w = 0, l = 0, d = 0;
    for (const g of games) (g.winner === 0 ? (w++) : g.winner === 1 ? (l++) : (d++));
    return `${w}W ${l}L ${d}D`;
  };

  return (
    <section className="card">
      <div className="row" style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>Games {genData ? `· generation ${genData.gen}` : ""}</h2>
        <div className="grow" />
        <button onClick={() => setPlaying((p) => !p)}>{playing ? "⏸ Pause" : "▶ Play"}</button>
        <label className="chk">speed <input type="range" min={4} max={60} value={fps} onChange={(e) => setFps(+e.target.value)} /></label>
        <span className="muted">{fps} fps</span>
        {picked && <button onClick={() => setPicked(null)} title="follow newest generation">↻ latest</button>}
      </div>

      <div className="gen-layout">
        <div className="gentable">
          <table>
            <thead><tr><th>gen</th><th>games</th><th>W/L/D</th><th>win%</th></tr></thead>
            <tbody>
              {gamesIndex.map((f) => {
                const wr = winRateByGen.get(f.gen);
                const active = genData && f.file === activeFile;
                return (
                  <tr key={f.file} className={active ? "active" : ""} onClick={() => setPicked(f.file)}>
                    <td>{f.gen}</td>
                    <td>{f.games.length}</td>
                    <td className="muted">{summarize(f.games)}</td>
                    <td style={{ textAlign: "right" }}>{wr != null ? wr.toFixed(2) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="board-grid">
          {genData
            ? genData.games.map((g, i) => <MiniBoard key={i} game={g} tick={tick} />)
            : <p className="muted">no games recorded yet</p>}
        </div>
      </div>
    </section>
  );
}
