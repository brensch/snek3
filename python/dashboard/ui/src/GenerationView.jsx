import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";
import MiniBoard from "./MiniBoard.jsx";

// Pick a generation (table) and show ALL its recorded games at once as a grid
// of mini-boards playing together on a shared clock.
export default function GenerationView({ run, gamesIndex, metrics }) {
  const [picked, setPicked] = useState(null); // file the user clicked (null = follow latest)
  const [genData, setGenData] = useState(null);
  const [playing, setPlaying] = useState(true);
  const [fps, setFps] = useState(18);
  const [tileWidth, setTileWidth] = useState(210);
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
  const summarizeSelfplay = (summary, fallbackGames) => {
    if (!summary?.completed_games) return summarize(fallbackGames);
    return `${summary.wins || 0}W ${summary.losses || 0}L ${summary.draws || 0}D`;
  };
  const pct = (value) => value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
  const num = (value, digits = 0) => value == null ? "—" : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });

  const gameGroups = useMemo(() => {
    const games = genData?.games || [];
    return [
      {
        key: "baseline",
        title: "Net vs baseline",
        games: games.filter((g) => g.opponent !== "net"),
      },
      {
        key: "net",
        title: "Net self-play",
        games: games.filter((g) => g.opponent === "net"),
      },
    ].filter((g) => g.games.length > 0);
  }, [genData]);

  const selfplay = genData?.selfplay;
  const maxBucket = Math.max(1, ...(selfplay?.length_histogram || []).map((b) => b.count || 0));

  return (
    <section className="card">
      <div className="row" style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>Games {genData ? `· generation ${genData.gen}` : ""}</h2>
        <div className="grow" />
        <button onClick={() => setPlaying((p) => !p)}>{playing ? "⏸ Pause" : "▶ Play"}</button>
        <label className="chk">speed <input type="range" min={4} max={60} value={fps} onChange={(e) => setFps(+e.target.value)} /></label>
        <span className="muted">{fps} fps</span>
        <label className="chk">tile <input type="range" min={140} max={320} value={tileWidth} onChange={(e) => setTileWidth(+e.target.value)} /></label>
        <span className="muted">{tileWidth}px</span>
        {picked && <button onClick={() => setPicked(null)} title="follow newest generation">↻ latest</button>}
      </div>

      {selfplay?.completed_games ? (
        <div className="game-summary">
          <div className="summary-stats">
            <span><b>{num(selfplay.completed_games)}</b><em>played</em></span>
            <span><b>{selfplay.wins || 0}/{selfplay.losses || 0}/{selfplay.draws || 0}</b><em>W/L/D</em></span>
            <span><b>{pct(selfplay.win_rate)}</b><em>total win</em></span>
            <span><b>{pct(selfplay.decisive_win_rate)}</b><em>decisive</em></span>
            <span><b>{num(selfplay.turns?.mean, 1)}</b><em>avg turns</em></span>
            <span><b>{num(selfplay.turns?.p50)} / {num(selfplay.turns?.p90)} / {num(selfplay.turns?.max)}</b><em>p50/p90/max</em></span>
            <span><b>{num(selfplay.short_draws)}</b><em>short draws</em></span>
            <span><b>{num(selfplay.overrun_draws)}</b><em>overruns</em></span>
          </div>
          <div className="length-hist">
            {(selfplay.length_histogram || []).map((bucket) => (
              <div className="hist-bin" key={`${bucket.min}-${bucket.max}`}>
                <i style={{ height: `${Math.max(4, (bucket.count / maxBucket) * 42)}px` }} />
                <span>{bucket.min}-{bucket.max}</span>
                <b>{bucket.count}</b>
              </div>
            ))}
          </div>
        </div>
      ) : null}

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
                    <td className="muted">{summarizeSelfplay(f.selfplay, f.games)}</td>
                    <td style={{ textAlign: "right" }}>{wr != null ? wr.toFixed(2) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="board-scroll">
          <div className="board-groups">
            {genData
              ? gameGroups.map((group) => (
                  <div className="board-group" key={group.key}>
                    <div className="board-group-head">
                      <h3>{group.title}</h3>
                      <span className="muted">{group.games.length} samples · {summarize(group.games)}</span>
                    </div>
                    <div
                      className="board-grid"
                      style={{ "--tile-width": `${tileWidth}px` }}
                    >
                      {group.games.map((g, i) => (
                        <MiniBoard
                          key={`${group.key}-${i}`}
                          game={g}
                          tick={tick}
                          playing={playing}
                          onPlay={() => setPlaying(true)}
                        />
                      ))}
                    </div>
                  </div>
                ))
              : <p className="muted">no games recorded yet</p>}
          </div>
        </div>
      </div>
    </section>
  );
}
