import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";
import MiniBoard from "./MiniBoard.jsx";

// Pick a generation (table) and show ALL its recorded games at once as a grid
// of mini-boards playing together on a shared clock.
export default function GenerationView({ run, gamesIndex, evalIndex = [] }) {
  const [picked, setPicked] = useState(null); // file the user clicked (null = follow latest)
  const [genData, setGenData] = useState(null);
  const [playing, setPlaying] = useState(true);
  const [fps, setFps] = useState(18);
  const [tileWidth, setTileWidth] = useState(210);
  const [tick, setTick] = useState(0);
  const [matchFilter, setMatchFilter] = useState("all"); // matchup-type filter
  const cache = useRef(new Map());

  useEffect(() => { setPicked(null); setGenData(null); cache.current.clear(); }, [run]);

  const latestFile = gamesIndex[0] ? gamesIndex[0].file : null;
  const activeFile = picked && gamesIndex.some((f) => f.file === picked) ? picked : latestFile;
  const activeGen = gamesIndex.find((f) => f.file === activeFile)?.gen;
  // The faithful eval artifact for the same generation (real games vs the pool),
  // if one exists — eval runs less often than self-play recording.
  const evalFile = activeGen != null ? evalIndex.find((e) => e.gen === activeGen)?.file : null;

  useEffect(() => {
    if (!activeFile) return;
    let alive = true;
    const cacheKey = `${activeFile}|${evalFile || ""}`;
    const load = async () => {
      let data = cache.current.get(cacheKey);
      if (!data) {
        const self = await api.gameFile(run, activeFile);
        let evalGames = [];
        if (evalFile) {
          try {
            const ev = await api.evalFile(run, evalFile);
            evalGames = ev.games || [];
          } catch (_) { /* eval still writing or absent */ }
        }
        // Self-play replays first, then the real eval games (tagged vs-baseline /
        // vs-uct) — all flow through the existing matchup filter + board grid.
        data = { ...self, games: [...(self.games || []), ...evalGames] };
        cache.current.set(cacheKey, data);
      }
      if (alive) setGenData(data);
    };
    load();
    return () => { alive = false; };
  }, [run, activeFile, evalFile]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setTick((t) => t + 1), 1000 / fps);
    return () => clearInterval(id);
  }, [playing, fps]);

  const drawPctFromGames = (games) => {
    const total = games?.length || 0;
    if (!total) return "—";
    const draws = games.filter((g) => Number(g.winner) < 0).length;
    return pct(draws / total);
  };
  const summarizeSelfplay = (summary, fallbackGames) => {
    if (!summary?.completed_games) return drawPctFromGames(fallbackGames);
    return pct(summary.draw_rate ?? ((summary.draws || 0) / Math.max(1, summary.completed_games)));
  };
  const pct = (value) => value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
  const num = (value, digits = 0) => value == null ? "—" : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });

  // Distinct matchup labels present this gen (for the filter bar), first-seen order.
  const matchTypes = useMemo(() => {
    const seen = [];
    for (const g of genData?.games || []) {
      const k = g.opponent || "baseline";
      if (!seen.includes(k)) seen.push(k);
    }
    return seen;
  }, [genData]);

  // All games in one flat list, filtered to the selected matchup ("all" = no filter).
  const shownGames = useMemo(() => {
    const games = genData?.games || [];
    return games
      .map((game, gameIndex) => ({ game, gameIndex }))
      .filter(({ game }) => matchFilter === "all" || (game.opponent || "baseline") === matchFilter);
  }, [genData, matchFilter]);

  const prettyMatch = (k) => {
    const named = {
      net: "self-play", baseline: "net vs baseline",
      proxy: "proxy self-play", response: "response self-play",
      "vs-baseline": "real vs baseline", "vs-uct": "real vs UCT",
    };
    return named[k] || k.replace(/-v-/g, " vs ");
  };

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
            <span><b>{num(selfplay.turns?.mean, 1)}</b><em>avg turns</em></span>
            <span><b>{pct(selfplay.draw_rate ?? ((selfplay.draws || 0) / Math.max(1, selfplay.completed_games)))}</b><em>draws</em></span>
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
            <thead><tr><th>gen</th><th>games</th><th>draws</th></tr></thead>
            <tbody>
              {gamesIndex.map((f) => {
                const active = genData && f.file === activeFile;
                return (
                  <tr key={f.file} className={active ? "active" : ""} onClick={() => setPicked(f.file)}>
                    <td>{f.gen}</td>
                    <td>{f.games.length}</td>
                    <td className="muted">{summarizeSelfplay(f.selfplay, f.games)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="board-scroll">
          {genData ? (
            <>
              <div className="match-filter">
                <button
                  className={matchFilter === "all" ? "active" : ""}
                  onClick={() => setMatchFilter("all")}
                >
                  all ({(genData.games || []).length})
                </button>
                {matchTypes.map((k) => (
                  <button
                    key={k}
                    className={matchFilter === k ? "active" : ""}
                    onClick={() => setMatchFilter(k)}
                  >
                    {prettyMatch(k)}
                  </button>
                ))}
              </div>
              <div className="board-grid" style={{ "--tile-width": `${tileWidth}px` }}>
                {shownGames.map(({ game: g, gameIndex }) => (
                  <MiniBoard
                    key={`${g.opponent}-${gameIndex}`}
                    game={g}
                    tick={tick}
                    playing={playing}
                    onPlay={() => setPlaying(true)}
                    context={{
                      run,
                      file: activeFile,
                      gen: activeGen,
                      gameIndex,
                    }}
                  />
                ))}
              </div>
            </>
          ) : (
            <p className="muted">no games recorded yet</p>
          )}
        </div>
      </div>
    </section>
  );
}
