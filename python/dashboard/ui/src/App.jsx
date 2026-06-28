import React, { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import MetricsChart from "./MetricsChart.jsx";
import GenerationView from "./GenerationView.jsx";

const runFromUrl = () => {
  const m = window.location.pathname.match(/^\/run\/(.+)$/);
  return m ? decodeURIComponent(m[1].replace(/\/$/, "")) : null;
};

export default function App() {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState(runFromUrl());
  const [status, setStatus] = useState({});
  const [meta, setMeta] = useState({});
  const [metrics, setMetrics] = useState([]);
  const [gamesIndex, setGamesIndex] = useState([]);
  const runRef = useRef(null);
  useEffect(() => { runRef.current = run; }, [run]);

  // Keep the URL in sync with the selected run (/run/<name>) so reloads/bookmarks
  // return to it.
  useEffect(() => {
    const want = run ? `/run/${encodeURIComponent(run)}` : "/";
    if (window.location.pathname !== want) window.history.replaceState(null, "", want);
  }, [run]);

  useEffect(() => {
    let alive = true;
    const refresh = async () => {
      try {
        const rs = await api.runs();
        if (!alive) return;
        setRuns(rs);
        let r = runRef.current;
        if (!r || !rs.includes(r)) r = rs[0] || null;
        runRef.current = r;
        setRun(r);
        if (!r) return;
        const [st, mt, me, gi] = await Promise.all([
          api.status(r), api.metrics(r), api.meta(r), api.games(r),
        ]);
        if (!alive) return;
        setStatus(st); setMetrics(mt); setMeta(me); setGamesIndex(gi);
      } catch (_) { /* transient; next tick retries */ }
    };
    refresh();
    const id = setInterval(refresh, 2500);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const running = status.running;

  return (
    <>
      <header>
        <h1>snek3<span className="dot">·</span>training</h1>
        <select value={run || ""} onChange={(e) => setRun(e.target.value)}>
          {runs.length ? runs.map((r) => <option key={r}>{r}</option>) : <option>(no runs)</option>}
        </select>
        <span className={"pill " + (running ? "live" : "done")}>
          {status.generation == null
            ? "no data"
            : `${running ? "● live" : "■ done"} · gen ${status.generation + 1}/${status.total_generations ?? "?"}`}
        </span>
        <div className="grow" />
        <span className="pill">
          {meta.board ? `${meta.board}×${meta.board} · ${meta.filters}f/${meta.blocks}b · depth ${meta.depth} · ${meta.device || ""}` : "—"}
        </span>
      </header>

      <main className="stacked">
        <section className="card">
          <h2>Training metrics</h2>
          <MetricsChart metrics={metrics} />
        </section>

        {run ? (
          <GenerationView run={run} gamesIndex={gamesIndex} metrics={metrics} />
        ) : (
          <section className="card"><h2>Games</h2><p className="muted">no run selected</p></section>
        )}
      </main>
    </>
  );
}
