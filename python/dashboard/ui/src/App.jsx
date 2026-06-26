import React, { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import MetricsChart from "./MetricsChart.jsx";
import GameViewer from "./GameViewer.jsx";

function Stat({ value, label }) {
  return (
    <div className="stat">
      <b>{value}</b>
      <span>{label}</span>
    </div>
  );
}

export default function App() {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState(null);
  const [status, setStatus] = useState({});
  const [meta, setMeta] = useState({});
  const [metrics, setMetrics] = useState([]);
  const [gamesIndex, setGamesIndex] = useState([]);
  const runRef = useRef(null);
  useEffect(() => { runRef.current = run; }, [run]);

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

  const m = status.last || {};
  const best = status.best_win_rate;
  const num = (v, d = 0) => (v == null ? "—" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));
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

      <main>
        <section className="card">
          <h2>Training metrics</h2>
          <div className="stats">
            <Stat value={m.gen ?? "—"} label="generation" />
            <Stat value={m.win_rate != null ? m.win_rate.toFixed(3) : "—"} label={`win-rate (best ${best != null ? best.toFixed(2) : "—"})`} />
            <Stat value={num(m.turns_per_sec)} label="turns / sec" />
            <Stat value={m.games_per_sec != null ? m.games_per_sec.toFixed(1) : "—"} label="games / sec" />
            <Stat value={m.policy_loss != null ? m.policy_loss.toFixed(3) : "—"} label="policy loss" />
            <Stat value={m.value_loss != null ? m.value_loss.toFixed(3) : "—"} label="value loss" />
            <Stat value={num(m.samples)} label="samples/gen" />
            <Stat value={num(m.buffer)} label="replay buffer" />
          </div>
          <MetricsChart metrics={metrics} />
        </section>

        {run ? <GameViewer key={run} run={run} gamesIndex={gamesIndex} /> : <section className="card"><h2>Live game stream</h2><p className="muted">no run selected</p></section>}
      </main>
    </>
  );
}
