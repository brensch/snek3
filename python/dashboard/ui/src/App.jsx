import React, { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import MetricsChart from "./MetricsChart.jsx";
import GenerationView from "./GenerationView.jsx";
import ControlPanel from "./ControlPanel.jsx";
import Home from "./Home.jsx";

const runFromUrl = () => {
  const m = window.location.pathname.match(/^\/run\/(.+)$/);
  return m ? decodeURIComponent(m[1].replace(/\/$/, "")) : null;
};

export default function App() {
  const [runs, setRuns] = useState([]);
  const [liveRun, setLiveRun] = useState(null);
  const [run, setRun] = useState(runFromUrl());
  const [status, setStatus] = useState({});
  const [meta, setMeta] = useState({});
  const [metrics, setMetrics] = useState([]);
  const [params, setParams] = useState({});
  const [liveKeys, setLiveKeys] = useState([]);
  const [lockedKeys, setLockedKeys] = useState([]);
  const [gamesIndex, setGamesIndex] = useState([]);
  const runRef = useRef(null);
  useEffect(() => { runRef.current = run; }, [run]);

  const isLive = run && run === liveRun;

  // Keep the URL in sync with the selected run.
  useEffect(() => {
    const want = run ? `/run/${encodeURIComponent(run)}` : "/";
    if (window.location.pathname !== want) window.history.replaceState(null, "", want);
  }, [run]);

  // Discover runs + which one is live; default to the live run.
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      const { runs: rs, live } = await api.runs();
      if (!alive) return;
      setRuns(rs || []);
      setLiveRun(live || null);
      // Don't auto-pick a run: the landing page (run === null) lets the user
      // start or choose one. Only clear a selection that has vanished.
      if (runRef.current && !(rs || []).includes(runRef.current)) {
        setRun(null);
      }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  // LIVE run: subscribe to the SSE stream.
  useEffect(() => {
    if (!isLive) return;
    const es = api.stream((ev) => {
      if (ev.type === "snapshot") {
        setMeta(ev.meta || {}); setStatus(ev.status || {});
        setParams(ev.params || {}); setMetrics(ev.metrics || []);
        setLiveKeys(ev.live_params || []); setLockedKeys(ev.locked_params || []);
      } else if (ev.type === "metric") {
        setMetrics((m) => [...m, ev.metric]);
      } else if (ev.type === "status") {
        setStatus(ev.status || {});
      } else if (ev.type === "params") {
        setParams(ev.params || {});
      }
    });
    return () => es.close();
  }, [isLive, run]);

  // ARCHIVED run: poll REST.
  useEffect(() => {
    if (isLive || !run) return;
    let alive = true;
    const tick = async () => {
      const [st, mt, me] = await Promise.all([api.status(run), api.metrics(run), api.meta(run)]);
      if (!alive) return;
      setStatus(st); setMetrics(mt); setMeta(me);
    };
    tick();
    const id = setInterval(tick, 4000);
    return () => { alive = false; clearInterval(id); };
  }, [isLive, run]);

  // Games index (both modes): refetch when the generation advances.
  const gen = status?.generation;
  useEffect(() => {
    if (!run) return;
    let alive = true;
    api.games(run).then((g) => { if (alive) setGamesIndex(g); });
    return () => { alive = false; };
  }, [run, gen]);

  const running = status.running;

  return (
    <>
      <header>
        <h1 className="brand" onClick={() => setRun(null)} title="Home">
          snek3<span className="dot">·</span>training
        </h1>
        <select value={run || ""} onChange={(e) => setRun(e.target.value || null)}>
          <option value="">＋ home / new run</option>
          {runs.map((r) => (
            <option key={r} value={r}>{r === liveRun ? `● ${r}` : r}</option>
          ))}
        </select>
        {run && (
          <span className={"pill " + (running ? "live" : "done")}>
            {status.generation == null ? "no data"
              : `${running ? (status.paused ? "❚❚ paused" : "● live") : "■ done"} · gen ${status.generation}`}
          </span>
        )}
        <div className="grow" />
        {run && (
          <span className="pill">
            {meta.board ? `${meta.board}×${meta.board} · ${meta.filters}f/${meta.blocks}b · depth ${meta.depth}` : "—"}
          </span>
        )}
      </header>

      {run ? (
        <main className="stacked">
          <section className="card">
            <h2>Control</h2>
            <ControlPanel status={status} params={params} liveKeys={liveKeys}
              lockedKeys={lockedKeys} live={isLive} />
          </section>

          <section className="card">
            <h2>Training metrics</h2>
            <MetricsChart metrics={metrics} />
          </section>

          <GenerationView run={run} gamesIndex={gamesIndex} metrics={metrics} />
        </main>
      ) : (
        <main className="stacked">
          <Home runs={runs} liveRun={liveRun} onSelect={setRun} />
        </main>
      )}
    </>
  );
}
