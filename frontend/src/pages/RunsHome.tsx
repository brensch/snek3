import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { control } from "../api/client";
import { ConfigFields } from "../components/ConfigFields";
import { useRunList } from "../hooks/useRunList";
import { since } from "../lib/format";
import type { RunConfig } from "../types";

// Home page: every run on disk, plus a control to start a fresh one.
export function RunsHome() {
  const { runs, loading, error } = useRunList();
  const [runId, setRunId] = useState("");
  const [busy, setBusy] = useState(false);
  const [showKnobs, setShowKnobs] = useState(false);
  const [config, setConfig] = useState<RunConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const navigate = useNavigate();

  // Seed the knob form with the trainer's default config. Retries on remount; a
  // failure here surfaces in the panel rather than silently disabling the button.
  useEffect(() => {
    control
      .config()
      .then((cfg) => {
        setConfig(cfg);
        setConfigError(null);
      })
      .catch((err) => setConfigError(err instanceof Error ? err.message : String(err)));
  }, []);

  const setField = (key: keyof RunConfig, value: number | boolean) =>
    setConfig((prev) => (prev ? { ...prev, [key]: value } : prev));

  // The GPU benchmark needs exclusive use of the GPU, so it is only offered when
  // no run is live (the server enforces this too).
  const anyRunning = runs.some((run) => run.running);

  async function startFresh() {
    setBusy(true);
    try {
      const { run_id } = await control.start(runId.trim() || null, true, config ?? undefined);
      navigate(`/runs/${encodeURIComponent(run_id)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-5 py-6">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">snek3 runs</h1>
          <p className="text-sm text-slate-500">Training runs on disk. Pick one to inspect its metrics and sample games.</p>
        </div>
        <div className="flex items-end gap-2">
          <label className="grid gap-1">
            <span className="text-[10px] uppercase tracking-wide text-slate-500">New run id (optional)</span>
            <input
              className="input w-52"
              value={runId}
              onChange={(e) => setRunId(e.target.value)}
              placeholder="auto-timestamp"
            />
          </label>
          <button
            type="button"
            className="btn"
            aria-pressed={showKnobs}
            onClick={() => setShowKnobs((v) => !v)}
          >
            {showKnobs ? "Hide knobs" : "Configure"}
          </button>
          <button
            type="button"
            className="btn"
            disabled={anyRunning}
            title={anyRunning ? "Stop the active run to benchmark the GPU" : undefined}
            onClick={() => navigate("/bench")}
          >
            Benchmark GPU
          </button>
          <button className="btn" disabled={busy} onClick={startFresh}>
            Start fresh run
          </button>
        </div>
      </header>

      {showKnobs && (
        <section className="panel mb-6">
          <div className="mb-3 flex items-center gap-2">
            <span className="section-title">Training knobs</span>
            <span className="text-xs text-slate-500">applied when the fresh run starts</span>
          </div>
          {config ? (
            <ConfigFields config={config} onChange={setField} disabled={busy} />
          ) : configError ? (
            <div className="text-sm text-red-300">Couldn't load defaults: {configError}</div>
          ) : (
            <div className="text-sm text-slate-500">Loading defaults…</div>
          )}
        </section>
      )}

      {error && <div className="mb-4 rounded border border-red-900 bg-red-950 p-3 text-sm text-red-200">{error}</div>}
      {loading && runs.length === 0 ? (
        <div className="text-sm text-slate-500">Loading runs…</div>
      ) : runs.length === 0 ? (
        <div className="text-sm text-slate-500">No runs yet. Start one above.</div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/60 text-left text-[10px] uppercase tracking-wide text-slate-500">
                <th className="px-3 py-2 font-medium">Run</th>
                <th className="px-3 py-2 text-right font-medium">Gen</th>
                <th className="px-3 py-2 font-medium">Board</th>
                <th className="px-3 py-2 text-right font-medium">Games</th>
                <th className="px-3 py-2 text-right font-medium">Policy loss</th>
                <th className="px-3 py-2 text-right font-medium">Value loss</th>
                <th className="px-3 py-2 text-right font-medium">Win rate</th>
                <th className="px-3 py-2 text-right font-medium">Updated</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr
                  key={run.runId}
                  onClick={() => navigate(`/runs/${encodeURIComponent(run.runId)}`)}
                  className="cursor-pointer border-b border-slate-800/60 text-slate-200 last:border-0 hover:bg-slate-800/40"
                >
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <span className="truncate font-semibold text-slate-100">{run.runId}</span>
                      {run.running ? (
                        <span className="rounded-full bg-green-500/15 px-2 py-0.5 text-[10px] font-medium text-green-400">live</span>
                      ) : null}
                    </div>
                  </td>
                  <td className="px-3 py-2 text-right font-mono">{run.generation}</td>
                  <td className="px-3 py-2 text-slate-300">{`${run.board}² · ${run.numSnakes}p`}</td>
                  <td className="px-3 py-2 text-right font-mono">{`${run.gameGenCount}`}</td>
                  <td className="px-3 py-2 text-right font-mono">{fmt(run.policyLoss)}</td>
                  <td className="px-3 py-2 text-right font-mono">{fmt(run.valueLoss)}</td>
                  <td className="px-3 py-2 text-right font-mono">{run.hasWinRate ? `${(run.winRate * 100).toFixed(0)}%` : "—"}</td>
                  <td className="px-3 py-2 text-right text-[11px] text-slate-500">{since(run.updatedUnixMs)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "—");
