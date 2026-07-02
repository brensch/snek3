import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { control } from "../api/client";
import { ConfigFields } from "../components/ConfigFields";
import { useRunList } from "../hooks/useRunList";
import { since } from "../lib/format";
import type { RunConfig } from "../types";

// Home: the live run first (one click back into the thing being monitored),
// then every run on disk, then the controls to start something new.
export function RunsHome() {
  const { runs, loading, error } = useRunList();
  const [runId, setRunId] = useState("");
  const [busy, setBusy] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [config, setConfig] = useState<RunConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const navigate = useNavigate();

  // Seed the knob form with the trainer's default config.
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

  const liveRun = runs.find((run) => run.running) ?? null;
  const open = (id: string) => navigate(`/runs/${encodeURIComponent(id)}`);

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
    <div className="mx-auto w-full max-w-5xl px-4 py-6 sm:px-6">
      <header className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold text-ink">snek3</h1>
          <p className="text-sm text-ink-3">Training runs on disk — pick one to monitor.</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="btn"
            disabled={!!liveRun}
            title={liveRun ? "Stop the active run to benchmark the GPU" : undefined}
            onClick={() => navigate("/bench")}
          >
            Benchmark GPU
          </button>
          <button type="button" className="btn-primary" aria-pressed={showNew} onClick={() => setShowNew((v) => !v)}>
            New run…
          </button>
        </div>
      </header>

      {showNew && (
        <section className="card mb-5 p-4">
          <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
            <label className="grid gap-1">
              <span className="card-title">Run id (optional)</span>
              <input
                className="input w-56"
                value={runId}
                onChange={(e) => setRunId(e.target.value)}
                placeholder="auto-timestamp"
              />
            </label>
            <button className="btn-primary" disabled={busy || !!liveRun} onClick={startFresh}>
              {busy ? "Starting…" : "Start fresh run"}
            </button>
          </div>
          {config ? (
            <ConfigFields config={config} onChange={setField} disabled={busy} />
          ) : configError ? (
            <div className="text-sm text-bad">Couldn't load defaults: {configError}</div>
          ) : (
            <div className="text-sm text-ink-3">Loading defaults…</div>
          )}
        </section>
      )}

      {liveRun && (
        <button
          onClick={() => open(liveRun.runId)}
          className="card mb-5 flex w-full flex-wrap items-center gap-x-6 gap-y-2 border-good/30 p-4 text-left transition-colors hover:border-good/60"
        >
          <span className="flex items-center gap-2">
            <span className="h-2 w-2 animate-pulse rounded-full bg-good" />
            <span className="font-mono text-base font-semibold text-ink">{liveRun.runId}</span>
          </span>
          <Fact label="gen" value={String(liveRun.generation)} />
          <Fact label="board" value={`${liveRun.board}² · ${liveRun.numSnakes}p`} />
          <Fact label="π loss" value={fmt(liveRun.policyLoss)} />
          <Fact label="v loss" value={fmt(liveRun.valueLoss)} />
          <span className="ml-auto text-sm text-good">monitor →</span>
        </button>
      )}

      {error && <div className="card mb-4 border-bad/40 p-3 text-sm text-bad">{error}</div>}
      {loading && runs.length === 0 ? (
        <div className="text-sm text-ink-3">Loading runs…</div>
      ) : runs.length === 0 ? (
        <div className="text-sm text-ink-3">No runs yet. Start one above.</div>
      ) : (
        <div className="card overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-white/10 text-left text-[10px] uppercase tracking-wider text-ink-3">
                <th className="px-3 py-2 font-medium">Run</th>
                <th className="px-3 py-2 text-right font-medium">Gen</th>
                <th className="px-3 py-2 font-medium">Board</th>
                <th className="px-3 py-2 text-right font-medium">π loss</th>
                <th className="px-3 py-2 text-right font-medium">v loss</th>
                <th className="px-3 py-2 text-right font-medium">Updated</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {runs.map((run) => (
                <tr
                  key={run.runId}
                  onClick={() => open(run.runId)}
                  className="cursor-pointer border-b border-white/5 last:border-0 hover:bg-white/5"
                >
                  <td className="px-3 py-2">
                    <span className="flex items-center gap-2 font-sans">
                      <span className="truncate font-mono font-semibold text-ink">{run.runId}</span>
                      {run.running && (
                        <span className="chip text-good">
                          <span className="h-1.5 w-1.5 rounded-full bg-good" />
                          live
                        </span>
                      )}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right text-ink-2">{run.generation}</td>
                  <td className="px-3 py-2 text-ink-3">{`${run.board}² · ${run.numSnakes}p`}</td>
                  <td className="px-3 py-2 text-right text-ink-2">{fmt(run.policyLoss)}</td>
                  <td className="px-3 py-2 text-right text-ink-2">{fmt(run.valueLoss)}</td>
                  <td className="px-3 py-2 text-right font-sans text-[11px] text-ink-3">{since(run.updatedUnixMs)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <span className="text-sm">
      <span className="text-ink-3">{label} </span>
      <span className="font-mono tabular-nums text-ink-2">{value}</span>
    </span>
  );
}

const fmt = (v: number) => (Number.isFinite(v) && v !== 0 ? v.toFixed(3) : "–");
