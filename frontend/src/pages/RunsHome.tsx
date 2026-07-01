import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { control } from "../api/client";
import { RunCard } from "../components/RunCard";
import { useRunList } from "../hooks/useRunList";

// Home page: every run on disk, plus a control to start a fresh one.
export function RunsHome() {
  const { runs, loading, error } = useRunList();
  const [runId, setRunId] = useState("");
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();

  async function startFresh() {
    setBusy(true);
    try {
      const { run_id } = await control.start(runId.trim() || null, true);
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
          <button className="btn" disabled={busy} onClick={startFresh}>
            Start fresh run
          </button>
        </div>
      </header>

      {error && <div className="mb-4 rounded border border-red-900 bg-red-950 p-3 text-sm text-red-200">{error}</div>}
      {loading && runs.length === 0 ? (
        <div className="text-sm text-slate-500">Loading runs…</div>
      ) : runs.length === 0 ? (
        <div className="text-sm text-slate-500">No runs yet. Start one above.</div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {runs.map((run) => (
            <RunCard key={run.runId} run={run} />
          ))}
        </div>
      )}
    </div>
  );
}
