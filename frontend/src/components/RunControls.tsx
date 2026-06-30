import { useState } from "react";
import type { RunList, RunState } from "../types";

type Props = {
  runs: RunList;
  state: RunState | null;
  onStart: (runId: string, fresh: boolean) => Promise<void>;
  onStop: () => Promise<void>;
};

export function RunControls({ runs, state, onStart, onStop }: Props) {
  const [runId, setRunId] = useState("");
  const [busy, setBusy] = useState(false);
  const running = state?.running;

  async function invoke(action: () => Promise<void>) {
    setBusy(true);
    try { await action(); } finally { setBusy(false); }
  }

  return (
    <section className="rounded border border-slate-800 bg-slate-900 p-4">
      <div className="flex flex-wrap items-end gap-3">
        <label className="grid gap-1">
          <span className="text-xs uppercase text-slate-500">Run id</span>
          <input className="w-56 rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm" value={runId} onChange={(e) => setRunId(e.target.value)} placeholder="new-run" />
        </label>
        <button className="btn" disabled={busy || !runId.trim()} onClick={() => invoke(() => onStart(runId.trim(), true))}>Start fresh</button>
        <button className="btn" disabled={busy || !runId.trim()} onClick={() => invoke(() => onStart(runId.trim(), false))}>Resume</button>
        <button className="btn-danger" disabled={busy || !running} onClick={() => invoke(onStop)}>Stop</button>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {runs.runs.map((run) => (
          <button key={run} className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-300" onClick={() => setRunId(run)}>
            {run}{run === runs.live ? " · live" : ""}
          </button>
        ))}
      </div>
    </section>
  );
}
