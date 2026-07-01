import { Link } from "react-router-dom";
import type { RunSummary } from "../gen/viewer_pb";
import { since } from "../lib/format";

// One run on the home grid. Links into the focused run view.
export function RunCard({ run }: { run: RunSummary }) {
  return (
    <Link
      to={`/runs/${encodeURIComponent(run.runId)}`}
      className="group grid gap-3 rounded-lg border border-slate-800 bg-slate-900 p-4 transition-colors hover:border-sky-600"
    >
      <div className="flex items-center gap-2">
        <span className="truncate font-semibold text-slate-100 group-hover:text-sky-300">{run.runId}</span>
        {run.running ? (
          <span className="rounded-full bg-green-500/15 px-2 py-0.5 text-[10px] font-medium text-green-400">live</span>
        ) : null}
        <span className="ml-auto text-[11px] text-slate-500">{since(run.updatedUnixMs)}</span>
      </div>
      <div className="grid grid-cols-3 gap-2 text-sm">
        <Stat label="Generation" value={String(run.generation)} />
        <Stat label="Board" value={`${run.board}² · ${run.numSnakes}p`} />
        <Stat label="Games" value={`${run.gameGenCount} gens`} />
        <Stat label="Policy loss" value={fmt(run.policyLoss)} />
        <Stat label="Value loss" value={fmt(run.valueLoss)} />
        <Stat label="Win rate" value={run.hasWinRate ? `${(run.winRate * 100).toFixed(0)}%` : "—"} />
      </div>
    </Link>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="font-mono text-slate-200">{value}</div>
    </div>
  );
}

const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "—");
