import type { StatsFrame } from "../gen/snek_pb";
import { number, percent, rate } from "../lib/format";
import { phaseLabel } from "../lib/phase";
import type { RunState } from "../types";

type Props = { stats: StatsFrame | null; state: RunState | null };

export function StatGrid({ stats, state }: Props) {
  const rows = [
    ["Generation", number(stats?.generation ?? state?.generation)],
    ["Phase", phaseLabel(stats?.phase ?? state?.phase)],
    ["Inf/s", rate(stats?.inferencesPerSec)],
    ["GPU rows/s", rate(stats?.gpuRowsPerSec)],
    ["Games/s", rate(stats?.gamesPerSec)],
    ["Done", number(stats ? Number(stats.completedGamesTotal) : undefined)],
    ["Samples", `${number(stats?.samplesCollected)} / ${number(stats?.samplesTarget)}`],
    ["GPU busy", percent(stats?.gpuBusyPct)],
    ["Batch rows", number(stats?.batchAvgRows)],
  ];
  return (
    <section className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5">
      {rows.map(([label, value]) => (
        <div key={label} className="rounded border border-slate-800 bg-slate-900 p-3">
          <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
          <div className="mt-1 font-mono text-lg text-slate-100">{value}</div>
        </div>
      ))}
    </section>
  );
}
