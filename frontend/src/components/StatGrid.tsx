import { number, percent, rate } from "../lib/format";
import type { RunState, StatsFrame } from "../types";

type Props = { stats: StatsFrame | null; state: RunState | null };

export function StatGrid({ stats, state }: Props) {
  const rows = [
    ["Generation", number(stats?.generation ?? state?.generation)],
    ["Phase", stats?.phase ?? state?.phase ?? "-"],
    ["Inf/s", rate(stats?.inferences_per_sec)],
    ["GPU rows/s", rate(stats?.gpu_rows_per_sec)],
    ["Games/s", rate(stats?.games_per_sec)],
    ["Done", number(stats?.completed_games_total)],
    ["Samples", `${number(stats?.samples_collected)} / ${number(stats?.samples_target)}`],
    ["GPU busy", percent(stats?.gpu_busy_pct)],
    ["Batch rows", number(stats?.batch_avg_rows)],
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
