import { useMemo } from "react";
import type { MetricRow } from "../../gen/viewer_pb";
import { series } from "../../lib/palette";
import { LineChart } from "../charts/LineChart";

// The LE-mode headline: is the net actually getting stronger? Sole-survival
// win rate against held-out baselines it never trains on — the Elo
// replacement (the checkpoint league is disabled in LE mode). Voronoi leads
// because it's the harder opponent and the one the whole lineage plateaued
// against; floodfill is the sanity floor. Sparse series (eval gens only).
export function StrengthHero({ metrics }: { metrics: MetricRow[] }) {
  const rows = useMemo(() => metrics.filter((m) => m.hasLeEval), [metrics]);
  if (rows.length === 0) return null;

  const gens = rows.map((m) => m.generation);
  const vor = rows.map((m) => m.leVorWinrate * 100);
  const ff = rows.map((m) => m.leFfWinrate * 100);

  return (
    <div className="card p-3">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <span className="card-title">Strength — held-out sole-survival %</span>
        <span className="text-[10px] text-ink-3">
          net's own equilibrium search vs baselines it never trains against · eval gens only
        </span>
      </div>
      <div className="grid gap-3 sm:grid-cols-[11rem_minmax(0,1fr)]">
        <div className="flex flex-row gap-2 sm:flex-col">
          <Headline label="vs voronoi" color={series.magenta} values={vor} gens={gens} />
          <Headline label="vs floodfill" color={series.green} values={ff} gens={gens} />
        </div>
        <LineChart
          title="Win rate by generation"
          height={168}
          domain={[0, 100]}
          format={(v) => `${v.toFixed(0)}%`}
          series={[
            { name: "vs voronoi", color: series.magenta, values: vor },
            { name: "vs floodfill", color: series.green, values: ff },
          ]}
          xValues={gens}
        />
      </div>
    </div>
  );
}

// One opponent's latest rate, big, with the delta against the previous eval
// point — the "is the newest checkpoint better" glance.
function Headline({
  label,
  color,
  values,
  gens,
}: {
  label: string;
  color: string;
  values: number[];
  gens: number[];
}) {
  const last = values[values.length - 1];
  const prev = values.length > 1 ? values[values.length - 2] : null;
  const delta = prev == null ? null : last - prev;
  return (
    <div className="flex-1 rounded-md border border-white/5 bg-inset px-3 py-2">
      <div className="flex items-center gap-1.5 text-[11px] text-ink-3">
        <span className="h-0.5 w-3 rounded" style={{ background: color }} />
        {label}
      </div>
      <div className="mt-0.5 flex items-baseline gap-2">
        <span className="font-mono text-3xl font-semibold tabular-nums text-ink">{last.toFixed(0)}%</span>
        {delta != null && (
          <span
            className={`font-mono text-xs tabular-nums ${
              delta > 0 ? "text-good" : delta < 0 ? "text-bad" : "text-ink-3"
            }`}
            title={`vs previous eval (gen ${gens[gens.length - 2]})`}
          >
            {delta > 0 ? "▲" : delta < 0 ? "▼" : "•"} {Math.abs(delta).toFixed(0)}
          </span>
        )}
      </div>
      <div className="text-[10px] text-ink-3">gen {gens[gens.length - 1]}</div>
    </div>
  );
}
