import { useMemo } from "react";
import type { MetricRow } from "../../gen/viewer_pb";
import { series } from "../../lib/palette";
import { LineChart } from "../charts/LineChart";

// The LE-mode headline: is the net actually getting stronger? Three lines:
//  - candidate vs voronoi@gate_sims — the learner's progress (the gate match)
//  - incumbent vs voronoi@gate_sims — the promoted net's level on the same
//    seeds; the candidate line crossing above it is what earns promotion
//  - incumbent vs voronoi@probe_sims — the STRONG opponent (the historical
//    20k-sim anchor). This is the super-heuristic goal line; expect it low.
// Headline stats: the probe (goal) and the candidate's gate rate with its
// delta, plus the promotion count so gating progress is visible at a glance.
export function StrengthHero({ metrics }: { metrics: MetricRow[] }) {
  const gates = useMemo(() => metrics.filter((m) => m.hasLeEval), [metrics]);
  const probes = useMemo(() => metrics.filter((m) => m.hasProbe), [metrics]);
  if (gates.length === 0) return null;

  const gens = gates.map((m) => m.generation);
  const cand = gates.map((m) => m.leVorWinrate * 100);
  const inc = gates.map((m) => m.leVorIncumbent * 100);
  // Probe is sparser than gates: align on gate gens, NaN where not measured
  // (the chart skips NaN points).
  const probeByGen = new Map(probes.map((m) => [m.generation, m.leVorProbe * 100]));
  const probe = gates.map((m) => probeByGen.get(m.generation) ?? NaN);
  const promotions = gates.filter((m) => m.gatePromoted).length;

  return (
    <div className="card p-3">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <span className="card-title">Strength — sole-survival % vs voronoi</span>
        <span className="text-[10px] text-ink-3">
          gate: paired vs voronoi@gate-sims · probe: incumbent vs the strong (20k-sim) anchor
        </span>
      </div>
      <div className="grid gap-3 sm:grid-cols-[11rem_minmax(0,1fr)]">
        <div className="flex flex-row gap-2 sm:flex-col">
          <Headline label="goal: vs strong voronoi" color={series.orange} values={probe.filter(Number.isFinite)} suffix="" />
          <Headline label="candidate (gate)" color={series.magenta} values={cand} suffix="" />
          <div className="flex-1 rounded-md border border-white/5 bg-inset px-3 py-2">
            <div className="text-[11px] text-ink-3">promotions</div>
            <div className="mt-0.5 font-mono text-3xl font-semibold tabular-nums text-ink">{promotions}</div>
            <div className="text-[10px] text-ink-3">gated checkpoints accepted</div>
          </div>
        </div>
        <LineChart
          title="Win rate by generation"
          height={168}
          domain={[0, 100]}
          format={(v) => `${v.toFixed(0)}%`}
          series={[
            { name: "candidate", color: series.magenta, values: cand },
            { name: "incumbent", color: series.blue, values: inc },
            { name: "vs strong voronoi", color: series.orange, values: probe },
          ]}
          xValues={gens}
        />
      </div>
    </div>
  );
}

// One line's latest value, big, with the delta against its previous point.
function Headline({
  label,
  color,
  values,
  suffix,
}: {
  label: string;
  color: string;
  values: number[];
  suffix: string;
}) {
  const last = values.length ? values[values.length - 1] : null;
  const prev = values.length > 1 ? values[values.length - 2] : null;
  const delta = last != null && prev != null ? last - prev : null;
  return (
    <div className="flex-1 rounded-md border border-white/5 bg-inset px-3 py-2">
      <div className="flex items-center gap-1.5 text-[11px] text-ink-3">
        <span className="h-0.5 w-3 rounded" style={{ background: color }} />
        {label}
      </div>
      <div className="mt-0.5 flex items-baseline gap-2">
        <span className="font-mono text-3xl font-semibold tabular-nums text-ink">
          {last == null ? "–" : `${last.toFixed(0)}%${suffix}`}
        </span>
        {delta != null && (
          <span
            className={`font-mono text-xs tabular-nums ${
              delta > 0 ? "text-good" : delta < 0 ? "text-bad" : "text-ink-3"
            }`}
          >
            {delta > 0 ? "▲" : delta < 0 ? "▼" : "•"} {Math.abs(delta).toFixed(0)}
          </span>
        )}
      </div>
    </div>
  );
}
