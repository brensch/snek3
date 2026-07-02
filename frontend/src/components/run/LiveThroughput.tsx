import type { StatsFrame } from "../../gen/snek_pb";
import { series } from "../../lib/palette";
import { compact } from "../charts/LineChart";

// The machine right now: the last ~60s of inference rate and GPU busy from
// the live stats stream. Two stacked metrics, each with its own scale.
export function LiveThroughput({ stats, history }: { stats: StatsFrame | null; history: StatsFrame[] }) {
  return (
    <div className="card flex flex-col gap-2.5 p-2.5">
      <div className="flex items-baseline justify-between">
        <span className="card-title">Live throughput</span>
        <span className="text-[10px] text-ink-3">{history.length > 1 ? "60s" : "idle"}</span>
      </div>
      <Metric
        label="inferences / s"
        value={stats ? compact(stats.inferencesPerSec) : "–"}
        values={history.map((f) => f.inferencesPerSec)}
        color={series.blue}
      />
      <Metric
        label="gpu busy"
        value={stats ? `${stats.gpuBusyPct.toFixed(0)}%` : "–"}
        values={history.map((f) => f.gpuBusyPct)}
        color={series.yellow}
        max={100}
      />
    </div>
  );
}

function Metric({
  label,
  value,
  values,
  color,
  max,
}: {
  label: string;
  value: string;
  values: number[];
  color: string;
  max?: number;
}) {
  return (
    <div className="min-w-0">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[11px] text-ink-3">{label}</span>
        <span className="font-mono text-base font-semibold leading-none text-ink">{value}</span>
      </div>
      <Spark values={values} color={color} max={max} />
    </div>
  );
}

function Spark({ values, color, max }: { values: number[]; color: string; max?: number }) {
  const finite = values.filter(Number.isFinite);
  const hi = max ?? ((finite.length ? Math.max(...finite) : 1) || 1);
  const pts = values
    .map((v, i) => {
      if (!Number.isFinite(v)) return null;
      const x = values.length <= 1 ? 50 : (i / (values.length - 1)) * 100;
      const y = 94 - (Math.min(v, hi) / hi) * 88;
      return `${x},${y}`;
    })
    .filter(Boolean);
  return (
    <span className="relative block h-9 w-full overflow-hidden rounded border border-white/5 bg-inset">
      {pts.length > 1 && (
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
          <polygon points={`0,100 ${pts.join(" ")} 100,100`} fill={color} opacity="0.12" />
          <polyline
            points={pts.join(" ")}
            fill="none"
            stroke={color}
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      )}
    </span>
  );
}
