import type { StatsFrame } from "../types";

type Props = { rows: StatsFrame[]; field: keyof StatsFrame; label: string };

export function MetricChart({ rows, field, label }: Props) {
  const values = rows.map((r) => Number(r[field])).filter(Number.isFinite);
  const fixedMax = field === "gpu_busy_pct" ? 100 : null;
  const max = fixedMax ?? niceMax(Math.max(1e-9, ...values));
  const points = rows.map((row, i) => {
    const x = rows.length <= 1 ? 0 : (i / (rows.length - 1)) * 100;
    const y = 100 - Math.max(0, Math.min(1, Number(row[field]) / max)) * 100;
    return `${x},${y}`;
  }).join(" ");
  const latest = values.length ? values[values.length - 1] : 0;

  return (
    <div className="rounded border border-slate-800 bg-slate-900 p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
        <div className="font-mono text-xs text-slate-300">{format(latest)}</div>
      </div>
      <div className="grid h-40 grid-cols-[3.25rem_minmax(0,1fr)] grid-rows-[minmax(0,1fr)_1.25rem] gap-x-2">
        <div className="row-start-1 flex h-full flex-col justify-between py-1 text-right font-mono text-[10px] text-slate-500">
          <span>{format(max)}</span>
          <span>{format(max / 2)}</span>
          <span>0</span>
        </div>
        <div className="relative min-w-0 overflow-hidden rounded border border-slate-800 bg-slate-950">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
            {[0, 50, 100].map((y) => (
              <line key={y} x1="0" x2="100" y1={y} y2={y} stroke="#1e293b" strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
            ))}
            {[0, 50, 100].map((x) => (
              <line key={x} x1={x} x2={x} y1="0" y2="100" stroke="#0f172a" strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
            ))}
            <polyline points={points} fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
          </svg>
        </div>
        <div className="col-start-2 row-start-2 flex justify-between pt-1 font-mono text-[10px] text-slate-500">
          <span>-{Math.max(0, rows.length - 1)}</span>
          <span>now</span>
        </div>
      </div>
    </div>
  );
}

function niceMax(value: number): number {
  const pow = 10 ** Math.floor(Math.log10(value));
  const n = value / pow;
  const step = n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return step * pow;
}

function format(value: number): string {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(1)}k`;
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2);
}
