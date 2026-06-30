import type { GenerationMetric } from "../types";

type Props = {
  rows: GenerationMetric[];
  field: keyof GenerationMetric;
  label: string;
};

export function GenerationChart({ rows, field, label }: Props) {
  const chartRows = rows.filter((row) => Number.isFinite(Number(row[field])));
  const values = chartRows.map((row) => Number(row[field]));
  const max = niceMax(Math.max(1e-9, ...values));
  const minGen = chartRows.length ? chartRows[0].generation : 0;
  const maxGen = chartRows.length ? chartRows[chartRows.length - 1].generation : 0;
  const points = chartRows
    .map((row, i) => {
      const value = Number(row[field]);
      const x = chartRows.length <= 1 ? 0 : (i / (chartRows.length - 1)) * 100;
      const y = 100 - Math.max(0, Math.min(1, value / max)) * 100;
      return `${x},${y}`;
    })
    .join(" ");
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
            <polyline points={points} fill="none" stroke="#f59e0b" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
          </svg>
        </div>
        <div className="col-start-2 row-start-2 flex justify-between pt-1 font-mono text-[10px] text-slate-500">
          <span>gen {minGen}</span>
          <span>gen {maxGen}</span>
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
  return value.toFixed(3);
}
