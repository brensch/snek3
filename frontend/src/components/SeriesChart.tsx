import { useState } from "react";

type Props = {
  values: number[];
  label: string;
  color?: string;
  fixedMax?: number | null;
  xLeft?: string;
  xRight?: string;
  digits?: number;
  // Optional x-axis labels per point (e.g. generation numbers) for the tooltip.
  xValues?: number[];
  xUnit?: string;
};

// A single-series sparkline with y-axis ticks and a hover crosshair/tooltip.
// Shared by the live-stats tiles and the per-generation metric charts.
export function SeriesChart({
  values,
  label,
  color = "#38bdf8",
  fixedMax = null,
  xLeft,
  xRight,
  digits = 2,
  xValues,
  xUnit = "gen",
}: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const finite = values.filter(Number.isFinite);
  const max = fixedMax ?? niceMax(Math.max(1e-9, ...finite));
  const px = (i: number) => (values.length <= 1 ? 0 : (i / (values.length - 1)) * 100);
  const py = (v: number) => 100 - Math.max(0, Math.min(1, v / max)) * 100;
  const points = values.map((v, i) => `${px(i)},${py(v)}`).join(" ");
  const latest = finite.length ? finite[finite.length - 1] : 0;

  const onMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    setHover(Math.max(0, Math.min(values.length - 1, Math.round(frac * (values.length - 1)))));
  };
  const hv = hover != null ? values[hover] : null;
  const hoverX = hover != null && xValues ? xValues[hover] : hover;

  return (
    <div className="rounded border border-slate-800 bg-slate-900 p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
        <div className="font-mono text-xs text-slate-300">{format(latest, digits)}</div>
      </div>
      <div className="grid h-40 grid-cols-[3.25rem_minmax(0,1fr)] grid-rows-[minmax(0,1fr)_1.25rem] gap-x-2">
        <div className="row-start-1 flex h-full flex-col justify-between py-1 text-right font-mono text-[10px] text-slate-500">
          <span>{format(max, digits)}</span>
          <span>{format(max / 2, digits)}</span>
          <span>0</span>
        </div>
        <div
          className="relative min-w-0 overflow-hidden rounded border border-slate-800 bg-slate-950"
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
            {[0, 50, 100].map((y) => (
              <line key={y} x1="0" x2="100" y1={y} y2={y} stroke="#1e293b" strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
            ))}
            <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
            {hover != null && hv != null && (
              <>
                <line x1={px(hover)} x2={px(hover)} y1="0" y2="100" stroke="#475569" strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
                <circle cx={px(hover)} cy={py(hv)} r="2.5" fill={color} vectorEffect="non-scaling-stroke" />
              </>
            )}
          </svg>
          {hover != null && hv != null && (
            <div
              className="pointer-events-none absolute top-1 z-10 -translate-x-1/2 whitespace-nowrap rounded border border-slate-700 bg-slate-950/95 px-1.5 py-0.5 font-mono text-[10px] text-slate-200"
              style={{ left: `${Math.max(12, Math.min(88, px(hover)))}%` }}
            >
              {xUnit} {hoverX}: {format(hv, digits)}
            </div>
          )}
        </div>
        <div className="col-start-2 row-start-2 flex justify-between pt-1 font-mono text-[10px] text-slate-500">
          <span>{xLeft ?? ""}</span>
          <span>{xRight ?? "now"}</span>
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

function format(value: number, digits: number): string {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(1)}k`;
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(digits);
}
