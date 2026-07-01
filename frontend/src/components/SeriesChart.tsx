import { useState } from "react";

// One line on the chart. `values` is the series; `color`/`name` are for drawing
// and the legend. Multiple series share the chart's y-axis.
export type Series = { values: number[]; color?: string; name?: string };

type Props = {
  label: string;
  // Single-series shorthand (kept for the many one-line callers).
  values?: number[];
  color?: string;
  // Multi-series: combine related metrics (e.g. policy + value loss) on one axis.
  series?: Series[];
  fixedMax?: number | null;
  xLeft?: string;
  xRight?: string;
  digits?: number;
  // Optional x-axis labels per point (e.g. generation numbers) for the tooltip.
  xValues?: number[];
  xUnit?: string;
  // Shorter chart body for the dense realtime row.
  compact?: boolean;
};

const DEFAULT_COLOR = "#38bdf8";

// A sparkline with y-axis ticks and a hover crosshair/tooltip. Draws one or more
// series on a shared axis; shared by the live-stats tiles and the per-generation
// metric charts.
export function SeriesChart({
  label,
  values,
  color = DEFAULT_COLOR,
  series,
  fixedMax = null,
  xLeft,
  xRight,
  digits = 2,
  xValues,
  xUnit = "gen",
  compact = false,
}: Props) {
  const [hover, setHover] = useState<number | null>(null);

  // Normalise to a series list so the rest of the component is series-agnostic.
  const lines: Series[] = series ?? [{ values: values ?? [], color, name: label }];
  const len = lines.reduce((n, s) => Math.max(n, s.values.length), 0);
  const multi = lines.length > 1;

  const finiteMax = Math.max(
    1e-9,
    ...lines.flatMap((s) => s.values.filter(Number.isFinite)),
  );
  const max = fixedMax ?? niceMax(finiteMax);
  const px = (i: number) => (len <= 1 ? 0 : (i / (len - 1)) * 100);
  const py = (v: number) => 100 - Math.max(0, Math.min(1, v / max)) * 100;

  const onMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    setHover(Math.max(0, Math.min(len - 1, Math.round(frac * (len - 1)))));
  };
  const hoverX = hover != null && xValues ? xValues[hover] : hover;

  const latest = (s: Series) => {
    const f = s.values.filter(Number.isFinite);
    return f.length ? f[f.length - 1] : 0;
  };

  return (
    <div className="rounded border border-slate-800 bg-slate-900 p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="truncate text-xs uppercase tracking-wide text-slate-500">{label}</div>
        {multi ? (
          <div className="flex shrink-0 items-center gap-2">
            {lines.map((s, i) => (
              <span key={i} className="flex items-center gap-1 font-mono text-[10px] text-slate-300">
                <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: s.color ?? DEFAULT_COLOR }} />
                <span className="text-slate-500">{s.name}</span>
                {format(latest(s), digits)}
              </span>
            ))}
          </div>
        ) : (
          <div className="font-mono text-xs text-slate-300">{format(latest(lines[0]), digits)}</div>
        )}
      </div>
      <div
        className={`grid grid-cols-[3.25rem_minmax(0,1fr)] grid-rows-[minmax(0,1fr)_1.25rem] gap-x-2 ${
          compact ? "h-24" : "h-40"
        }`}
      >
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
            {lines.map((s, li) => (
              <polyline
                key={li}
                points={s.values.map((v, i) => `${px(i)},${py(v)}`).join(" ")}
                fill="none"
                stroke={s.color ?? DEFAULT_COLOR}
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
            ))}
            {hover != null && (
              <>
                <line x1={px(hover)} x2={px(hover)} y1="0" y2="100" stroke="#475569" strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
                {lines.map((s, li) => {
                  const v = s.values[hover];
                  return v != null && Number.isFinite(v) ? (
                    <circle key={li} cx={px(hover)} cy={py(v)} r="2.5" fill={s.color ?? DEFAULT_COLOR} vectorEffect="non-scaling-stroke" />
                  ) : null;
                })}
              </>
            )}
          </svg>
          {hover != null && (
            <div
              className="pointer-events-none absolute top-1 z-10 -translate-x-1/2 whitespace-nowrap rounded border border-slate-700 bg-slate-950/95 px-1.5 py-0.5 font-mono text-[10px] text-slate-200"
              style={{ left: `${Math.max(12, Math.min(88, px(hover)))}%` }}
            >
              <span className="text-slate-500">
                {xUnit} {hoverX}
              </span>
              {lines.map((s, li) => {
                const v = s.values[hover];
                if (v == null || !Number.isFinite(v)) return null;
                return (
                  <span key={li} className="ml-1.5">
                    {multi && <span style={{ color: s.color ?? DEFAULT_COLOR }}>{s.name} </span>}
                    {format(v, digits)}
                  </span>
                );
              })}
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
