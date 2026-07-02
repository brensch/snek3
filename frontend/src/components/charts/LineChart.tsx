import { useMemo, useState } from "react";
import { chrome } from "../../lib/palette";

export type ChartSeries = {
  name: string;
  color: string;
  values: number[];
};

type Props = {
  title: string;
  series: ChartSeries[];
  /** X label per point (e.g. generation numbers); index used when omitted. */
  xValues?: number[];
  xUnit?: string;
  /** Plot height in px (the card grows around it). */
  height?: number;
  /** Value formatter shared by axis, legend and tooltip. */
  format?: (v: number) => string;
  /** Wash the area under a single series (10% of the hue). */
  area?: boolean;
  /** Pin the y-domain (e.g. 0–100 for percentages). */
  domain?: [number, number];
  className?: string;
};

// The one line chart. 2px lines, hairline solid gridlines, a crosshair that
// snaps to the nearest x and a tooltip listing every series at that x (values
// lead, names follow, keyed by a short line of the series color). Markers get
// a 2px surface ring. A legend with latest values is shown for ≥2 series; a
// single series is named by the title alone.
export function LineChart({
  title,
  series,
  xValues,
  xUnit = "gen",
  height = 128,
  format = compact,
  area = false,
  domain,
  className = "",
}: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const len = series.reduce((n, s) => Math.max(n, s.values.length), 0);
  const multi = series.length > 1;

  const [lo, hi] = useMemo(() => {
    if (domain) return domain;
    const finite = series.flatMap((s) => s.values.filter(Number.isFinite));
    if (!finite.length) return [0, 1];
    const max = Math.max(...finite);
    const min = Math.min(...finite);
    // Anchor at zero unless the data goes negative (Elo can).
    return [Math.min(0, niceFloor(min)), niceCeil(max)] as [number, number];
  }, [series, domain]);

  const px = (i: number) => (len <= 1 ? 50 : (i / (len - 1)) * 100);
  const py = (v: number) => 100 - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo || 1)) * 100;

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    setHover(Math.max(0, Math.min(len - 1, Math.round(frac * (len - 1)))));
  };
  const latest = (s: ChartSeries) => {
    for (let i = s.values.length - 1; i >= 0; i--) {
      if (Number.isFinite(s.values[i])) return s.values[i];
    }
    return null;
  };
  const xLabel = (i: number) => (xValues ? xValues[i] : i);
  const gridYs = [0, 25, 50, 75, 100];

  return (
    <div className={`card p-2.5 ${className}`}>
      <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
        <span className="card-title">{title}</span>
        <div className="flex flex-wrap items-baseline justify-end gap-x-3 gap-y-0.5">
          {series.map((s, i) => {
            const v = latest(s);
            return (
              <span key={i} className="flex items-baseline gap-1.5 text-xs">
                {multi && <span className="h-0.5 w-3 self-center rounded" style={{ background: s.color }} />}
                {multi && <span className="text-ink-3">{s.name}</span>}
                <span className="font-mono text-ink">{v == null ? "–" : format(v)}</span>
              </span>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-[2.5rem_minmax(0,1fr)] gap-x-1.5">
        <div className="flex flex-col justify-between py-0.5 text-right font-mono text-[10px] tabular-nums text-ink-3" style={{ height }}>
          <span>{format(hi)}</span>
          <span>{format(lo + (hi - lo) / 2)}</span>
          <span>{format(lo)}</span>
        </div>

        <div
          className="relative min-w-0 touch-none col-start-2"
          style={{ height }}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        >
          <div className="absolute inset-0 overflow-hidden rounded border border-white/5 bg-inset">
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
              {gridYs.map((y) => (
                <line key={y} x1="0" x2="100" y1={y} y2={y} stroke={chrome.grid} strokeWidth="1" vectorEffect="non-scaling-stroke" />
              ))}
              {area && series.length === 1 && len > 1 && (
                <polygon
                  points={`0,100 ${series[0].values.map((v, i) => `${px(i)},${py(v)}`).join(" ")} 100,100`}
                  fill={series[0].color}
                  opacity="0.1"
                />
              )}
              {series.map((s, i) => (
                <polyline
                  key={i}
                  points={s.values
                    .map((v, j) => (Number.isFinite(v) ? `${px(j)},${py(v)}` : null))
                    .filter(Boolean)
                    .join(" ")}
                  fill="none"
                  stroke={s.color}
                  strokeWidth="2"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  vectorEffect="non-scaling-stroke"
                />
              ))}
              {hover != null && (
                <line x1={px(hover)} x2={px(hover)} y1="0" y2="100" stroke={chrome.axis} strokeWidth="1" vectorEffect="non-scaling-stroke" />
              )}
            </svg>
            {/* Markers as HTML so they stay round despite the stretched SVG. */}
            {hover != null &&
              series.map((s, i) => {
                const v = s.values[hover];
                if (v == null || !Number.isFinite(v)) return null;
                return (
                  <span
                    key={i}
                    className="pointer-events-none absolute h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full"
                    style={{
                      left: `${px(hover)}%`,
                      top: `${py(v)}%`,
                      background: s.color,
                      boxShadow: `0 0 0 2px ${chrome.inset}`,
                    }}
                  />
                );
              })}
          </div>

          {hover != null && (
            <div
              className="pointer-events-none absolute top-1 z-10 whitespace-nowrap rounded-md border border-white/10 bg-surface px-2 py-1 text-[11px] shadow-xl"
              style={tooltipPos(px(hover))}
            >
              <div className="text-[10px] text-ink-3">
                {xUnit} {xLabel(hover)}
              </div>
              {series.map((s, i) => {
                const v = s.values[hover];
                if (v == null || !Number.isFinite(v)) return null;
                return (
                  <div key={i} className="flex items-baseline gap-1.5">
                    <span className="h-0.5 w-3 self-center rounded" style={{ background: s.color }} />
                    <span className="font-mono font-semibold text-ink">{format(v)}</span>
                    {multi && <span className="text-ink-3">{s.name}</span>}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="col-start-2 flex justify-between pt-1 font-mono text-[10px] tabular-nums text-ink-3">
          <span>{len ? `${xUnit} ${xLabel(0)}` : ""}</span>
          <span>{len ? `${xUnit} ${xLabel(len - 1)}` : ""}</span>
        </div>
      </div>
    </div>
  );
}

// Anchor the tooltip to the hovered x, flipping alignment near the edges so it
// never overflows out of view.
function tooltipPos(xPct: number): React.CSSProperties {
  if (xPct < 25) return { left: `${xPct}%` };
  if (xPct > 75) return { left: `${xPct}%`, transform: "translateX(-100%)" };
  return { left: `${xPct}%`, transform: "translateX(-50%)" };
}

function niceCeil(v: number): number {
  if (v <= 0) return 0;
  const pow = 10 ** Math.floor(Math.log10(v));
  const n = v / pow;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * pow;
}

function niceFloor(v: number): number {
  return v >= 0 ? 0 : -niceCeil(-v);
}

export function compact(v: number): string {
  if (!Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (a >= 10_000) return `${(v / 1000).toFixed(0)}k`;
  if (a >= 1000) return `${(v / 1000).toFixed(1)}k`;
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a === 0) return "0";
  if (a < 0.01) return v.toExponential(0);
  return v.toFixed(2);
}
