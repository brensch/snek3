import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

// Series plotted on the unit axis (0..1, read straight off the left scale).
// Win rates vs each opponent + the issue indicators that live in [0,1].
const UNIT_SERIES = [
  { key: "response_vs_uct", color: "#34d399", label: "response vs UCT" },
  { key: "response_vs_baseline", color: "#22d3ee", label: "response vs baseline" },
  { key: "proxy_vs_uct", color: "#a3e635", label: "proxy vs UCT", dash: true },
  { key: "proxy_vs_baseline", color: "#38bdf8", label: "proxy vs baseline", dash: true },
  { key: "win_rate", color: "#34d399", label: "win rate" }, // legacy MCTS runs
  { key: "proxy_draw_rate", color: "#f87171", label: "draw rate ⚠", issue: true },
  { key: "proxy_len_frac", color: "#c084fc", label: "game length / cap ⚠", issue: true },
];

// Series plotted on the shared normalized axis (value/max). Training-health
// diagnostics; a value loss or target entropy heading to 0 signals collapse.
const NORM_SERIES = [
  { key: "proxy_value_loss", color: "#60a5fa", label: "proxy value loss ⚠", dash: true, issue: true },
  { key: "response_value_loss", color: "#818cf8", label: "response value loss", dash: true },
  { key: "proxy_policy_loss", color: "#f59e0b", label: "proxy policy loss", dash: true },
  { key: "response_policy_loss", color: "#fbbf24", label: "response policy loss", dash: true },
  { key: "target_entropy", color: "#ec4899", label: "target entropy ⚠", issue: true },
  { key: "policy_loss", color: "#f59e0b", label: "policy loss", dash: true }, // legacy
  { key: "value_loss", color: "#60a5fa", label: "value loss", dash: true }, // legacy
];

const present = (metrics, key) => metrics.some((m) => m[key] != null);

// Fixed-scale line chart, x = generation, pan/zoom horizontally.
export default function MetricsChart({ metrics }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const dragRef = useRef(null);
  const [width, setWidth] = useState(900);
  const [pxPerGen, setPxPerGen] = useState(18);
  const [panX, setPanX] = useState(0);
  const [followLatest, setFollowLatest] = useState(true);
  const [hover, setHover] = useState(null);

  // Which series actually have data in this run.
  const unit = useMemo(() => UNIT_SERIES.filter((s) => present(metrics, s.key)), [metrics]);
  const norm = useMemo(() => NORM_SERIES.filter((s) => present(metrics, s.key)), [metrics]);

  const plot = useMemo(() => {
    if (!metrics.length) return { gmin: 0, gmax: 1, contentWidth: 0, maxPan: 0 };
    const gens = metrics.map((m) => m.gen);
    const gmin = Math.min(...gens);
    const gmax = Math.max(...gens, gmin + 1);
    const contentWidth = Math.max(1, (gmax - gmin) * pxPerGen);
    const plotWidth = Math.max(1, width - 42 - 14);
    return { gmin, gmax, contentWidth, maxPan: Math.max(0, contentWidth - plotWidth) };
  }, [metrics, pxPerGen, width]);

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => setWidth(Math.max(320, Math.floor(entries[0].contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setPanX((p) => followLatest ? plot.maxPan : Math.min(Math.max(0, p), plot.maxPan));
  }, [followLatest, plot.maxPan]);

  const beginDrag = (event) => {
    const el = wrapRef.current;
    if (!el || event.button === 1 || event.target?.closest?.("input,button")) return;
    dragRef.current = { x: event.clientX, panX };
    setFollowLatest(false);
    el.setPointerCapture?.(event.pointerId);
    el.classList.add("dragging");
  };

  const drag = (event) => {
    const el = wrapRef.current;
    const start = dragRef.current;
    if (!el || !start) return;
    setPanX(Math.min(Math.max(0, start.panX - (event.clientX - start.x)), plot.maxPan));
  };

  const endDrag = (event) => {
    const el = wrapRef.current;
    dragRef.current = null;
    if (!el) return;
    el.releasePointerCapture?.(event.pointerId);
    el.classList.remove("dragging");
  };

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const W = c.width, H = c.height, padL = 42, padR = 14, padT = 14, padB = 24;
    ctx.clearRect(0, 0, W, H);
    if (!metrics.length) {
      ctx.fillStyle = "#8b949e";
      ctx.font = "12px ui-sans-serif";
      ctx.fillText("no metrics yet", padL, H / 2);
      return;
    }
    const { gmin, gmax } = plot;
    const x = (g) => padL + (g - gmin) * pxPerGen - panX;
    ctx.strokeStyle = "#21262d";
    ctx.fillStyle = "#8b949e";
    ctx.lineWidth = 1;
    ctx.font = "11px ui-sans-serif";
    for (let i = 0; i <= 4; i++) {
      const yy = padT + (H - padT - padB) * i / 4;
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W - padR, yy); ctx.stroke();
      ctx.fillText((1 - i / 4).toFixed(2), 6, yy + 3);
    }
    // Shared normalized scale for the diagnostic (loss/entropy) series.
    const normVals = metrics.flatMap((m) => norm.map((s) => m[s.key])).filter((v) => v != null);
    const lmax = Math.max(0.001, ...normVals);
    const yU = (v) => padT + (H - padT - padB) * (1 - v);
    const yL = (v) => padT + (H - padT - padB) * (1 - v / lmax);
    const drawLine = (key, color, dash, yFn) => {
      const pts = metrics.filter((m) => m[key] != null).map((m) => [x(m.gen), yFn(m[key])]);
      if (!pts.length) return;
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash ? [4, 4] : []);
      ctx.beginPath();
      pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = color;
      pts.forEach((p) => { ctx.beginPath(); ctx.arc(p[0], p[1], 2.2, 0, 7); ctx.fill(); });
    };
    norm.forEach((s) => drawLine(s.key, s.color, s.dash, yL));
    unit.forEach((s) => drawLine(s.key, s.color, s.dash, yU));

    if (hover?.metric) {
      const m = hover.metric;
      const hx = x(m.gen);
      ctx.strokeStyle = "rgba(230, 237, 243, 0.42)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(hx, padT); ctx.lineTo(hx, H - padB); ctx.stroke();
      ctx.setLineDash([]);
      const dot = (value, yFn, color) => {
        if (value == null) return;
        ctx.fillStyle = "#0b0e13"; ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(hx, yFn(value), 5, 0, 7); ctx.fill(); ctx.stroke();
      };
      norm.forEach((s) => dot(m[s.key], yL, s.color));
      unit.forEach((s) => dot(m[s.key], yU, s.color));
    }
    ctx.fillStyle = "#8b949e";
    ctx.fillText("gen " + gmin, padL, H - 7);
    ctx.fillText("gen " + gmax, W - padR - 44, H - 7);
  }, [metrics, width, hover, pxPerGen, panX, plot, unit, norm]);

  const updateHover = (event) => {
    const c = canvasRef.current;
    if (!c || !metrics.length) return;
    const rect = c.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (c.width / rect.width);
    const py = (event.clientY - rect.top) * (c.height / rect.height);
    const padL = 42;
    const x = (g) => padL + (g - plot.gmin) * pxPerGen - panX;
    const nearest = metrics.reduce((best, metric) => {
      const dist = Math.abs(x(metric.gen) - px);
      return dist < best.dist ? { metric, dist } : best;
    }, { metric: metrics[0], dist: Infinity }).metric;
    setHover({ metric: nearest, x: x(nearest.gen), y: py });
  };

  const fmt = (value, digits = 3) => {
    if (value == null || Number.isNaN(Number(value))) return null;
    return Number(value).toFixed(digits);
  };

  // Tooltip: every plotted series with a value at the hovered gen, plus a few
  // raw extras that are handy but not lines.
  const tooltipRows = hover?.metric
    ? [
        ...unit.map((s) => [s.label, fmt(hover.metric[s.key])]),
        ...norm.map((s) => [s.label, fmt(hover.metric[s.key], 4)]),
        ["mean turns", fmt(hover.metric.proxy_mean_turns, 1)],
        ["samples", hover.metric.samples != null ? Number(hover.metric.samples).toLocaleString() : null],
        ["gen sec", fmt(hover.metric.gen_seconds, 1)],
      ].filter(([, value]) => value != null)
    : [];

  const tooltipLeft = hover ? Math.min(Math.max(hover.x + 12, 8), Math.max(8, width - 200)) : 0;
  const tooltipTop = hover ? (hover.y > 145 ? Math.max(8, hover.y - 150) : Math.min(210, hover.y + 12)) : 0;

  const legendItems = [...unit, ...norm];

  return (
    <div
      ref={wrapRef}
      className="chart-panel"
      onPointerDown={beginDrag}
      onPointerMove={drag}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      <div className="chart-toolbar">
        <label className="chk">width <input type="range" min={6} max={48} value={pxPerGen} onChange={(e) => setPxPerGen(+e.target.value)} /></label>
        <span className="muted">{pxPerGen}px/gen</span>
        <button
          type="button"
          className={"chart-lock" + (followLatest ? " active" : "")}
          onClick={() => setFollowLatest((locked) => !locked)}
          title={followLatest ? "Following latest generation" : "Lock view to latest generation"}
        >
          latest
        </button>
        {plot.maxPan > 0 && <span className="muted">drag chart to pan</span>}
      </div>
      <div className="chart-wrap">
        <canvas
          ref={canvasRef}
          width={width}
          height={260}
          style={{ width: "100%" }}
          onPointerMove={updateHover}
          onPointerLeave={() => setHover(null)}
        />
        {hover?.metric && (
          <div className="chart-tooltip" style={{ left: tooltipLeft, top: tooltipTop }}>
            <b>gen {hover.metric.gen}</b>
            {tooltipRows.map(([label, value]) => (
              <span key={label}>
                <em>{label}</em>
                <strong>{value}</strong>
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="legend">
        {legendItems.map((s) => (
          <span key={s.key}>
            <i className="swatch" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
        <span className="muted">solid = 0–1 scale · dashed = normalized · ⚠ = issue indicator</span>
      </div>
    </div>
  );
}
