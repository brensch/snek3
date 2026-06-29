import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

// Grouped charts. Each chart is a list of series; a series is plotted on the
// chart's y-treatment (`kind`):
//   unit    -> value on a fixed 0..1 axis (win rates, draw rate)
//   percent -> value on a fixed 0..100 axis
//   shared  -> value / (max of this chart's series)  (losses share a scale)
//   series  -> value / (this series' own max), so each series peaks at the top
//              (mixed scales; tooltip shows the real value)
//   lanes   -> each series gets its own horizontal band, min..max normalized
//              within the band. Lets wildly-different-scale parameters (lr 1e-3,
//              buffer 5e5, draw_value -0.25) all be read for *when they change*.
// Series keys may be dotted to reach nested fields, e.g. "params.sims".
// Only series present in the run are drawn, so legacy runs still render.
const CHARTS = [
  {
    title: "Losses (shared scale)",
    kind: "shared",
    series: [
      { key: "value_loss", label: "value loss", color: "#60a5fa" },
      { key: "policy_loss", label: "policy loss", color: "#f59e0b" },
    ],
  },
  {
    title: "Policy targets",
    kind: "series",
    series: [
      { key: "target_entropy", label: "target entropy (visit-count π)", color: "#ec4899" },
      { key: "target_max_prob", label: "target max prob", color: "#c084fc" },
    ],
  },
  {
    title: "Game length",
    kind: "shared",
    series: [
      { key: "avg_game_len", label: "avg game turns", color: "#34d399" },
      { key: "proxy_game_len", label: "proxy game turns", color: "#22d3ee", dash: true },
      { key: "completed_games", label: "completed games", color: "#f472b6" },
    ],
  },
  {
    title: "Draw monitoring",
    kind: "unit",
    series: [
      { key: "terminal_draw_rate", label: "terminal draws", color: "#fb923c" },
      { key: "overrun_draw_rate", label: "turn-cap draws", color: "#f87171" },
      { key: "proxy_draw_rate", label: "proxy draws", color: "#fbbf24", dash: true },
    ],
  },
  {
    title: "Generation time",
    kind: "shared",
    series: [
      { key: "gen_seconds", label: "self-play seconds", color: "#94a3b8" },
      { key: "train_seconds", label: "train seconds", color: "#60a5fa" },
      { key: "save_seconds", label: "save seconds", color: "#c084fc" },
    ],
  },
  {
    title: "Throughput",
    kind: "series",
    series: [
      { key: "samples_per_sec", label: "samples/sec", color: "#34d399" },
      { key: "turns_per_sec", label: "turns/sec", color: "#f59e0b" },
      { key: "games_per_sec", label: "games/sec", color: "#f472b6" },
      { key: "inference_per_sec", label: "inferences/sec", color: "#60a5fa" },
    ],
  },
  {
    title: "Utilization",
    kind: "percent",
    series: [
      { key: "gpu_busy_pct", label: "GPU busy %", color: "#34d399" },
      { key: "selfplay_gpu_pct", label: "self-play GPU %", color: "#22d3ee" },
    ],
  },
  {
    title: "Parameters (per-lane min–max)",
    kind: "lanes",
    series: [
      { key: "params.sims", label: "sims", color: "#f59e0b" },
      { key: "params.lr", label: "lr", color: "#60a5fa" },
      { key: "params.c_puct", label: "c_puct", color: "#34d399" },
      { key: "params.train_steps", label: "train steps", color: "#c084fc" },
      { key: "params.batch_size", label: "batch size", color: "#f472b6" },
      { key: "params.target_samples", label: "target samples", color: "#22d3ee" },
      { key: "params.buffer_size", label: "buffer size", color: "#fbbf24" },
      { key: "params.max_turns", label: "max turns", color: "#94a3b8" },
      { key: "params.exploration_prob", label: "exploration prob", color: "#fb923c" },
      { key: "params.draw_value", label: "draw value", color: "#f87171" },
      { key: "params.keep_games", label: "keep games", color: "#a3e635" },
      { key: "params.sample_games", label: "sample games", color: "#e879f9" },
      { key: "params.value_weight", label: "value weight", color: "#818cf8" },
      { key: "params.recency", label: "recency", color: "#fca5a5" },
    ],
  },
];

// Resolve possibly-dotted keys (e.g. "params.sims") against a metric row.
const getVal = (m, key) => {
  if (m == null) return null;
  if (key.indexOf(".") === -1) return m[key];
  let o = m;
  for (const k of key.split(".")) { if (o == null) return null; o = o[k]; }
  return o;
};
const present = (metrics, key) => metrics.some((m) => getVal(m, key) != null);
const fmt = (v) => {
  if (v == null || Number.isNaN(+v)) return null;
  const a = Math.abs(+v);
  if (a !== 0 && a < 0.01) return (+v).toExponential(1);
  return a >= 10 ? (+v).toFixed(0) : (+v).toFixed(3);
};

function Chart({ metrics, title, kind, series, hub }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const geomRef = useRef(null); // imperative draw geometry shared with the crosshair painter
  const [width, setWidth] = useState(380);
  const [hover, setHover] = useState(null); // drives only the DOM tooltip

  const used = useMemo(() => series.filter((s) => present(metrics, s.key)), [metrics, series]);

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((e) => setWidth(Math.max(220, Math.floor(e[0].contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const view = useMemo(() => {
    if (!metrics.length) return { gmin: 0, gmax: 1, sharedMax: 1, seriesMax: {}, seriesMin: {} };
    const gens = metrics.map((m) => m.gen);
    const gmin = Math.min(...gens);
    const gmax = Math.max(...gens, gmin + 1);
    let sharedMax = 0.001;
    const seriesMax = {};
    const seriesMin = {};
    if (kind === "shared") {
      for (const m of metrics) for (const s of used) { const v = getVal(m, s.key); if (v != null) sharedMax = Math.max(sharedMax, v); }
    }
    if (kind === "series" || kind === "lanes") {
      for (const s of used) {
        let mx = -Infinity, mn = Infinity;
        for (const m of metrics) { const v = getVal(m, s.key); if (v != null) { mx = Math.max(mx, v); mn = Math.min(mn, v); } }
        seriesMax[s.key] = mx === -Infinity ? 0.001 : mx;
        seriesMin[s.key] = mn === Infinity ? 0 : mn;
      }
    }
    return { gmin, gmax, sharedMax, seriesMax, seriesMin };
  }, [metrics, used, kind]);

  // value -> 0..1 fraction of the chart height for series `s` at used-index `i`.
  const frac = (s, v, i) => {
    if (v == null) return null;
    if (kind === "unit") return Math.min(1, Math.max(0, v));
    if (kind === "percent") return Math.min(1, Math.max(0, v / 100));
    if (kind === "shared") return Math.min(1, Math.max(0, v / view.sharedMax));
    if (kind === "lanes") {
      const mn = view.seriesMin[s.key], mx = view.seriesMax[s.key];
      const norm = mx > mn ? (v - mn) / (mx - mn) : 0.5;
      const n = used.length || 1;
      // lane i counts from the top; 12% margin, 76% usable band.
      return 1 - (i + 0.12 + 0.76 * norm) / n;
    }
    return Math.min(1, Math.max(0, v / (view.seriesMax[s.key] || 1))); // series: own max
  };
  const topLabel = kind === "shared" ? view.sharedMax.toFixed(2) : kind === "unit" ? "1.00" : kind === "percent" ? "100" : "max";

  const H = 190;
  const padL = 38, padR = 10, padT = 12, padB = 20;

  // Base layer: grid + series. Redraws only on data/size/scale changes — NOT on
  // hover, so pointer movement never triggers an expensive series redraw.
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    if (!metrics.length || !used.length) {
      geomRef.current = null;
      ctx.fillStyle = "#8b949e"; ctx.font = "12px ui-sans-serif";
      ctx.fillText("no data", padL, H / 2);
      return;
    }
    const { gmin, gmax } = view;
    const plotW = Math.max(1, width - padL - padR);
    const x = (g) => padL + ((g - gmin) / (gmax - gmin)) * plotW;
    const y = (f) => padT + (H - padT - padB) * (1 - f);

    ctx.strokeStyle = "#21262d"; ctx.fillStyle = "#8b949e"; ctx.lineWidth = 1; ctx.font = "10px ui-sans-serif";
    if (kind === "lanes") {
      for (let i = 1; i < used.length; i++) {
        const yy = padT + (H - padT - padB) * i / used.length;
        ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(width - padR, yy); ctx.stroke();
      }
    } else {
      for (let i = 0; i <= 4; i++) {
        const yy = padT + (H - padT - padB) * i / 4;
        ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(width - padR, yy); ctx.stroke();
      }
      ctx.fillText(topLabel, 4, padT + 4);
      ctx.fillText("0", 4, H - padB + 2);
    }

    used.forEach((s, i) => {
      const pts = [];
      for (const m of metrics) { const v = getVal(m, s.key); if (v != null) pts.push([x(m.gen), y(frac(s, v, i))]); }
      if (!pts.length) return;
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash ? [4, 4] : []);
      ctx.beginPath(); pts.forEach((p, j) => (j ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.stroke();
      ctx.setLineDash([]);
      if (kind !== "lanes") { // per-point dots are noise across many lanes
        ctx.fillStyle = s.color;
        pts.forEach((p) => { ctx.beginPath(); ctx.arc(p[0], p[1], 2, 0, 7); ctx.fill(); });
      }
    });

    // Publish geometry for the imperative crosshair painter.
    const byGen = new Map();
    for (const m of metrics) byGen.set(m.gen, m);
    geomRef.current = { x, y, frac, used, byGen, padT, padB, H };
    // Repaint the crosshair so it stays aligned after a resize / live append.
    if (hub.gen != null) paintCrosshair(hub.gen);
  }, [metrics, used, width, view, kind]);

  // Crosshair layer: vertical line + per-series markers at the hovered gen.
  // Driven imperatively by the hub (no React state) so a hover repaints every
  // chart's overlay with just a clear + a line + a few dots.
  const paintCrosshair = useCallback((gen) => {
    const o = overlayRef.current; if (!o) return;
    const octx = o.getContext("2d");
    octx.clearRect(0, 0, o.width, o.height);
    const g = geomRef.current;
    if (gen == null || !g) return;
    const hx = g.x(gen);
    octx.strokeStyle = "rgba(230,237,243,0.5)"; octx.lineWidth = 1; octx.setLineDash([3, 3]);
    octx.beginPath(); octx.moveTo(hx, g.padT); octx.lineTo(hx, g.H - g.padB); octx.stroke(); octx.setLineDash([]);
    const m = g.byGen.get(gen);
    if (!m) return;
    g.used.forEach((s, i) => {
      const v = getVal(m, s.key); if (v == null) return;
      octx.fillStyle = "#0b0e13"; octx.strokeStyle = s.color; octx.lineWidth = 2;
      octx.beginPath(); octx.arc(hx, g.y(g.frac(s, v, i)), 3.5, 0, 7); octx.fill(); octx.stroke();
    });
  }, []);

  // Register this chart's painter with the shared hub for the run's lifetime.
  useEffect(() => hub.subscribe(paintCrosshair), [hub, paintCrosshair]);

  const onMove = (e) => {
    if (!metrics.length || !used.length) return;
    const c = canvasRef.current, r = c.getBoundingClientRect();
    const px = (e.clientX - r.left) * (c.width / r.width);
    const py = (e.clientY - r.top) * (c.height / r.height);
    const { gmin, gmax } = view;
    const plotW = Math.max(1, width - padL - padR);
    const x = (g) => padL + ((g - gmin) / (gmax - gmin)) * plotW;
    const y = (f) => padT + (H - padT - padB) * (1 - f);
    const metric = metrics.reduce((b, m) => (Math.abs(x(m.gen) - px) < b.d ? { m, d: Math.abs(x(m.gen) - px) } : b), { m: metrics[0], d: Infinity }).m;
    // closest series by vertical distance at this gen
    let nearKey = null, best = Infinity;
    used.forEach((s, i) => {
      const v = getVal(metric, s.key); if (v == null) return;
      const d = Math.abs(y(frac(s, v, i)) - py);
      if (d < best) { best = d; nearKey = s.key; }
    });
    hub.emit(metric.gen);                          // crosshair across ALL charts (imperative)
    setHover({ metric, x: x(metric.gen), nearKey }); // tooltip on THIS chart (DOM)
  };
  const onLeave = () => { hub.emit(null); setHover(null); };

  // tooltip rows: every present series at this gen, ALPHABETICAL by label.
  const rows = hover?.metric
    ? used
        .filter((s) => getVal(hover.metric, s.key) != null)
        .map((s) => ({ key: s.key, label: s.label, color: s.color, val: fmt(getVal(hover.metric, s.key)) }))
        .sort((a, b) => a.label.localeCompare(b.label))
    : [];
  const ttLeft = hover ? Math.min(Math.max(hover.x + 10, 6), Math.max(6, width - 190)) : 0;

  return (
    <div className="chart-card" ref={wrapRef}>
      <div className="chart-title">{title}</div>
      <div className="chart-wrap" style={{ position: "relative" }}>
        <canvas ref={canvasRef} width={width} height={H} style={{ width: "100%" }}
          onPointerMove={onMove} onPointerLeave={onLeave} />
        <canvas ref={overlayRef} width={width} height={H}
          style={{ width: "100%", position: "absolute", left: 0, top: 0, pointerEvents: "none", background: "transparent" }} />
        {hover?.metric && rows.length > 0 && (
          <div className="chart-tooltip" style={{ left: ttLeft, top: 6 }}>
            <b>gen {hover.metric.gen}</b>
            {rows.map((r) => (
              <span key={r.key} className={"tt-row" + (r.key === hover.nearKey ? " near" : "")}>
                <i className="swatch" style={{ background: r.color }} />
                <em>{r.label}</em>
                <strong>{r.val}</strong>
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="legend">
        {used.map((s) => (
          <span key={s.key}><i className="swatch" style={{ background: s.color }} />{s.label}</span>
        ))}
      </div>
    </div>
  );
}

export default function MetricsChart({ metrics }) {
  // Shared crosshair hub: charts subscribe their imperative painters; a pointer
  // move emits the hovered gen to all of them. Lives for the component's life so
  // hovering never re-renders sibling charts.
  const hubRef = useRef(null);
  if (!hubRef.current) {
    const subs = new Set();
    hubRef.current = {
      gen: null,
      subscribe(fn) { subs.add(fn); return () => subs.delete(fn); },
      emit(gen) { this.gen = gen; for (const fn of subs) fn(gen); },
    };
  }
  return (
    <div className="charts-grid">
      {CHARTS.map((c) => (
        <Chart key={c.title} hub={hubRef.current} metrics={metrics} title={c.title} kind={c.kind} series={c.series} />
      ))}
    </div>
  );
}
