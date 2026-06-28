import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

// Three grouped charts. Each chart is a list of series; a series is plotted on
// the chart's y-treatment (`kind`):
//   unit   -> value on a fixed 0..1 axis (win rates, draw rate)
//   shared -> value / (max of this chart's series)  (losses share a scale)
//   series -> value / (this series' own max), so each series peaks at the top
//             (mixed scales; tooltip shows the real value)
// Only series present in the run are drawn, so legacy MCTS runs still render.
const CHARTS = [
  {
    title: "Results — win rate (0–1)",
    kind: "unit",
    series: [
      // Faithful eval: deployed agent (proxy ONNX + serve search) vs the pool.
      { key: "vs_base", label: "vs baseline", color: "#34d399" },
      { key: "vs_uct", label: "vs UCT", color: "#22d3ee" },
      // Legacy keys (older runs that recorded win-rate in metrics.jsonl).
      { key: "response_vs_baseline", label: "response vs baseline", color: "#38bdf8", dash: true },
      { key: "response_vs_uct", label: "response vs UCT", color: "#a3e635", dash: true },
      { key: "proxy_vs_baseline", label: "proxy vs baseline", color: "#0ea5e9", dash: true },
      { key: "proxy_vs_uct", label: "proxy vs UCT", color: "#84cc16", dash: true },
      { key: "win_rate", label: "win rate", color: "#34d399" }, // legacy
    ],
  },
  {
    title: "Losses (shared scale)",
    kind: "shared",
    series: [
      { key: "proxy_value_loss", label: "proxy value loss", color: "#60a5fa" },
      { key: "proxy_policy_loss", label: "proxy policy loss", color: "#f59e0b" },
      { key: "response_value_loss", label: "response value loss", color: "#818cf8", dash: true },
      { key: "response_policy_loss", label: "response policy loss", color: "#fbbf24", dash: true },
      { key: "value_loss", label: "value loss", color: "#60a5fa" }, // legacy
      { key: "policy_loss", label: "policy loss", color: "#f59e0b" }, // legacy
    ],
  },
  {
    title: "Game shape / degeneracy",
    kind: "series",
    series: [
      { key: "proxy_draw_rate", label: "draw rate ⚠", color: "#f87171" },
      { key: "target_entropy", label: "target entropy ⚠", color: "#ec4899" },
      { key: "proxy_game_len", label: "avg game length", color: "#c084fc" },
      { key: "proxy_mean_turns", label: "turn-activity ratio", color: "#7c6f9c", dash: true },
    ],
  },
  {
    title: "Throughput / utilization",
    kind: "series",
    series: [
      { key: "proxy_games", label: "games finished", color: "#f472b6" },
      { key: "selfplay_gpu_pct", label: "self-play GPU %", color: "#34d399" },
      { key: "gpu_busy_pct", label: "GPU busy %", color: "#34d399" }, // legacy
      { key: "inference_per_sec", label: "inferences/sec", color: "#60a5fa" },
      { key: "samples_per_sec", label: "samples/sec", color: "#f59e0b" },
      { key: "gpu_peak_gb", label: "GPU peak GB", color: "#a78bfa" },
      { key: "gen_seconds", label: "gen seconds", color: "#94a3b8" },
    ],
  },
];

const present = (metrics, key) => metrics.some((m) => m[key] != null);
const fmt = (v) => (v == null || Number.isNaN(+v) ? null : Math.abs(+v) >= 10 ? (+v).toFixed(0) : (+v).toFixed(3));

function Chart({ metrics, title, kind, series }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const [width, setWidth] = useState(380);
  const [hover, setHover] = useState(null);

  const used = useMemo(() => series.filter((s) => present(metrics, s.key)), [metrics, series]);

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((e) => setWidth(Math.max(220, Math.floor(e[0].contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const view = useMemo(() => {
    if (!metrics.length) return { gmin: 0, gmax: 1, sharedMax: 1, seriesMax: {} };
    const gens = metrics.map((m) => m.gen);
    const gmin = Math.min(...gens);
    const gmax = Math.max(...gens, gmin + 1);
    let sharedMax = 0.001;
    const seriesMax = {};
    if (kind === "shared") {
      for (const m of metrics) for (const s of used) if (m[s.key] != null) sharedMax = Math.max(sharedMax, m[s.key]);
    }
    if (kind === "series") {
      for (const s of used) {
        let mx = 0.001;
        for (const m of metrics) if (m[s.key] != null) mx = Math.max(mx, m[s.key]);
        seriesMax[s.key] = mx;
      }
    }
    return { gmin, gmax, sharedMax, seriesMax };
  }, [metrics, used, kind]);

  // value -> 0..1 fraction of the chart height for a given series.
  const frac = (s, v) => {
    if (v == null) return null;
    if (kind === "unit") return Math.min(1, Math.max(0, v));
    if (kind === "shared") return Math.min(1, Math.max(0, v / view.sharedMax));
    return Math.min(1, Math.max(0, v / (view.seriesMax[s.key] || 1))); // series: own max
  };
  const topLabel = kind === "shared" ? view.sharedMax.toFixed(2) : kind === "unit" ? "1.00" : "max";

  const H = 190;
  const padL = 38, padR = 10, padT = 12, padB = 20;

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    if (!metrics.length || !used.length) {
      ctx.fillStyle = "#8b949e"; ctx.font = "12px ui-sans-serif";
      ctx.fillText("no data", padL, H / 2);
      return;
    }
    const { gmin, gmax } = view;
    const plotW = Math.max(1, width - padL - padR);
    const x = (g) => padL + ((g - gmin) / (gmax - gmin)) * plotW;
    const y = (f) => padT + (H - padT - padB) * (1 - f);

    ctx.strokeStyle = "#21262d"; ctx.fillStyle = "#8b949e"; ctx.lineWidth = 1; ctx.font = "10px ui-sans-serif";
    for (let i = 0; i <= 4; i++) {
      const yy = padT + (H - padT - padB) * i / 4;
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(width - padR, yy); ctx.stroke();
    }
    ctx.fillText(topLabel, 4, padT + 4);
    ctx.fillText("0", 4, H - padB + 2);

    for (const s of used) {
      const pts = metrics.filter((m) => m[s.key] != null).map((m) => [x(m.gen), y(frac(s, m[s.key]))]);
      if (!pts.length) continue;
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash ? [4, 4] : []);
      ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle = s.color;
      pts.forEach((p) => { ctx.beginPath(); ctx.arc(p[0], p[1], 2, 0, 7); ctx.fill(); });
    }

    if (hover?.metric) {
      const m = hover.metric, hx = x(m.gen);
      ctx.strokeStyle = "rgba(230,237,243,0.4)"; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(hx, padT); ctx.lineTo(hx, H - padB); ctx.stroke(); ctx.setLineDash([]);
      for (const s of used) {
        if (m[s.key] == null) continue;
        ctx.fillStyle = "#0b0e13"; ctx.strokeStyle = s.color; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(hx, y(frac(s, m[s.key])), s.key === hover.nearKey ? 5 : 3.5, 0, 7); ctx.fill(); ctx.stroke();
      }
      if (hover.nearKey) {
        const s = used.find((u) => u.key === hover.nearKey);
        if (s && m[s.key] != null) { ctx.strokeStyle = "#e6edf3"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(hx, y(frac(s, m[s.key])), 7.5, 0, 7); ctx.stroke(); }
      }
    }
  }, [metrics, used, width, hover, view]);

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
    for (const s of used) {
      if (metric[s.key] == null) continue;
      const d = Math.abs(y(frac(s, metric[s.key])) - py);
      if (d < best) { best = d; nearKey = s.key; }
    }
    setHover({ metric, x: x(metric.gen), y: py, nearKey });
  };

  // tooltip rows: every present series at this gen, ALPHABETICAL by label.
  const rows = hover?.metric
    ? used
        .filter((s) => hover.metric[s.key] != null)
        .map((s) => ({ key: s.key, label: s.label, color: s.color, val: fmt(hover.metric[s.key]) }))
        .sort((a, b) => a.label.localeCompare(b.label))
    : [];
  const ttLeft = hover ? Math.min(Math.max(hover.x + 10, 6), Math.max(6, width - 190)) : 0;

  return (
    <div className="chart-card" ref={wrapRef}>
      <div className="chart-title">{title}</div>
      <div className="chart-wrap">
        <canvas ref={canvasRef} width={width} height={H} style={{ width: "100%" }}
          onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
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
  return (
    <div className="charts-grid">
      {CHARTS.map((c) => (
        <Chart key={c.title} metrics={metrics} title={c.title} kind={c.kind} series={c.series} />
      ))}
    </div>
  );
}
