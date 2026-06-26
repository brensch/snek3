import React, { useEffect, useLayoutEffect, useRef, useState } from "react";

// Full-width responsive line chart: win-rate (0..1) + normalized policy/value
// losses, x = generation. Resizes to its container.
export default function MetricsChart({ metrics }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const [width, setWidth] = useState(900);
  const [hover, setHover] = useState(null);

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => setWidth(Math.max(320, Math.floor(entries[0].contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

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
    const gens = metrics.map((m) => m.gen);
    const gmin = Math.min(...gens), gmax = Math.max(...gens, gmin + 1);
    const x = (g) => padL + (W - padL - padR) * (g - gmin) / ((gmax - gmin) || 1);
    ctx.strokeStyle = "#21262d";
    ctx.fillStyle = "#8b949e";
    ctx.lineWidth = 1;
    ctx.font = "11px ui-sans-serif";
    for (let i = 0; i <= 4; i++) {
      const yy = padT + (H - padT - padB) * i / 4;
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W - padR, yy); ctx.stroke();
      ctx.fillText((1 - i / 4).toFixed(2), 6, yy + 3);
    }
    const losses = metrics.flatMap((m) => [m.policy_loss, m.value_loss, m.target_entropy]).filter((v) => v != null);
    const lmax = Math.max(0.001, ...losses);
    const yU = (v) => padT + (H - padT - padB) * (1 - v);
    const yL = (v) => padT + (H - padT - padB) * (1 - v / lmax);
    const line = (pts, color, dash) => {
      if (!pts.length) return;
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash ? [4, 4] : []);
      ctx.beginPath();
      pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = color;
      pts.forEach((p) => { ctx.beginPath(); ctx.arc(p[0], p[1], 2.4, 0, 7); ctx.fill(); });
    };
    line(metrics.filter((m) => m.policy_loss != null).map((m) => [x(m.gen), yL(m.policy_loss)]), "#f59e0b", true);
    line(metrics.filter((m) => m.value_loss != null).map((m) => [x(m.gen), yL(m.value_loss)]), "#60a5fa", true);
    line(metrics.filter((m) => m.target_entropy != null).map((m) => [x(m.gen), yL(m.target_entropy)]), "#ec4899", false);
    line(metrics.filter((m) => m.win_rate != null).map((m) => [x(m.gen), yU(m.win_rate)]), "#34d399", false);
    if (hover?.metric) {
      const m = hover.metric;
      const hx = x(m.gen);
      ctx.strokeStyle = "rgba(230, 237, 243, 0.42)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(hx, padT); ctx.lineTo(hx, H - padB); ctx.stroke();
      ctx.setLineDash([]);

      const highlight = (value, yFn, color) => {
        if (value == null) return;
        ctx.fillStyle = "#0b0e13";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(hx, yFn(value), 5, 0, 7); ctx.fill(); ctx.stroke();
      };
      highlight(m.policy_loss, yL, "#f59e0b");
      highlight(m.value_loss, yL, "#60a5fa");
      highlight(m.target_entropy, yL, "#ec4899");
      highlight(m.win_rate, yU, "#34d399");
    }
    ctx.fillStyle = "#8b949e";
    ctx.fillText("gen " + gmin, padL, H - 7);
    ctx.fillText("gen " + gmax, W - padR - 44, H - 7);
  }, [metrics, width, hover]);

  const updateHover = (event) => {
    const c = canvasRef.current;
    if (!c || !metrics.length) return;
    const rect = c.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (c.width / rect.width);
    const py = (event.clientY - rect.top) * (c.height / rect.height);
    const W = c.width, padL = 42, padR = 14;
    const gens = metrics.map((m) => m.gen);
    const gmin = Math.min(...gens), gmax = Math.max(...gens, gmin + 1);
    const x = (g) => padL + (W - padL - padR) * (g - gmin) / ((gmax - gmin) || 1);
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

  const tooltipRows = hover?.metric ? [
    ["win rate", fmt(hover.metric.win_rate)],
    ["policy loss", fmt(hover.metric.policy_loss)],
    ["value loss", fmt(hover.metric.value_loss)],
    ["target entropy", fmt(hover.metric.target_entropy)],
    ["target max prob", fmt(hover.metric.target_max_prob)],
    ["samples", hover.metric.samples != null ? Number(hover.metric.samples).toLocaleString() : null],
    ["buffer", hover.metric.buffer_size != null ? Number(hover.metric.buffer_size).toLocaleString() : null],
  ].filter(([, value]) => value != null) : [];

  const tooltipLeft = hover ? Math.min(Math.max(hover.x + 12, 8), Math.max(8, width - 190)) : 0;
  const tooltipTop = hover ? (hover.y > 145 ? Math.max(8, hover.y - 124) : Math.min(210, hover.y + 12)) : 0;

  return (
    <div ref={wrapRef} className="chart-wrap">
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
      <div className="legend">
        <span><i className="swatch" style={{ background: "#34d399" }} />win-rate vs baseline (0–1)</span>
        <span><i className="swatch" style={{ background: "#f59e0b" }} />policy loss</span>
        <span><i className="swatch" style={{ background: "#60a5fa" }} />value loss</span>
        <span><i className="swatch" style={{ background: "#ec4899" }} />target entropy</span>
      </div>
    </div>
  );
}
