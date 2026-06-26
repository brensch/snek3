import React, { useEffect, useLayoutEffect, useRef, useState } from "react";

// Full-width responsive line chart: win-rate (0..1) + normalized policy/value
// losses, x = generation. Resizes to its container.
export default function MetricsChart({ metrics }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const [width, setWidth] = useState(900);

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
    const losses = metrics.flatMap((m) => [m.policy_loss, m.value_loss]).filter((v) => v != null);
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
    line(metrics.filter((m) => m.win_rate != null).map((m) => [x(m.gen), yU(m.win_rate)]), "#34d399", false);
    ctx.fillStyle = "#8b949e";
    ctx.fillText("gen " + gmin, padL, H - 7);
    ctx.fillText("gen " + gmax, W - padR - 44, H - 7);
  }, [metrics, width]);

  return (
    <div ref={wrapRef} style={{ width: "100%" }}>
      <canvas ref={canvasRef} width={width} height={260} style={{ width: "100%" }} />
      <div className="legend">
        <span><i className="swatch" style={{ background: "#34d399" }} />win-rate vs baseline (0–1)</span>
        <span><i className="swatch" style={{ background: "#f59e0b" }} />policy loss</span>
        <span><i className="swatch" style={{ background: "#60a5fa" }} />value loss</span>
      </div>
    </div>
  );
}
