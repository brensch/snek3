// Shared board renderer used by every mini-board.

export const ROLE_COLORS = {
  net: "#3b82f6",
  baseline: "#22c55e",
};
export const COLORS = [ROLE_COLORS.net, ROLE_COLORS.baseline, "#f59e0b", "#ec4899", "#a78bfa", "#f87171", "#2dd4bf", "#facc15"];

export function snakeRole(opponent, index) {
  if (opponent === "net") return "net";
  // "agent-v-opp" labels (e.g. proxy-v-uct): snake 0 is our agent, snake 1 the opponent.
  const parts = (opponent || "").split("-v-");
  if (parts.length === 2) return index === 0 ? parts[0] : parts[1];
  return index === 0 ? "net" : "baseline";
}

export function snakeColor(opponent, index) {
  // Snake 0 (our agent) blue, snake 1 (opponent) green; fall back by index.
  return ROLE_COLORS[snakeRole(opponent, index)] || COLORS[index % COLORS.length];
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export function boardLayout(canvas, fr) {
  if (!canvas || !fr) return null;
  const W = fr.width, H = fr.height;
  const cell = Math.floor(Math.min(canvas.width, canvas.height) / Math.max(W, H));
  const ox = Math.floor((canvas.width - cell * W) / 2);
  const oy = Math.floor((canvas.height - cell * H) / 2);
  return {
    W, H, cell, ox, oy,
    px: (x) => ox + x * cell,
    py: (y) => oy + (H - 1 - y) * cell,
  };
}

export function eventToBoardCell(canvas, fr, event) {
  const layout = boardLayout(canvas, fr);
  if (!layout) return null;
  const rect = canvas.getBoundingClientRect();
  const cx = (event.clientX - rect.left) * (canvas.width / rect.width);
  const cy = (event.clientY - rect.top) * (canvas.height / rect.height);
  const x = Math.floor((cx - layout.ox) / layout.cell);
  const row = Math.floor((cy - layout.oy) / layout.cell);
  const y = layout.H - 1 - row;
  if (x < 0 || x >= layout.W || y < 0 || y >= layout.H) return null;
  return { x, y, canvasX: cx, canvasY: cy };
}

export function snakeAtCell(fr, x, y) {
  if (!fr?.snakes) return null;
  for (let si = fr.snakes.length - 1; si >= 0; si--) {
    const s = fr.snakes[si];
    if ((s.body || []).some(([bx, by]) => bx === x && by === y)) return si;
  }
  return null;
}

// Draw one snapshot frame onto a square canvas.
export function drawFrame(canvas, fr, opponent = "baseline", hoverSnake = null) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!fr) {
    ctx.fillStyle = "#8b949e";
    ctx.font = "11px ui-sans-serif";
    ctx.fillText("…", 8, 16);
    return;
  }
  const { W, H, cell, ox, oy, px, py } = boardLayout(canvas, fr);

  ctx.fillStyle = "#0b0e13";
  ctx.fillRect(ox, oy, W * cell, H * cell);
  ctx.strokeStyle = "#161b22";
  ctx.lineWidth = 1;
  for (let i = 0; i <= W; i++) { ctx.beginPath(); ctx.moveTo(px(0) + i * cell, oy); ctx.lineTo(px(0) + i * cell, oy + H * cell); ctx.stroke(); }
  for (let j = 0; j <= H; j++) { ctx.beginPath(); ctx.moveTo(ox, oy + j * cell); ctx.lineTo(ox + W * cell, oy + j * cell); ctx.stroke(); }

  (fr.food || []).forEach(([x, y]) => { ctx.fillStyle = "#ef4444"; ctx.beginPath(); ctx.arc(px(x) + cell / 2, py(y) + cell / 2, cell * 0.22, 0, 7); ctx.fill(); });
  (fr.hazards || []).forEach(([x, y]) => { ctx.fillStyle = "rgba(234,179,8,0.12)"; ctx.fillRect(px(x), py(y), cell, cell); });
  fr.snakes.forEach((s, si) => {
    const col = snakeColor(opponent, si);
    ctx.globalAlpha = s.alive ? 1 : 0.22;
    s.body.forEach(([x, y], bi) => {
      ctx.fillStyle = col;
      const pad = Math.max(1, cell * 0.12);
      roundRect(ctx, px(x) + pad, py(y) + pad, cell - 2 * pad, cell - 2 * pad, cell * 0.25);
      ctx.fill();
      if (si === hoverSnake) {
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = Math.max(1.5, cell * 0.08);
        ctx.stroke();
      }
      if (bi === 0 && cell >= 12) {
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        ctx.font = `bold ${Math.floor(cell * 0.55)}px ui-sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(si), px(x) + cell / 2, py(y) + cell / 2);
      }
    });
    ctx.globalAlpha = 1;
  });
}

export function resultOf(winner) {
  return winner === 0 ? ["W", "win"] : winner === 1 ? ["L", "loss"] : winner === -1 ? ["D", "draw"] : ["·", ""];
}
