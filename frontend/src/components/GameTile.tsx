import { useEffect, useState } from "react";
import type { Game, SnakeFrame } from "../gen/viewer_pb";
import { MOVE_ARROW, snakeColor } from "../lib/moves";
import { Board } from "./Board";

// Ticks to hold on the final outcome frame before the loop restarts.
const HOLD = 10;

// One recorded game as a self-contained tile: a board, its own play/pause and
// turn scrubber, the value/length/health of each snake, and the per-direction
// policy on hover. `intervalMs`/`cell` come from the grid-wide speed/size sliders.
export function GameTile({ game, intervalMs, cell }: { game: Game; intervalMs: number; cell: number }) {
  const frames = game.frames;
  const len = frames.length;
  const cycle = len + HOLD;
  const [phase, setPhase] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [hovered, setHovered] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!playing || len === 0) return;
    const timer = window.setInterval(() => setPhase((p) => (p + 1) % cycle), intervalMs);
    return () => window.clearInterval(timer);
  }, [playing, intervalMs, cycle, len]);

  if (len === 0) return null;
  const idx = Math.min(phase, len - 1);
  const frame = frames[idx];
  const seek = (i: number) => {
    setPlaying(false);
    setPhase(Math.max(0, Math.min(len - 1, i)));
  };

  // Copy the visible turn as self-describing JSON — everything needed to make
  // sense of the position (bodies, health, the search policy/value, the moves
  // played) without cross-referencing the proto schema.
  const copyFrame = async () => {
    const pt = (p: { x: number; y: number }) => [p.x, p.y];
    const MOVES = ["up", "down", "left", "right"];
    const doc = {
      coords: "x right, y up, origin bottom-left; bodies head first",
      turn: frame.turn,
      frame_index: idx,
      board: {
        width: frame.width,
        height: frame.height,
        food: frame.food.map(pt),
        hazards: frame.hazards.map(pt),
      },
      snakes: frame.snakes.map((s, i) => ({
        index: i,
        alive: s.alive,
        health: s.health,
        length: s.body.length,
        body: s.body.map(pt),
        chosen_move: MOVES[s.chosenMove] ?? s.chosenMove,
        search_policy: Object.fromEntries(
          MOVES.map((m, j) => [m, Number((s.policy[j] ?? 0).toFixed(4))]),
        ),
        value: Number(s.value.toFixed(4)),
      })),
      game: { winner: game.winner, num_turns: game.numTurns },
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(doc, null, 2));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard unavailable (e.g. non-secure context) */
    }
  };

  // Shared column template so the header labels line up with each snake row:
  // colour dot, value bar (-1..1), length, health.
  const statCols = "grid grid-cols-[0.5rem_minmax(0,1fr)_0.9rem_1.4rem] items-center gap-1";

  return (
    <div className="overflow-hidden rounded border border-slate-800 bg-slate-900 p-2">
      <div className="relative">
        <Board
          width={frame.width}
          height={frame.height}
          snakes={frame.snakes}
          food={frame.food}
          hazards={frame.hazards}
          cell={cell}
          highlight={hovered}
          onHoverSnake={setHovered}
        />
        {hovered != null && frame.snakes[hovered] && <PolicyPopover snake={frame.snakes[hovered]} index={hovered} />}
      </div>

      <div className="mt-1.5 flex min-w-0 items-center gap-1 text-[10px] text-slate-400">
        <button onClick={() => setPlaying((p) => !p)} className="w-4 shrink-0 text-slate-300">
          {playing ? "❚❚" : "▶"}
        </button>
        <button
          onClick={copyFrame}
          title="Copy this turn as JSON"
          className={`w-4 shrink-0 ${copied ? "text-emerald-400" : "text-slate-300"}`}
        >
          {copied ? "✓" : "⧉"}
        </button>
        <button onClick={() => seek(idx - 1)} className="w-3 shrink-0">
          ‹
        </button>
        <input
          type="range"
          min={0}
          max={len - 1}
          value={idx}
          onChange={(e) => seek(Number(e.target.value))}
          className="h-1 w-0 min-w-0 flex-1 accent-sky-500"
        />
        <button onClick={() => seek(idx + 1)} className="w-3 shrink-0">
          ›
        </button>
        <span className="w-6 shrink-0 text-right font-mono">{frame.turn}</span>
      </div>

      <div className="mt-1 grid gap-0.5">
        <div className={`${statCols} px-0.5 text-[9px] uppercase text-slate-600`}>
          <span />
          <span className="pl-0.5 normal-case">value</span>
          <span className="text-right" title="length">L</span>
          <span className="text-right" title="health">♥</span>
        </div>
        {frame.snakes.map((s, i) => (
          <div
            key={i}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            className={`${statCols} rounded px-0.5 text-[10px] ${hovered === i ? "bg-slate-800" : ""} ${s.alive ? "" : "opacity-40"}`}
          >
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: snakeColor(i) }} />
            <ValueBar v={s.value} showValue={hovered === i} />
            <span className="text-right font-mono text-slate-400" title={`length ${s.body.length}`}>
              {s.body.length}
            </span>
            <span className="text-right font-mono text-slate-400" title={`health ${s.health}`}>
              {s.health}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// A value bar spanning -1..1 with the zero point centred: red fills to the left
// for negative, green to the right for positive. Hovering the snake's row
// (`showValue`) overlays the numeric value on the bar.
function ValueBar({ v, showValue }: { v: number; showValue?: boolean }) {
  const pct = Math.max(0, Math.min(100, (v + 1) * 50));
  const label = `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
  return (
    <span
      className="relative block h-2 w-full min-w-0 overflow-hidden rounded bg-slate-950"
      title={`value ${label}`}
    >
      <span className="absolute inset-y-0 left-1/2 w-px bg-slate-600" />
      <span
        className="absolute inset-y-0"
        style={{
          left: `${Math.min(50, pct)}%`,
          width: `${Math.abs(pct - 50)}%`,
          background: v >= 0 ? "#22c55e" : "#ef4444",
        }}
      />
      {showValue && (
        <span
          className="absolute inset-0 flex items-center justify-center font-mono text-[8px] leading-none text-slate-100"
          style={{ textShadow: "0 0 3px rgba(0,0,0,0.95), 0 0 3px rgba(0,0,0,0.95)" }}
        >
          {label}
        </span>
      )}
    </span>
  );
}

// The four-direction search policy for one snake, shown only on hover.
// `pointer-events-none` so it never steals the hover from the element beneath
// (which otherwise caused the readout to flicker on/off).
function PolicyPopover({ snake, index }: { snake: SnakeFrame; index: number }) {
  return (
    <div className="pointer-events-none absolute left-1 top-1 z-10 rounded border border-slate-700 bg-slate-950/95 p-1.5 shadow-lg">
      <div className="mb-1 flex items-center gap-1 text-[10px] text-slate-300">
        <span className="h-2 w-2 rounded-full" style={{ background: snakeColor(index) }} />
        snake {index}
      </div>
      {[0, 1, 2, 3].map((m) => {
        const p = snake.policy[m] ?? 0;
        const played = snake.chosenMove === m;
        return (
          <div key={m} className="flex items-center gap-1 text-[10px]">
            <span className={`w-3 ${played ? "text-sky-300" : "text-slate-400"}`}>{MOVE_ARROW[m]}</span>
            <span className="relative h-1.5 w-14 overflow-hidden rounded bg-slate-800">
              <span
                className={`absolute inset-y-0 left-0 ${played ? "bg-sky-500" : "bg-slate-500"}`}
                style={{ width: `${Math.round(Math.max(0, Math.min(1, p)) * 100)}%` }}
              />
            </span>
            <span className="w-7 text-right font-mono text-slate-300">{(p * 100).toFixed(0)}%</span>
          </div>
        );
      })}
    </div>
  );
}
