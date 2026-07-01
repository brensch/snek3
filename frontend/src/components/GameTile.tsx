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

  return (
    <div className="rounded border border-slate-800 bg-slate-900 p-2">
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

      <div className="mt-1.5 flex items-center gap-1 text-[10px] text-slate-400">
        <button onClick={() => setPlaying((p) => !p)} className="w-4 text-slate-300">
          {playing ? "❚❚" : "▶"}
        </button>
        <button onClick={() => seek(idx - 1)} className="w-3">
          ‹
        </button>
        <input
          type="range"
          min={0}
          max={len - 1}
          value={idx}
          onChange={(e) => seek(Number(e.target.value))}
          className="h-1 flex-1 accent-sky-500"
        />
        <button onClick={() => seek(idx + 1)} className="w-3">
          ›
        </button>
        <span className="w-6 text-right font-mono">{frame.turn}</span>
      </div>

      <div className="mt-1 grid gap-0.5">
        {frame.snakes.map((s, i) => (
          <div
            key={i}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            className={`flex items-center gap-1 rounded px-0.5 text-[10px] ${hovered === i ? "bg-slate-800" : ""} ${s.alive ? "" : "opacity-40"}`}
          >
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: snakeColor(i) }} />
            <ValueBar v={s.value} />
            <span className={`w-9 shrink-0 text-right font-mono ${s.value >= 0 ? "text-green-400" : "text-red-400"}`}>
              {s.value >= 0 ? "+" : ""}
              {s.value.toFixed(2)}
            </span>
            <span className="w-6 shrink-0 text-right text-slate-400" title="length">
              {s.body.length}L
            </span>
            <span className="w-8 shrink-0 text-right text-slate-400" title="health">
              ♥{s.health}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// A value bar spanning -1..1 with the zero point centred: red fills to the left
// for negative, green to the right for positive.
function ValueBar({ v }: { v: number }) {
  const pct = Math.max(0, Math.min(100, (v + 1) * 50));
  return (
    <span className="relative h-2 flex-1 overflow-hidden rounded bg-slate-950">
      <span className="absolute inset-y-0 left-1/2 w-px bg-slate-600" />
      <span
        className="absolute inset-y-0"
        style={{
          left: `${Math.min(50, pct)}%`,
          width: `${Math.abs(pct - 50)}%`,
          background: v >= 0 ? "#22c55e" : "#ef4444",
        }}
      />
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
