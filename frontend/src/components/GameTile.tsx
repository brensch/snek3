import { useEffect, useState } from "react";
import type { Game, SnakeFrame } from "../gen/viewer_pb";
import { MOVE_ARROW } from "../lib/moves";
import { snakeColor, status } from "../lib/palette";
import { Board } from "./Board";

// Ticks to hold on the final outcome frame before the loop restarts.
const HOLD = 10;

// One recorded game as a self-contained tile: a board, its own play/pause and
// turn scrubber, a copy-as-JSON button, the value/length/health of each snake,
// and the per-direction policy on hover. `intervalMs`/`cell` come from the
// grid-wide speed/size sliders; `colors` overrides seat colors with stable
// per-player colors (league games).
export function GameTile({
  game,
  intervalMs,
  cell,
  colors,
  onFrame,
}: {
  game: Game;
  intervalMs: number;
  cell: number;
  colors?: string[];
  /** Reports the currently displayed frame index (fullscreen view renders a
      synced odds panel from it). */
  onFrame?: (idx: number) => void;
}) {
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

  const idx = Math.min(phase, Math.max(0, len - 1));
  useEffect(() => {
    onFrame?.(idx);
  }, [idx, onFrame]);

  if (len === 0) return null;
  const frame = frames[idx];

  // Which seats a built-in heuristic (voronoi) plays this game, from the
  // per-game bitmask. Empty for ordinary net-vs-net self-play and league games.
  const isHeuristic = (i: number): boolean => ((game.heurMask >> i) & 1) === 1;
  const heuristicSeats = frame.snakes.map((_, i) => isHeuristic(i));
  const voronoiSeats = heuristicSeats.flatMap((h, i) => (h ? [i] : []));
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
        heuristic: isHeuristic(i) ? "voronoi" : false,
        search_policy: Object.fromEntries(MOVES.map((m, j) => [m, Number((s.policy[j] ?? 0).toFixed(4))])),
        value: Number(s.value.toFixed(4)),
      })),
      game: { winner: game.winner, num_turns: game.numTurns, heur_mask: game.heurMask },
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
    <div className="card overflow-hidden p-2">
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
          colors={colors}
          heuristic={heuristicSeats}
        />
        {hovered != null && frame.snakes[hovered] && (
          <PolicyPopover snake={frame.snakes[hovered]} index={hovered} color={colors?.[hovered]} />
        )}
      </div>

      {/* The timeline gets a full line of its own, directly under the board it
          scrubs — the whole tile width is grabbable (mobile-friendly). */}
      <input
        type="range"
        min={0}
        max={len - 1}
        value={idx}
        onChange={(e) => seek(Number(e.target.value))}
        className="mt-1 h-6 w-full min-w-0 touch-manipulation accent-accent"
        aria-label="turn"
      />

      {/* Transport row: play + steppers left, where-am-I centre, copy right.
          All buttons are 28px tap targets with crisp SVG icons. */}
      <div className="flex min-w-0 items-center gap-0.5 text-[10px] text-ink-3">
        <IconButton onClick={() => setPlaying((p) => !p)} label={playing ? "pause" : "play"}>
          {playing ? <PauseIcon /> : <PlayIcon />}
        </IconButton>
        <IconButton onClick={() => seek(idx - 1)} label="previous turn">
          <ChevronIcon dir={-1} />
        </IconButton>
        <IconButton onClick={() => seek(idx + 1)} label="next turn">
          <ChevronIcon dir={1} />
        </IconButton>
        <span className="min-w-0 flex-1 truncate text-center font-mono tabular-nums">
          {frame.turn}
          <span className="text-ink-3/60"> / {game.numTurns}</span>
        </span>
        <IconButton onClick={copyFrame} label="copy this turn as JSON" className={copied ? "text-good" : undefined}>
          {copied ? <CheckIcon /> : <CopyIcon />}
        </IconButton>
      </div>

      {voronoiSeats.length > 0 && (
        <div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-ink-3">
          <span className="inline-block h-2 w-2 rounded-full border border-dashed border-white/80" />
          <span>
            voronoi (heuristic):{" "}
            {voronoiSeats.map((i, k) => (
              <span key={i}>
                {k > 0 && ", "}
                <span
                  className="font-medium"
                  style={{ color: colors?.[i] ?? snakeColor(i) }}
                >
                  snake {i}
                </span>
              </span>
            ))}
          </span>
        </div>
      )}

      <div className="mt-1 grid gap-0.5">
        <div className={`${statCols} px-0.5 text-[9px] uppercase text-ink-3/70`}>
          <span />
          <span className="pl-0.5 normal-case">value</span>
          <span className="text-right" title="length">
            L
          </span>
          <span className="text-right" title="health">
            ♥
          </span>
        </div>
        {frame.snakes.map((s, i) => (
          <div
            key={i}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            className={`${statCols} rounded px-0.5 text-[10px] ${hovered === i ? "bg-inset" : ""} ${s.alive ? "" : "opacity-40"}`}
          >
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{
                background: colors?.[i] ?? snakeColor(i),
                boxShadow: isHeuristic(i) ? "0 0 0 1.5px #fff" : undefined,
              }}
              title={isHeuristic(i) ? "voronoi (heuristic sparring partner)" : undefined}
            />
            <ValueBar v={s.value} showValue={hovered === i} />
            <span className="text-right font-mono tabular-nums text-ink-3" title={`length ${s.body.length}`}>
              {s.body.length}
            </span>
            <span className="text-right font-mono tabular-nums text-ink-3" title={`health ${s.health}`}>
              {s.health}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// 28px tap-target button wrapping a small SVG icon — the one style for every
// transport control so the row reads as a unit.
function IconButton({
  onClick,
  label,
  className = "",
  children,
}: {
  onClick: () => void;
  label: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      className={`flex h-7 w-7 shrink-0 touch-manipulation select-none items-center justify-center rounded text-ink-2 hover:bg-white/5 hover:text-ink active:bg-white/10 ${className}`}
    >
      {children}
    </button>
  );
}

const ICON = "h-3.5 w-3.5";

function PlayIcon() {
  return (
    <svg viewBox="0 0 16 16" className={ICON} fill="currentColor" aria-hidden>
      <path d="M4.5 2.8a.7.7 0 0 1 1.06-.6l8 5.2a.7.7 0 0 1 0 1.2l-8 5.2a.7.7 0 0 1-1.06-.6V2.8z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg viewBox="0 0 16 16" className={ICON} fill="currentColor" aria-hidden>
      <rect x="3.5" y="2.5" width="3.2" height="11" rx="1" />
      <rect x="9.3" y="2.5" width="3.2" height="11" rx="1" />
    </svg>
  );
}

function ChevronIcon({ dir }: { dir: -1 | 1 }) {
  return (
    <svg
      viewBox="0 0 16 16"
      className={ICON}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      style={dir === 1 ? { transform: "scaleX(-1)" } : undefined}
    >
      <path d="M10 3.5 5.5 8l4.5 4.5" />
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      className={ICON}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="5.5" y="5.5" width="8" height="8" rx="1.5" />
      <path d="M10.5 3.5v-.5A1.5 1.5 0 0 0 9 1.5H4A1.5 1.5 0 0 0 2.5 3v5A1.5 1.5 0 0 0 4 9.5h.5" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      className={ICON}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="m3 8.5 3.2 3.2L13 4.5" />
    </svg>
  );
}

// A value bar spanning -1..1 with the zero point centred: red fills to the left
// for negative, green to the right for positive. Hovering the snake's row
// (`showValue`) overlays the numeric value on the bar.
function ValueBar({ v, showValue }: { v: number; showValue?: boolean }) {
  const pct = Math.max(0, Math.min(100, (v + 1) * 50));
  const label = `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
  return (
    <span className="relative block h-2 w-full min-w-0 overflow-hidden rounded bg-page" title={`value ${label}`}>
      <span className="absolute inset-y-0 left-1/2 w-px bg-axis" />
      <span
        className="absolute inset-y-0"
        style={{
          left: `${Math.min(50, pct)}%`,
          width: `${Math.abs(pct - 50)}%`,
          background: v >= 0 ? status.good : status.critical,
        }}
      />
      {showValue && (
        <span
          className="absolute inset-0 flex items-center justify-center font-mono text-[8px] leading-none text-ink"
          style={{ textShadow: "0 0 3px rgba(0,0,0,0.95), 0 0 3px rgba(0,0,0,0.95)" }}
        >
          {label}
        </span>
      )}
    </span>
  );
}

// The four-direction search policy for one snake, shown only on hover.
// `pointer-events-none` so it never steals the hover from the element beneath.
function PolicyPopover({ snake, index, color }: { snake: SnakeFrame; index: number; color?: string }) {
  return (
    <div className="pointer-events-none absolute left-1 top-1 z-10 rounded-md border border-white/10 bg-page/95 p-1.5 shadow-lg">
      <div className="mb-1 flex items-center gap-1 text-[10px] text-ink-2">
        <span className="h-2 w-2 rounded-full" style={{ background: color ?? snakeColor(index) }} />
        snake {index}
      </div>
      {[0, 1, 2, 3].map((m) => {
        const p = snake.policy[m] ?? 0;
        const played = snake.chosenMove === m;
        return (
          <div key={m} className="flex items-center gap-1 text-[10px]">
            <span className={`w-3 ${played ? "text-accent" : "text-ink-3"}`}>{MOVE_ARROW[m]}</span>
            <span className="relative h-1.5 w-14 overflow-hidden rounded bg-inset">
              <span
                className={`absolute inset-y-0 left-0 rounded ${played ? "bg-accent" : "bg-axis"}`}
                style={{ width: `${Math.round(Math.max(0, Math.min(1, p)) * 100)}%` }}
              />
            </span>
            <span className="w-7 text-right font-mono tabular-nums text-ink-2">{(p * 100).toFixed(0)}%</span>
          </div>
        );
      })}
    </div>
  );
}
