import React, { useEffect, useRef, useState } from "react";
import { drawFrame, eventToBoardCell, resultOf, snakeColor, snakeRole, snakesAtCell } from "./board.js";

const MOVE_LABELS = ["Up", "Down", "Left", "Right"];

function fmtPct(value) {
  return value == null || !Number.isFinite(Number(value)) ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
}

function fmtValue(value) {
  return value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(3);
}

function SnakePolicyBlock({ snake, snakeIndex, opponent }) {
  const policy = Array.isArray(snake?.policy) ? snake.policy : null;
  const playPolicy = Array.isArray(snake?.play_policy) ? snake.play_policy : null;
  const maxP = policy || playPolicy
    ? Math.max(0.001, ...(policy || []).map((v) => Number(v) || 0), ...(playPolicy || []).map((v) => Number(v) || 0))
    : 1;

  return (
    <div className={"snake-tooltip-block " + (snake.alive ? "alive" : "dead")}>
      <div className="snake-tooltip-head">
        <span>
          <i className="swatch" style={{ background: snakeColor(opponent, snakeIndex) }} />
          {snakeRole(opponent, snakeIndex)} {snakeIndex}
          {!snake.alive && <em>dead</em>}
        </span>
        <b>
          h {snake.health ?? "—"} · {snake.chosen_move != null ? MOVE_LABELS[snake.chosen_move] : "—"} · v {fmtValue(snake.value)}
        </b>
      </div>
      {policy || playPolicy ? (
        <div className="policy-rows">
          {MOVE_LABELS.map((label, i) => {
            const p = Number(policy?.[i]) || 0;
            const pp = Number(playPolicy?.[i]) || 0;
            return (
              <div className="policy-row" key={label}>
                <em>{label}</em>
                <i><span style={{ width: `${Math.max(3, ((policy ? p : pp) / maxP) * 100)}%` }} /></i>
                <b>{fmtPct(p)}</b>
                <strong>{fmtPct(playPolicy ? pp : null)}</strong>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="policy-empty">no search target</div>
      )}
    </div>
  );
}

function PolicyTooltip({ frame, hover, opponent }) {
  if (!frame || !hover?.snakes?.length) return null;
  const snakeIndexes = hover.snakes.filter((si) => frame.snakes?.[si]);
  if (!snakeIndexes.length) return null;
  const left = Math.min(Math.max(hover.left + 12, 6), 112);
  const top = Math.min(Math.max(hover.top + 12, 6), 112);

  return (
    <div className="snake-tooltip" style={{ left, top }}>
      <div className="policy-headings"><span>move</span><span>target</span><span>played</span></div>
      {snakeIndexes.map((si) => (
        <SnakePolicyBlock key={si} snake={frame.snakes[si]} snakeIndex={si} opponent={opponent} />
      ))}
    </div>
  );
}

// One game's replay; `tick` is a shared clock so every board on the page plays
// together (each loops over its own length).
export default function MiniBoard({ game, tick, playing, onPlay, context = {} }) {
  const ref = useRef(null);
  const [autoplay, setAutoplay] = useState(true);
  const [manualFrame, setManualFrame] = useState(0);
  const [hover, setHover] = useState(null);
  const [copied, setCopied] = useState(false);
  const n = game.frames.length;
  const holdTicks = 10;
  const autoplayFrame = n ? Math.min(tick % (n + holdTicks), n - 1) : 0;
  const frame = autoplay ? autoplayFrame : Math.min(manualFrame, Math.max(0, n - 1));
  const fr = game.frames[frame];

  useEffect(() => {
    setAutoplay(true);
    setManualFrame(0);
    setHover(null);
  }, [game]);

  useEffect(() => {
    if (ref.current) drawFrame(ref.current, fr, game.opponent, hover?.snakes ?? []);
  }, [fr, game.opponent, hover]);

  const [r, cls] = resultOf(game.winner);
  const type = game.opponent === "net" ? "net self-play"
    : (game.opponent || "net vs baseline").replace(/-v-/g, " vs ");
  const isPlaying = autoplay && playing && n > 1;
  const stopAutoplay = () => {
    setManualFrame(autoplayFrame);
    setAutoplay(false);
  };
  const setFrame = (value) => {
    setAutoplay(false);
    setManualFrame(Number(value));
  };
  const toggleAutoplay = () => {
    if (autoplay && playing) {
      stopAutoplay();
    } else {
      setAutoplay(true);
      onPlay?.();
    }
  };
  const handlePointerMove = (e) => {
    const canvas = ref.current;
    if (!canvas || !fr) return;
    const cell = eventToBoardCell(canvas, fr, e);
    if (!cell) {
      setHover(null);
      return;
    }
    const snakes = snakesAtCell(fr, cell.x, cell.y);
    if (!snakes.length) {
      setHover(null);
      return;
    }
    if (autoplay) stopAutoplay();
    const rect = canvas.getBoundingClientRect();
    setHover({
      snakes,
      left: e.clientX - rect.left,
      top: e.clientY - rect.top,
    });
  };
  const clearHover = () => setHover(null);
  const copyContext = async () => {
    if (autoplay) stopAutoplay();
    const payload = {
      kind: "snek3-replay-frame",
      run: context.run ?? null,
      game_file: context.file ?? null,
      generation: context.gen ?? null,
      game_index: context.gameIndex ?? null,
      frame_index: frame,
      frame_number: frame + 1,
      total_frames: n,
      board_turn: fr?.turn ?? null,
      opponent: game.opponent ?? null,
      winner: game.winner,
      move_order: MOVE_LABELS,
      frame: fr,
      next_frame: game.frames?.[frame + 1] ?? null,
    };
    const snakeLines = (fr?.snakes || []).map((s, i) => {
      const pol = Array.isArray(s.policy)
        ? `target_policy: ${MOVE_LABELS.map((m, j) => `${m}=${fmtPct(s.policy[j])}`).join(", ")}`
        : "policy unavailable";
      const playPol = Array.isArray(s.play_policy)
        ? `play_policy: ${MOVE_LABELS.map((m, j) => `${m}=${fmtPct(s.play_policy[j])}`).join(", ")}`
        : "play_policy unavailable";
      const chosen = s.chosen_move == null ? "chosen=unavailable" : `chosen=${MOVE_LABELS[s.chosen_move]}`;
      const head = s.body?.[0] ? `[${s.body[0][0]},${s.body[0][1]}]` : "none";
      return `- snake ${i} (${snakeRole(game.opponent, i)}): alive=${!!s.alive}, head=${head}, health=${s.health}, value=${fmtValue(s.value)}, ${chosen}, ${pol}, ${playPol}`;
    }).join("\n");
    const text = [
      "SNEK3_REPLAY_CONTEXT",
      `run=${payload.run} generation=${payload.generation} game_file=${payload.game_file} game_index=${payload.game_index}`,
      `frame=${payload.frame_number}/${payload.total_frames} frame_index=${payload.frame_index} board_turn=${payload.board_turn} opponent=${payload.opponent} winner=${payload.winner}`,
      `move_order=${MOVE_LABELS.join(", ")}`,
      snakeLines,
      "JSON:",
      JSON.stringify(payload, null, 2),
    ].join("\n");
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div className="board-cell">
      <div className="board-canvas-wrap">
        <canvas
          ref={ref}
          width={210}
          height={210}
          onPointerMove={handlePointerMove}
          onPointerLeave={clearHover}
        />
        <PolicyTooltip frame={fr} hover={hover} opponent={game.opponent} />
      </div>
      <div className="replay-controls">
        <button
          className="replay-play"
          type="button"
          onClick={toggleAutoplay}
          title={isPlaying ? "Stop autoplay" : "Resume autoplay"}
          aria-label={isPlaying ? "Stop autoplay" : "Resume autoplay"}
        >
          {isPlaying ? "↻" : "▶"}
        </button>
        <input
          className="replay-scrub"
          type="range"
          min="0"
          max={Math.max(0, n - 1)}
          value={frame}
          disabled={n <= 1}
          onPointerDown={stopAutoplay}
          onChange={(e) => setFrame(e.target.value)}
          aria-label="Replay frame"
        />
      </div>
      <div className="board-meta">
        <span className="type">{type}</span>
        <span className={"badge " + cls}>{r}</span>
        <button
          className="copy-frame"
          type="button"
          onClick={copyContext}
          title="Copy replay context for this frame"
          aria-label="Copy replay context"
        >
          {copied ? "copied" : "copy"}
        </button>
        <span className="muted turn">{frame + 1}/{n}</span>
      </div>
    </div>
  );
}
