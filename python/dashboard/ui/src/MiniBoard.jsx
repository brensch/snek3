import React, { useEffect, useRef, useState } from "react";
import { drawFrame, resultOf, snakeColor, snakeRole } from "./board.js";

// One game's replay; `tick` is a shared clock so every board on the page plays
// together (each loops over its own length).
export default function MiniBoard({ game, tick, playing, onPlay }) {
  const ref = useRef(null);
  const [autoplay, setAutoplay] = useState(true);
  const [manualFrame, setManualFrame] = useState(0);
  const n = game.frames.length;
  const holdTicks = 10;
  const autoplayFrame = n ? Math.min(tick % (n + holdTicks), n - 1) : 0;
  const frame = autoplay ? autoplayFrame : Math.min(manualFrame, Math.max(0, n - 1));
  const fr = game.frames[frame];

  useEffect(() => {
    setAutoplay(true);
    setManualFrame(0);
  }, [game]);

  useEffect(() => {
    if (ref.current) drawFrame(ref.current, fr, game.opponent);
  }, [fr, game.opponent]);

  const [r, cls] = resultOf(game.winner);
  const type = game.opponent === "net" ? "net self-play" : "net vs baseline";
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

  return (
    <div className="board-cell">
      <canvas ref={ref} width={210} height={210} />
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
        <span className="muted turn">{frame + 1}/{n}</span>
      </div>
      <div className="board-snakes">
        {game.frames[0].snakes.map((_, i) => (
          <span key={i}>
            <i className="swatch" style={{ background: snakeColor(game.opponent, i) }} />
            {snakeRole(game.opponent, i)}
            {game.winner === i ? " ✓" : ""}
          </span>
        ))}
      </div>
    </div>
  );
}
