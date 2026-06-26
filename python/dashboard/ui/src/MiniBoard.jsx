import React, { useEffect, useRef } from "react";
import { drawFrame, resultOf, COLORS } from "./board.js";

// One game's replay; `tick` is a shared clock so every board on the page plays
// together (each loops over its own length).
export default function MiniBoard({ game, tick }) {
  const ref = useRef(null);
  const n = game.frames.length;
  const frame = n ? tick % n : 0;
  const fr = game.frames[frame];

  useEffect(() => {
    if (ref.current) drawFrame(ref.current, fr);
  }, [fr]);

  const [r, cls] = resultOf(game.winner);
  const type = game.opponent === "net" ? "self-play" : "vs baseline";

  return (
    <div className="board-cell">
      <canvas ref={ref} width={210} height={210} />
      <div className="board-meta">
        <span className="type">{type}</span>
        <span className={"badge " + cls}>{r}</span>
        <span className="muted turn">t{fr ? fr.turn : 0} · {frame + 1}/{n}</span>
      </div>
      <div className="board-snakes">
        {game.frames[0].snakes.map((_, i) => (
          <span key={i}>
            <i className="swatch" style={{ background: COLORS[i] }} />
            {i === 0 ? "net" : game.opponent === "net" ? "net" : "baseline"}
            {game.winner === i ? " ✓" : ""}
          </span>
        ))}
      </div>
    </div>
  );
}
