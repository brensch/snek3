import React from "react";
import { MOVE_DELTA } from "./api.js";

// A snake-game board rendered as SVG. Battlesnake coords put (0,0) bottom-left
// with y increasing upward, so we flip y for screen space. `snakes` is a list of
// { body: [[x,y],...] head-first, alive, health }. `youIndex` is emphasized;
// `chosenMove` (0..3) draws an intent arrow from that snake's head.
const PALETTE = [
  "#3b82f6", // blue  — our snake
  "#ef4444", // red
  "#22c55e", // green
  "#eab308", // amber
  "#a855f7", // purple
  "#ec4899", // pink
  "#14b8a6", // teal
  "#f97316", // orange
];

export function snakeColor(i) {
  return PALETTE[i % PALETTE.length];
}

export default function Board({
  width,
  height,
  snakes = [],
  food = [],
  hazards = [],
  youIndex = 0,
  chosenMove = null,
  cell = 34,
  highlight = null, // [x,y] cell to ring (e.g. a head of interest)
}) {
  if (!width || !height) return null;
  const W = width * cell;
  const H = height * cell;
  const sx = (x) => x * cell;
  const sy = (y) => (height - 1 - y) * cell; // flip
  const cx = (x) => sx(x) + cell / 2;
  const cy = (y) => sy(y) + cell / 2;

  const cells = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      cells.push(
        <rect
          key={`c${x}-${y}`}
          x={sx(x)}
          y={sy(y)}
          width={cell}
          height={cell}
          className={(x + y) % 2 ? "cell cell-b" : "cell cell-a"}
        />
      );
    }
  }

  return (
    <svg
      className="board"
      viewBox={`0 0 ${W} ${H}`}
      width={W}
      height={H}
      shapeRendering="crispEdges"
    >
      {cells}
      {hazards.map(([x, y], i) => (
        <rect
          key={`h${i}`}
          x={sx(x)}
          y={sy(y)}
          width={cell}
          height={cell}
          className="hazard"
        />
      ))}
      {food.map(([x, y], i) => (
        <circle key={`f${i}`} cx={cx(x)} cy={cy(y)} r={cell * 0.22} className="food" />
      ))}
      {snakes.map((s, i) => {
        if (!s.body || !s.body.length) return null;
        const color = snakeColor(i);
        const isYou = i === youIndex;
        const pts = s.body.map(([x, y]) => `${cx(x)},${cy(y)}`).join(" ");
        const [hx, hy] = s.body[0];
        return (
          <g key={`s${i}`} className={s.alive === false ? "snake dead" : "snake"}>
            <polyline
              points={pts}
              fill="none"
              stroke={color}
              strokeWidth={cell * 0.62}
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity={isYou ? 0.95 : 0.78}
            />
            <circle
              cx={cx(hx)}
              cy={cy(hy)}
              r={cell * 0.3}
              fill={color}
              stroke={isYou ? "#fff" : "#0b0f17"}
              strokeWidth={isYou ? 2.5 : 1.5}
            />
            {isYou && chosenMove != null && (() => {
              const [dx, dy] = MOVE_DELTA[chosenMove];
              return (
                <line
                  x1={cx(hx)}
                  y1={cy(hy)}
                  x2={cx(hx) + dx * cell * 0.62}
                  y2={cy(hy) + dy * cell * 0.62}
                  stroke="#fff"
                  strokeWidth={3}
                  markerEnd="url(#arrow)"
                />
              );
            })()}
          </g>
        );
      })}
      {highlight && (
        <rect
          x={sx(highlight[0])}
          y={sy(highlight[1])}
          width={cell}
          height={cell}
          className="cell-highlight"
        />
      )}
      <defs>
        <marker
          id="arrow"
          viewBox="0 0 10 10"
          refX="7"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#fff" />
        </marker>
      </defs>
    </svg>
  );
}
