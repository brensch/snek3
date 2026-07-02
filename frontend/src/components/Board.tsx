import { snakeColor } from "../lib/moves";

export type Coord = { x: number; y: number };

export type BoardSnake = {
  body: Coord[]; // head first
  alive: boolean;
};

type Props = {
  width: number;
  height: number;
  snakes: BoardSnake[];
  food?: Coord[];
  hazards?: Coord[];
  cell?: number;
  highlight?: number | null; // snake index to emphasize
  onHoverSnake?: (index: number | null) => void;
};

// Battlesnake coords put (0,0) bottom-left with y increasing upward, so we flip
// y for screen space. Rendered as a single SVG; snakes are polylines with a head
// dot.
export function Board({
  width,
  height,
  snakes,
  food = [],
  hazards = [],
  cell = 30,
  highlight = null,
  onHoverSnake,
}: Props) {
  if (!width || !height) return null;
  const W = width * cell;
  const H = height * cell;
  const sx = (x: number) => x * cell;
  const sy = (y: number) => (height - 1 - y) * cell;
  const cx = (x: number) => sx(x) + cell / 2;
  const cy = (y: number) => sy(y) + cell / 2;

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
          fill={(x + y) % 2 ? "#0b1220" : "#0e1729"}
        />,
      );
    }
  }

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      className="max-w-[560px] rounded border border-slate-800 bg-slate-950"
      shapeRendering="crispEdges"
    >
      {cells}
      {hazards.map((p, i) => (
        <rect key={`h${i}`} x={sx(p.x)} y={sy(p.y)} width={cell} height={cell} fill="#7c2d1240" />
      ))}
      {food.map((p, i) => (
        <circle key={`f${i}`} cx={cx(p.x)} cy={cy(p.y)} r={cell * 0.22} fill="#f43f5e" />
      ))}
      {snakes.map((s, i) => {
        if (!s.body?.length) return null;
        const color = snakeColor(i);
        const emphasized = highlight === i;
        const dimmed = highlight != null && highlight !== i;
        const pts = s.body.map((p) => `${cx(p.x)},${cy(p.y)}`).join(" ");
        const head = s.body[0];
        return (
          <g
            key={`s${i}`}
            opacity={s.alive ? (dimmed ? 0.3 : 1) : 0.22}
            onMouseEnter={() => onHoverSnake?.(i)}
            onMouseLeave={() => onHoverSnake?.(null)}
            style={{ cursor: onHoverSnake ? "pointer" : "default" }}
          >
            <polyline
              points={pts}
              fill="none"
              stroke={color}
              strokeWidth={cell * (emphasized ? 0.7 : 0.58)}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            <circle
              cx={cx(head.x)}
              cy={cy(head.y)}
              r={cell * 0.3}
              fill={color}
              stroke={emphasized ? "#fff" : "#0b0f17"}
              strokeWidth={emphasized ? 2.5 : 1.5}
            />
          </g>
        );
      })}
    </svg>
  );
}
