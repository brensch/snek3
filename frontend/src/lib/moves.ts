// Move indices are stable across the whole stack: 0=Up 1=Down 2=Left 3=Right,
// matching the Rust engine's `Move` discriminants and the policy vector order.
export const MOVE_ARROW = ["↑", "↓", "←", "→"] as const;

const PALETTE = [
  "#3b82f6", // blue
  "#ef4444", // red
  "#22c55e", // green
  "#eab308", // amber
  "#a855f7", // purple
  "#ec4899", // pink
  "#14b8a6", // teal
  "#f97316", // orange
];

export function snakeColor(i: number): string {
  return PALETTE[i % PALETTE.length];
}
