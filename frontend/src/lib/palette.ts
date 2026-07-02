// Design tokens — the single source of color for the dashboard.
//
// The categorical slots are a validated dark-surface palette (lightness band,
// chroma floor, ≥3:1 contrast on #1a1a19; adjacent-pair CVD ΔE 10.3, which is
// the floor band — legal because identity never rides on color alone here:
// series carry named legends and snakes carry seat legends). Slot order is the
// CVD-safety mechanism: assign in order, never cycle new hues.
export const CATEGORICAL = [
  "#3987e5", // 1 blue
  "#199e70", // 2 aqua
  "#c98500", // 3 yellow
  "#008300", // 4 green
  "#9085e9", // 5 violet
  "#e66767", // 6 red
  "#d55181", // 7 magenta
  "#d95926", // 8 orange
] as const;

// Named series slots, so a metric keeps its hue everywhere it appears.
export const series = {
  blue: CATEGORICAL[0],
  aqua: CATEGORICAL[1],
  yellow: CATEGORICAL[2],
  green: CATEGORICAL[3],
  violet: CATEGORICAL[4],
  red: CATEGORICAL[5],
  magenta: CATEGORICAL[6],
  orange: CATEGORICAL[7],
} as const;

// Status colors are reserved for good/bad meaning, never for a series.
export const status = {
  good: "#0ca30c",
  warning: "#fab219",
  serious: "#ec835a",
  critical: "#d03b3b",
} as const;

// Chart chrome (dark mode; the app is dark-only).
export const chrome = {
  page: "#0d0d0d",
  surface: "#1a1a19",
  inset: "#141413",
  grid: "#2c2c2a",
  axis: "#383835",
  ink: "#ffffff",
  ink2: "#c3c2b7",
  ink3: "#898781",
} as const;

// Snake identity on boards: the categorical order, fixed by seat index.
export function snakeColor(seat: number): string {
  return CATEGORICAL[seat % CATEGORICAL.length];
}
