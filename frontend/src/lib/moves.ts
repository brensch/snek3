// Move indices are stable across the whole stack: 0=Up 1=Down 2=Left 3=Right,
// matching the Rust engine's `Move` discriminants and the policy vector order.
export const MOVE_ARROW = ["↑", "↓", "←", "→"] as const;
