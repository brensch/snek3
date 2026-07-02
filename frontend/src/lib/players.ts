// League player ids are checkpoint generations, except the fixed flood-fill
// MCTS baseline (crates/snek-heuristic), which the trainer's league registers
// under u32::MAX so it can never collide with a real generation.
export const HEURISTIC_GEN = 0xffffffff;

export const isHeuristic = (gen: number) => gen === HEURISTIC_GEN;

/** Short label: "g42", or "floodfill" for the baseline. */
export const playerName = (gen: number) => (isHeuristic(gen) ? "floodfill" : `g${gen}`);

/** Long label: "gen_0042", or "floodfill" for the baseline. */
export const playerNameLong = (gen: number) =>
  isHeuristic(gen) ? "floodfill" : `gen_${String(gen).padStart(4, "0")}`;
