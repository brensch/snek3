// League player ids are checkpoint generations, except external players
// (built-in heuristic agents and configured API snakes), which the trainer's
// league registers at ids counting down from u32::MAX so they can never
// collide with a real generation.
export const HEURISTIC_GEN = 0xffffffff;
export const VORONOI_GEN = 0xfffffffe;

/** Everything at or above this id is an external player, not a checkpoint. */
export const EXTERNAL_GEN_MIN = 0xffffff00;

export const isExternal = (gen: number) => gen >= EXTERNAL_GEN_MIN;

/** @deprecated floodfill only — prefer isExternal for "not a checkpoint". */
export const isHeuristic = (gen: number) => gen === HEURISTIC_GEN;

// External players' display names come from the backend (LeagueRating.name /
// live players). Components that only have an id (match placements) resolve
// through this registry, populated whenever league data loads.
const names = new Map<number, string>([
  [HEURISTIC_GEN, "floodfill"],
  [VORONOI_GEN, "voronoi"],
]);

/** Feed known names (e.g. RunDetail.league) into the id → name registry. */
export const registerPlayerNames = (players: { gen: number; name?: string }[]) => {
  for (const p of players) {
    if (p.name && isExternal(p.gen)) names.set(p.gen, p.name);
  }
};

/** Short label: "g42", or the external player's name. */
export const playerName = (gen: number) =>
  isExternal(gen) ? (names.get(gen) ?? `api_${HEURISTIC_GEN - gen}`) : `g${gen}`;

/** Long label: "gen_0042", or the external player's name. */
export const playerNameLong = (gen: number) =>
  isExternal(gen)
    ? (names.get(gen) ?? `api_${HEURISTIC_GEN - gen}`)
    : `gen_${String(gen).padStart(4, "0")}`;
