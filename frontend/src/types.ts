// Protobuf types (StatsFrame, RunState, Phase) come from the generated
// ../gen/snek_pb. RunConfig is the JSON control-plane payload, which has no proto
// schema (it is a large serde struct served as JSON), so it stays hand-typed.
import type { Phase } from "./gen/snek_pb";

export type RunConfig = {
  board: number;
  num_snakes: number;
  sims: number;
  c_puct: number;
  // Games per GPU forward. The forward tensor has this many rows × num_snakes
  // (one row per snake); concurrent self-play games are derived from it in the
  // backend (a double buffer), so this is the only GPU dial.
  gpu_batch_games: number;
  samples_per_gen: number;
  exploration_prob: number;
  max_turns: number;
  draw_value: number;
  skip_short_draw_turns: number;
  trunk_channels: number;
  trunk_blocks: number;
  train_steps: number;
  batch_size: number;
  recency: number;
  buffer_size: number;
  value_weight: number;
  search_threads: number;
  sample_games: number;
  // Continuous CPU evaluation league: a checkpoint joins the pool every this
  // many gens; game pairs run back-to-back while the run is active. 0 disables.
  league_entrant_gens: number;
  eval_sims: number;
  eval_cores: number;
};

// The JSON shape returned by GET /api/state. `phase` is the Phase enum value
// (a number over the wire); `run_id`/`device` are snake_case JSON, so this is
// kept distinct from the proto RunState (which the SSE/stats path uses).
export type RunState = {
  phase: Phase;
  generation: number;
  run_id: string;
  running: boolean;
  device?: string;
};
