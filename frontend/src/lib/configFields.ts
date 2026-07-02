import type { RunConfig } from "../types";

export const configFields: Array<{ key: keyof RunConfig; label: string; hint?: string; kind?: "bool" }> = [
  { key: "board", label: "Board" },
  { key: "num_snakes", label: "Snakes" },
  // The only GPU dial. This is games per forward; the actual GPU batch is this
  // multiplied by the number of snakes (one tensor row per snake). The count of
  // concurrent self-play games is derived from it (double-buffered) in the backend.
  { key: "gpu_batch_games", label: "GPU batch size", hint: "games/forward · ×snakes rows" },
  { key: "samples_per_gen", label: "Samples" },
  { key: "sims", label: "Sims" },
  { key: "c_puct", label: "C PUCT" },
  { key: "exploration_prob", label: "Explore" },
  { key: "draw_value", label: "Draw value" },
  { key: "max_turns", label: "Max turns" },
  { key: "skip_short_draw_turns", label: "Skip draws" },
  { key: "trunk_channels", label: "Channels" },
  { key: "trunk_blocks", label: "Blocks" },
  { key: "gpool_every", label: "GPool every" },
  { key: "train_steps", label: "Train steps" },
  { key: "batch_size", label: "Batch" },
  { key: "recency", label: "Recency" },
  { key: "buffer_size", label: "Buffer" },
  { key: "value_weight", label: "Value weight" },
  { key: "search_threads", label: "Threads" },
  { key: "sample_games", label: "Sample games", hint: "recorded per gen" },
  // Concurrent CPU arena eval (current vs eval_turns-old checkpoint).
  { key: "eval_turns", label: "Eval every", hint: "gens · 0 disables" },
  { key: "eval_games", label: "Eval games", hint: "per eval point" },
  { key: "eval_sims", label: "Eval sims", hint: "per move, CPU" },
  { key: "eval_opponents", label: "Eval opponents", hint: "1x,2x,4x… turns back" },
  { key: "eval_cores", label: "Eval cores", hint: "per side" },
];
