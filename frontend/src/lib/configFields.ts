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
  { key: "sample_turns", label: "Sample turns", hint: "τ=1 opening, argmax after" },
  { key: "dirichlet_frac", label: "Root noise", hint: "frac at search root" },
  { key: "dirichlet_alpha", label: "Root α" },
  { key: "draw_value", label: "Draw value" },
  { key: "max_turns", label: "Max turns" },
  { key: "skip_short_draw_turns", label: "Skip draws" },
  { key: "trunk_channels", label: "Channels" },
  { key: "trunk_blocks", label: "Blocks" },
  { key: "train_steps", label: "Train steps" },
  { key: "batch_size", label: "Batch" },
  { key: "recency", label: "Recency", hint: "1 = uniform sampling" },
  { key: "buffer_size", label: "Buffer" },
  { key: "value_weight", label: "Value weight" },
  { key: "entropy_floor", label: "Entropy floor", hint: "nats · coef 0 = off" },
  { key: "entropy_coef", label: "Entropy coef" },
  { key: "weight_decay", label: "Weight decay", hint: "AdamW · AZ L2 = 1e-4" },
  { key: "lr_half_life_samples", label: "LR half-life", hint: "samples per LR halving" },
  { key: "search_threads", label: "Threads" },
  { key: "sample_games", label: "Sample games", hint: "recorded per gen" },
  // Continuous CPU evaluation league (checkpoint-vs-checkpoint games running
  // back-to-back in concurrent slots while the run is active).
  { key: "league_entrant_gens", label: "League entrant", hint: "every N gens · 0 off" },
  { key: "eval_sims", label: "League sims", hint: "per move, CPU" },
  { key: "league_games", label: "League games", hint: "concurrent arena games" },
];
