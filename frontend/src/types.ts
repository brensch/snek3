export type Phase = "idle" | "playing" | "training" | "checkpoint" | "stopping" | "stopped";

export type RunConfig = {
  board: number;
  num_snakes: number;
  count: number;
  sims: number;
  c_puct: number;
  gpu_batch_games: number;
  samples_per_gen: number;
  exploration_prob: number;
  max_turns: number;
  draw_value: number;
  skip_short_draw_turns: number;
  bootstrap_value: boolean;
  trunk_channels: number;
  trunk_blocks: number;
  gpool_every: number;
  train_steps: number;
  batch_size: number;
  lr: number;
  recency: number;
  buffer_size: number;
  value_weight: number;
  search_threads: number;
  eval_every: number;
  eval_games: number;
};

export type RunState = {
  phase: Phase;
  generation: number;
  run_id: string;
  running: boolean;
  device?: string;
};

export type StatsFrame = {
  t_unix_ms: number;
  generation: number;
  phase: Phase;
  inferences_per_sec: number;
  games_per_sec: number;
  completed_games_total: number;
  samples_collected: number;
  samples_target: number;
  gpu_busy_pct: number;
  batch_avg_rows: number;
  policy_loss: number;
  value_loss: number;
  target_entropy: number;
  gpu_rows_per_sec: number;
};

export type GenerationMetric = {
  generation: number;
  policy_loss: number;
  value_loss: number;
  win_rate?: number;
  completed_games?: number;
  seconds?: number;
};

export type HistoryResponse = { metrics: GenerationMetric[] };

export type RunList = { runs: string[]; live: string | null };
