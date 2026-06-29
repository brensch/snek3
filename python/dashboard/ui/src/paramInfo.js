export const PARAM_INFO = {
  count: {
    name: "parallel game slots",
    summary: "How many self-play games are advanced in parallel while collecting a generation.",
    details:
      "This is not a target for finished games. The generator keeps this many game slots active and stops the generation once enough training samples have been selected. Games that finish before the cutoff can contribute samples; games still running at the cutoff are not used for final-outcome training in that generation.",
    faster:
      "Lowering this usually does not reduce work as directly as lowering samples or sims, and can hurt GPU batching if it drops utilization.",
  },
  samples: {
    name: "samples per generation",
    summary: "How many training positions to collect before moving from self-play to training.",
    details:
      "This is the biggest iteration-speed lever. More samples means more self-play positions and usually more stable training targets, but each generation takes longer. With length balancing enabled, reaching this target may require finishing more long games.",
    faster: "Drop this first for faster generations.",
  },
  sims: {
    name: "MCTS simulations per move",
    summary: "How much tree search is run for each move decision.",
    details:
      "Higher values make policy targets stronger and less noisy, but self-play cost scales almost directly with this. Lower values iterate faster but can reinforce bad early-net evaluations because search has less chance to correct them.",
    faster: "Use 24 as a moderate speed setting, or 16 for debugging.",
  },
  c_puct: {
    name: "search exploration",
    summary: "How strongly MCTS explores moves with high prior probability.",
    details:
      "Higher values push search to keep trying policy-prior moves; lower values make search lean harder on currently estimated value. This changes target quality more than raw speed.",
  },
  lr: {
    name: "learning rate",
    summary: "Optimizer step size during training.",
    details:
      "Higher values adapt faster but can destabilize policy/value learning. Lower values are steadier but may need more generations to move.",
  },
  train_steps: {
    name: "training steps",
    summary: "Number of optimizer updates after each self-play generation.",
    details:
      "This controls how much the net trains on the replay buffer per generation. In the current run, self-play is much slower than training, so this is a secondary speed lever unless training time becomes large.",
    faster: "Drop this after samples if total generation time still feels too slow.",
  },
  batch_size: {
    name: "training batch size",
    summary: "Replay samples used per optimizer update.",
    details:
      "Larger batches give smoother gradients and better GPU utilization during training, but use more memory. This mostly affects the training phase, not self-play.",
  },
  recency: {
    name: "replay recency bias",
    summary: "How strongly training minibatches favor newer replay-buffer samples.",
    details:
      "A value of 1 samples the retained replay window uniformly. Values above 1 bias toward recent games while still leaving a tail of older positions; 2 is a moderate recent-game bias.",
  },
  exploration_prob: {
    name: "played-move exploration",
    summary: "Uniform legal-move noise mixed into the action actually played.",
    details:
      "This is separate from the search target. It makes self-play try legal alternatives even when MCTS strongly prefers one move, which improves coverage but can create visible games where the played move differs from the target policy.",
  },
  draw_value: {
    name: "draw value",
    summary: "Value target used for snakes that truly draw.",
    details:
      "A draw means the snake is one of the snakes still alive at the final terminal state. Snakes eliminated earlier are losses even if the game later ends as a draw.",
  },
  board: {
    name: "board size",
    summary: "Width and height of the square Battlesnake board.",
    details:
      "This is baked into the network input shape for a run. Changing it requires starting a fresh run.",
  },
  num_snakes: {
    name: "snake count",
    summary: "Number of snakes spawned at the start of each game.",
    details:
      "This changes the game shape and network inputs. It is fixed for a run.",
  },
  filters: {
    name: "legacy trunk width",
    summary: "Older name for network channel width.",
    details:
      "If supported by the launcher, this controls model width. Wider models are stronger but slower and heavier. Current metadata may call this trunk_channels.",
  },
  blocks: {
    name: "legacy trunk depth",
    summary: "Older name for number of residual blocks.",
    details:
      "If supported by the launcher, this controls model depth. Deeper models are stronger but slower and heavier. Current metadata may call this trunk_blocks.",
  },
  depth: {
    name: "legacy search depth",
    summary: "Older search-depth field kept for compatibility.",
    details:
      "Current AlphaZero self-play is controlled primarily by sims rather than a fixed depth. Prefer sims when tuning current runs.",
  },
  trunk_channels: {
    name: "network width",
    summary: "Number of channels in the grid trunk.",
    details:
      "Wider networks can represent more patterns, but inference and training are heavier. This is fixed for a run.",
  },
  trunk_blocks: {
    name: "network depth",
    summary: "Number of residual blocks in the grid trunk.",
    details:
      "Deeper networks can compute richer board features, but inference and training are heavier. This is fixed for a run.",
  },
  generations: {
    name: "generation limit",
    summary: "How many generations the run should execute before stopping.",
    details:
      "This is a run-length cap. It does not affect the amount of work inside a single generation.",
  },
  max_turns: {
    name: "turn cap",
    summary: "Maximum game length before forcing a draw; 0 means play until terminal.",
    details:
      "A positive cap prevents very long games from consuming unlimited self-play time, but forced overrun draws can make value targets less informative.",
  },
  sample_games: {
    name: "recorded replays",
    summary: "Number of self-play games saved to JSON for the dashboard each recorded generation.",
    details:
      "This affects introspection and disk use, not the replay buffer. Lower it if dashboard files are too large; raise it if you need more examples to inspect.",
  },
  sample_every: {
    name: "replay interval",
    summary: "Save dashboard replay games every N generations.",
    details:
      "A value of 1 records every generation. Higher values reduce JSON output and UI load without changing training data.",
  },
  skip_short_draw_turns: {
    name: "short draw filter",
    summary: "Drop terminal draw games up to this turn length from replay training; 0 disables it.",
    details:
      "This can suppress low-value trivial draw positions, but setting it too high can hide real early-game failures.",
  },
  buffer_size: {
    name: "replay buffer size",
    summary: "Maximum number of samples retained for training.",
    details:
      "A larger buffer preserves more history but adapts more slowly to the latest net. This affects training data mix more than self-play speed.",
  },
  keep_games: {
    name: "saved replay files",
    summary: "How many recent game JSON files to keep for dashboard replay.",
    details:
      "This controls dashboard storage only. It does not change the replay buffer used for training.",
  },
  eval_batch_size: {
    name: "inference batch size",
    summary: "Leaf observations evaluated per neural-net inference chunk during self-play.",
    details:
      "Larger batches can improve GPU efficiency but use more memory. Lower this if self-play runs out of GPU memory.",
  },
};

export function paramInfo(key) {
  return PARAM_INFO[key] || {
    name: key,
    summary: "Training parameter.",
    details: "No detailed description has been added for this parameter yet.",
  };
}
