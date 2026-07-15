use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub board: i8,
    pub num_snakes: usize,
    pub sims: usize,
    pub c_puct: f32,
    /// Games batched into one GPU forward. The forward's tensor is this many rows
    /// times `num_snakes` (one row per snake). Everything else about self-play
    /// concurrency is derived from this — see [`RunConfig::concurrent_games`].
    pub gpu_batch_games: usize,
    pub samples_per_gen: usize,
    /// Opening turns whose move is sampled from the visit-count policy
    /// (temperature 1); from this turn on play is strict argmax. AlphaZero's
    /// move-selection temperature schedule — they sampled 30 of ~80 chess
    /// plies; snek games run ~60–130 turns, so 20 keeps a similar fraction.
    #[serde(default = "default_sample_turns")]
    pub sample_turns: usize,
    /// AlphaZero root exploration noise, applied INSIDE the search: at the
    /// root of every self-play search, per snake,
    /// prior = (1-frac)*prior + frac*Dir(alpha) over the masked-legal moves,
    /// sampled fresh each turn. Because the noise shapes where simulations go,
    /// the visit-count training target itself explores moves the raw prior
    /// dislikes — the mechanism AlphaZero relies on to escape policy local
    /// optima. (Noise applied only to the played move — what this trainer and
    /// the archived Python one did through snek3-14 — diversifies states but
    /// never the targets; that run's Elo froze by gen ~70.) 0 disables.
    #[serde(default = "default_dirichlet_frac")]
    pub dirichlet_frac: f32,
    #[serde(default = "default_dirichlet_alpha")]
    pub dirichlet_alpha: f32,
    pub draw_value: f32,
    pub skip_short_draw_turns: usize,
    pub trunk_channels: i64,
    pub trunk_blocks: i64,
    pub train_steps: usize,
    pub batch_size: usize,
    pub recency: f64,
    pub buffer_size: usize,
    pub value_weight: f64,
    /// Decoupled (AdamW) weight decay. AlphaZero's loss carries an L2 term
    /// (c=1e-4); without any regularizer the net is free to drift toward the
    /// idiosyncrasies of the newest self-play shards long after real progress
    /// has stopped — the run-15 lineage's post-peak Elo slide. 0 disables
    /// (the pre-run-20 behavior; old run configs deserialize to 0).
    #[serde(default)]
    pub weight_decay: f64,
    /// The LR halves every this many training samples seen (smooth decay from
    /// 1e-3, floored at 1e-4). AlphaZero steps its LR down ~10x twice across a
    /// run; the old fixed 5M half-life (~420 gens/halving at 12k samples/gen)
    /// left updates near full size at the gen ~200 strength peak, so the net
    /// kept taking near-base-LR steps on noisy targets long past convergence.
    /// Old run configs deserialize to the 5M they trained with.
    #[serde(default = "default_lr_half_life_samples")]
    pub lr_half_life_samples: f64,
    pub search_threads: usize,
    /// How many self-play games to record as browsable samples each generation.
    #[serde(default = "default_sample_games")]
    pub sample_games: usize,
    /// A new checkpoint joins the evaluation league every this many
    /// generations; league matches between pool members then run back-to-back
    /// on CPU for as long as the run is active. 0 disables the league.
    /// (`alias`: pre-league configs called this eval_turns.)
    #[serde(default = "default_league_entrant_gens", alias = "eval_turns")]
    pub league_entrant_gens: usize,
    /// Fixed MCTS sims per eval move (deterministic, CPU).
    #[serde(default = "default_eval_sims")]
    pub eval_sims: usize,
    /// Concurrent league games. Each is a slot that picks fresh opponents,
    /// plays one game on unpinned CPU threads (one search thread per living
    /// snake), records it, and immediately starts the next.
    #[serde(default = "default_league_games")]
    pub league_games: usize,
    /// GPU burst arena: every `league_entrant_gens` generations — right after
    /// the new entrant's checkpoint is saved — the trainer borrows the GPU for
    /// one cycle and plays this many league games with per-net batched
    /// forwards (hundreds of times the CPU slots' throughput). The games land
    /// in the same summary/ratings as the CPU league's. 0 disables bursts.
    #[serde(default = "default_burst_games")]
    pub burst_games: usize,
    /// Sims per move in burst games. 0 means `eval_sims`, keeping burst games
    /// statistically interchangeable with the CPU league's.
    #[serde(default)]
    pub burst_sims: usize,

    // ---- Logit-Equilibrium ("correct game mode") self-play ----
    /// Use the Albatross-faithful fixed-depth Logit-Equilibrium search for
    /// self-play instead of decoupled-PUCT MCTS. Produces mixed-strategy
    /// policies + per-player equilibrium values for the simultaneous-move game.
    #[serde(default)]
    pub le_mode: bool,
    /// Plies of the fixed-depth joint-move equilibrium tree. Depth-1 is a 1-ply
    /// equilibrium best-response over the net's leaf values (cheap, 4-player
    /// friendly); depth-2 is stronger but ~O(cands^n) more leaves.
    #[serde(default = "default_le_depth")]
    pub le_depth: u32,
    /// Selective deepening: expand the top-K joint successors (by equilibrium
    /// reach probability) one extra ply and re-solve. 0 = fixed depth-1 (no
    /// deepening). See docs/le-selective-depth.md.
    #[serde(default)]
    pub le_top_k: usize,
    /// Leaf-eval forward chunk size in ROWS (board-seats). 0 = one big forward
    /// (old behaviour). >0 splits each turn's value forward into fixed-size,
    /// benchmark-friendly chunks near the GPU throughput sweet spot (~512),
    /// pipelined so the GPU stays fed. See docs/le-throughput.md.
    #[serde(default)]
    pub le_fwd_chunk: usize,
    /// Stochastic-fictitious-play iterations per equilibrium solve.
    #[serde(default = "default_le_iters")]
    pub le_iters: usize,
    /// Fraction of NEW self-play games seeded from a curriculum scenario
    /// (snek_core::scenario) instead of the official start. 0 disables.
    #[serde(default)]
    pub scenario_prob: f32,
    /// Which scenarios to draw from (uniformly). Empty = every registered
    /// scenario. Unknown names are ignored with a log line.
    #[serde(default)]
    pub scenarios: Vec<String>,
    /// Per-episode inverse-temperature (rationality) range the proxy is
    /// conditioned on: high = rational/Nash-like, low = near-uniform. A τ is
    /// sampled uniformly from [tau_min, tau_max] each game.
    #[serde(default = "default_tau_min")]
    pub tau_min: f32,
    #[serde(default = "default_tau_max")]
    pub tau_max: f32,
    /// Rational responder inverse-temperature for the exploit path (Stage B).
    #[serde(default = "default_response_tau")]
    pub response_tau: f32,
    /// Generations of proxy-only self-play before the response/exploit path
    /// begins (Stage B).
    #[serde(default = "default_response_after")]
    pub response_after: usize,
    /// Blend of game outcome into the LE value target: 0 = pure equilibrium
    /// bootstrap (validated Albatross, at depth 2), 1 = pure game outcome. A
    /// blend grounds the depth-1 bootstrap against its cold-start (early leaf
    /// values are near-random, so a purely self-referential target is weak).
    #[serde(default = "default_le_outcome_weight")]
    pub le_outcome_weight: f32,
    /// External API players holding permanent league seats, as "name=url"
    /// entries. The url is the base of a Battlesnake-protocol HTTP server
    /// (the arena POSTs {url}/move). The name is the player's stable league
    /// identity: its rating id is assigned on first sight and persisted in
    /// eval/players.json, so entries can be reordered or removed and later
    /// re-added without corrupting history.
    #[serde(default)]
    pub league_api_players: Vec<String>,
    /// Probability that a self-play game seats one or more built-in heuristic
    /// opponents ("sparring partners") instead of being pure net-vs-net. Pure
    /// self-play only ever trains the net against its own style, so a strategy
    /// it never plays (e.g. voronoi space-control) is a permanent blind spot;
    /// seating the heuristic forces the net to generate winning lines against
    /// it. 0 disables (pure self-play — the default and all pre-existing runs).
    #[serde(default)]
    pub heuristic_opponent_prob: f64,
    /// When a game is chosen to include heuristic opponents, how many of its
    /// seats they take (clamped to leave at least one net seat). The rest stay
    /// net-controlled, so the net still generates net-vs-net data within the
    /// same game and never over-specialises into an anti-heuristic one-trick.
    #[serde(default = "default_heuristic_opponent_seats")]
    pub heuristic_opponent_seats: usize,
    /// Which built-in heuristic the sparring partners play: "voronoi" (the
    /// strong space-control agent) or "floodfill". Ignored when the probability
    /// is 0.
    #[serde(default = "default_heuristic_opponent_kind")]
    pub heuristic_opponent_kind: String,
    /// MCTS simulations the sparring partner runs for the move it actually
    /// plays (and which the net's search forces at its root). Voronoi sims are
    /// cheap CPU flood-fills (~100k/s/core) and run while the GPU is busy on the
    /// net's forwards, so a budget comparable to the net's own `sims` is nearly
    /// free — turning the opponent from a 1-ply pushover into a real challenge.
    /// 0 falls back to the 1-ply greedy (the cheapest possible). Deeper tree
    /// nodes always use the 1-ply greedy regardless (a per-node MCTS is
    /// infeasible); only the played/root move gets the full budget.
    #[serde(default = "default_heuristic_opponent_sims")]
    pub heuristic_opponent_sims: usize,
}

fn default_heuristic_opponent_sims() -> usize {
    800
}

fn default_le_depth() -> u32 {
    1
}
fn default_le_iters() -> usize {
    120
}
fn default_tau_min() -> f32 {
    0.5
}
fn default_tau_max() -> f32 {
    10.0
}
fn default_response_tau() -> f32 {
    12.0
}
fn default_response_after() -> usize {
    30
}
fn default_le_outcome_weight() -> f32 {
    0.5
}

fn default_heuristic_opponent_seats() -> usize {
    1
}

fn default_heuristic_opponent_kind() -> String {
    "voronoi".to_string()
}

fn default_sample_turns() -> usize {
    20
}

fn default_dirichlet_frac() -> f32 {
    0.25
}

fn default_dirichlet_alpha() -> f32 {
    0.3
}

fn default_sample_games() -> usize {
    8
}

fn default_league_entrant_gens() -> usize {
    5
}

fn default_eval_sims() -> usize {
    64
}

fn default_league_games() -> usize {
    4
}

fn default_burst_games() -> usize {
    256
}

fn default_lr_half_life_samples() -> f64 {
    // The schedule runs at this half-life. Pre-knob it lived in train.rs as
    // LR_HALF_LIFE_SAMPLES = 5M; old configs (no field) must keep training at
    // the rate they always did.
    5_000_000.0
}

/// How many GPU-batch-sized groups of games are kept in flight at once. Two is a
/// double buffer: while one batch is on the GPU, the other is being built on the
/// CPU. Self-play is GPU-forward-bound (the GPU never idles under the lock
/// handoff), so two is enough to saturate it and more only wastes memory.
const GPU_PIPELINE_BUFFERS: usize = 2;

impl RunConfig {
    /// The heuristic sparring partner to seat in self-play, or `None` when the
    /// feature is off (`heuristic_opponent_prob == 0`) or the kind string is
    /// unrecognised. Parsed once per generation, not per game.
    pub fn heuristic_opponent(&self) -> Option<snek_heuristic::Baseline> {
        if self.heuristic_opponent_prob <= 0.0 {
            return None;
        }
        snek_heuristic::Baseline::parse(&self.heuristic_opponent_kind)
    }

    /// Total number of games played concurrently in one self-play generation.
    /// Derived from the GPU batch size rather than configured directly: it is just
    /// enough games to keep the GPU saturated via double buffering.
    pub fn concurrent_games(&self) -> usize {
        self.gpu_batch_games.max(1) * GPU_PIPELINE_BUFFERS
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            board: 11,
            num_snakes: 4,
            sims: 24,
            c_puct: 1.5,
            gpu_batch_games: 128,
            samples_per_gen: 12_000,
            sample_turns: default_sample_turns(),
            dirichlet_frac: default_dirichlet_frac(),
            dirichlet_alpha: default_dirichlet_alpha(),
            draw_value: -0.25,
            skip_short_draw_turns: 0,
            trunk_channels: 96,
            trunk_blocks: 8,
            train_steps: 128,
            batch_size: 2048,
            // 1.0 = uniform over the buffer window, AlphaZero's sampling. A
            // recency skew (the old 2.0) doubles down on the newest shards —
            // the noisiest data — which amplifies late-run drift.
            recency: 1.0,
            buffer_size: 500_000,
            value_weight: 1.0,
            // AlphaZero's L2 coefficient, as decoupled AdamW decay.
            weight_decay: 1e-4,
            // ~125 gens per halving at 12k samples/gen: near-base LR through
            // the fast-improvement phase, ~1e-4 (the floor) past gen ~400.
            lr_half_life_samples: 1_500_000.0,
            search_threads: 0,
            sample_games: default_sample_games(),
            league_entrant_gens: default_league_entrant_gens(),
            eval_sims: default_eval_sims(),
            league_games: default_league_games(),
            burst_games: default_burst_games(),
            burst_sims: 0,
            league_api_players: Vec::new(),
            heuristic_opponent_prob: 0.0,
            heuristic_opponent_seats: default_heuristic_opponent_seats(),
            heuristic_opponent_kind: default_heuristic_opponent_kind(),
            heuristic_opponent_sims: default_heuristic_opponent_sims(),
            le_mode: false,
            le_depth: default_le_depth(),
            le_top_k: 0,
            le_fwd_chunk: 0,
            le_iters: default_le_iters(),
            scenario_prob: 0.0,
            scenarios: Vec::new(),
            tau_min: default_tau_min(),
            tau_max: default_tau_max(),
            response_tau: default_response_tau(),
            response_after: default_response_after(),
            le_outcome_weight: default_le_outcome_weight(),
        }
    }
}

impl RunConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn save_atomic(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }
}
