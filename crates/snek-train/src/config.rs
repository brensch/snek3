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
    pub max_turns: usize,
    pub draw_value: f32,
    pub skip_short_draw_turns: usize,
    pub trunk_channels: i64,
    pub trunk_blocks: i64,
    pub train_steps: usize,
    pub batch_size: usize,
    pub recency: f64,
    pub buffer_size: usize,
    pub value_weight: f64,
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
    /// Fixed MCTS sims per eval move (deterministic, CPU). The league's CPU
    /// allotment and concurrent-game count are derived from the machine, not
    /// configured — see `eval::league_layout`.
    #[serde(default = "default_eval_sims")]
    pub eval_sims: usize,
    /// External API players holding permanent league seats, as "name=url"
    /// entries. The url is the base of a Battlesnake-protocol HTTP server
    /// (the arena POSTs {url}/move). The name is the player's stable league
    /// identity: its rating id is assigned on first sight and persisted in
    /// eval/players.json, so entries can be reordered or removed and later
    /// re-added without corrupting history.
    #[serde(default)]
    pub league_api_players: Vec<String>,
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

/// How many GPU-batch-sized groups of games are kept in flight at once. Two is a
/// double buffer: while one batch is on the GPU, the other is being built on the
/// CPU. Self-play is GPU-forward-bound (the GPU never idles under the lock
/// handoff), so two is enough to saturate it and more only wastes memory.
const GPU_PIPELINE_BUFFERS: usize = 2;

impl RunConfig {
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
            max_turns: 0, // 0 = uncapped (games run to a natural terminal)
            draw_value: -0.25,
            skip_short_draw_turns: 0,
            trunk_channels: 96,
            trunk_blocks: 8,
            train_steps: 128,
            batch_size: 2048,
            recency: 2.0,
            buffer_size: 500_000,
            value_weight: 1.0,
            search_threads: 0,
            sample_games: default_sample_games(),
            league_entrant_gens: default_league_entrant_gens(),
            eval_sims: default_eval_sims(),
            league_api_players: Vec::new(),
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
