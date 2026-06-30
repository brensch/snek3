use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub board: i8,
    pub num_snakes: usize,
    pub count: usize,
    pub sims: usize,
    pub c_puct: f32,
    pub gpu_batch_games: usize,
    pub samples_per_gen: usize,
    pub exploration_prob: f32,
    pub max_turns: usize,
    pub draw_value: f32,
    pub skip_short_draw_turns: usize,
    pub bootstrap_value: bool,
    pub trunk_channels: i64,
    pub trunk_blocks: i64,
    pub gpool_every: i64,
    pub train_steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub recency: f64,
    pub buffer_size: usize,
    pub value_weight: f64,
    pub search_threads: usize,
    pub record_games: usize,
    pub eval_every: usize,
    pub eval_games: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            board: 11,
            num_snakes: 4,
            count: 512,
            sims: 24,
            c_puct: 1.5,
            gpu_batch_games: 128,
            samples_per_gen: 12_000,
            exploration_prob: 0.15,
            max_turns: 200,
            draw_value: -0.25,
            skip_short_draw_turns: 0,
            bootstrap_value: false,
            trunk_channels: 96,
            trunk_blocks: 8,
            gpool_every: 3,
            train_steps: 128,
            batch_size: 2048,
            lr: 1e-3,
            recency: 2.0,
            buffer_size: 500_000,
            value_weight: 1.0,
            search_threads: 0,
            record_games: 4,
            eval_every: 5,
            eval_games: 32,
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
