use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainerState {
    pub generation: u32,
    pub seed: u64,
    pub best_win_rate: f64,
    pub samples_seen: u64,
}

impl Default for TrainerState {
    fn default() -> Self {
        Self {
            generation: 0,
            seed: 1,
            best_win_rate: 0.0,
            samples_seen: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RunPaths {
    pub root: PathBuf,
    pub config: PathBuf,
    pub trainer_state: PathBuf,
    pub net: PathBuf,
    pub replay: PathBuf,
    pub metrics: PathBuf,
    pub games: PathBuf,
}

impl RunPaths {
    pub fn new(runs_dir: &Path, run_id: &str) -> Self {
        let root = runs_dir.join(run_id);
        Self {
            config: root.join("config.json"),
            trainer_state: root.join("trainer_state.json"),
            net: root.join("net.safetensors"),
            replay: root.join("buffer"),
            metrics: root.join("metrics.jsonl"),
            games: root.join("games"),
            root,
        }
    }

    pub fn ensure(&self) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.root)?;
        std::fs::create_dir_all(&self.replay)?;
        Ok(())
    }
}

pub fn load_trainer_state(path: &Path) -> anyhow::Result<TrainerState> {
    if !path.exists() {
        return Ok(TrainerState::default());
    }
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

pub fn save_trainer_state(path: &Path, state: &TrainerState) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(state)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}
