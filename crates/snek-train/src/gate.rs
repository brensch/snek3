//! Checkpoint gating: only a candidate that PROVES itself replaces the
//! self-play data generator.
//!
//! Without gating, the freshly-trained net becomes the data generator every
//! generation unconditionally — so the moment training takes a bad step, the
//! regressed net starts generating its own (worse) training data. Measured on
//! snek3-le-6: checkpoints 32 gens apart swung ±40 points of true strength
//! and the held-out mean eroded ~50%→~35% over 500 gens while every internal
//! health metric stayed green. Self-play grades its own homework; gating
//! brings in an examiner.
//!
//! # Design (AlphaGo-style, adapted for LE)
//!
//! Two nets live in the trainer:
//! - the **incumbent** — frozen; generates ALL self-play data and is what
//!   serving publishes. Changes only by promotion.
//! - the **candidate** (the live net) — trained every generation on the
//!   incumbent's data; between gates it is a pure learner.
//!
//! Every `gate_gens` generations both nets play `gate_games` games against
//! voronoi at `gate_sims` **from the same start seeds** (a paired comparison:
//! start-position luck — the dominant variance source — cancels). The
//! candidate is promoted iff it wins strictly more than the incumbent plus
//! `gate_margin`. LE nets can't play each other head-to-head (one net's value
//! head drives the whole equilibrium solve), so a shared external opponent is
//! the honest referee — and voronoi at mid-strength sims is chosen so win
//! rates land mid-range, where a paired gate has maximum signal.
//!
//! The gate seed rotates every gate so repeated gating can't select for
//! start-position specialists.

use crate::config::RunConfig;
use crate::le_eval::{eval_le_vs_baseline, RecordedEvalGame};
use snek_heuristic::Baseline;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::AtomicU32;
use tch::Device;

/// Persisted gate state (`runs/<id>/gate.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateMeta {
    /// Generation whose weights the incumbent currently holds.
    pub incumbent_gen: u32,
    /// Total promotions so far (0 = still the run's founding incumbent).
    pub promotions: u32,
    /// Consecutive failed gates since the last promotion — sustained high
    /// values mean the learner has stopped improving on the incumbent.
    pub failed_gates: u32,
    /// Seed for the NEXT gate's paired games; advanced every gate.
    pub next_seed: u64,
}

impl GateMeta {
    pub fn founding(gen: u32, seed: u64) -> Self {
        GateMeta { incumbent_gen: gen, promotions: 0, failed_gates: 0, next_seed: seed }
    }

    pub fn load(path: &Path) -> Option<Self> {
        serde_json::from_slice(&std::fs::read(path).ok()?).ok()
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }
}

/// Everything one gate produced, for logs/metrics/the promotion decision.
pub struct GateReport {
    pub candidate_wins: usize,
    pub incumbent_wins: usize,
    pub games: usize,
    pub promoted: bool,
    /// Candidate's win rates for the strength chart.
    pub candidate_vor: f64,
    pub incumbent_vor: f64,
    /// Candidate's recorded games (viewer audit trail).
    pub recorded: Vec<RecordedEvalGame>,
}

/// The promotion rule, separated for testability: strictly more wins than the
/// incumbent plus the margin. Ties keep the incumbent — stability is the
/// point, and the candidate gets another `gate_gens` of training either way.
pub fn promote(candidate_wins: usize, incumbent_wins: usize, margin: usize) -> bool {
    candidate_wins > incumbent_wins + margin
}

/// Play one gate: candidate and incumbent each play `gate_games` vs voronoi at
/// `gate_sims` from the same seed. Pure measurement — the caller owns weight
/// copies and persistence based on `report.promoted`.
#[allow(clippy::too_many_arguments)]
pub fn run_gate(
    candidate: &snek_tch::AZNet,
    incumbent: &snek_tch::AZNet,
    device: Device,
    cfg: &RunConfig,
    meta: &mut GateMeta,
    gen: u32,
    progress: Option<&AtomicU32>,
    record_games: usize,
) -> GateReport {
    let vor = Baseline::parse("voronoi").expect("voronoi baseline exists");
    let games = cfg.gate_games.max(8);
    let seed = meta.next_seed;
    // Same seed => same start positions for both sides of the pair.
    let (cand_rate, recorded) = eval_le_vs_baseline(
        candidate, device, cfg, vor, cfg.gate_sims, games, cfg.response_tau, seed,
        progress, gen, crate::eval::VORONOI_GEN, record_games,
    );
    let (inc_rate, _) = eval_le_vs_baseline(
        incumbent, device, cfg, vor, cfg.gate_sims, games, cfg.response_tau, seed,
        progress, meta.incumbent_gen, crate::eval::VORONOI_GEN, 0,
    );
    let candidate_wins = (cand_rate * games as f64).round() as usize;
    let incumbent_wins = (inc_rate * games as f64).round() as usize;
    let promoted = promote(candidate_wins, incumbent_wins, cfg.gate_margin);

    // Rotate the seed so no start set is ever reused; update streaks.
    meta.next_seed = meta.next_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(gen as u64);
    if promoted {
        meta.incumbent_gen = gen;
        meta.promotions += 1;
        meta.failed_gates = 0;
    } else {
        meta.failed_gates += 1;
    }

    GateReport {
        candidate_wins,
        incumbent_wins,
        games,
        promoted,
        candidate_vor: cand_rate,
        incumbent_vor: inc_rate,
        recorded,
    }
}

/// The strong probe: the incumbent vs voronoi at `probe_sims` — the
/// "super-heuristic" goal line. voronoi-20k is the historical AZ-league
/// anchor; voronoi at the old 64-sim eval barely searched past its root.
pub fn run_probe(
    incumbent: &snek_tch::AZNet,
    device: Device,
    cfg: &RunConfig,
    meta: &GateMeta,
    games: usize,
    seed: u64,
    progress: Option<&AtomicU32>,
) -> (f64, Vec<RecordedEvalGame>) {
    let vor = Baseline::parse("voronoi").expect("voronoi baseline exists");
    eval_le_vs_baseline(
        incumbent, device, cfg, vor, cfg.probe_sims, games.max(8), cfg.response_tau, seed,
        progress, meta.incumbent_gen, crate::eval::VORONOI_GEN, 2,
    )
}

/// One-off checkpoint sweep for picking a gating incumbent: every listed
/// checkpoint plays `games` vs voronoi@gate_sims from the SAME start seeds
/// (paired — differences are checkpoint skill, not opening luck). Prints a
/// table and the winner, then exits. Run with training STOPPED (owns the GPU).
pub fn sweep(paths: &crate::state::RunPaths, gens: &[u32], games: usize) -> anyhow::Result<()> {
    use crate::config::RunConfig;
    anyhow::ensure!(!gens.is_empty(), "no checkpoint gens given");
    let cfg = RunConfig::load(&paths.config)?;
    anyhow::ensure!(cfg.le_mode, "le-sweep is for LE runs");
    let device = Device::Cuda(0);
    let in_ch = snek_core::NUM_CHANNELS_TEMP as i64;
    let vor = Baseline::parse("voronoi").expect("voronoi baseline exists");
    let seed = 0xC0FFEE; // fixed: every checkpoint sees identical starts
    println!(
        "LE sweep: {} checkpoints x {} games vs voronoi-{} (tau {})",
        gens.len(), games, cfg.gate_sims, cfg.response_tau
    );
    let mut best: Option<(u32, f64)> = None;
    for &gen in gens {
        let path = paths.checkpoint_net(gen);
        anyhow::ensure!(path.exists(), "missing checkpoint {}", path.display());
        let mut vs = tch::nn::VarStore::new(device);
        let net = snek_tch::AZNet::new(&vs.root(), in_ch, cfg.trunk_channels, cfg.trunk_blocks);
        vs.load(&path)?;
        let t0 = std::time::Instant::now();
        let (rate, _) = eval_le_vs_baseline(
            &net, device, &cfg, vor, cfg.gate_sims, games, cfg.response_tau, seed,
            None, gen, crate::eval::VORONOI_GEN, 0,
        );
        println!(
            "  gen {gen:5}: {:5.1}%  ({} wins / {games}, {:.0}s)",
            rate * 100.0,
            (rate * games as f64).round() as usize,
            t0.elapsed().as_secs_f64(),
        );
        if best.map(|(_, b)| rate > b).unwrap_or(true) {
            best = Some((gen, rate));
        }
    }
    let (gen, rate) = best.expect("at least one checkpoint");
    println!("WINNER: gen {gen} at {:.1}% — use as founding incumbent", rate * 100.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_requires_strictly_more_wins() {
        assert!(!promote(10, 10, 0), "tie keeps the incumbent");
        assert!(promote(11, 10, 0));
        assert!(!promote(11, 10, 1), "margin raises the bar");
        assert!(promote(12, 10, 1));
        assert!(!promote(0, 0, 0), "0-0 keeps the incumbent");
    }

    #[test]
    fn gate_meta_roundtrips_and_rotates() {
        let dir = std::env::temp_dir().join(format!("gate-meta-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gate.json");
        let meta = GateMeta::founding(1760, 42);
        meta.save(&path).unwrap();
        let loaded = GateMeta::load(&path).unwrap();
        assert_eq!(loaded.incumbent_gen, 1760);
        assert_eq!(loaded.promotions, 0);
        assert_eq!(loaded.next_seed, 42);
        std::fs::remove_dir_all(&dir).ok();
    }
}
