use crate::config::RunConfig;
use crate::metrics::Counters;
use crate::replay::{ReplayBuffer, Samples};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::atomic::Ordering;
use tch::nn::OptimizerConfig;
use tch::{nn, Device, Kind, Reduction, Tensor};

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainMetrics {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub target_entropy: f64,
}

pub fn train_steps(
    net: &snek_tch::AZNet,
    vs: &nn::VarStore,
    opt: &mut nn::Optimizer,
    replay: &ReplayBuffer,
    cfg: &RunConfig,
    seed: u64,
    counters: &Counters,
) -> anyhow::Result<TrainMetrics> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut last = TrainMetrics::default();
    counters
        .train_steps_total
        .store(cfg.train_steps as u32, Ordering::Relaxed);
    counters.train_step.store(0, Ordering::Relaxed);
    for step in 0..cfg.train_steps {
        let Some(batch) =
            replay.sample_batch(cfg.batch_size.min(replay.len()), cfg.recency, &mut rng)
        else {
            break;
        };
        last = train_one(net, vs.device(), opt, &batch, cfg.value_weight)?;
        counters
            .train_step
            .store((step + 1) as u32, Ordering::Relaxed);
    }
    Ok(last)
}

pub fn build_optimizer(vs: &nn::VarStore, lr: f64) -> anyhow::Result<nn::Optimizer> {
    Ok(nn::Adam::default().build(vs, lr)?)
}

fn train_one(
    net: &snek_tch::AZNet,
    device: Device,
    opt: &mut nn::Optimizer,
    batch: &Samples,
    value_weight: f64,
) -> anyhow::Result<TrainMetrics> {
    let b = batch.len();
    let [c, h, w] = batch.obs_shape;
    let obs = Tensor::from_slice(&batch.obs)
        .reshape([b as i64, c as i64, h as i64, w as i64])
        .to_device(device);
    let target_pol = Tensor::from_slice(&batch.pol)
        .reshape([b as i64, 4])
        .to_device(device);
    let target_z = Tensor::from_slice(&batch.z).to_device(device);
    let (logits, value) = net.forward(&obs);
    let logp = logits.log_softmax(-1, Kind::Float);
    let policy_loss = -(target_pol.shallow_clone() * logp)
        .sum_dim_intlist(&[1i64][..], false, Kind::Float)
        .mean(Kind::Float);
    let value_loss = value.mse_loss(&target_z, Reduction::Mean);
    let loss = &policy_loss + &value_loss * value_weight;
    opt.backward_step(&loss);
    let entropy = -(target_pol.shallow_clone() * target_pol.clamp_min(1e-8).log())
        .sum_dim_intlist(&[1i64][..], false, Kind::Float)
        .mean(Kind::Float);
    Ok(TrainMetrics {
        policy_loss: policy_loss.double_value(&[]),
        value_loss: value_loss.double_value(&[]),
        target_entropy: entropy.double_value(&[]),
    })
}
