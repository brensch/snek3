//! The snek3 AlphaZero net (`arch="grid"`) expressed in burn, matching
//! the archived `azsnek/net.py`: stem conv -> N KataGo-style pre-activation residual
//! blocks (every `gpool_every`-th carries a global-pooling bias) -> GroupNorm ->
//! LeakyReLU -> global pool (concat mean+max over space) -> policy/value heads.
//!
//! Built to measure cubecl/CUDA inference throughput on the exact net shape we
//! run today (14 channels, 11x11, trunk 96x8). Numerics need not match PyTorch
//! bit-for-bit for the throughput question; the architecture does.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::leaky_relu;

const GN_GROUPS: usize = 8;
const SLOPE: f64 = 0.01;

/// Concat of channel-wise mean and max over the spatial dims -> `[B, 2C]`.
fn global_pool<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 2> {
    let [b, c, _h, _w] = x.dims();
    let mean = x.clone().mean_dim(3).mean_dim(2).reshape([b, c]);
    let max = x.max_dim(3).max_dim(2).reshape([b, c]);
    Tensor::cat(vec![mean, max], 1)
}

#[derive(Module, Debug)]
pub struct GPoolResBlock<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    gp: Option<Linear<B>>,
}

impl<B: Backend> GPoolResBlock<B> {
    pub fn new(ch: usize, gpool: bool, device: &B::Device) -> Self {
        let conv = || {
            Conv2dConfig::new([ch, ch], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device)
        };
        Self {
            norm1: GroupNormConfig::new(GN_GROUPS, ch).init(device),
            conv1: conv(),
            norm2: GroupNormConfig::new(GN_GROUPS, ch).init(device),
            conv2: conv(),
            gp: gpool.then(|| LinearConfig::new(2 * ch, ch).init(device)),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self
            .conv1
            .forward(leaky_relu(self.norm1.forward(x.clone()), SLOPE));
        let y = if let Some(gp) = &self.gp {
            let [b, c, _h, _w] = y.dims();
            let bias = gp.forward(global_pool(y.clone())).reshape([b, c, 1, 1]);
            y + bias
        } else {
            y
        };
        let y = self
            .conv2
            .forward(leaky_relu(self.norm2.forward(y), SLOPE));
        x + y
    }
}

#[derive(Module, Debug)]
pub struct AZNet<B: Backend> {
    stem: Conv2d<B>,
    blocks: Vec<GPoolResBlock<B>>,
    trunk_norm: GroupNorm<B>,
    policy_head: Linear<B>,
    value_h1: Linear<B>,
    value_h2: Linear<B>,
}

impl<B: Backend> AZNet<B> {
    pub fn new(
        in_ch: usize,
        trunk_ch: usize,
        trunk_blocks: usize,
        gpool_every: usize,
        device: &B::Device,
    ) -> Self {
        let blocks = (0..trunk_blocks)
            .map(|i| GPoolResBlock::new(trunk_ch, (i + 1) % gpool_every == 0, device))
            .collect();
        Self {
            stem: Conv2dConfig::new([in_ch, trunk_ch], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device),
            blocks,
            trunk_norm: GroupNormConfig::new(GN_GROUPS, trunk_ch).init(device),
            policy_head: LinearConfig::new(2 * trunk_ch, 4).init(device),
            value_h1: LinearConfig::new(2 * trunk_ch, trunk_ch).init(device),
            value_h2: LinearConfig::new(trunk_ch, 1).init(device),
        }
    }

    /// Returns (policy_logits `[B,4]`, value `[B]` in [-1,1]).
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let mut h = self.stem.forward(x);
        for block in &self.blocks {
            h = block.forward(h);
        }
        let h = leaky_relu(self.trunk_norm.forward(h), SLOPE);
        let pooled = global_pool(h); // [B, 2*ch]
        let [b, _] = pooled.dims();
        let p = self.policy_head.forward(pooled.clone());
        let v = self
            .value_h2
            .forward(leaky_relu(self.value_h1.forward(pooled), SLOPE))
            .tanh()
            .reshape([b]);
        (p, v)
    }
}
