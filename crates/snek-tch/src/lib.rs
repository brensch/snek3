//! The snek3 AZNet (`arch="grid"`) in tch-rs (libtorch), matching
//! the archived `azsnek/net.py`. Uses libtorch/cuDNN and is shared by training
//! and serving.

use tch::{nn, nn::Module, Kind, Tensor};

const GN: i64 = 8;

// NOTE on init: net.py uses orthogonal(gain=sqrt(2)) for conv/linear weights and
// zeros for biases. tch 0.24's `Init::Orthogonal` is broken for >2-D tensors (it
// stores the flattened 2-D weight, corrupting conv shapes), so we build with the
// default init here and apply a correct orthogonal pass post-construction via
// [`init_orthogonal`]. Bias init is Const(0) below; GroupNorm affine (1/0) is
// tch's default. (Bit-exact parity with PyTorch is impossible regardless — the
// RNG streams differ — so what matters is matching the init *method*.)
fn conv(p: nn::Path, inc: i64, outc: i64) -> nn::Conv2D {
    nn::conv2d(
        p,
        inc,
        outc,
        3,
        nn::ConvConfig {
            padding: 1,
            bias: false,
            ..Default::default()
        },
    )
}

fn lin(p: nn::Path, inc: i64, outc: i64) -> nn::Linear {
    nn::linear(
        p,
        inc,
        outc,
        nn::LinearConfig {
            bs_init: Some(nn::Init::Const(0.)),
            bias: true,
            ..Default::default()
        },
    )
}

/// Apply PyTorch-style `nn.init.orthogonal_(gain)` to every conv/linear weight
/// (≥2-D `*.weight`), reshaping back to the original shape. GroupNorm (1-D) and
/// biases are left as constructed (1/0). Call once after [`AZNet::new`].
pub fn init_orthogonal(vs: &nn::VarStore, gain: f64) {
    tch::no_grad(|| {
        for (name, mut w) in vs.variables() {
            if !name.ends_with(".weight") {
                continue;
            }
            let dims = w.size();
            if dims.len() < 2 {
                continue; // GroupNorm weight stays at 1
            }
            let rows = dims[0];
            let cols: i64 = dims[1..].iter().product();
            let flat = Tensor::randn([rows, cols], (Kind::Float, w.device()));
            let flat = if rows < cols {
                flat.transpose(0, 1)
            } else {
                flat
            };
            let (q, r) = Tensor::linalg_qr(&flat, "reduced");
            // Sign-correct columns of Q by sign(diag(R)) (PyTorch convention).
            let signs = r.diagonal(0, 0, 1).sign();
            let q = q * signs.unsqueeze(0);
            let q = if rows < cols { q.transpose(0, 1) } else { q };
            w.copy_(&(q.reshape(&dims) * gain));
        }
    });
}

/// concat(mean_HW, max_HW) -> [B, 2C].
fn global_pool(x: &Tensor) -> Tensor {
    let mean = x.mean_dim(&[2i64, 3][..], false, Kind::Float);
    let max = x.amax(&[2i64, 3][..], false);
    Tensor::cat(&[mean, max], 1)
}

struct GPoolBlock {
    norm1: nn::GroupNorm,
    conv1: nn::Conv2D,
    norm2: nn::GroupNorm,
    conv2: nn::Conv2D,
    gp: Option<nn::Linear>,
}

impl GPoolBlock {
    fn new(p: &nn::Path, ch: i64, gpool: bool) -> Self {
        Self {
            norm1: nn::group_norm(p / "n1", GN, ch, Default::default()),
            conv1: conv(p / "c1", ch, ch),
            norm2: nn::group_norm(p / "n2", GN, ch, Default::default()),
            conv2: conv(p / "c2", ch, ch),
            gp: gpool.then(|| lin(p / "gp", 2 * ch, ch)),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let y = self.conv1.forward(&self.norm1.forward(x).leaky_relu());
        let y = match &self.gp {
            Some(gp) => {
                let bias = gp.forward(&global_pool(&y)).unsqueeze(-1).unsqueeze(-1);
                y + bias
            }
            None => y,
        };
        let y = self.conv2.forward(&self.norm2.forward(&y).leaky_relu());
        x + y
    }
}

pub struct AZNet {
    stem: nn::Conv2D,
    blocks: Vec<GPoolBlock>,
    trunk_norm: nn::GroupNorm,
    policy_head: nn::Linear,
    value_h1: nn::Linear,
    value_h2: nn::Linear,
}

impl AZNet {
    pub fn new(
        p: &nn::Path,
        in_ch: i64,
        trunk_ch: i64,
        trunk_blocks: i64,
        gpool_every: i64,
    ) -> Self {
        let blocks = (0..trunk_blocks)
            .map(|i| {
                GPoolBlock::new(
                    &(p / format!("block{i}")),
                    trunk_ch,
                    (i + 1) % gpool_every == 0,
                )
            })
            .collect();
        Self {
            stem: conv(p / "stem", in_ch, trunk_ch),
            blocks,
            trunk_norm: nn::group_norm(p / "tnorm", GN, trunk_ch, Default::default()),
            policy_head: lin(p / "ph", 2 * trunk_ch, 4),
            value_h1: lin(p / "vh1", 2 * trunk_ch, trunk_ch),
            value_h2: lin(p / "vh2", trunk_ch, 1),
        }
    }

    /// (policy_logits [B,4], value [B]).
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut h = self.stem.forward(x);
        for b in &self.blocks {
            h = b.forward(&h);
        }
        let h = self.trunk_norm.forward(&h).leaky_relu();
        let pooled = global_pool(&h);
        let p = self.policy_head.forward(&pooled);
        let v = self
            .value_h2
            .forward(&self.value_h1.forward(&pooled).leaky_relu())
            .tanh()
            .squeeze_dim(-1);
        (p, v)
    }
}
