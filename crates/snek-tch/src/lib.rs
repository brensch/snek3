//! The snek3 AZNet in tch-rs (libtorch), shared by training and serving. The
//! trunk (conv stem + gpool residual blocks) matches the archived
//! `azsnek/net.py`; the policy head is spatial (see [`AZNet::forward`]): a
//! per-cell "move my head onto this cell" logit map read out at the four
//! neighbours of the snake's head, instead of the archived global-pool linear.

use tch::{nn, nn::Module, Kind, Tensor};

#[cfg(snek_cuda)]
pub mod cudagraph;

const GN: i64 = 8;

/// Insert a global-pool context block every Nth residual block (see
/// [`GPoolBlock`]). A fixed architecture choice matching the archived
/// `azsnek/net.py`, not a training knob — changing it makes existing checkpoints
/// weight-incompatible.
const GPOOL_EVERY: i64 = 3;

/// Observation channel holding the one-hot `my_head` plane (schema v1, see
/// snek-core `encode.rs`). The policy readout locates the head through it.
const HEAD_PLANE: i64 = 0;

/// Logit assigned to policy reads that fall off the board. Finite rather than
/// -inf: a fully trapped snake's search target can put real probability on an
/// off-board move (every move is kept as a candidate when none is legal), and
/// the training cross-entropy must stay finite. exp(-8) keeps structurally
/// illegal moves at ~0 probability.
const OFF_BOARD_LOGIT: f64 = -8.0;

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

/// 1x1 conv with a zero-init bias (not norm-followed, so it keeps its bias,
/// like the linear layers).
fn conv1x1(p: nn::Path, inc: i64, outc: i64) -> nn::Conv2D {
    nn::conv2d(
        p,
        inc,
        outc,
        1,
        nn::ConvConfig {
            bs_init: nn::Init::Const(0.),
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
    policy_c1: nn::Conv2D,
    policy_c2: nn::Conv2D,
    value_h1: nn::Linear,
    value_h2: nn::Linear,
}

impl AZNet {
    pub fn new(p: &nn::Path, in_ch: i64, trunk_ch: i64, trunk_blocks: i64) -> Self {
        let blocks = (0..trunk_blocks)
            .map(|i| {
                GPoolBlock::new(
                    &(p / format!("block{i}")),
                    trunk_ch,
                    (i + 1) % GPOOL_EVERY == 0,
                )
            })
            .collect();
        Self {
            stem: conv(p / "stem", in_ch, trunk_ch),
            blocks,
            trunk_norm: nn::group_norm(p / "tnorm", GN, trunk_ch, Default::default()),
            policy_c1: conv1x1(p / "pc1", trunk_ch, trunk_ch),
            policy_c2: conv1x1(p / "pc2", trunk_ch, 1),
            value_h1: lin(p / "vh1", 2 * trunk_ch, trunk_ch),
            value_h2: lin(p / "vh2", trunk_ch, 1),
        }
    }

    /// (policy_logits [B,4], value [B]).
    ///
    /// The policy is spatial: the head convs turn the trunk into a per-cell
    /// "move my head onto this cell" logit map, and the four move logits are
    /// that map read at the four neighbours of this snake's head — one shared
    /// judgment evaluated at four cells, computed where the consequence is,
    /// rather than four unrelated rows decoded from a globally pooled vector.
    /// The value stays a global question and keeps the pooled MLP head.
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut h = self.stem.forward(x);
        for b in &self.blocks {
            h = b.forward(&h);
        }
        let h = self.trunk_norm.forward(&h).leaky_relu();
        let map = self.policy_c2.forward(&self.policy_c1.forward(&h).leaky_relu());
        let p = head_neighbor_logits(&map, x);
        let pooled = global_pool(&h);
        let v = self
            .value_h2
            .forward(&self.value_h1.forward(&pooled).leaky_relu())
            .tanh()
            .squeeze_dim(-1);
        (p, v)
    }
}

/// Read the per-cell logit `map` ([B,1,H,W]) at the four neighbours of each
/// row's head cell, returning move logits ([B,4]) ordered as `Move::ALL`
/// (Up, Down, Left, Right; y grows upward, cell index is `y*w + x`).
///
/// The head is located via the one-hot [`HEAD_PLANE`] of the observation. A
/// dead snake's plane is all zeros; argmax then falls back to cell 0, whose
/// logits are never trained on or played (dead rows are skipped/dummied
/// upstream). The map is padded with one ring of [`OFF_BOARD_LOGIT`] so
/// neighbour reads never leave the canvas.
fn head_neighbor_logits(map: &Tensor, obs: &Tensor) -> Tensor {
    let (b, _, h, w) = map.size4().expect("policy map must be [B,1,H,W]");
    let head = obs.select(1, HEAD_PLANE).reshape([b, -1]).argmax(1, false);
    let y = head.floor_divide_scalar(w);
    let x = head.remainder(w);
    let wp = w + 2; // padded canvas width; padded coords are shifted by +1
    let flat = map
        .pad([1, 1, 1, 1], "constant", OFF_BOARD_LOGIT)
        .reshape([b, (h + 2) * wp]);
    let base = (y + 1) * wp + (x + 1);
    let idx = Tensor::stack(&[&base + wp, &base - wp, &base - 1, &base + 1], 1);
    flat.gather(1, &idx, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3x3 map whose cell (y,x) holds `off + y*3 + x`, plus an obs whose
    /// HEAD_PLANE is one-hot at (hy,hx).
    fn map_and_obs(heads: &[(i64, i64)], off: f32) -> (Tensor, Tensor) {
        let b = heads.len() as i64;
        let cells: Vec<f32> = (0..b)
            .flat_map(|r| (0..9).map(move |i| off * r as f32 + i as f32))
            .collect();
        let map = Tensor::from_slice(&cells).reshape([b, 1, 3, 3]);
        let mut obs = vec![0.0f32; (b * 9) as usize];
        for (r, &(hy, hx)) in heads.iter().enumerate() {
            obs[r * 9 + (hy * 3 + hx) as usize] = 1.0;
        }
        let obs = Tensor::from_slice(&obs).reshape([b, 1, 3, 3]);
        (map, obs)
    }

    #[test]
    fn head_neighbor_readout_matches_move_order() {
        // Row 0: head mid-board (1,1). Row 1: head at corner (0,0).
        let (map, obs) = map_and_obs(&[(1, 1), (0, 0)], 100.0);
        let p = head_neighbor_logits(&map, &obs);
        assert_eq!(p.size(), vec![2, 4]);
        let got: Vec<f32> = p.reshape([8]).try_into().unwrap();
        // Move::ALL order: Up (+y), Down (-y), Left (-x), Right (+x).
        assert_eq!(&got[..4], &[7.0, 1.0, 3.0, 5.0]);
        // Corner head: Down/Left fall off the board -> OFF_BOARD_LOGIT.
        let off = OFF_BOARD_LOGIT as f32;
        assert_eq!(&got[4..], &[103.0, off, off, 101.0]);
    }

    #[test]
    fn dead_snake_row_reads_something_finite() {
        let (map, mut obs) = map_and_obs(&[(2, 2)], 0.0);
        obs = obs.zeros_like(); // dead snake: all-zero head plane
        let p = head_neighbor_logits(&map, &obs);
        assert_eq!(p.size(), vec![1, 4]);
        let got: Vec<f32> = p.reshape([4]).try_into().unwrap();
        assert!(got.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_shapes() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let net = AZNet::new(&vs.root(), 14, 32, 2);
        init_orthogonal(&vs, 2f64.sqrt());
        let x = Tensor::zeros([6, 14, 11, 11], (Kind::Float, tch::Device::Cpu));
        let (p, v) = net.forward(&x);
        assert_eq!(p.size(), vec![6, 4]);
        assert_eq!(v.size(), vec![6]);
    }
}
