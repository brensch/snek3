//! Structural + init parity check vs the PyTorch reference.
use snek_tch::{init_orthogonal, AZNet};
use tch::{nn, Device, Kind, Tensor};

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let net = AZNet::new(&vs.root(), 14, 96, 8, 3);
    init_orthogonal(&vs, 2f64.sqrt());

    let mut vars: Vec<(String, Vec<i64>)> = vs
        .variables()
        .into_iter()
        .map(|(n, t)| (n, t.size()))
        .collect();
    vars.sort();
    let mut total = 0i64;
    let (mut convs, mut linears, mut norms) = (0, 0, 0);
    for (_n, size) in &vars {
        total += size.iter().product::<i64>();
        match size.len() {
            4 => convs += 1,
            2 => linears += 1,
            1 => norms += 1,
            _ => {}
        }
    }
    println!("TOTAL_PARAMS {total}  NUM_TENSORS {}", vars.len());
    println!("conv4d={convs} linear2d={linears} norm1d={norms}");
    println!(
        "stem.weight shape = {:?}",
        vs.variables()["stem.weight"].size()
    );

    // Orthogonality: for stem [96,126] (rows<cols) the rows are orthonormal*gain,
    // so W Wᵀ ≈ gain² I. Check max off-diagonal and mean diagonal.
    let w = vs.variables()["stem.weight"]
        .shallow_clone()
        .reshape([96, 126]);
    let g = w.matmul(&w.transpose(0, 1)); // [96,96]
    let eye = Tensor::eye(96, (Kind::Float, Device::Cpu));
    let diag_mean = (&g * &eye).sum(Kind::Float).double_value(&[]) / 96.0;
    let offdiag_max = (&g - &(&g * &eye)).abs().max().double_value(&[]);
    println!(
        "stem WWᵀ: diag_mean={diag_mean:.4} (expect gain²={:.4})  offdiag_max={offdiag_max:.2e}",
        2.0
    );

    // Forward must run and give [B,4],[B].
    let x = Tensor::zeros([3, 14, 11, 11], (Kind::Float, Device::Cpu));
    let (p, v) = net.forward(&x);
    println!("forward: policy={:?} value={:?}", p.size(), v.size());
}
