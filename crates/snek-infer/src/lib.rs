//! Thin ONNX-Runtime (via `ort`) inference wrapper for the AlphaZero net.
//!
//! Loads an exported AZNet ONNX model and runs batched forward passes on the GPU
//! (CUDA execution provider, falling back to CPU). Used by both Rust self-play
//! generation and the Rust `/move` API server — no PyTorch / Python in the loop.

use ndarray::{ArrayView1, ArrayView4};

pub struct Net {
    session: ort::session::Session,
}

impl Net {
    /// Load an ONNX model, preferring CUDA (silently falls back to CPU if the
    /// CUDA execution provider is unavailable). Set `SNEK_CPU_ONLY=1` to force the
    /// CPU execution provider — used by the offline evaluator so it runs on
    /// otherwise-idle CPU (the deployment target) instead of stealing the GPU
    /// that the trainer is using.
    pub fn load(path: &str) -> ort::Result<Net> {
        let cpu_only = std::env::var("SNEK_CPU_ONLY")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        let builder = ort::session::Session::builder()?;
        let builder = if cpu_only {
            let cpu = ort::execution_providers::CPUExecutionProvider::default();
            // The evaluator runs many worker threads, each with its own `Net`. If
            // every session also spawned an intra-op thread pool sized to the core
            // count we'd massively oversubscribe the CPU (every core pinned at
            // 100%). Default each CPU session to a single intra-op thread so the
            // parallelism comes purely from the worker count; override with
            // SNEK_ORT_THREADS if a session should fan out internally.
            let threads = std::env::var("SNEK_ORT_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(1);
            builder
                .with_intra_threads(threads)?
                .with_execution_providers([cpu.build()])?
        } else {
            let cuda = ort::execution_providers::CUDAExecutionProvider::default();
            builder.with_execution_providers([cuda.build()])?
        };
        let session = builder.commit_from_file(path)?;
        Ok(Net { session })
    }

    /// Forward pass over `n` observations laid out `[n, c, h, w]` row-major.
    /// Returns `(policy_probs [n*4], value [n])`; the policy logits are softmaxed
    /// into per-row probabilities (what the MCTS uses as priors).
    pub fn forward(
        &mut self,
        obs: &[f32],
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> ort::Result<(Vec<f32>, Vec<f32>)> {
        self.forward_temp(obs, None, n, c, h, w)
    }

    /// Like [`forward`], but optionally supplies a per-observation `temp` slice
    /// (length `n`) for temperature-conditioned (Albatross) ONNX models that
    /// were exported with a second `temp` input.
    pub fn forward_temp(
        &mut self,
        obs: &[f32],
        temp: Option<&[f32]>,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> ort::Result<(Vec<f32>, Vec<f32>)> {
        let arr = ArrayView4::from_shape((n, c, h, w), obs)
            .expect("obs length matches n*c*h*w")
            .to_owned();
        let outputs = match temp {
            Some(t) => {
                let tarr = ArrayView1::from_shape((n,), t)
                    .expect("temp length matches n")
                    .to_owned();
                self.session.run(ort::inputs![
                    "obs" => ort::value::Tensor::from_array(arr)?,
                    "temp" => ort::value::Tensor::from_array(tarr)?,
                ])?
            }
            None => self
                .session
                .run(ort::inputs!["obs" => ort::value::Tensor::from_array(arr)?])?,
        };

        let (_pshape, plogits) = outputs["policy_logits"].try_extract_tensor::<f32>()?;
        let (_vshape, vdata) = outputs["value"].try_extract_tensor::<f32>()?;
        let value: Vec<f32> = vdata.to_vec();

        // Softmax each row of 4 logits -> probabilities.
        let mut pol: Vec<f32> = plogits.to_vec();
        for row in pol.chunks_mut(4) {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0f32;
            for x in row.iter_mut() {
                *x = (*x - m).exp();
                s += *x;
            }
            if s > 0.0 {
                for x in row.iter_mut() {
                    *x /= s;
                }
            }
        }
        Ok((pol, value))
    }
}
