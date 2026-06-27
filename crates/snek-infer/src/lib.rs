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
    /// CUDA execution provider is unavailable).
    pub fn load(path: &str) -> ort::Result<Net> {
        let cuda = ort::execution_providers::CUDAExecutionProvider::default();
        let session = ort::session::Session::builder()?
            .with_execution_providers([cuda.build()])?
            .commit_from_file(path)?;
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
