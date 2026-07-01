//! CUDA-graph capture/replay for the fixed-shape AZNet forward.
//!
//! For a net this small the ~50 per-op kernel launches of one forward are a
//! large slice of the wall time. Because the engine pins every forward to one
//! constant batch shape reusing fixed buffers, we can *capture* that forward as
//! a CUDA graph once and *replay* it as a single launch — same math, none of the
//! per-launch CPU/driver overhead.
//!
//! tch doesn't expose `at::cuda::CUDAGraph`, so this goes through the tiny C ABI
//! in `csrc/graph_shim.cpp`. Everything runs on one dedicated (non-default) CUDA
//! stream, which capture requires.
//!
//! Usage:
//! ```ignore
//! let g = GraphedForward::capture(&net, device, rows, c, h, w);
//! g.run(&obs, &mut pol, &mut val); // repeat; obs -> pol/val, one graph launch
//! ```

use crate::AZNet;
use std::os::raw::c_void;
use tch::{Device, Kind, Tensor};

extern "C" {
    fn snek_stream_new() -> *mut c_void;
    fn snek_stream_set_current(s: *mut c_void);
    fn snek_stream_sync(s: *mut c_void);
    fn snek_stream_free(s: *mut c_void);
    fn snek_graph_new() -> *mut c_void;
    fn snek_graph_capture_begin(g: *mut c_void);
    fn snek_graph_capture_end(g: *mut c_void);
    fn snek_graph_replay(g: *mut c_void);
    fn snek_graph_free(g: *mut c_void);
}

/// A captured forward+softmax for a fixed `(rows, c, h, w)` input. Not `Send`:
/// it pins a CUDA stream as current on its owning thread.
pub struct GraphedForward {
    stream: *mut c_void,
    graph: *mut c_void,
    input: Tensor, // fixed-address input the graph reads
    probs: Tensor, // fixed-address graph output (softmaxed policy)
    value: Tensor, // fixed-address graph output
    rows: i64,
    c: i64,
    h: i64,
    w: i64,
}

impl GraphedForward {
    /// Warm up and capture `net`'s forward+softmax at the given fixed shape. All
    /// work happens on a dedicated side stream (a capture requirement); that
    /// stream stays current on this thread for subsequent [`run`](Self::run)s.
    pub fn capture(net: &AZNet, device: Device, rows: i64, c: i64, h: i64, w: i64) -> Self {
        unsafe {
            let stream = snek_stream_new();
            snek_stream_set_current(stream);

            let input = Tensor::zeros([rows, c, h, w], (Kind::Float, device));

            // Warm up: let cuDNN pick algorithms and the caching allocator settle
            // before the region we capture.
            tch::no_grad(|| {
                for _ in 0..8 {
                    let (logits, value) = net.forward(&input);
                    let _p = logits.softmax(-1, Kind::Float);
                    let _keep = (&_p, &value);
                }
            });
            snek_stream_sync(stream);

            let graph = snek_graph_new();
            snek_graph_capture_begin(graph);
            let (probs, value) = tch::no_grad(|| {
                let (logits, value) = net.forward(&input);
                (logits.softmax(-1, Kind::Float), value)
            });
            snek_graph_capture_end(graph);
            snek_stream_sync(stream);

            GraphedForward {
                stream,
                graph,
                input,
                probs,
                value,
                rows,
                c,
                h,
                w,
            }
        }
    }

    /// Copy `obs` into the static input, replay the captured graph, and read the
    /// policy (softmaxed, `rows*4`) and value (`rows`) back into the host slices.
    pub fn run(&mut self, obs: &[f32], pol: &mut [f32], val: &mut [f32]) {
        let src = Tensor::from_slice(obs).reshape([self.rows, self.c, self.h, self.w]);
        // H2D into the exact address the graph was captured against.
        self.input.copy_(&src);
        unsafe {
            snek_graph_replay(self.graph);
            snek_stream_sync(self.stream);
        }
        let p = self.probs.to_device(Device::Cpu);
        let v = self.value.to_device(Device::Cpu);
        p.copy_data(pol, (self.rows * 4) as usize);
        v.copy_data(val, self.rows as usize);
    }

    pub fn rows(&self) -> i64 {
        self.rows
    }
}

impl Drop for GraphedForward {
    fn drop(&mut self) {
        // Free the graph first (releases its private mempool refs), then the
        // stream; the captured output tensors drop with the struct afterwards.
        unsafe {
            snek_graph_free(self.graph);
            snek_stream_free(self.stream);
        }
    }
}
