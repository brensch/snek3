//! The shared immutable net guarded by a mutex — the mutex *is* the inference
//! queue: the instant one worker unlocks, another that has finished its CPU phase
//! is already blocked on it and takes over, so the GPU never idles.

use tch::{no_grad, Device, Kind, Tensor};

pub(super) struct Gpu {
    pub(super) net: *const snek_tch::AZNet,
    pub(super) device: Device,
    pub(super) c: i64,
    pub(super) h: i64,
    pub(super) w: i64,
}
// Safe: every access is serialised by the GPU mutex, and the scoped workers all
// join before `generate` returns, so this borrow of the net cannot dangle.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Gpu {
    pub(super) fn forward(&self, obs: &[f32], rows: usize, pol: &mut [f32], val: &mut [f32]) {
        let net = unsafe { &*self.net };
        no_grad(|| {
            let x = Tensor::from_slice(obs)
                .reshape([rows as i64, self.c, self.h, self.w])
                .to_device(self.device);
            let (logits, value) = net.forward(&x);
            let probs = logits.softmax(-1, Kind::Float).to_device(Device::Cpu);
            let value = value.to_device(Device::Cpu);
            probs.copy_data(pol, rows * 4);
            value.copy_data(val, rows);
        });
    }
}
