use tch::{nn, no_grad, Device, Kind, Tensor};

pub struct Net {
    _vs: nn::VarStore,
    net: snek_tch::AZNet,
    device: Device,
}

impl Net {
    /// Load with device and trunk shape from the environment (`SNEK_CPU_ONLY`,
    /// `SNEK_TRUNK_CHANNELS`, `SNEK_TRUNK_BLOCKS`) — the standalone serving
    /// path. In-process callers that know the run's config use [`Net::load_on`].
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let device = if std::env::var("SNEK_CPU_ONLY").ok().as_deref() == Some("1") {
            Device::Cpu
        } else if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        Self::load_on(
            path,
            device,
            env_or("SNEK_TRUNK_CHANNELS", 96i64),
            env_or("SNEK_TRUNK_BLOCKS", 8i64),
        )
    }

    /// Load onto an explicit device with an explicit trunk shape (which must
    /// match how the checkpoint was trained).
    pub fn load_on(
        path: &str,
        device: Device,
        trunk_channels: i64,
        trunk_blocks: i64,
    ) -> anyhow::Result<Self> {
        let mut vs = nn::VarStore::new(device);
        let net = snek_tch::AZNet::new(
            &vs.root(),
            snek_core::NUM_CHANNELS as i64,
            trunk_channels,
            trunk_blocks,
        );
        vs.load(path)?;
        Ok(Self {
            _vs: vs,
            net,
            device,
        })
    }

    pub fn forward(
        &mut self,
        obs: &[f32],
        rows: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let x = Tensor::from_slice(obs)
            .reshape([rows as i64, c as i64, h as i64, w as i64])
            .to_device(self.device);
        let (logits, value) = no_grad(|| self.net.forward(&x));
        let probs = logits.softmax(-1, Kind::Float).to_device(Device::Cpu);
        let value = value.to_device(Device::Cpu);
        let mut pol = vec![0.0f32; rows * 4];
        let mut val = vec![0.0f32; rows];
        let pol_len = pol.len();
        let val_len = val.len();
        probs.copy_data(&mut pol, pol_len);
        value.copy_data(&mut val, val_len);
        Ok((pol, val))
    }
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
