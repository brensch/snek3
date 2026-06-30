use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Samples {
    pub obs: Vec<f32>,
    pub pol: Vec<f32>,
    pub z: Vec<f32>,
    pub obs_shape: [usize; 3],
    pub turns: usize,
    pub games: usize,
}

impl Samples {
    pub fn len(&self) -> usize {
        self.z.len()
    }

    pub fn obs_len(&self) -> usize {
        self.obs_shape.iter().product()
    }
}

#[derive(Default)]
pub struct ReplayBuffer {
    capacity: usize,
    shards: VecDeque<Samples>,
    len: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ..Self::default()
        }
    }

    pub fn add(&mut self, samples: Samples) {
        self.len += samples.len();
        self.shards.push_back(samples);
        while self.len > self.capacity && self.shards.len() > 1 {
            if let Some(old) = self.shards.pop_front() {
                self.len -= old.len();
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn sample_batch<R: Rng>(&self, batch: usize, recency: f64, rng: &mut R) -> Option<Samples> {
        let first = self.shards.front()?;
        let obs_len = first.obs_len();
        let mut obs = Vec::with_capacity(batch * obs_len);
        let mut pol = Vec::with_capacity(batch * 4);
        let mut z = Vec::with_capacity(batch);
        for _ in 0..batch {
            let idx = self.draw_index(recency, rng);
            let (s, local) = self.locate(idx)?;
            obs.extend_from_slice(&s.obs[local * obs_len..(local + 1) * obs_len]);
            pol.extend_from_slice(&s.pol[local * 4..(local + 1) * 4]);
            z.push(s.z[local]);
        }
        apply_random_d4(&mut obs, &mut pol, batch, first.obs_shape, rng);
        Some(Samples {
            obs,
            pol,
            z,
            obs_shape: first.obs_shape,
            turns: 0,
            games: 0,
        })
    }

    pub fn save_shard(&self, dir: &Path, gen: u32, samples: &Samples) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        let final_path = dir.join(format!("gen_{gen:06}_n{}.json", samples.len()));
        let tmp = final_path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(samples)?)?;
        std::fs::rename(tmp, final_path)?;
        Ok(())
    }

    pub fn restore(dir: &Path, capacity: usize) -> anyhow::Result<Self> {
        let mut out = Self::new(capacity);
        if !dir.exists() {
            return Ok(out);
        }
        let mut files = std::fs::read_dir(dir)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect::<Vec<_>>();
        files.sort();
        for path in files {
            let samples: Samples = serde_json::from_slice(&std::fs::read(path)?)?;
            out.add(samples);
        }
        Ok(out)
    }

    fn draw_index<R: Rng>(&self, recency: f64, rng: &mut R) -> usize {
        let u = rng.gen::<f64>().powf(recency.max(0.0001));
        let from_end = (u * self.len as f64).floor() as usize;
        self.len
            .saturating_sub(1)
            .saturating_sub(from_end.min(self.len.saturating_sub(1)))
    }

    fn locate(&self, mut idx: usize) -> Option<(&Samples, usize)> {
        for s in &self.shards {
            if idx < s.len() {
                return Some((s, idx));
            }
            idx -= s.len();
        }
        None
    }
}

fn apply_random_d4<R: Rng>(
    obs: &mut [f32],
    pol: &mut [f32],
    batch: usize,
    shape: [usize; 3],
    rng: &mut R,
) {
    let [c, h, w] = shape;
    if h != w {
        return;
    }
    let transform = rng.gen_range(0..8);
    let obs_len = c * h * w;
    let src = obs.to_vec();
    for b in 0..batch {
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let (yy, xx) = map_d4(transform, y, x, h);
                    obs[b * obs_len + ch * h * w + yy * w + xx] =
                        src[b * obs_len + ch * h * w + y * w + x];
                }
            }
        }
        let old = [pol[b * 4], pol[b * 4 + 1], pol[b * 4 + 2], pol[b * 4 + 3]];
        let perm = move_perm(transform);
        for m in 0..4 {
            pol[b * 4 + perm[m]] = old[m];
        }
    }
}

fn map_d4(t: usize, y: usize, x: usize, n: usize) -> (usize, usize) {
    let (mut y, mut x) = (y, x);
    if t & 1 != 0 {
        y = n - 1 - y;
    }
    if t & 2 != 0 {
        x = n - 1 - x;
    }
    if t & 4 != 0 {
        std::mem::swap(&mut y, &mut x);
    }
    (y, x)
}

fn move_perm(t: usize) -> [usize; 4] {
    const OFF: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, -1), (0, 1)];
    let mut out = [0; 4];
    for (i, (mut y, mut x)) in OFF.into_iter().enumerate() {
        if t & 1 != 0 {
            y = -y;
        }
        if t & 2 != 0 {
            x = -x;
        }
        if t & 4 != 0 {
            std::mem::swap(&mut y, &mut x);
        }
        out[i] = OFF.iter().position(|&o| o == (y, x)).unwrap();
    }
    out
}
