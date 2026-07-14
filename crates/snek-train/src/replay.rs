use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Samples {
    pub obs: Vec<f32>,
    pub pol: Vec<f32>,
    pub z: Vec<f32>,
    /// Absolute game turn per sample (parallel to `z`). Defaulted for old shards.
    #[serde(default)]
    pub turn: Vec<u32>,
    /// Per-sample inverse-temperature τ (parallel to `z`), for the
    /// Logit-Equilibrium mode's τ-conditioned net. Empty in the AZ path.
    #[serde(default)]
    pub temp: Vec<f32>,
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
        let mut temp = Vec::with_capacity(batch);
        for _ in 0..batch {
            let idx = self.draw_index(recency, rng);
            let (s, local) = self.locate(idx)?;
            obs.extend_from_slice(&s.obs[local * obs_len..(local + 1) * obs_len]);
            pol.extend_from_slice(&s.pol[local * 4..(local + 1) * 4]);
            z.push(s.z[local]);
            if !s.temp.is_empty() {
                temp.push(s.temp[local]);
            }
        }
        apply_random_d4(&mut obs, &mut pol, batch, first.obs_shape, rng);
        Some(Samples {
            obs,
            pol,
            z,
            turn: Vec::new(),
            temp,
            obs_shape: first.obs_shape,
            turns: 0,
            games: 0,
        })
    }

    /// Average absolute game turn across every sample currently in the buffer.
    /// Indicates how deep into games the buffered positions are.
    pub fn avg_turn(&self) -> f64 {
        let mut sum = 0u64;
        let mut count = 0u64;
        for shard in &self.shards {
            for &t in &shard.turn {
                sum += t as u64;
                count += 1;
            }
        }
        if count > 0 {
            sum as f64 / count as f64
        } else {
            0.0
        }
    }

    pub fn save_shard(&self, dir: &Path, gen: u32, samples: &Samples) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        // Drop any earlier shard for this generation first: a resumed run can
        // regenerate a gen with a different sample count (hence a different
        // filename), and leaving the stale file would double-count it on the next
        // restore.
        remove_shards_for_gen(dir, gen);
        let final_path = dir.join(format!("gen_{gen:06}_n{}.json.zst", samples.len()));
        let tmp = final_path.with_extension("tmp");
        std::fs::write(&tmp, zstd::encode_all(&*serde_json::to_vec(samples)?, 3)?)?;
        std::fs::rename(tmp, final_path)?;
        self.prune_evicted(dir);
        Ok(())
    }

    /// Delete on-disk shards that have slid out of the sample window, mirroring
    /// the in-memory eviction rule in `add`: drop from the oldest end while the
    /// total exceeds capacity and more than one shard remains.
    fn prune_evicted(&self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut shards: Vec<(u32, usize, std::path::PathBuf)> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter_map(|p| shard_meta(&p).map(|(gen, n)| (gen, n, p)))
            .collect();
        shards.sort();
        let mut total: usize = shards.iter().map(|(_, n, _)| n).sum();
        let mut evict = 0;
        while total > self.capacity && shards.len() - evict > 1 {
            total -= shards[evict].1;
            let _ = std::fs::remove_file(&shards[evict].2);
            evict += 1;
        }
    }

    /// Rebuild the buffer from on-disk shards, keeping only committed generations.
    /// `up_to_gen` is the resume generation (`trainer_state.generation`): a shard
    /// for that gen or later was written mid-generation before the trainer state
    /// advanced, so the run is about to regenerate it — loading it here would
    /// double-count a generation's worth of samples into the buffer.
    pub fn restore(dir: &Path, capacity: usize, up_to_gen: u32) -> anyhow::Result<Self> {
        let mut out = Self::new(capacity);
        if !dir.exists() {
            return Ok(out);
        }
        let mut files = std::fs::read_dir(dir)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("zst"))
            .collect::<Vec<_>>();
        files.sort();
        for path in files {
            if shard_gen(&path).is_some_and(|gen| gen >= up_to_gen) {
                continue;
            }
            let samples: Samples =
                serde_json::from_slice(&zstd::decode_all(&*std::fs::read(path)?)?)?;
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

/// Parse a `gen_{gen:06}_n{len}.json.zst` shard path into (generation, sample count).
fn shard_meta(path: &Path) -> Option<(u32, usize)> {
    let name = path.file_name()?.to_str()?;
    let rest = name.strip_prefix("gen_")?.strip_suffix(".json.zst")?;
    let (gen, len) = rest.split_once("_n")?;
    Some((gen.parse().ok()?, len.parse().ok()?))
}

fn shard_gen(path: &Path) -> Option<u32> {
    shard_meta(path).map(|(gen, _)| gen)
}

/// Delete every shard file for a given generation (there is normally at most one,
/// but a differing sample count changes the filename).
fn remove_shards_for_gen(dir: &Path, gen: u32) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for path in entries.filter_map(Result::ok).map(|e| e.path()) {
        if shard_gen(&path) == Some(gen) {
            let _ = std::fs::remove_file(path);
        }
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
    let obs_len = c * h * w;
    let src = obs.to_vec();
    for b in 0..batch {
        // One independent transform per sample — a single draw for the whole
        // batch would give every step just one orientation of the board.
        let transform = rng.gen_range(0..8);
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
        for (m, &dst) in perm.iter().enumerate() {
            pol[b * 4 + dst] = old[m];
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

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(n: usize, turn0: u32) -> Samples {
        Samples {
            obs: vec![0.0; n * 4],
            pol: vec![0.25; n * 4],
            z: vec![0.5; n],
            turn: (turn0..turn0 + n as u32).collect(),
            temp: Vec::new(),
            obs_shape: [1, 2, 2],
            turns: n,
            games: 1,
        }
    }

    #[test]
    fn save_restore_prune_roundtrip() {
        let dir = std::env::temp_dir().join(format!("snek-replay-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Capacity 25 keeps at most two 10-sample shards; older ones must be
        // pruned from disk as they are evicted.
        let buf = ReplayBuffer::new(25);
        for gen in 0..4 {
            buf.save_shard(&dir, gen, &samples(10, gen * 100)).unwrap();
        }
        let names: Vec<String> = {
            let mut v: Vec<String> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect();
            v.sort();
            v
        };
        assert_eq!(
            names,
            ["gen_000002_n10.json.zst", "gen_000003_n10.json.zst"]
        );
        // Resuming at gen 3 skips the in-flight gen-3 shard, leaving only gen 2.
        let restored = ReplayBuffer::restore(&dir, 15, 3).unwrap();
        assert_eq!(restored.len(), 10);
        assert_eq!(restored.shards.front().unwrap().turn[0], 200);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
