# LE self-play forward throughput

## The finding

A synthetic sweep of the value net's forward throughput (11×11, trunk 96/8) on
this GPU shows it **peaks at a small batch and falls off at large batches** —
the opposite of the usual "bigger batch = faster":

| batch (rows) | forward-only | with H2D/D2H |
|---:|---:|---:|
| 256 | 63.4k | 61.1k |
| **512** | **67.8k (peak)** | **65.7k** |
| 1024 | 56.3k | 55.3k |
| 2048 | 51.2k | 49.6k |
| 4096 | 51.4k | 49.3k |

(cuDNN benchmark ON at 512 ≈ 71k = the AZ ceiling; the τ channel costs ~nothing.)
Resident (no-copy) mode drops the same way, so it's the **forward** that's slower
per-row at large batch (the small net saturates the GPU at ~512 rows; beyond that
it's memory/occupancy-bound), not the PCIe copy (~3-5%).

LE self-play batches *all* live games into one ~3,600-row forward per turn → it
runs on the ~49k tail, ~25% below the ~66k peak. Its per-turn batch also varies
(candidate pruning, dying snakes), so throughput wanders across the steep part of
the curve → the "blips".

## The fix: `le_fwd_chunk`

Split each turn's leaf-value forward into fixed `le_fwd_chunk`-row slices near the
sweet spot (512). Key points for it to actually pay off:

- **No per-chunk sync.** One H2D of the whole (padded) batch, then every slice
  forward is *enqueued* on the CUDA stream with no `D2H`/sync between them (value
  tensors stay resident), then ONE `cat` + ONE D2H. The GPU pipelines the slices
  back-to-back with no host stalls. (A first attempt synced per chunk — 7
  syncs/turn — and was *slower* than the baseline; the sync stalls ate the
  small-batch win.)
- **Static shape.** The batch is zero-padded to a whole number of chunks so every
  forward is the identical `[le_fwd_chunk, C, H, W]` shape → cuDNN **benchmark
  back ON** (autotunes once), +~9%. Padding rows are ignored on scatter.
- **Minimal copies + parallel encode.** Encode is parallel (`par_chunks_mut`);
  the forward path is one H2D + one D2H, no per-chunk copies.

## Measured

`le_fwd_chunk=512` (chunk = 128 boards × 4 seats), le-6, gen ~1340:

```
baseline plain:            ~45k inf/s
threaded per-chunk-sync:   ~43k inf/s  (rejected)
no-sync sliced (this):     ~57.7k inf/s   → +28% end-to-end
```

Below the raw ~45% the pure bench suggests because self-play also spends ~7% of
the play phase on encode + SFP backup + step (unchanged). Health (entropy,
avg-turn, win-rate) is identical — the values are the same, just computed in
pipelined slices.

## Notes / follow-ups

- `le_fwd_chunk=0` keeps the old single-forward path (and benchmark off).
- The sweet spot (512) is GPU-specific; re-sweep `crates/snek-tch/examples/bench_tch`
  (`SNEK_BENCH_INCH=15 SNEK_CUDNN_BENCH=1 bench_tch 256,384,512,768,1024`) on new
  hardware.
- This is a **GPU self-play** win only. The CPU serve device has no saturation
  cliff and serves one game at a time, so chunking there is moot.
- A further ~2% is available by overlapping the parallel encode with the forward
  (producer/consumer), and a CUDA graph over the fixed slice could add more — not
  worth the complexity at current gen times.
