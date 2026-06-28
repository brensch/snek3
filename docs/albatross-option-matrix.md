# Albatross option matrix

A living, honest log of what we're doing, what we've measured, and what we could
try. Update the **Status** and **Result** columns as things are tried — don't
delete failed attempts, that's the point of the file.

Last updated: 2026-06-27.

---

## The goal and the core problem

**Goal:** a snek bot that actually *beats* opponents (flood-fill baseline, the CPU
UCT agent, eventually real bots), not just plays a balanced mirror match.

**The plateau we keep hitting:** plain AlphaZero self-play converges to the
Nash/minimax equilibrium — optimal *if the opponent is also optimal*. It does
**not** learn to *exploit* a fixed/weaker opponent, so win-rate vs the baseline
stalls (~0.47) even while self-play losses keep dropping. This is exactly the gap
the Albatross paper targets (temperature-conditioned proxy + best-response model
+ opponent-temperature estimation).

---

## Decision log (chronological, honest)

| When | Change | Result |
|---|---|---|
| — | MCTS self-play, flat MC value target | Short-game collapse: 9-turn mutual-suicide draws (~58%). |
| — | **Bootstrap value target** (`--bootstrap-value`: value = search root value) | **Fixed short games.** Mean turns 9→100, win-rate vs baseline 0.18→0.47. |
| — | (same run continued) | **Overshot** into mutual-survival: ~94% 200-turn timeout draws, then **win-rate plateaued/regressed** (0.47→~0.15). Diagnosed as the Nash-can't-exploit limit. |
| — | Recorder fix: replays used value-only equilibrium search (dropped policy head) → looked weak. Switched to MCTS (both heads). | Replays now reflect real strength. |
| — | **Full Albatross build** (egocentric, symmetry, per-agent τ, proxy, response, MLE, UCT agent) | Built + unit/smoke tested. **Not yet validated at scale.** |
| — | Egocentric encoding 11×11 → 21×21 | Believed useful (positional hints); **measured 3.6× more conv FLOPs/inference** → big throughput cost. User chose to keep it. |
| — | Tried "fast" Rust ORT LE self-play to recover speed | **Did NOT help** (~190 vs ~210 samples/s). Bottleneck is search-breadth × obs-size, not torch-vs-ORT. My "6× faster" claim was wrong. |

---

## Current configuration (the active experiment)

- **Encoding:** egocentric 21×21 (head-centred), 9 channels + symmetry (D4) aug. **Kept** for positional signal (accepting ~3.6× slower inference).
- **Learning signal:** Albatross — temperature-conditioned **proxy** trained on logit-equilibrium policy+value targets at a per-generation τ∈[0.5,10]; **response** net (after gen 30) best-responds at τ_R=12 to a weak opponent at τ_opp via the heterogeneous-τ search over frozen-proxy values; **opponent-τ MLE** for test-time exploitation.
- **Search:** fixed-depth equilibrium, depth 2, 120 SFP iters.
- **Net:** 64 filters, 6 blocks, AdaptiveAvgPool (size-agnostic), temperature plane input.
- **Opponents:** eval pool = flood-fill baseline + CPU UCT. Self-play = net-vs-net (UCT-in-self-play is built via `uct_opp_frac` but **off/untested**).
- **Historical entry point:** the earlier Albatross trainer used a separate Python
  proxy-generation path; current container training uses `python -m azsnek.train`.
- **Run cadence:** ~146 s/gen at count=512; eval every 5 gens; response starts gen 30 (~70 min in).

### Measured facts (so we stop guessing)
- Inference: **~10–17k inf/sec** (egocentric 21×21). Old 11×11 run: ~101k. The 3.6× input size explains most of the drop; the rest is impl overhead.
- Throughput: **~190–210 samples/sec** (Python and Rust LE paths ≈ equal).
- GPU peak: **~7.5 GB / 16 GB** (after `expandable_segments` + batch caps). Healthy.
- CPU: **~89% idle** during self-play (search is GPU-bound; LE backup is light CPU).

---

## Option matrix

Status key: ✅ done · 🟡 built-not-validated · ⬜ not tried · ❌ tried, didn't help · ▶️ active.

### A. Learning signal (the plateau fix)
| Option | Effect | Cost/Risk | Status | Honest take |
|---|---|---|---|---|
| Bootstrap value target | fixed short games | low | ✅ | Worked, but self-play alone then plateaus. |
| Proxy (LE-at-τ targets) | temperature-conditioned equilibrium net | med | ▶️ | Trains fine; **alone does not beat the plateau** (still self-play). |
| Response + opponent-τ MLE | best-respond to/exploit weaker opponents | high | 🟡 | **The actual differentiator. Unproven in our setup.** Starts gen 30; batched eval uses a representative τ (true per-game MLE only in serving). |
| Train against opponent pool (league: baseline/UCT/past ckpts) | direct gradient to beat fixed opponents | med | ⬜ | The most *direct* anti-plateau lever; also uses idle CPU. Strongly worth trying alongside/instead of response. |

### B. Encoding
| Option | Effect | Cost/Risk | Status | Honest take |
|---|---|---|---|---|
| Egocentric 21×21 | positional/translation-invariant signal | **3.6× slower inference** | ▶️ | Kept by choice. Real cost, plausible benefit on relative positioning. |
| Absolute 11×11 | 3.6× faster | loses head-relative signal | ⬜ | The single biggest speedup if we ever want it back. |
| Symmetry (D4) aug | ~8× data, symmetry prior | ~free | ✅ | Pure win; keep regardless. |

### C. Opponents / CPU usage
| Option | Effect | Cost/Risk | Status | Honest take |
|---|---|---|---|---|
| CPU UCT agent | strong non-net opponent on idle CPU | built | ✅ | Beats baseline ~64%, ~free. Currently **eval-only**. |
| UCT as self-play opponent (`uct_opp_frac`) | trains vs strong fixed opp + uses CPU | low-med | 🟡 | Built into `generate_selfplay_le`, **untested & unwired** into training. This is "the idle-CPU idea" — still needs validating. |
| Past-checkpoint league | opponent diversity | med | ⬜ | Standard, complements response. |

### D. Search
| Option | Effect | Cost/Risk | Status | Honest take |
|---|---|---|---|---|
| Depth 2 full-width | ~81 leaves/move | baseline | ▶️ | Main inference cost driver alongside obs size. |
| Depth 1 | ~9 leaves/move (~9× cheaper) | shallower targets | ⬜ | Big speed lever if speed matters; weaker lookahead. |
| MCTS instead of full-width | fewer leaves/move, proven-fast pipeline | not Albatross-equilibrium | ⬜ | The old fast path; abandons LE targets. |

### E. Speed / implementation
| Option | Effect | Cost/Risk | Status | Honest take |
|---|---|---|---|---|
| Rust ORT LE loop | hoped ~6× | — | ❌ | **No speedup** — bottleneck is breadth×obs, not language. |
| CPU/GPU double-buffering | overlap idle GPU during build/backup | med Rust | ⬜ | Maybe 1.5–2×; only worth it if we keep this path. |
| Bigger eval_chunk / ORT tuning | better GPU utilization | OOM risk at 21×21 | ⬜ | 8192 OOM'd; 2048 safe. Tunable with headroom. |
| `expandable_segments` + batch caps | fit VRAM, no spill | — | ✅ | Done; peak ~7.5 GB. |

---

## Proven vs unproven (be honest)
- **Proven:** bootstrap fixes short games; egocentric costs 3.6×; everything compiles, smoke-tests pass, memory fits; proxy trains (losses drop, entropy stable).
- **Unproven / open:** whether the **response + MLE** actually beats the plateau in *our* setup; whether **egocentric** measurably improves play; whether **UCT-in-self-play (league)** helps — none of these have a real training-run result yet.

## How to run & what to watch
- Run: `make albatross FRESH=1 RUN_ID=<name>` · dashboard: `make dashboard`.
- **The signal that we're beating the plateau:** `response_vs_uct` / `response_vs_baseline` (start gen 30) climbing **above** `proxy_vs_*`.
- **Issue ⚠ lines:** `draw_rate`/`len_frac` (collapse either direction), `target_entropy`→0 (policy collapse), `proxy_value_loss`→0 (degenerate targets).
- Pace: ~146 s/gen → gen 30 (~70 min), meaningful read ~gen 60–100 (~2.5–4 h).

## Honest recommendation (2026-06-27)
Run it — the proxy will train and the machinery is sound — **but treat the
response/MLE as the real hypothesis under test, not a sure thing.** In parallel I
think the **league/opponent-pool** lever (A row 4 / C row 2 — your CPU UCT idea
in self-play) is the most *direct* attack on the plateau and the best use of the
idle CPU, and it's the next thing I'd validate if the response alone underwhelms.
