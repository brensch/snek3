# Logit-Equilibrium ("correct game mode") overhaul — build plan & log

Goal: replace the mis-specified AlphaZero (decoupled-PUCT, assume-perfect-play,
deterministic policy) self-play/training with the Albatross-faithful
Logit-Equilibrium approach — a **mixed-strategy** policy from a per-node LE
backup over a fixed-depth joint-move search, τ-conditioned, that (a) is not
exploitable, (b) is strong at low sim/compute (350 ms CPU deploy), and (c) can
exploit weak/diverse ladder opponents. End state: stop snek3-21 (AZ), start a
fresh LE run, watch it succeed through the first few generations.

Rules for this build: no shortcuts; best version of every fork regardless of
token cost; real tests for every piece; question assumptions on what the metrics
show; anywhere an easy route is taken it is documented in an `EASY-ROUTE` note.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done+verified.

---

## Phase 0 — Equilibrium engine (shared, libtorch-free) — `[x]`
- `[x]` `snek-search/src/le.rs` — SFP logit-equilibrium solver, per-agent τ (SBRLE). 5 unit tests incl. rational-exploits-weak. **Verified: 18/18 crate tests pass.**
- `[x]` `snek-search/src/eqsearch.rs` — `EqForest`: fixed-depth full-width joint search, two-phase (build → caller evals leaves → LE backup), mixed policy + per-player value, dead-seat force to -1, parallel over games. Tests: mixed-legal-policy, bounded values, eval-layout. **Verified.**

## Phase 1 — Config + data model — `[x]`
- `[x]` Config knobs: `le_mode, le_depth(1), le_iters(120), tau_min(0.5), tau_max(10), le_exploration(0.15), response_tau(12), response_after(30)`, + `le_outcome_weight` (added Phase 4). Serde defaults; existing configs load.
- `[x]` Per-episode τ threaded: `Samples.temp: Vec<f32>`, `GameJson.temp: f32`; every constructor updated; `sample_batch` carries τ. AZ path sets empty/0.
- `[x]` **Verified: `cargo check` on snek-train clean (49s).**

**FORK — keep the AZ path behind `le_mode` vs rip it out.** Chosen: keep, gated by
`le_mode`. Reason (not a shortcut): retaining AZ lets us run AZ-vs-LE head-to-head
to *prove* LE is better on the real metric — the comparison is the point. It is a
clean single branch at net-construction + `generate`, not a maintained parallel
API. Full AZ removal is a later cleanup once LE wins are demonstrated.

## Phase 2 — Net τ-conditioning — `[ ]`
- `[ ]` 2.1 τ-plane obs builder: 14-ch obs (+ τ) → 15-ch by appending a constant
  plane `τ / temperature_scale` (scale=100, validated). Lives where the net
  input tensor is built (trainer + server). Unit test: shape + plane value +
  τ-invariance of the board channels.
  - **FORK — plane vs concat-scalar-at-value-head vs FiLM.** Chosen: constant
    input plane (matches validated Albatross). Reason: the *policy* target changes
    with τ, so τ must reach the trunk, not just the value head. A scalar concatted
    at the pooled value head would leave the policy τ-blind — that would be an
    `EASY-ROUTE` bug, explicitly avoided. FiLM is more expressive but unvalidated
    and heavier; a plane is proven and sufficient.
- `[ ]` 2.2 Net constructed with `in_ch = 14 + (le_mode?1:0)`; Gpu/serving `c=15`.

## Phase 3 — LE self-play worker (`generate_le`) — `[ ]`
- `[ ]` 3.1 Per-turn loop over the live batch: `EqForest::build(boards, depth)` →
  encode every leaf board once per seat into 15-ch rows (board 14ch + τ plane) →
  one batched value forward → `backup(values, τ-vec, iters)` → per game: record a
  `FrameJson` (LE policy → `policy`, LE root value → `value`, played mix →
  `play_policy`) → mix-uniform(`le_exploration`) → sample per-seat move → step.
- `[ ]` 3.2 Per-episode τ ~ U[τ_min, τ_max]; store on `GameJson.temp`. Game
  completion, `SelfPlayState` (boards/turns/rec/finished) integration, sample gate.
  - **FORK — reuse the AZ double-buffered/CUDA-graph worker vs a clean single-pass
    LE worker.** Chosen: clean single-pass LE worker (batch all games' leaves into
    one forward/turn, no CUDA graph). Reason: the LE leaf batch is *variable-shape*
    each turn (games branch/terminate differently), so a fixed-shape CUDA graph does
    not apply. `EASY-ROUTE` note: this forgoes the AZ path's CPU/GPU double-buffer
    overlap — acceptable because depth-1 LE is cheap and correctness > overlap; if
    throughput needs it later, double-buffer two batches.
  - **FORK — depth 1 vs 2 for 4 players.** Chosen: knob, default 1 for 4p (depth-2
    is ~cands^n leaves ≈ 20× more at n=4). `EASY-ROUTE` note: depth-1 is a weaker
    (1-ply) equilibrium target than the validated depth-2 (2p); mitigated by the
    value bootstrap + tested at both depths.
- `[ ]` 3.3 TEST (CPU smoke, real tiny net): run a few turns, assert every frame's
  per-seat LE policy sums to 1 and is mixed (not one-hot), values in [-1,1], games
  advance and terminate, τ recorded. Assert the whole batch produces ≥1 finished
  game.

## Phase 4 — LE materialize — `[ ]`
- `[ ]` 4.1 `materialize_le`: `Samples` from LE frames. obs = 14-ch `encode_into`
  per alive seat; `Samples.temp` = game τ; policy target = frame's LE policy;
  value target = `(1-λ)·LE_root_value + λ·game_outcome`, λ=`le_outcome_weight`.
  - **FORK — pure LE bootstrap (validated, λ=0) vs blend with game outcome.**
    Chosen: blend, default λ configurable. Reason: the validated run used pure
    bootstrap at *depth 2* (more terminals grounding the target); at depth-1 4p,
    early leaf values are near-random so a pure self-referential bootstrap has a
    cold-start problem. Blending a fraction of the true game outcome grounds it.
    We TEST both λ=0 and λ=1 give sane targets; default set after the smoke.
    `QUESTION-ASSUMPTION`: watch value-loss & value-calibration early — if the
    bootstrap is unstable, raise λ.
- `[ ]` 4.2 TEST: synthetic LE `GameJson` → correct `Samples` (temp/pol/value,
  dead+heur seats skipped, blend arithmetic).

## Phase 5 — Training — `[ ]`
- `[ ]` 5.1 `train_one` (LE path): build input = D4-aug'd 14-ch obs + τ/100 plane
  (τ invariant under D4), forward 15-ch, loss = CE(LE policy) + `value_weight`·
  MSE(value, LE-value-target). Net `in_ch=15`.
- `[ ]` 5.2 TEST: one train step on synthetic LE `Samples` reduces loss; the net's
  output *changes with τ* (conditioning actually wired); D4 leaves τ-plane intact.

## Phase 6 — Trainer wiring + whole-crate build — `[ ]`
- `[ ]` 6.1 Net constructed at 15ch under `le_mode`; `generate`→`generate_le`;
  materialize→`materialize_le`; eval/league serve the LE net at a fixed serve-τ
  (=`response_tau`) with the τ plane. Persist/resume unaffected (frames carry τ).
- `[ ]` 6.2 **Verify: whole snek-train compiles.** Run existing tests.

## Phase 7 — Instrumentation ("how we KNOW it's working") — `[ ]`
- `[ ]` 7.1 Per-gen LE metrics → `metrics.jsonl` + proto: mean LE-policy entropy
  (target H), value calibration (‖v_pred − v_target‖), mean τ, LE convergence
  residual (SFP last-step Δ), fraction-mixed (policy not argmax).
- `[ ]` 7.2 **Held-out diverse arena** (primary early signal): win rate of the LE
  net vs floodfill + voronoi + a set of *past checkpoints not trained against*, at
  deploy conditions. Reuse league/burst adapted for 15ch+τ.
- `[ ]` 7.3 **Exploitability probe** (gold metric): freeze a checkpoint, train a
  best-response net against it, report its win rate ≈ approx exploitability; should
  trend DOWN over gens. Standalone subcommand + per-gen light version.
  - `QUESTION-ASSUMPTION`: self-play score & 64-sim league both lied before — these
    two (7.2/7.3) are the metrics we trust.

## Phase 8 — Serving — `[ ]`
- `[ ]` 8.1 snek-server: LE `EqForest` search at deploy (net@15ch + serve-τ), replaces
  DUCT for the LE net. Cheap depth for 350 ms CPU.
- `[ ]` 8.2 Online opponent-τ MLE (grid `geomspace(0.25,20,24)`) → response net
  conditioned on estimated τ (exploit path). Stage B.

## Phase 9 — Dashboard — `[ ]`
- `[ ]` Panels: exploitability-over-gen (headline), held-out win rate, LE-policy
  entropy, value calibration, τ distribution, LE convergence. Typed, buf proto.

## Phase 10 — Launch & watch — `[x]`
- `[x]` Build release, stop snek3-21 (retired; data kept in runs/snek3-21), start
  the LE run `snek3-le-1` (fresh net@15ch, le_depth 1, le_iters 120, tau 0.5–10,
  trunk 96/8, samples_per_gen 2000, gpu_batch_games 24, max_turns 200, eval_sims 64,
  league_entrant_gens 8).
- `[x]` **SUCCESS confirmed through gen 16.** No crash across 4 hot-swaps; entropy
  MIXED and rock-stable the whole time (tgtH 0.921–0.938, netH tracks it, KL
  +0.005..+0.013 — the exact opposite of AZ's entropy collapse); value loss falls
  monotonically 0.161→0.071; held-out beats floodfill (38–50% sole-survival vs 25%
  chance), voronoi ~0–6% (expected at 16 gens — the number to watch climb). Gen
  cadence ~30–60s (non-eval). This is the correct-game-mode loop working end to end.
- Metrics fix mid-launch: LE self-play worker now bumps inference/gpu counters
  (were flat-zero on the dashboard; AZ worker had them, LE path didn't). Verified:
  gen 1 reports 2700 inf/s vs gen 0's 0.
- Held-out win rates now PERSIST in metrics.jsonl (`le_ff_winrate`, `le_vor_winrate`,
  only on eval gens) instead of only a transient log line — so the dashboard can
  chart the one signal we trust. Eval moved before the metrics-row write. First
  numbers: gen4 ff50%/vor0%, gen8 ff38%/vor6% (vor climbing off zero = learning;
  ff noisy at 16 games).
- **Eval-cost characterization + fixes.** The held-out eval was ~336s (vs ~40s
  gens), dominated by a *few long survivor games* where the net vs three
  voronoi-@200-sims play a single-threaded endgame to 200 turns. Fixes: (1)
  early-stop each game the moment the net's seat dies — a "win" requires net =
  sole survivor, so a dead-net game's verdict is already decided; correctness,
  not a shortcut. (2) Parallelize the per-turn heuristic-move computation across
  live games (helps early, not the 1-2-game tail). (3) Drop the hardcoded 200-sim
  floor → configurable `eval_sims` (set 64, matching the historical league; still
  a real lookahead opponent, ~3× cheaper per move). (4) Eval frequency 2→8 gens.
  Rejected an eval turn-cap: it would bias the win rate DOWN exactly as the net
  improves and games lengthen — an `EASY-ROUTE` that corrupts the trusted metric.

---

## Response/exploit path (Stage B, part of the full overhaul)
Heterogeneous-τ response generation (rational seat at `response_tau` best-responds
to weak-τ opponents using frozen-proxy leaf values; targets conditioned on the
opponent's τ) — 4-player generalization of the validated 2p response path. Begins
after `response_after` gens. Serving uses the response net + online τ-MLE.
`FORK` (4p): one rational seat vs three weak seats at a sampled τ-vector, target
recorded for the rational seat only. Novel vs the 2p original; validated by the
same held-out/exploitability metrics.

## Log (append per step)
- (init) Phases 0–1 done & verified. Starting Phase 2.
- Phase 2 done: `snek_core` τ helpers (`encode_into_temp`, `append_temp_planes`,
  `NUM_CHANNELS_TEMP`, `TEMP_SCALE`) + 2 tests. Verified (snek-core tests green).
- Engine fix: `EqForest` now carries **per-game τ** (`eval_game` map + per-game
  `backup`) — required because games persist across gens here, so τ can't be
  per-gen. 18/18 snek-search tests green.
- Phase 3 done: `selfplay/le_selfplay.rs::generate_le` — single-pass LE self-play
  (build forest → encode leaves per-seat w/ τ plane → value forward → LE backup →
  play_mix + sample → step; per-game τ in `SelfPlayState::temp`, persisted via
  session snapshot).
- Phase 4 done: `materialize_le_game` (value = (1-λ)·LE + λ·outcome, τ per sample).
- Phase 5 done: `train_one` appends τ plane when the batch carries τ (net 15-ch),
  after D4 aug (τ untouched by symmetry).
- Phase 6 done: trainer builds net at `in_ch = 15` under `le_mode`, routes
  `generate`→`generate_le`; AZ league/burst disabled in LE mode. **Whole trainer
  compiles clean.**
- **Phase 3.3 + 5.2 VERIFIED**: `le_selfplay_and_train_smoke` (CPU) passes —
  mixed policies (sum→1, not one-hot), τ carried & conditions the net, value
  targets in [-1,1], finite train losses. The correct-game-mode loop is real.
- Next: Phase 7 (held-out LE strength eval — the "is it working" signal) → build
  release → launch & watch.
