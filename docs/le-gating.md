# LE regime: gating, per-seat τ, sharpened play, strong voronoi

Why this exists: after ~2,800 generations, snek3-le-6 plateaued at ~50% and
then eroded to ~35% against a **weak** opponent (voronoi-64 ≈ 1-ply greedy),
while every internal health metric stayed green. The audit found four
structural causes; this regime fixes all four. The measured evidence:

- **No gating**: the freshly-trained net became the self-play data generator
  unconditionally every gen — a regressed net immediately poisons its own
  data. Checkpoints 32 gens apart swung ±40 points of true strength.
- **LR floored at 1e-4 since gen ~408**: constant Adam step = the weights
  random-walked a constant ~3.7% L2 per 32 gens, forever. Never consolidated.
- **Exploration mis-aimed**: root Dirichlet noise only perturbed the *played*
  move (never the search or the target) — redundant with the sampled mixed
  equilibrium, and it forced uncorrected opening blunders. Meanwhile the real
  target-side levers were untouched: all seats shared one τ (so the net never
  learned to *exploit* weaker seats — the exact skill the eval measures), the
  eval/serve τ (12) sat outside the training range (≤10), and soft-τ games
  played near-randomly to the very end, making their 90%-outcome value labels
  noise.
- **The benchmark lied by omission**: voronoi-64 can't even expand its root's
  joint children. The historical "voronoi Elo" anchor was the same agent at
  20,000 sims — ~300× more search.

## What runs now, generation by generation

Two nets live in the trainer:

| net | role | changes when |
|---|---|---|
| **incumbent** (`incumbent.safetensors`) | generates ALL self-play data; published to serving | only by winning a gate |
| **candidate** (`net.safetensors`, the live `VarStore`) | trained every gen on the incumbent's data | every generation |

**Every generation**
1. Self-play: the **incumbent** plays `samples_per_gen` samples of LE
   self-play. Each new game draws **per-seat** τ i.i.d. from
   `U[tau_min, tau_max]` — most games are rationality-asymmetric, so the
   solved equilibria contain exploit-the-weak-seat behaviour, and each seat's
   leaves are encoded at *its own* τ. For the first `sample_turns` turns the
   played move samples the raw equilibrium mix (opening diversity); after
   that the played distribution is sharpened (`p^play_sharpness`) so
   endgames — and therefore the outcome-dominated value labels — reflect
   intentful play. The TRAINING target is always the clean equilibrium; no
   noise, no sharpening ever touches a label. (Root Dirichlet is gone from
   the LE path; curriculum scenario seeders carry opening-state diversity.)
2. Train: the **candidate** takes `train_steps` × `batch_size` uniform draws
   from the buffer at `lr = max(1e-3 · 0.5^(seen/half_life), lr_floor)`.
3. Checkpoint: candidate → `net.safetensors` + `checkpoints/net_NNNN`;
   serving publishes the **incumbent**.

**Every `gate_gens` generations — the gate**
1. Floodfill sanity line: candidate vs greedy floodfill, 16 games.
2. Paired gate match: candidate plays `gate_games` vs voronoi-`gate_sims`
   from seed S; the incumbent plays the SAME seed S (identical starts, so
   opening luck cancels). Promotion iff
   `candidate_wins > incumbent_wins + gate_margin`. Ties keep the incumbent.
3. On promotion: incumbent ← candidate weights, `incumbent.safetensors`
   saved, serving re-published, `gate.json` updated (incumbent gen,
   promotion count, next rotated seed). On failure: nothing — the candidate
   just keeps training; `failed_gates` counts the streak (visible stall
   signal).
4. Seed S rotates every gate so repeated gating can't select for
   start-position specialists.

**Every `probe_gens` generations — the probe**
The incumbent plays 32 games vs voronoi-`probe_sims` (default 20,000 = the
historical strong anchor). This is the **super-heuristic goal line** on the
dashboard. Expect it near zero initially; it is the number that must climb.

## What can happen at runtime, and what we expect

- **Candidate never promotes** (`failed_gates` grows): the learner can't beat
  its own teacher's data — the honest stall signal that was previously
  invisible. Levers then: LR, capacity, data mix. Nothing degrades: the
  incumbent (and serving) keep the best-known net.
- **Every gate promotes**: strength ratchets; the strength chart's incumbent
  line becomes monotone-ish by construction.
- **Gate flakiness**: at 48 paired games a promotion needs strictly more
  wins; paired seeds cut opening variance, and rotating seeds mean a lucky
  promotion isn't reinforced next gate. A falsely-kept incumbent only delays
  progress `gate_gens`.
- **Cost**: gate ≈ 96 games (2 sides) + 16 ff every 64 gens; probe ≈ 32
  slow games every 128 gens. Order minutes; a few percent of wall time.
- **Resume**: `incumbent.safetensors` + `gate.json` restore the exact gating
  state. A pre-gating run resuming under this build founds the incumbent from
  its live net. An old `selfplay.json` (per-game τ) fails to parse and is
  discarded with a warning — costs a few in-flight games, never training data.

## The knobs (`RunConfig`)

| knob | default | meaning |
|---|---|---|
| `gate_gens` | 64 | gate cadence (0 = gating off: pre-regime behaviour) |
| `gate_games` | 48 | games per side per gate (paired seeds) |
| `gate_margin` | 0 | extra wins required beyond the incumbent's |
| `gate_sims` | 256 | gate opponent strength (aim for mid-range win rates) |
| `probe_gens` | 128 | strong-probe cadence (0 = off) |
| `probe_sims` | 20000 | probe opponent strength (the historical anchor) |
| `lr_floor` | 2.5e-5 | permanent minimum Adam step |
| `play_sharpness` | 2.0 | post-opening played-policy exponent (1 = off) |
| `tau_min`/`tau_max` | run-set | per-SEAT τ range; must cover `response_tau` |
| `value_weight` | run-set | 0.25 in the new regime (policy loss dominates) |

## Picking the founding incumbent

`snek-train --le-sweep "g1,g2,…" --run-id <run>` plays every listed
checkpoint against voronoi-`gate_sims` from identical start seeds and prints
a table + winner. Copy the winner over both `net.safetensors` and let the
founding logic seed `incumbent.safetensors` from it (or copy explicitly),
then resume. Run it with training stopped — it owns the GPU.
