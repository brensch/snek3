# Albatross training — current architecture

How the live `train_albatross` run actually works (the Python equilibrium path).

## Why the GPU stays busy (and the CPU mostly doesn't)
Each self-play move is **serial**, not pipelined:

`prepare_search` (Rust builds the tree) → **net forward on all leaves (GPU)** → `backup_search` (Rust solves the equilibrium) → step.

The GPU reads ~95% busy because the leaf evaluation is a **huge batch** — the
full-width depth-2 tree across all 256 games on the 21×21 egocentric board is
~40k leaf positions per move, so the GPU forward dominates each move's wall-time.
The Rust parts (tree build + logit-equilibrium solve) are fast enough that the
GPU doesn't idle long between moves. So it's *GPU-dominated work helped by fast
Rust* — **not** a continuous Rust→GPU feeder pipeline (that's the unused
`generate_selfplay_le` fast path). The CPU is still largely idle.

## Diagram

```mermaid
flowchart TB
  subgraph GEN["ONE GENERATION — train_albatross (Python loop)"]
    direction TB
    P["PROXY self-play (generate_proxy)<br/>sample one τ ∈ [0.5,10] for the gen"]
    PB["ReplayBuffer (+τ per sample)"]
    PT["train PROXY net<br/>256 steps · D4 symmetry aug · temp=τ"]
    R["RESPONSE self-play (generate_response)<br/>agent τ_R=12 vs frozen proxy at opp τ"]
    RT["train RESPONSE net"]
    E["EVAL + RECORD<br/>proxy/response vs baseline, UCT<br/>+ proxy-v-proxy, response-v-proxy<br/>→ metrics.jsonl + games/"]
    P --> PB --> PT --> R --> RT --> E -->|next gen| P
  end

  subgraph MOVE["INNER LOOP — one move of self-play (run_search), repeated until 8000 samples"]
    direction LR
    GB["GameBatch boards<br/>(Rust snek-core engine)"]
    PS["prepare_search(depth=2, draw=-0.9)<br/>build full-width tree, emit ALL leaf obs<br/>egocentric 21×21 · ~40k leaves<br/>RUST (snek-search) · CPU"]
    NET["AZNet forward on leaves<br/>VALUE head · temp=τ<br/>GPU (torch, chunked @2048)"]
    BK["backup_search(τ, iters=120)<br/>logit-equilibrium solve per node (SFP)<br/>→ root policy + root value<br/>RUST (le.rs) · CPU"]
    REC["record (obs, LE policy, LE value, τ)<br/>sample move (+explore) · step<br/>Python + Rust"]
    GB --> PS --> NET --> BK --> REC --> GB
  end

  P -. uses .-> MOVE
  R -. uses .-> MOVE

  subgraph COMP["Components"]
    direction LR
    C1["snek-core (Rust)<br/>engine · egocentric encoding · flood-fill baseline"]
    C2["snek-search (Rust)<br/>fixed-depth search · le.rs LE solver · UCT agent"]
    C3["AZNet (Python/torch)<br/>temp-conditioned ResNet<br/>policy + value heads"]
    C4["dashboard (FastAPI + React)<br/>reads metrics.jsonl + games/"]
  end
```

## Notes
- **τ (temperature) is the whole self-play**: proxy plays the logit equilibrium
  at a sampled τ; response best-responds (τ_R) to the frozen proxy at a sampled
  opponent τ. Test-time exploitation = estimate opponent τ by MLE → feed the
  response.
- **baseline / UCT are test opponents only** — used in EVAL and recorded
  replays, never in self-play (Albatross trains purely via self-play; this is
  also "GPU purity"). Training against them would be the optional "league" lever.
- **draw_value = −0.9** in the equilibrium search (was hardcoded 0, which made
  mutual head-to-head death "free" → proxy-vs-proxy suicide-draws).
- **Speed**: bottleneck is the leaf-eval inference volume (full-width depth-2 ×
  21×21 egocentric). The faster future path is the Rust ORT loop
  (`generate_selfplay_le`) with GPU-worker double-buffering — natural fit for a
  cloud H100 to cut iteration time.

See [[albatross-overhaul]] (memory) for the build history and
`docs/albatross-option-matrix.md` for the levers/options.
