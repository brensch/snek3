# Selective-depth LE search (progressive deepening)

## Problem

The LE search is **fixed-depth, full-width**: at each node it builds the
normal-form game over the *joint* move space and solves a logit equilibrium
(SFP). With 4 simultaneous snakes and ~2.5 candidates each (after the suicide
mask), one ply is already `b^4 ≈ 38` leaves, so depth grows as `b^(4·depth)`:

| depth | leaves/game (b=2.5) | ×48 games/turn |
|------:|--------------------:|---------------:|
| 1     | ~38                 | ~1.8k          |
| 2     | ~1.5k  (38×)        | ~71k → OOM     |
| 3     | ~56k                | —              |

Depth-1 therefore never *sees* multi-move traps — e.g. a length-21 snake whose
only "good"-looking move steps into a 4-cell pocket that closes over 3-4 turns.
The value net is supposed to encode that, but its judgement of the confined
successor isn't sharp enough, and depth-1 has no way to look further. Observed
in-run: the equilibrium put 56% on a move that walks into a dead-end.

## What others do (research)

- **Albatross** (SBRLE, arXiv 2402.03136): fixed-depth full-width + NFG-per-node
  + SFP. No selectivity. Explicitly: *"the tree search becomes a weak
  improvement operator if only a fraction of nodes can be evaluated"* — i.e. go
  shallow-but-complete rather than deep-but-sparse. So going selective is an
  extension beyond the base method, and must be done carefully.
- **NN-CCE** (arXiv 2406.10411): full-width, but **masks dominated strategies**
  before the equilibrium solve (generalises our suicide mask) and trains
  per-agent policies to avoid the joint blow-up.
- **SM-MCTS progressive widening** (Lisý/Winands): schedule child expansion by
  visit count; a knob interpolates width↔depth. → *expand where play goes.*
- **Best-first / iterative deepening + transposition + move ordering** (classic
  alpha-beta): deepen the important lines first; the PV move is usually still
  best next iteration.
- **ReBeL / depth-limited solving** (arXiv 2007.13544): build a depth-limited
  subgame, solve the equilibrium iteratively, value-net the leaves; grow the
  tree where it matters.

## Our design: reach-probability-prioritised selective deepening

Full anytime best-first (expand→re-solve→expand) is the theoretical ideal but is
**inherently sequential**, which kills batching — and batching is 99% of GPU
self-play cost *and* the only way ~hundreds of leaf evals fit a 350ms **CPU**
serve. So we use a **two-phase batched** scheme:

1. **Build depth-1** (full-width, suicide-masked). Collect depth-1 leaves.
   → one batched value forward → `vals1`.
2. **Solve + select.** Solve the depth-1 LE at each root. For each non-terminal
   joint successor compute its **reach probability** `∏_i π_i(a_i)` under the
   equilibrium. Deepen only the **top-K** successors per root (optionally gated
   by a decision-margin: skip roots whose best move already dominates — spend
   compute only where the outcome is uncertain).
3. **Expand selected** one more ply (full-width) → the *only* new leaves.
   → a second batched value forward → `vals2`.
4. **Re-solve** each root's LE with the refined (depth-2) values for the K
   deepened successors and `vals1` for the rest.

Leaves per game ≈ `38 + K·38`. With **K=4 that's ~190 (≈5×)**, not 38× — the
GPU/CPU can afford it. Two forwards, both batched.

Why reach-probability is the right selector here: the trap move is *high*
reach-probability precisely because it looks good at depth-1 (the equilibrium
put 56% on it). Deepening it exposes the closing pocket, the refined value drops,
and the re-solve down-weights it. We spend the depth exactly on the lines the
equilibrium believes in — which are the lines whose value most moves the root
decision. The margin-gate is the "sick" part for CPU: most positions are
unambiguous → skip deepening → average serve cost stays ~depth-1, and the ~190-
leaf cost is only paid on genuinely close/uncertain decisions.

## Deploy shape (CPU, 350ms)

Same search runs at serve. Knobs: `le_top_k` (K) and a decision-margin
threshold. On CPU the two forwards are batched (one ~38-row, one ~K·38-row);
the margin-gate means the deep pass usually doesn't fire. Train and serve use
the *same* search so there's no train/serve mismatch (per ReBeL/AZ). Self-play
can run depth-1 (cheap) or selective depth-2 (`le_top_k>0`) via config; if serve
is depth-2 the value net should ideally have trained under depth-2 too, so the
plan is: validate depth-2 in self-play, then enable it for serve.

## Implementation notes

- `eqsearch`: `build(depth=1)` unchanged; add `deepen_topk(&vals1, tau, iters,
  k)` (solve→select→expand, returns new leaves) and make `backup` a **recursive
  post-order** (selective expansion breaks the children-before-parent id order).
- Config: `le_top_k` (0 = current fixed depth-1).
- Callers: `le_selfplay` and `le_eval` gain the second forward pass.
- Keep suicide mask (`le_candidates`) at every expanded node.
