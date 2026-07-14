//! `snek-search`: shared simultaneous-move search over a policy+value net, used
//! by both self-play and serving.
//!
//! - [`eqsearch`] — the Albatross-faithful fixed-depth joint-move search with a
//!   per-node Logit-Equilibrium backup ([`EqForest`]). The "correct game mode":
//!   produces mixed-strategy policies + per-player values for a
//!   simultaneous-move, multi-player game.
//! - [`le`] — the logit-equilibrium (Stochastic Fictitious Play) solver.
//! - [`mcts`] — the legacy batched DUCT-PUCT forest ([`MctsForest`]) (retained
//!   for the AlphaZero path / reference).
//! - `search` — shared board helpers (legal candidates, terminal values).

pub mod eqsearch;
pub mod le;
pub mod mcts;
mod search;

pub use eqsearch::{EqForest, RootEq};
pub use mcts::{
    forced_move, mask_obvious_immediate_deaths, obvious_immediate_death, ActionStat, ChildEdge,
    MctsForest, NodeSnake, TreeNodeSnapshot, TreeSnapshot,
};
