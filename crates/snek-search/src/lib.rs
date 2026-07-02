//! `snek-search`: simultaneous-move Monte-Carlo Tree Search (decoupled-PUCT) over
//! a policy+value net — the single search used by both self-play and serving.
//!
//! - [`mcts`] — the batched DUCT-PUCT forest ([`MctsForest`]).
//! - `search` — shared board helpers (legal candidates, terminal values).

pub mod mcts;
mod search;

pub use mcts::{
    forced_move, mask_obvious_immediate_deaths, obvious_immediate_death, ActionStat, ChildEdge,
    MctsForest, NodeSnake, TreeNodeSnapshot, TreeSnapshot,
};
