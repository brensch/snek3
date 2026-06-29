//! `snek-search`: fixed-depth, full-width search with a per-node Logit
//! Equilibrium solve (Stochastic Fictitious Play) at a high fixed temperature.
//!
//! - [`le`] — the equilibrium solver for a single normal-form game.
//! - [`search`] — the fixed-depth tree with a two-phase batched-eval interface.

pub mod le;
pub mod mcts;
pub mod search;
pub mod uct;

pub use le::{solve as solve_le, LeSolution};
pub use mcts::{
    mask_obvious_immediate_deaths, obvious_immediate_death, ActionStat, ChildEdge, MctsForest,
    NodeSnake, TreeNodeSnapshot, TreeSnapshot,
};
pub use search::Forest;
pub use uct::uct_actions;
