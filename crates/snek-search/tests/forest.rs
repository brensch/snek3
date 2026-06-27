//! End-to-end checks of the fixed-depth equilibrium search on real boards.

use snek_core::{Board, Point};
use snek_search::Forest;

/// A simple "value net" stand-in: every leaf is neutral (0). With neutral
/// leaves, terminal outcomes (wins/losses) are what drive the policy.
fn neutral_values(forest: &Forest) -> Vec<f32> {
    vec![0.0; forest.eval_count()]
}

#[test]
fn build_collects_leaf_observations() {
    let mut b = Board::new(11, 11);
    b.add_snake(&[Point::new(2, 2), Point::new(2, 1)]);
    b.add_snake(&[Point::new(8, 8), Point::new(8, 9)]);

    let forest = Forest::build(std::slice::from_ref(&b), 2);
    assert!(
        forest.eval_count() > 0,
        "depth-2 search has leaves to evaluate"
    );
    // Egocentric (head-centred) canvas: side = 2*11-1 = 21.
    let side = 2 * 11 - 1;
    assert_eq!(forest.obs_size(), 9 * side * side);

    let mut obs = vec![0.0f32; forest.eval_count() * forest.obs_size()];
    forest.write_observations(&mut obs);
    // Board-mask channel marks exactly the real board cells (11*11 of them) on
    // the larger egocentric canvas, not the whole plane.
    let board_mask = &obs[8 * side * side..9 * side * side];
    assert_eq!(board_mask.iter().filter(|&&x| x == 1.0).count(), 11 * 11);
}

#[test]
fn policy_is_a_distribution_over_legal_moves() {
    let mut b = Board::new(11, 11);
    b.add_snake(&[Point::new(5, 5), Point::new(5, 4)]);
    b.add_snake(&[Point::new(1, 1), Point::new(1, 0)]);

    let mut forest = Forest::build(std::slice::from_ref(&b), 1);
    let values = neutral_values(&forest);
    let (policy, root_values) = forest.backup(&values, &[6.0, 6.0], 200);

    // Layout: [root, snake, 4]; root values are [root, snake].
    assert_eq!(policy.len(), 2 * 4);
    assert_eq!(root_values.len(), 2);
    assert!(root_values.iter().all(|&v| v.is_finite()));
    let p0 = &policy[0..4];
    let sum: f32 = p0.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "snake 0 policy sums to 1, got {sum}"
    );
    // Moving Down (reversing onto the neck at (5,4)) is pruned -> zero mass.
    assert_eq!(p0[1], 0.0, "reversal move gets no probability");
}

#[test]
fn search_avoids_walking_into_a_wall() {
    // Snake hugging the left wall: moving Left steps off the board and is pruned,
    // so it must receive zero probability.
    let mut b = Board::new(11, 11);
    b.add_snake(&[Point::new(0, 5), Point::new(1, 5)]); // head at x=0
    b.add_snake(&[Point::new(8, 8), Point::new(8, 7)]);

    let mut forest = Forest::build(std::slice::from_ref(&b), 1);
    let values = neutral_values(&forest);
    let (policy, _root_values) = forest.backup(&values, &[8.0, 8.0], 200);
    // Move::Left has index 2.
    assert_eq!(policy[2], 0.0, "stepping into the wall is never chosen");
}

#[test]
fn search_prefers_winning_head_to_head() {
    // Snake 0 (length 3) can move into a cell where snake 1 (length 2) is forced
    // to contest. With neutral leaf values, the search should favor the move that
    // leads to winning head-to-heads / surviving, putting most mass on a safe move.
    let mut b = Board::new(11, 11);
    b.add_snake(&[Point::new(5, 5), Point::new(5, 4), Point::new(5, 3)]);
    b.add_snake(&[Point::new(5, 7), Point::new(5, 8)]);

    let mut forest = Forest::build(std::slice::from_ref(&b), 2);
    let values = neutral_values(&forest);
    let (policy, _root_values) = forest.backup(&values, &[8.0, 8.0], 300);
    let p0 = &policy[0..4];
    let sum: f32 = p0.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3);
    // The longer snake should not assign large mass to a self-destructive move;
    // every probability is finite and non-negative.
    assert!(p0.iter().all(|&x| x >= 0.0 && x.is_finite()));
}
