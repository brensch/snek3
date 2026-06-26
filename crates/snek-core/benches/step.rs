use criterion::{criterion_group, criterion_main, Criterion};
use snek_core::{Board, Move, Point};
use std::hint::black_box;

/// A simple deterministic move policy: walk in a small cycle so games run long
/// without immediately dying. Good enough to measure raw `step` throughput.
fn policy(turn: u32) -> Move {
    match turn % 4 {
        0 => Move::Up,
        1 => Move::Right,
        2 => Move::Down,
        _ => Move::Left,
    }
}

fn bench_step(c: &mut Criterion) {
    c.bench_function("duel_step", |b| {
        b.iter(|| {
            let mut board = Board::new(11, 11);
            board.add_snake(&[Point::new(1, 1), Point::new(1, 1), Point::new(1, 1)]);
            board.add_snake(&[Point::new(9, 9), Point::new(9, 9), Point::new(9, 9)]);
            // Sprinkle food so snakes keep eating and bodies stay nontrivial.
            for i in 0..11 {
                board.food.push(Point::new(i, 5));
            }
            let mut steps = 0u64;
            while !board.is_terminal() && board.turn < 200 {
                let mv = policy(board.turn);
                board.step(&[mv, mv]);
                steps += 1;
            }
            black_box(steps)
        })
    });
}

criterion_group!(benches, bench_step);
criterion_main!(benches);
