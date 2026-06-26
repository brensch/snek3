//! Standard board initialization, following the official fixed-placement setup:
//! snakes start coiled (three segments stacked) on a set of fixed points, with
//! one food diagonally next to each snake and one in the centre.

use crate::{Board, Point, SNAKE_START_HEALTH};
use rand::seq::SliceRandom;
use rand::Rng;

/// Create a standard `width` x `height` board with `num_snakes` snakes placed at
/// fixed start points (shuffled) and the standard starting food.
pub fn standard_start<R: Rng>(width: i8, height: i8, num_snakes: usize, rng: &mut R) -> Board {
    let mut board = Board::new(width, height);
    let starts = fixed_start_points(width, height, rng);
    assert!(
        num_snakes <= starts.len(),
        "too many snakes for a {width}x{height} board"
    );

    for &p in starts.iter().take(num_snakes) {
        // Coiled: three segments stacked on the start point.
        board.add_snake(&[p, p, p]);
    }
    for s in &mut board.snakes {
        s.health = SNAKE_START_HEALTH;
    }

    place_start_food(&mut board);
    board
}

/// The official fixed start points: four corners and four cardinal mid-edges,
/// each set shuffled independently, corners first.
fn fixed_start_points<R: Rng>(width: i8, height: i8, rng: &mut R) -> Vec<Point> {
    let mn = 1i8;
    let md_x = (width - 1) / 2;
    let md_y = (height - 1) / 2;
    let mx_x = width - 2;
    let mx_y = height - 2;

    let mut corners = vec![
        Point::new(mn, mn),
        Point::new(mn, mx_y),
        Point::new(mx_x, mn),
        Point::new(mx_x, mx_y),
    ];
    let mut cardinals = vec![
        Point::new(mn, md_y),
        Point::new(md_x, mn),
        Point::new(md_x, mx_y),
        Point::new(mx_x, md_y),
    ];
    corners.shuffle(rng);
    cardinals.shuffle(rng);
    corners.extend(cardinals);
    corners
}

/// Place one food diagonally adjacent to each snake (toward the board centre)
/// plus one food in the exact centre.
fn place_start_food(board: &mut Board) {
    let cx = (board.width - 1) / 2;
    let cy = (board.height - 1) / 2;

    let occupied = |b: &Board, p: Point| {
        b.snakes.iter().any(|s| s.body.iter().any(|q| q == p)) || b.food.contains(&p)
    };

    let heads: Vec<Point> = board.snakes.iter().map(|s| s.head()).collect();
    for head in heads {
        // Prefer the diagonal neighbour that points toward the centre.
        let dx = if head.x < cx { 1 } else { -1 };
        let dy = if head.y < cy { 1 } else { -1 };
        let candidates = [
            Point::new(head.x + dx, head.y + dy),
            Point::new(head.x + dx, head.y - dy),
            Point::new(head.x - dx, head.y + dy),
        ];
        for c in candidates {
            if board.in_bounds(c) && !occupied(board, c) {
                board.food.push(c);
                break;
            }
        }
    }

    let center = Point::new(cx, cy);
    if board.in_bounds(center) && !occupied(board, center) {
        board.food.push(center);
    }
}
