//! A classic flood-fill / area-control baseline policy. For each snake it picks
//! the non-suicidal move that maximizes reachable free space (Voronoi-style
//! survival heuristic), breaking ties toward the nearest food. It ignores
//! opponents' simultaneous moves, which is exactly the kind of agent the
//! equilibrium search should learn to beat.

use crate::{Board, Move, Point};

/// Flood-fill the count of free cells reachable from `start`, treating all
/// current snake-body cells as obstacles.
fn reachable_area(board: &Board, start: Point, occupied: &[bool]) -> i32 {
    let w = board.width as i32;
    let h = board.height as i32;
    if start.x < 0 || start.x >= board.width || start.y < 0 || start.y >= board.height {
        return 0;
    }
    let idx = |x: i32, y: i32| (y * w + x) as usize;
    let mut visited = vec![false; (w * h) as usize];
    let start_i = idx(start.x as i32, start.y as i32);
    if occupied[start_i] {
        return 0;
    }
    let mut stack = vec![(start.x as i32, start.y as i32)];
    visited[start_i] = true;
    let mut count = 0;
    while let Some((x, y)) = stack.pop() {
        count += 1;
        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || nx >= w || ny < 0 || ny >= h {
                continue;
            }
            let ni = idx(nx, ny);
            if !visited[ni] && !occupied[ni] {
                visited[ni] = true;
                stack.push((nx, ny));
            }
        }
    }
    count
}

fn nearest_food_dist(board: &Board, p: Point) -> i32 {
    board
        .food
        .iter()
        .map(|f| (f.x as i32 - p.x as i32).abs() + (f.y as i32 - p.y as i32).abs())
        .min()
        .unwrap_or(i32::MAX)
}

/// Choose a baseline move for snake `i`. Returns `Move::Up` as a last resort if
/// the snake is dead or trapped.
pub fn baseline_action(board: &Board, i: usize) -> Move {
    let snake = &board.snakes[i];
    if !snake.alive() {
        return Move::Up;
    }

    // Obstacle map: every current body cell of every snake.
    let w = board.width as usize;
    let h = board.height as usize;
    let mut occupied = vec![false; w * h];
    for s in &board.snakes {
        if !s.alive() {
            continue;
        }
        for seg in s.body.iter() {
            if seg.x >= 0 && seg.x < board.width && seg.y >= 0 && seg.y < board.height {
                occupied[seg.y as usize * w + seg.x as usize] = true;
            }
        }
    }

    let head = snake.head();
    let neck = if snake.len() >= 2 {
        Some(snake.body.get(1))
    } else {
        None
    };

    let mut best: Option<(Move, i32, i32)> = None; // (move, area, -food_dist)
    for m in Move::ALL {
        let nh = m.apply(head);
        if Some(nh) == neck || !board.in_bounds(nh) {
            continue;
        }
        let area = reachable_area(board, nh, &occupied);
        if area == 0 {
            continue; // immediately fatal (into a body)
        }
        let food_score = -nearest_food_dist(board, nh);
        let key = (m, area, food_score);
        match best {
            Some((_, ba, bf)) if (area, food_score) <= (ba, bf) => {}
            _ => best = Some(key),
        }
    }

    best.map(|(m, _, _)| m).unwrap_or(Move::Up)
}
