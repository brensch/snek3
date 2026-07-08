//! Standard board initialization, following the official fixed-placement setup
//! (rules `PlaceSnakesFixed` + `PlaceFoodFixed`): snakes start coiled (three
//! segments stacked) on a set of fixed points, with one food diagonally next
//! to each snake (away from the centre) and one in the centre.

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

    place_start_food(&mut board, rng);
    board
}

/// The official fixed start points: four corners and four cardinal mid-edges,
/// each set shuffled independently, then a coin flip decides which set is
/// drawn from first (so 4-snake games start all-corner or all-mid-edge with
/// equal probability, like the real engine).
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
    if rng.gen_bool(0.5) {
        corners.extend(cardinals);
        corners
    } else {
        cardinals.extend(corners);
        cardinals
    }
}

/// The official `PlaceFoodFixed`: one food on a random diagonal neighbour of
/// each snake that sits further from the centre than the head on at least one
/// axis — never the centre square, a corner, or on existing food — plus one
/// food in the exact centre. (The official small-board >4-snake skip never
/// triggers at 11x11, so it is not reproduced here.)
fn place_start_food<R: Rng>(board: &mut Board, rng: &mut R) {
    let cx = (board.width - 1) / 2;
    let cy = (board.height - 1) / 2;
    let center = Point::new(cx, cy);

    let heads: Vec<Point> = board.snakes.iter().map(|s| s.head()).collect();
    for head in heads {
        let candidates = [
            Point::new(head.x - 1, head.y - 1),
            Point::new(head.x - 1, head.y + 1),
            Point::new(head.x + 1, head.y - 1),
            Point::new(head.x + 1, head.y + 1),
        ];
        let valid: Vec<Point> = candidates
            .into_iter()
            .filter(|&p| board.in_bounds(p) && p != center && !board.food.contains(&p))
            // Further than the head from the centre on at least one axis.
            .filter(|&p| {
                (p.x < head.x && head.x < cx)
                    || (cx < head.x && head.x < p.x)
                    || (p.y < head.y && head.y < cy)
                    || (cy < head.y && head.y < p.y)
            })
            .filter(|&p| {
                !((p.x == 0 || p.x == board.width - 1) && (p.y == 0 || p.y == board.height - 1))
            })
            .collect();
        if !valid.is_empty() {
            board.food.push(valid[rng.gen_range(0..valid.len())]);
        }
    }

    let occupied = board
        .snakes
        .iter()
        .any(|s| s.body.iter().any(|q| q == center))
        || board.food.contains(&center);
    if board.in_bounds(center) && !occupied {
        board.food.push(center);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// Pin the official `PlaceSnakesFixed` + `PlaceFoodFixed` semantics: starts
    /// are all-corner or all-mid-edge (roughly 50/50), each snake's food is a
    /// diagonal neighbour further from the centre on at least one axis (never
    /// a corner or the centre), and the centre food is present.
    #[test]
    fn standard_start_matches_official_rules() {
        let (w, h) = (11i8, 11i8);
        let (cx, cy) = ((w - 1) / 2, (h - 1) / 2);
        let corners = [(1, 1), (1, 9), (9, 1), (9, 9)];
        let mut corner_games = 0;
        let n_games = 2000;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        for _ in 0..n_games {
            let board = standard_start(w, h, 4, &mut rng);

            let on_corners = board
                .snakes
                .iter()
                .all(|s| corners.contains(&(s.head().x, s.head().y)));
            let on_cardinals = board
                .snakes
                .iter()
                .all(|s| !corners.contains(&(s.head().x, s.head().y)));
            assert!(on_corners || on_cardinals, "mixed corner/cardinal start");
            corner_games += on_corners as usize;

            // One food per snake plus the centre one.
            assert_eq!(board.food.len(), board.snakes.len() + 1);
            assert!(board.food.contains(&Point::new(cx, cy)));

            for s in &board.snakes {
                let head = s.head();
                let food = board
                    .food
                    .iter()
                    .find(|p| (p.x - head.x).abs() == 1 && (p.y - head.y).abs() == 1)
                    .unwrap_or_else(|| panic!("no diagonal food for head {head:?}"));
                let away = (food.x < head.x && head.x < cx)
                    || (cx < head.x && head.x < food.x)
                    || (food.y < head.y && head.y < cy)
                    || (cy < head.y && head.y < food.y);
                assert!(away, "food {food:?} not away from centre for head {head:?}");
                assert!(
                    !((food.x == 0 || food.x == w - 1) && (food.y == 0 || food.y == h - 1)),
                    "food in corner: {food:?}"
                );
            }
        }

        // Coin flip: both start geometries must occur, roughly evenly.
        assert!(
            (700..1300).contains(&corner_games),
            "corner starts way off 50%: {corner_games}/{n_games}"
        );
    }
}
