//! Curriculum seeders: programmatically generated positions demonstrating a
//! situation self-play under-visits (KataGo-style seeded starts).
//!
//! Self-play only learns what its own distribution touches. When a blind spot
//! is found (e.g. the net starving next to food because standard-start games
//! die in combat long before the hunger clock matters), the fix is to seed a
//! slice of games from positions that exhibit the situation and let them play
//! out — real games, real outcomes, real gradient; only the starting
//! distribution changes. No reward shaping, no synthetic targets.
//!
//! # Adding a scenario
//!
//! Write one generator function and register it:
//!
//! ```ignore
//! fn my_situation(width: i8, height: i8, num_snakes: usize, rng: &mut dyn RngCore) -> Board {
//!     // either delegate to `random_midgame` with tuned `MidgameOpts`,
//!     // or hand-build any valid Board.
//! }
//! // ...and add to SCENARIOS:
//! Scenario { name: "my-situation", describe: "…", generate: my_situation },
//! ```
//!
//! The name is what configs reference (`scenarios: ["hunger", …]`), so keep it
//! stable once a run has used it.

use crate::setup::standard_start;
use crate::{Board, Point};
use rand::seq::SliceRandom;
use rand::{Rng, RngCore};

/// One registered curriculum seeder.
pub struct Scenario {
    /// Stable config/log identifier.
    pub name: &'static str,
    /// One-line human description (shown in logs/docs).
    pub describe: &'static str,
    /// Position generator. Must return a mechanically valid board.
    pub generate: fn(i8, i8, usize, &mut dyn RngCore) -> Board,
}

/// Every known scenario. Order is cosmetic; selection is uniform among the
/// configured subset.
pub const SCENARIOS: &[Scenario] = &[Scenario {
    name: "hunger",
    describe: "low health, sparse food — learn the starvation gradient",
    generate: hunger,
}];

/// Look a scenario up by config name.
pub fn scenario(name: &str) -> Option<&'static Scenario> {
    SCENARIOS.iter().find(|s| s.name == name)
}

/// A "hungry mid-game": 1-2 VICTIM snakes on a live starvation clock while the
/// rest are healthy — mirroring the observed failure (our net starving among
/// thriving opponents who can also deny food). Standard starts almost never
/// reach starvation (snakes die in combat first), so the value net never
/// learns the hunger gradient.
///
/// Every victim is dealt a SOLVABLE puzzle: at least one food within
/// `health - 1` free-space BFS steps of its head (placing one if needed).
/// A doomed victim only teaches "hunger = death"; the blindness this scenario
/// exists to fix needs the contrast case — hunger + reachable food + eating =
/// survival. (BFS over the spawn board is conservative: tails vacate as play
/// advances, so a statically reachable food stays reachable.)
fn hunger(width: i8, height: i8, num_snakes: usize, mut rng: &mut dyn RngCore) -> Board {
    let mut b = random_midgame(
        width,
        height,
        num_snakes,
        &MidgameOpts {
            len_range: (4, 14),
            health_range: (40, 100), // healthy baseline; victims overridden below
            food_range: (1, 3),
            turn_range: (40, 160),
        },
        &mut rng,
    );
    let n = b.snakes.len();
    let victims = (&mut rng).gen_range(1..=2usize.min(n));
    let first = (&mut rng).gen_range(0..n);
    for k in 0..victims {
        let v = (first + k) % n;
        let health = (&mut rng).gen_range(5..=30i16);
        b.snakes[v].health = health;
        ensure_reachable_food(&mut b, v, (health - 1) as u32, &mut rng);
    }
    b
}

/// Guarantee snake `v` has at least one food within `budget` BFS steps through
/// free cells; place one at a random reachable cell if not. The victim's head
/// always has an escape square (engine invariant), so distance-1 cells exist.
fn ensure_reachable_food(b: &mut Board, v: usize, budget: u32, rng: &mut &mut dyn RngCore) {
    let dist = free_space_bfs(b, b.snakes[v].head());
    let idx = |p: Point| (p.y as usize) * (b.width as usize) + p.x as usize;
    if b.food.iter().any(|&f| dist[idx(f)].is_some_and(|d| d <= budget)) {
        return;
    }
    // Prefer a genuinely urgent puzzle: food at 2..=budget steps when possible.
    let mut cells: Vec<Point> = (0..b.width)
        .flat_map(|x| (0..b.height).map(move |y| Point::new(x, y)))
        .filter(|&p| !b.food.contains(&p))
        .filter(|&p| dist[idx(p)].is_some_and(|d| (2..=budget).contains(&d)))
        .collect();
    if cells.is_empty() {
        // Cramped board: fall back to anything reachable in budget (>= 1).
        cells = (0..b.width)
            .flat_map(|x| (0..b.height).map(move |y| Point::new(x, y)))
            .filter(|&p| !b.food.contains(&p))
            .filter(|&p| dist[idx(p)].is_some_and(|d| (1..=budget).contains(&d)))
            .collect();
    }
    if !cells.is_empty() {
        let p = cells[rng.gen_range(0..cells.len())];
        b.food.push(p);
    }
}

/// BFS distance from `start` through cells no snake occupies (bodies as of the
/// spawn board; `start` itself is the only occupied cell allowed). `None` =
/// unreachable.
fn free_space_bfs(b: &Board, start: Point) -> Vec<Option<u32>> {
    let (w, h) = (b.width as usize, b.height as usize);
    let idx = |p: Point| (p.y as usize) * w + p.x as usize;
    let mut occupied = vec![false; w * h];
    for s in &b.snakes {
        for c in s.body.iter() {
            occupied[idx(c)] = true;
        }
    }
    let mut dist = vec![None; w * h];
    dist[idx(start)] = Some(0);
    let mut queue = std::collections::VecDeque::from([start]);
    while let Some(cur) = queue.pop_front() {
        let d = dist[idx(cur)].expect("queued cells have distances");
        for (dx, dy) in CARDINALS {
            let p = Point::new(cur.x + dx, cur.y + dy);
            if b.in_bounds(p) && !occupied[idx(p)] && dist[idx(p)].is_none() {
                dist[idx(p)] = Some(d + 1);
                queue.push_back(p);
            }
        }
    }
    dist
}

/// Tunable shape of a synthetic mid-game position — the shared engine most
/// scenarios delegate to (bodies as self-avoiding random walks, then sampled
/// health/food/turn).
#[derive(Clone, Copy, Debug)]
pub struct MidgameOpts {
    /// Per-snake body length, sampled uniformly (min 3).
    pub len_range: (usize, usize),
    /// Per-snake health, sampled uniformly.
    pub health_range: (i16, i16),
    /// Foods on the board, sampled uniformly (may be 0 — the spawner tops up).
    pub food_range: (usize, usize),
    /// Cosmetic board turn, sampled uniformly (affects nothing mechanical).
    pub turn_range: (u32, u32),
}

/// Generate a random VALID mid-game position: mutually disjoint contiguous
/// bodies (head-first self-avoiding walks), every head with an escape square,
/// health/food/turn from `opts`. Falls back to the official start if the opts
/// are unsatisfiable (e.g. bodies larger than the board) rather than looping.
///
/// Takes `dyn RngCore` so scenario generators stay plain `fn` pointers; the
/// generic engine below does the work (`Rng`'s methods need `Sized`).
pub fn random_midgame(
    width: i8,
    height: i8,
    num_snakes: usize,
    opts: &MidgameOpts,
    mut rng: &mut dyn RngCore,
) -> Board {
    midgame(width, height, num_snakes, opts, &mut rng)
}

fn midgame<R: Rng>(
    width: i8,
    height: i8,
    num_snakes: usize,
    opts: &MidgameOpts,
    rng: &mut R,
) -> Board {
    'attempt: for _ in 0..64 {
        let mut board = Board::new(width, height);
        let mut occupied: Vec<Point> = Vec::new();
        for _ in 0..num_snakes {
            let target_len = rng.gen_range(opts.len_range.0.max(3)..=opts.len_range.1.max(3));
            let Some(body) = grow_body(&board, &occupied, target_len, rng) else {
                continue 'attempt; // couldn't fit this snake; rebuild the position
            };
            occupied.extend(body.iter().copied());
            board.add_snake(&body);
        }
        // Every head needs at least one free in-board neighbour, or the snake
        // spawns with only losing moves — wasted (and misleading) data.
        for s in &board.snakes {
            let h = s.head();
            let free = CARDINALS.iter().any(|&(dx, dy)| {
                let p = Point::new(h.x + dx, h.y + dy);
                board.in_bounds(p) && !occupied.contains(&p)
            });
            if !free {
                continue 'attempt;
            }
        }
        for s in &mut board.snakes {
            s.health = rng.gen_range(opts.health_range.0..=opts.health_range.1);
        }
        let n_food = rng.gen_range(opts.food_range.0..=opts.food_range.1);
        let mut free_cells: Vec<Point> = (0..width)
            .flat_map(|x| (0..height).map(move |y| Point::new(x, y)))
            .filter(|p| !occupied.contains(p))
            .collect();
        free_cells.shuffle(rng);
        board.food.extend(free_cells.into_iter().take(n_food));
        board.turn = rng.gen_range(opts.turn_range.0..=opts.turn_range.1);
        return board;
    }
    standard_start(width, height, num_snakes, rng)
}

const CARDINALS: [(i8, i8); 4] = [(0, 1), (0, -1), (-1, 0), (1, 0)];

/// Self-avoiding random walk from a random free head cell; returns head-first
/// segments. Accepts a shorter body (>= 3) when the walk boxes itself in.
fn grow_body<R: Rng>(
    board: &Board,
    occupied: &[Point],
    target_len: usize,
    rng: &mut R,
) -> Option<Vec<Point>> {
    let free =
        |p: Point, own: &[Point]| board.in_bounds(p) && !occupied.contains(&p) && !own.contains(&p);
    'head: for _ in 0..16 {
        let head = Point::new(rng.gen_range(0..board.width), rng.gen_range(0..board.height));
        if !free(head, &[]) {
            continue 'head;
        }
        let mut body = vec![head];
        while body.len() < target_len {
            let cur = *body.last().unwrap();
            let mut dirs = CARDINALS;
            dirs.shuffle(rng);
            let Some(&(dx, dy)) = dirs
                .iter()
                .find(|&&(dx, dy)| free(Point::new(cur.x + dx, cur.y + dy), &body))
            else {
                break; // boxed in — accept what we have if long enough
            };
            body.push(Point::new(cur.x + dx, cur.y + dy));
        }
        if body.len() >= 3 {
            return Some(body);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// Every registered scenario must produce mechanically valid positions:
    /// disjoint contiguous bodies, live snakes with an escape square, food on
    /// free cells. Hammer each with seeds — one invalid board would poison
    /// self-play.
    #[test]
    fn every_scenario_generates_valid_positions() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        for sc in SCENARIOS {
            for _ in 0..300 {
                let b = (sc.generate)(11, 11, 4, &mut rng);
                assert_eq!(b.snakes.len(), 4, "{}", sc.name);
                let mut seen = std::collections::HashSet::new();
                for s in &b.snakes {
                    assert!(s.alive(), "{}", sc.name);
                    assert!(s.len() >= 3, "{}", sc.name);
                    let cells: Vec<Point> = s.body.iter().collect();
                    for w in cells.windows(2) {
                        let d = (w[0].x - w[1].x).abs() + (w[0].y - w[1].y).abs();
                        assert!(d <= 1, "{}: body must be contiguous or coiled", sc.name);
                    }
                    for c in &cells {
                        assert!(b.in_bounds(*c), "{}", sc.name);
                        // Coiled stacks (standard-start fallback) repeat cells;
                        // only distinct cells must be disjoint across snakes.
                        seen.insert((c.x, c.y));
                    }
                    let h = s.head();
                    assert!(
                        CARDINALS.iter().any(|&(dx, dy)| {
                            let p = Point::new(h.x + dx, h.y + dy);
                            b.in_bounds(p)
                                && !b.snakes.iter().any(|t| t.body.iter().any(|q| q == p))
                        }),
                        "{}: head must have an escape square",
                        sc.name
                    );
                }
                for f in &b.food {
                    assert!(b.in_bounds(*f), "{}", sc.name);
                    assert!(!seen.contains(&(f.x, f.y)), "{}: food on a body", sc.name);
                }
            }
        }
    }

    /// The hunger scenario's whole point: 1-2 victims on a live starvation
    /// clock, the rest healthy (they mirror thriving opponents), food sparse.
    #[test]
    fn hunger_scenario_has_victims_among_healthy() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        for _ in 0..200 {
            let b = (scenario("hunger").unwrap().generate)(11, 11, 4, &mut rng);
            let victims = b.snakes.iter().filter(|s| s.health <= 30).count();
            let healthy = b.snakes.iter().filter(|s| s.health >= 40).count();
            assert!((1..=2).contains(&victims), "victims: {victims}");
            assert_eq!(victims + healthy, 4, "no snake in the dead zone between");
            // Base 1-3, plus up to one placed per victim to guarantee a
            // reachable meal.
            assert!((1..=5).contains(&b.food.len()), "foods: {}", b.food.len());
        }
    }

    /// No doomed victims: every hungry snake must be dealt a SOLVABLE puzzle —
    /// at least one food statically reachable (free-space BFS through spawn
    /// bodies) within its health budget. A victim that cannot possibly eat
    /// only teaches "hunger = death"; the contrast case is the point.
    #[test]
    fn hunger_victims_can_always_reach_food() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(23);
        for i in 0..500 {
            let b = (scenario("hunger").unwrap().generate)(11, 11, 4, &mut rng);
            let idx = |p: Point| (p.y as usize) * (b.width as usize) + p.x as usize;
            for (si, s) in b.snakes.iter().enumerate() {
                if s.health > 30 {
                    continue; // not a victim
                }
                let dist = free_space_bfs(&b, s.head());
                let best = b.food.iter().filter_map(|&f| dist[idx(f)]).min();
                assert!(
                    best.is_some_and(|d| d <= (s.health - 1) as u32),
                    "board {i}: victim {si} (h={}) has no reachable food within budget (best {best:?})",
                    s.health
                );
            }
        }
    }

    #[test]
    fn lookup_by_name() {
        assert!(scenario("hunger").is_some());
        assert!(scenario("nope").is_none());
    }
}
