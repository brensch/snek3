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
pub const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "hunger",
        describe: "low health, sparse food — learn the starvation gradient",
        generate: hunger,
    },
    Scenario {
        name: "tail-trap",
        describe: "direct tail chase that turns fatal when the leader eats the bait by its head",
        generate: tail_trap,
    },
    Scenario {
        name: "pocket",
        describe: "baited width-1 dead-end — space death several moves past the search horizon",
        generate: pocket,
    },
];

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
    for s in b.snakes.iter().filter(|s| s.alive()) {
        // A just-eliminated snake can carry an out-of-bounds head segment;
        // only in-bounds cells of LIVING snakes block space.
        for c in s.body.iter().filter(|&c| b.in_bounds(c)) {
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

/// Direct (margin-0) tail chase that turns fatal when the leader eats: the
/// victim sits at a corridor mouth (leader's body above, wall below) with a
/// free choice — chase the tail cell-for-cell as it vacates (legal under the
/// official simultaneous-vacate rule, and the classic "tails are safe" line)
/// or turn into open space. The leader's head is 2-3 moves from a bait food;
/// eating delays every subsequent tail pop by one turn, so the committed
/// victim arrives one turn early and dies at turn 3-5 — beyond the depth-2
/// search horizon, so only the value net can learn the pattern "tail chase +
/// leader near food = danger".
fn tail_trap(width: i8, height: i8, num_snakes: usize, mut rng: &mut dyn RngCore) -> Board {
    match tail_trap_layout(width, height, num_snakes, &mut rng) {
        Some(l) => l.board,
        None => fallback_midgame(width, height, num_snakes, &mut rng),
    }
}

/// Baited width-1 dead-end: a pocket sealed by another snake's HEAD-side body
/// (drains last, so the walls persist), food a couple of cells inside. Width 1
/// means entering is committing — the victim reaches the end and dies at turn
/// ~6-8, far beyond the search horizon. At turn 0 there's a genuine choice:
/// take the bait, or turn into open space.
fn pocket(width: i8, height: i8, num_snakes: usize, mut rng: &mut dyn RngCore) -> Board {
    match pocket_layout(width, height, num_snakes, &mut rng) {
        Some(l) => l.board,
        None => fallback_midgame(width, height, num_snakes, &mut rng),
    }
}

fn fallback_midgame(width: i8, height: i8, num_snakes: usize, rng: &mut &mut dyn RngCore) -> Board {
    random_midgame(
        width,
        height,
        num_snakes,
        &MidgameOpts {
            len_range: (4, 14),
            health_range: (40, 100),
            food_range: (1, 3),
            turn_range: (40, 160),
        },
        rng,
    )
}

/// A constructed trap position plus everything a simulation test needs to
/// PROVE it: the victim's doom line, the leader's bait line, the escape move,
/// and the deadline the doom line must kill by.
struct TrapLayout {
    board: Board,
    /// Seat index of the snake being taught.
    victim: usize,
    /// Seat index of the trap-owning snake (leader/sealer).
    leader: usize,
    /// Victim's scripted trap line (cells to walk, in order).
    doom_path: Vec<Point>,
    /// Leader's scripted line to the bait (empty when the leader just sits).
    leader_path: Vec<Point>,
    /// Victim's t0 alternative — one move into genuinely open space.
    escape: Point,
    /// The doom line must kill the victim by this turn (and never before 3).
    death_by: u32,
    /// Trap-machinery cells; sim opponents avoid them so they can't
    /// accidentally spring or dismantle the trap.
    reserved: Vec<Point>,
}

/// One of the 8 square-board symmetries, so every generated trap comes in all
/// orientations (the net must learn the pattern, not the corner).
#[derive(Clone, Copy)]
struct Sym {
    rot: u8,
    mirror: bool,
    s: i8,
}

impl Sym {
    fn random<R: Rng>(s: i8, rng: &mut R) -> Self {
        Sym { rot: rng.gen_range(0..4), mirror: rng.gen(), s }
    }

    fn map(&self, p: Point) -> Point {
        let mut q = p;
        if self.mirror {
            q = Point::new(self.s - 1 - q.x, q.y);
        }
        for _ in 0..self.rot {
            q = Point::new(q.y, self.s - 1 - q.x);
        }
        q
    }

    fn map_all(&self, ps: &[Point]) -> Vec<Point> {
        ps.iter().map(|&p| self.map(p)).collect()
    }
}

/// Shared assembly for both trap layouts: transform the canonical geometry
/// through a random symmetry, grow `n - 2` filler snakes in the reserved-free
/// zone, shuffle seat order, and validate every head has an escape square.
#[allow(clippy::too_many_arguments)]
fn assemble_trap<R: Rng>(
    width: i8,
    height: i8,
    num_snakes: usize,
    rng: &mut R,
    victim_cells: Vec<Point>,
    leader_cells: Vec<Point>,
    foods: Vec<Point>,
    doom_path: Vec<Point>,
    leader_path: Vec<Point>,
    escape: Point,
    death_by: u32,
    filler_region: &dyn Fn(Point) -> bool,
) -> Option<TrapLayout> {
    let sym = Sym::random(width, rng);
    let victim_t = sym.map_all(&victim_cells);
    let leader_t = sym.map_all(&leader_cells);

    // Fillers grow in canonical coords inside `filler_region`, then transform
    // — so the region constraint stays simple canonical geometry.
    let bounds_board = Board::new(width, height);
    let mut occupied: Vec<Point> = Vec::new();
    occupied.extend(victim_cells.iter().copied());
    occupied.extend(leader_cells.iter().copied());
    let mut filler_ts: Vec<Vec<Point>> = Vec::new();
    for _ in 2..num_snakes {
        let len = rng.gen_range(4..=8);
        let body = grow_body_in(&bounds_board, &occupied, len, rng, filler_region)?;
        occupied.extend(body.iter().copied());
        filler_ts.push(sym.map_all(&body));
    }

    // Seat order is shuffled so the victim isn't always snake 0. Spec 0 is
    // the victim, spec 1 the leader/sealer, the rest fillers.
    let mut specs: Vec<Vec<Point>> = vec![victim_t, leader_t];
    specs.extend(filler_ts);
    let mut order: Vec<usize> = (0..specs.len()).collect();
    order.shuffle(rng);

    let mut board = Board::new(width, height);
    let mut victim_seat = 0usize;
    let mut leader_seat = 0usize;
    for (seat, &spec_i) in order.iter().enumerate() {
        board.add_snake(&specs[spec_i]);
        match spec_i {
            0 => victim_seat = seat,
            1 => leader_seat = seat,
            _ => {}
        }
    }
    for s in &mut board.snakes {
        s.health = rng.gen_range(40..=100);
    }
    board.food = sym.map_all(&foods);
    board.turn = rng.gen_range(40..=160);

    if !heads_have_escape(&board) {
        return None;
    }
    // Trap-machinery zone = everything outside the filler region, transformed.
    let reserved: Vec<Point> = (0..width)
        .flat_map(|x| (0..height).map(move |y| Point::new(x, y)))
        .filter(|&p| !filler_region(p))
        .map(|p| sym.map(p))
        .collect();
    Some(TrapLayout {
        board,
        victim: victim_seat,
        leader: leader_seat,
        doom_path: sym.map_all(&doom_path),
        leader_path: sym.map_all(&leader_path),
        escape: sym.map(escape),
        death_by,
        reserved,
    })
}

fn heads_have_escape(b: &Board) -> bool {
    b.snakes.iter().all(|s| {
        let h = s.head();
        CARDINALS.iter().any(|&(dx, dy)| {
            let p = Point::new(h.x + dx, h.y + dy);
            b.in_bounds(p) && !b.snakes.iter().any(|t| t.body.iter().any(|q| q == p))
        })
    })
}

/// Canonical tail-trap geometry (see `tail_trap`): corridor floor on row 0,
/// leader ceiling on row 1, one free floor cell at the mouth so the chase is
/// margin-1, leader head rising toward a bait food K moves up.
fn tail_trap_layout<R: Rng>(width: i8, height: i8, n: usize, rng: &mut R) -> Option<TrapLayout> {
    if width != height || width < 10 || n < 2 {
        return None;
    }
    for _ in 0..32 {
        let xm = rng.gen_range(4..=6i8); // leader's tail column = the chase entry
        let k = rng.gen_range(2..=3i8); // leader's moves to the bait
        let max_lv = (width - 2 - xm).min(4).max(3);
        let lv = rng.gen_range(3..=max_lv); // victim body cells behind its head

        // Leader head-first: head above the corridor, along the ceiling,
        // hooking into the floor — its TAIL (xm, 0) sits directly ahead of the
        // victim, so the chase is margin-0 (onto the cell as it vacates —
        // legal under the official simultaneous-vacate rule). One eat delays
        // every pop by a turn, so the victim arrives a turn early and dies.
        let mut leader = vec![Point::new(xm, 2)];
        leader.extend((1..=xm).rev().map(|x| Point::new(x, 1)));
        leader.extend((1..=xm).map(|x| Point::new(x, 0)));
        let victim: Vec<Point> = (0..=lv).map(|i| Point::new(xm + 1 + i, 0)).collect();
        let bait = Point::new(xm, 2 + k);
        let leader_path: Vec<Point> = (3..=2 + k).map(|y| Point::new(xm, y)).collect();
        let doom_path: Vec<Point> = (1..=xm).rev().map(|x| Point::new(x, 0)).collect();

        let band = 2 + k + 2;
        if let Some(l) = assemble_trap(
            width,
            height,
            n,
            rng,
            victim,
            leader,
            vec![bait],
            doom_path,
            leader_path,
            Point::new(xm + 1, 1),
            (k + 3) as u32,
            &move |p| p.y >= band,
        ) {
            return Some(l);
        }
    }
    None
}

/// Canonical pocket geometry (see `pocket`): width-1 dead-end on row 0 sealed
/// by the sealer's head-side body on row 1 (its tail rises far up the mouth
/// column, so the walls outlive the victim), bait food deep inside.
fn pocket_layout<R: Rng>(width: i8, height: i8, n: usize, rng: &mut R) -> Option<TrapLayout> {
    if width != height || width < 10 || n < 2 {
        return None;
    }
    for _ in 0..32 {
        let xd = rng.gen_range(4..=6i8); // pocket depth (mouth at x = xd)
        let bx = rng.gen_range(1..=xd - 2); // bait column, deep enough to commit
        let max_lv = (width - 2 - xd).min(4).max(3);
        let lv = rng.gen_range(3..=max_lv);

        // Sealer head-first: head tucked in the corner (its wall cells are
        // head-side, draining LAST), ceiling row 1, tail rising at the mouth.
        let mut sealer = vec![Point::new(0, 2), Point::new(0, 1)];
        sealer.extend((1..=xd).map(|x| Point::new(x, 1)));
        sealer.extend((2..=4).map(|y| Point::new(xd, y)));
        let victim: Vec<Point> = (0..=lv).map(|i| Point::new(xd + 1 + i, 0)).collect();
        let doom_path: Vec<Point> = (0..=xd).rev().map(|x| Point::new(x, 0)).collect();
        // Sim-scripted safe lane for the sealer: straight up the corner column
        // into open space, so a dumb pilot can't self-trap and dismantle the
        // walls mid-proof. Fillers keep out of that lane (region below).
        let sealer_path: Vec<Point> = (3..=7).map(|y| Point::new(0, y)).collect();

        if let Some(l) = assemble_trap(
            width,
            height,
            n,
            rng,
            victim,
            sealer,
            vec![Point::new(bx, 0)],
            doom_path,
            sealer_path,
            Point::new(xd + 1, 1),
            xd as u32 + 4,
            &|p| p.y >= 6 && p.x >= 2,
        ) {
            return Some(l);
        }
    }
    None
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
            let Some(body) = grow_body_in(&board, &occupied, target_len, rng, &|_| true) else {
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

/// Self-avoiding random walk from a random free head cell within `region`;
/// returns head-first segments. Accepts a shorter body (>= 3) when the walk
/// boxes itself in.
fn grow_body_in<R: Rng>(
    board: &Board,
    occupied: &[Point],
    target_len: usize,
    rng: &mut R,
    region: &dyn Fn(Point) -> bool,
) -> Option<Vec<Point>> {
    let free = |p: Point, own: &[Point]| {
        board.in_bounds(p) && region(p) && !occupied.contains(&p) && !own.contains(&p)
    };
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
    use crate::Move;
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
        assert!(scenario("tail-trap").is_some());
        assert!(scenario("pocket").is_some());
        assert!(scenario("nope").is_none());
    }

    // ---- simulation harness: prove traps with the real engine ------------
    //
    // Structural checks can't prove a trap is REAL; only playing it out can.
    // Each layout is simulated twice with the actual `Board::step`:
    //   doom line   — victim scripted into the trap, leader scripted to the
    //                 bait: the victim MUST die, and no earlier than turn 3
    //                 (beyond the depth-2 search horizon — that's the point).
    //   escape line — victim takes the escape move then plays safe-greedy:
    //                 it MUST survive, proving the trap was a choice, not a
    //                 death sentence.

    type Script = Box<dyn FnMut(&Board, usize) -> Move>;

    fn move_toward(from: Point, to: Point) -> Option<Move> {
        Move::ALL.into_iter().find(|m| m.apply(from) == to)
    }

    /// Follow `path` cell by cell; safe-greedy once it's exhausted/blocked.
    fn path_script(path: Vec<Point>, avoid: Vec<Point>) -> Script {
        let mut cursor = 0usize;
        Box::new(move |b: &Board, s: usize| {
            let head = b.snakes[s].head();
            while cursor < path.len() && path[cursor] == head {
                cursor += 1;
            }
            if cursor < path.len() {
                if let Some(m) = move_toward(head, path[cursor]) {
                    cursor += 1;
                    return m;
                }
            }
            safe_greedy(b, s, &avoid)
        })
    }

    fn greedy_script(avoid: Vec<Point>) -> Script {
        Box::new(move |b: &Board, s: usize| safe_greedy(b, s, &avoid))
    }

    /// Pick the move into a free cell with the largest free region, shunning
    /// `avoid` cells (the trap machinery) and cells in reach of a longer-or-
    /// equal opponent's head, unless nothing better is legal.
    fn safe_greedy(b: &Board, s: usize, avoid: &[Point]) -> Move {
        let head = b.snakes[s].head();
        let my_len = b.snakes[s].len();
        let occupied = |p: Point| {
            b.snakes.iter().filter(|t| t.alive()).any(|t| t.body.iter().any(|q| q == p))
        };
        let risky = |p: Point| {
            b.snakes.iter().enumerate().any(|(i, t)| {
                i != s
                    && t.alive()
                    && t.len() >= my_len
                    && (t.head().x - p.x).abs() + (t.head().y - p.y).abs() == 1
            })
        };
        // (not head-to-head risky, outside avoid, area) lexicographic.
        let mut best: Option<(bool, bool, usize, Move)> = None;
        for m in Move::ALL {
            let p = m.apply(head);
            if !b.in_bounds(p) || occupied(p) {
                continue;
            }
            let area = free_space_bfs(b, p).iter().flatten().count();
            let cand = (!risky(p), !avoid.contains(&p), area, m);
            if best.as_ref().is_none_or(|bst| (cand.0, cand.1, cand.2) > (bst.0, bst.1, bst.2)) {
                best = Some(cand);
            }
        }
        best.map(|(_, _, _, m)| m).unwrap_or(Move::Up)
    }

    /// Step the engine with per-seat scripts until `max_turns`; returns each
    /// seat's death turn (1-based, relative to the sim start).
    fn run_sim(mut b: Board, mut scripts: Vec<Script>, max_turns: u32) -> Vec<Option<u32>> {
        b.turn = 0; // relative clock so assertions are absolute
        let n = b.snakes.len();
        let mut deaths: Vec<Option<u32>> = vec![None; n];
        for t in 1..=max_turns {
            let moves: Vec<Move> = (0..n)
                .map(|s| if b.snakes[s].alive() { scripts[s](&b, s) } else { Move::Up })
                .collect();
            b.step(&moves);
            for s in 0..n {
                if !b.snakes[s].alive() && deaths[s].is_none() {
                    deaths[s] = Some(t);
                }
            }
            if b.alive_count() <= 1 {
                break;
            }
        }
        deaths
    }

    /// The trap theorem is about the two trap parties; fillers are live-player
    /// noise that can perturb either side of the proof. Sim on a copy with the
    /// fillers eliminated (their structural validity is asserted separately).
    fn two_party(l: &TrapLayout) -> Board {
        let mut b = l.board.clone();
        for (s, snake) in b.snakes.iter_mut().enumerate() {
            if s != l.victim && s != l.leader {
                snake.eliminated = Some(crate::EliminatedCause::Collision);
            }
        }
        b
    }

    fn scripts_for(l: &TrapLayout, victim_path: Vec<Point>) -> Vec<Script> {
        (0..l.board.snakes.len())
            .map(|s| {
                if s == l.victim {
                    path_script(victim_path.clone(), l.reserved.clone())
                } else if s == l.leader && !l.leader_path.is_empty() {
                    path_script(l.leader_path.clone(), l.reserved.clone())
                } else {
                    greedy_script(l.reserved.clone())
                }
            })
            .collect()
    }

    fn assert_trap_real(
        mut make: impl FnMut(&mut Xoshiro256PlusPlus) -> Option<TrapLayout>,
        name: &str,
        seeds: u64,
    ) {
        let mut built = 0;
        for seed in 0..seeds {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let Some(l) = make(&mut rng) else { continue };
            built += 1;

            // Doom line: victim into the trap, leader to the bait.
            let scripts = scripts_for(&l, l.doom_path.clone());
            let deaths = run_sim(two_party(&l), scripts, l.death_by + 2);
            let died = deaths[l.victim];
            assert!(
                died.is_some_and(|t| (3..=l.death_by).contains(&t)),
                "{name} seed {seed}: doom line should kill the victim in turns 3..={} (beyond \
                 the search horizon), got {died:?}",
                l.death_by
            );
            assert!(
                deaths[l.leader].is_none(),
                "{name} seed {seed}: the trap owner must outlive its own trap"
            );

            // Escape line: one move to the escape cell, then safe-greedy.
            let scripts = scripts_for(&l, vec![l.escape]);
            let deaths = run_sim(two_party(&l), scripts, 8);
            assert!(
                deaths[l.victim].is_none(),
                "{name} seed {seed}: escape line should survive 8 turns, died at {:?}",
                deaths[l.victim]
            );
        }
        assert!(built >= seeds / 2, "{name}: only {built}/{seeds} layouts built");
    }

    #[test]
    fn tail_trap_is_a_real_trap_with_a_real_escape() {
        assert_trap_real(|rng| tail_trap_layout(11, 11, 4, rng), "tail-trap", 150);
    }

    #[test]
    fn pocket_is_a_real_trap_with_a_real_escape() {
        assert_trap_real(|rng| pocket_layout(11, 11, 4, rng), "pocket", 150);
    }


    /// The pocket's bait must actually sit on the doom path, deep enough that
    /// taking it commits the victim (not the first cell).
    #[test]
    fn pocket_bait_is_inside_the_pocket() {
        for seed in 0..100u64 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let Some(l) = pocket_layout(11, 11, 4, &mut rng) else { continue };
            let bait = l.board.food[0];
            let pos = l.doom_path.iter().position(|&p| p == bait);
            assert!(pos.is_some_and(|i| i >= 2), "bait {bait:?} not deep in {:?}", l.doom_path);
        }
    }
}
