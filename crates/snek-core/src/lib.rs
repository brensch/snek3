//! `snek-core`: a fast, faithful Battlesnake rules engine.
//!
//! The `step` pipeline is a direct port of the official standard ruleset
//! (`BattlesnakeOfficial/rules`, `standard.go`), preserving the exact ordering:
//! move → reduce health → hazard damage → feed (grow) → eliminate.
//!
//! Faithfulness notes that callers rely on:
//! - Feeding happens *before* elimination, so eating food on the same turn as a
//!   head-to-head increases length and can win the tie.
//! - Snakes eliminated by wall/starvation this turn are not collision targets.
//! - Collision eliminations are gathered, then applied together, so two snakes
//!   can eliminate each other simultaneously.
//! - Growth duplicates the tail; this is what lets a snake legally chase a tail
//!   that is moving away, but not one that just ate.

pub mod baseline;
pub mod body;
pub mod encode;
pub mod setup;

#[cfg(feature = "json")]
pub mod json;

pub use body::{Body, MAX_BODY};
pub use encode::{encode_into, obs_h, obs_len, obs_side, obs_w, NUM_CHANNELS};
pub use setup::standard_start;

/// Maximum number of snakes supported in a single game.
pub const MAX_SNAKES: usize = 8;

/// Health a snake is reset to after eating, and the starting health.
pub const SNAKE_MAX_HEALTH: i16 = 100;
pub const SNAKE_START_HEALTH: i16 = 100;

use arrayvec::ArrayVec;
use rand::Rng;

/// A board coordinate. Stored as signed bytes so a head can transiently step
/// out of bounds (to -1 or `width`) before the elimination step removes it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Point {
    pub x: i8,
    pub y: i8,
}

impl Point {
    #[inline]
    pub const fn new(x: i8, y: i8) -> Self {
        Point { x, y }
    }
}

/// The four moves. Discriminants are stable and used as policy indices.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Move {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl Move {
    pub const ALL: [Move; 4] = [Move::Up, Move::Down, Move::Left, Move::Right];

    /// Apply this move to a head, returning the new head (may be out of bounds).
    /// y grows upward, matching the official ruleset (Up = +Y).
    #[inline]
    pub fn apply(self, head: Point) -> Point {
        match self {
            Move::Up => Point::new(head.x, head.y + 1),
            Move::Down => Point::new(head.x, head.y - 1),
            Move::Left => Point::new(head.x - 1, head.y),
            Move::Right => Point::new(head.x + 1, head.y),
        }
    }

    #[inline]
    pub fn from_index(i: usize) -> Move {
        Move::ALL[i]
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EliminatedCause {
    Collision,
    SelfCollision,
    OutOfBounds,
    OutOfHealth,
    HeadToHead,
    Hazard,
}

#[derive(Clone, Debug)]
pub struct Snake {
    pub body: Body,
    pub health: i16,
    pub eliminated: Option<EliminatedCause>,
}

impl Snake {
    #[inline]
    pub fn alive(&self) -> bool {
        self.eliminated.is_none()
    }

    #[inline]
    pub fn head(&self) -> Point {
        self.body.head()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.body.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct Board {
    pub width: i8,
    pub height: i8,
    pub turn: u32,
    pub snakes: ArrayVec<Snake, MAX_SNAKES>,
    pub food: Vec<Point>,
    pub hazards: Vec<Point>,
    /// Damage applied to a snake whose head sits on a hazard (0 disables hazards).
    pub hazard_damage: i16,
    /// Food the board is topped up to each turn (official default 1).
    pub min_food: i32,
    /// Percent chance per turn to spawn one extra food once at/above min (15).
    pub food_spawn_chance: i32,
}

impl Board {
    pub fn new(width: i8, height: i8) -> Self {
        Board {
            width,
            height,
            turn: 0,
            snakes: ArrayVec::new(),
            food: Vec::new(),
            hazards: Vec::new(),
            hazard_damage: 14,
            min_food: 1,
            food_spawn_chance: 15,
        }
    }

    /// Construct a snake from an ordered list of body points (head first) and
    /// add it to the board with full health.
    pub fn add_snake(&mut self, segments: &[Point]) {
        let mut body = Body::new();
        body.init_from_head_first(segments);
        self.snakes.push(Snake {
            body,
            health: SNAKE_START_HEALTH,
            eliminated: None,
        });
    }

    #[inline]
    pub fn in_bounds(&self, p: Point) -> bool {
        p.x >= 0 && p.x < self.width && p.y >= 0 && p.y < self.height
    }

    /// Number of snakes still alive.
    pub fn alive_count(&self) -> usize {
        self.snakes.iter().filter(|s| s.alive()).count()
    }

    /// Standard game-over: one or zero snakes remain.
    pub fn is_terminal(&self) -> bool {
        self.alive_count() <= 1
    }

    /// The sole surviving snake index, if the game is over with a winner.
    /// Returns `None` for a draw (zero alive) or an ongoing game.
    pub fn winner(&self) -> Option<usize> {
        if self.alive_count() == 1 {
            self.snakes.iter().position(|s| s.alive())
        } else {
            None
        }
    }

    /// Advance the board one turn. `moves[i]` is the move for snake `i`; the
    /// length must equal the number of snakes. Moves for eliminated snakes are
    /// ignored. This mutates the board in place.
    pub fn step(&mut self, moves: &[Move]) {
        debug_assert_eq!(moves.len(), self.snakes.len());
        self.move_snakes(moves);
        self.reduce_health();
        self.damage_hazards();
        self.feed_snakes();
        self.eliminate_snakes();
        self.turn += 1;
    }

    /// Like [`Board::step`] but also spawns food (the official `SpawnFoodStandard`
    /// behavior). Use this to advance a *real* game; the search uses the plain
    /// deterministic `step` so its lookahead transitions don't branch on random
    /// food appearing.
    pub fn step_and_spawn<R: Rng>(&mut self, moves: &[Move], rng: &mut R) {
        self.step(moves);
        self.spawn_food(rng);
    }

    /// Top up to `min_food`, otherwise spawn one food with `food_spawn_chance`%.
    fn spawn_food<R: Rng>(&mut self, rng: &mut R) {
        let cur = self.food.len() as i32;
        if cur < self.min_food {
            self.place_food_randomly(self.min_food - cur, rng);
        } else if self.food_spawn_chance > 0 && rng.gen_range(0..100) < self.food_spawn_chance {
            self.place_food_randomly(1, rng);
        }
    }

    fn place_food_randomly<R: Rng>(&mut self, count: i32, rng: &mut R) {
        for _ in 0..count {
            let pts = self.unoccupied_points();
            if !pts.is_empty() {
                self.food.push(pts[rng.gen_range(0..pts.len())]);
            }
        }
    }

    /// Cells not occupied by a body, existing food, or adjacent to a live head
    /// (matches the official `GetUnoccupiedPoints(b, false, false)`).
    fn unoccupied_points(&self) -> Vec<Point> {
        let (w, h) = (self.width as usize, self.height as usize);
        let mut occ = vec![false; w * h];
        let mark = |occ: &mut [bool], x: i8, y: i8| {
            if x >= 0 && x < self.width && y >= 0 && y < self.height {
                occ[y as usize * w + x as usize] = true;
            }
        };
        for &f in &self.food {
            mark(&mut occ, f.x, f.y);
        }
        for s in &self.snakes {
            if !s.alive() {
                continue;
            }
            for p in s.body.iter() {
                mark(&mut occ, p.x, p.y);
            }
            let head = s.head();
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                mark(&mut occ, head.x + dx, head.y + dy);
            }
        }
        let mut pts = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if !occ[y as usize * w + x as usize] {
                    pts.push(Point::new(x, y));
                }
            }
        }
        pts
    }

    fn move_snakes(&mut self, moves: &[Move]) {
        for (i, snake) in self.snakes.iter_mut().enumerate() {
            if snake.eliminated.is_some() {
                continue;
            }
            let new_head = moves[i].apply(snake.body.head());
            snake.body.advance(new_head);
        }
    }

    fn reduce_health(&mut self) {
        for snake in self.snakes.iter_mut() {
            if snake.eliminated.is_none() {
                snake.health -= 1;
            }
        }
    }

    fn damage_hazards(&mut self) {
        if self.hazards.is_empty() || self.hazard_damage == 0 {
            return;
        }
        for i in 0..self.snakes.len() {
            let snake = &mut self.snakes[i];
            if snake.eliminated.is_some() {
                continue;
            }
            let head = snake.body.head();
            // Hazard damage is skipped if the head is also on food this turn.
            if self.hazards.contains(&head) && !self.food.contains(&head) {
                snake.health -= self.hazard_damage;
                snake.health = snake.health.clamp(0, SNAKE_MAX_HEALTH);
                if snake.health <= 0 {
                    snake.eliminated = Some(EliminatedCause::Hazard);
                }
            }
        }
    }

    fn feed_snakes(&mut self) {
        if self.food.is_empty() {
            return;
        }
        let mut remaining = Vec::with_capacity(self.food.len());
        for &food in &self.food {
            let mut eaten = false;
            for snake in self.snakes.iter_mut() {
                if snake.eliminated.is_some() || snake.body.is_empty() {
                    continue;
                }
                if snake.body.head() == food {
                    snake.body.grow();
                    snake.health = SNAKE_MAX_HEALTH;
                    eaten = true;
                }
            }
            if !eaten {
                remaining.push(food);
            }
        }
        self.food = remaining;
    }

    fn eliminate_snakes(&mut self) {
        // First pass: out of health, then out of bounds. These are applied
        // immediately so the collision pass does not treat them as targets.
        let (width, height) = (self.width, self.height);
        for snake in self.snakes.iter_mut() {
            if snake.eliminated.is_some() {
                continue;
            }
            if snake.health <= 0 {
                snake.eliminated = Some(EliminatedCause::OutOfHealth);
                continue;
            }
            let head = snake.body.head();
            let in_bounds = head.x >= 0 && head.x < width && head.y >= 0 && head.y < height;
            if !in_bounds {
                snake.eliminated = Some(EliminatedCause::OutOfBounds);
            }
        }

        // Collision pass: gather eliminations, then apply, so collisions are
        // resolved against this turn's positions simultaneously.
        let mut eliminations: ArrayVec<(usize, EliminatedCause), MAX_SNAKES> = ArrayVec::new();
        for i in 0..self.snakes.len() {
            if self.snakes[i].eliminated.is_some() {
                continue;
            }
            let head = self.snakes[i].body.head();

            // Self-collision: head against own body, excluding the head segment.
            if self.snakes[i].body.collides_excluding_head(head) {
                eliminations.push((i, EliminatedCause::SelfCollision));
                continue;
            }

            // Body collision against any other live snake's non-head segments.
            let mut body_collided = false;
            for j in 0..self.snakes.len() {
                if i == j || self.snakes[j].eliminated.is_some() {
                    continue;
                }
                if self.snakes[j].body.collides_excluding_head(head) {
                    eliminations.push((i, EliminatedCause::Collision));
                    body_collided = true;
                    break;
                }
            }
            if body_collided {
                continue;
            }

            // Head-to-head: lose if heads coincide and we are not strictly longer.
            let my_len = self.snakes[i].body.len();
            for j in 0..self.snakes.len() {
                if i == j || self.snakes[j].eliminated.is_some() {
                    continue;
                }
                if self.snakes[j].body.head() == head && my_len <= self.snakes[j].body.len() {
                    eliminations.push((i, EliminatedCause::HeadToHead));
                    break;
                }
            }
        }

        for (i, cause) in eliminations {
            self.snakes[i].eliminated = Some(cause);
        }
    }
}

#[cfg(test)]
mod food_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn food_is_replenished_during_play() {
        let mut b = Board::new(11, 11);
        b.add_snake(&[Point::new(5, 5), Point::new(5, 4), Point::new(5, 3)]);
        b.food.clear(); // start with none
        let mut rng = StdRng::seed_from_u64(1);
        // min_food default 1, so within a few turns food should appear and persist.
        let mut seen_food = false;
        for _ in 0..10 {
            b.step_and_spawn(&[Move::Up], &mut rng);
            if !b.food.is_empty() {
                seen_food = true;
            }
        }
        assert!(seen_food, "food should spawn to maintain the minimum");
        assert!(!b.food.is_empty());
    }

    #[test]
    fn food_never_spawns_on_a_snake_or_next_to_a_head() {
        let mut b = Board::new(7, 7);
        b.add_snake(&[Point::new(3, 3), Point::new(3, 2)]);
        b.food.clear();
        b.min_food = 5; // force several placements
        let mut rng = StdRng::seed_from_u64(7);
        b.step_and_spawn(&[Move::Up], &mut rng);
        let head = b.snakes[0].head();
        for &f in &b.food {
            assert!(b.in_bounds(f));
            // not on a body cell
            assert!(!b.snakes[0].body.iter().any(|p| p == f));
            // not orthogonally adjacent to the head
            let adj = (f.x - head.x).abs() + (f.y - head.y).abs() == 1;
            assert!(!adj, "food spawned adjacent to head");
        }
    }
}
