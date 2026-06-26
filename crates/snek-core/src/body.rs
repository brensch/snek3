//! A fixed-capacity ring buffer for a snake body.
//!
//! The head is at logical index 0; increasing indices walk toward the tail.
//! Movement (prepend head, drop tail) and growth (duplicate tail) are O(1) and
//! the whole struct is `Copy`, so cloning a board for search is cheap and
//! allocation-free.

use crate::Point;

/// Maximum body length. An 11x11 board has 121 cells; 128 leaves headroom and
/// keeps the index math a cheap power-of-two mask.
pub const MAX_BODY: usize = 128;
const MASK: usize = MAX_BODY - 1;

#[derive(Clone, Copy, Debug)]
pub struct Body {
    buf: [Point; MAX_BODY],
    /// Physical index of the head in `buf`.
    head: u16,
    len: u16,
}

impl Body {
    pub fn new() -> Self {
        Body {
            buf: [Point::new(0, 0); MAX_BODY],
            head: 0,
            len: 0,
        }
    }

    /// Initialize from a head-first slice of segments (index 0 is the head).
    pub fn init_from_head_first(&mut self, segments: &[Point]) {
        debug_assert!(segments.len() <= MAX_BODY);
        self.head = 0;
        self.len = segments.len() as u16;
        for (i, &p) in segments.iter().enumerate() {
            self.buf[i] = p;
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn phys(&self, logical: usize) -> usize {
        (self.head as usize + logical) & MASK
    }

    /// Segment at logical index (0 = head). Caller ensures `i < len`.
    #[inline]
    pub fn get(&self, i: usize) -> Point {
        self.buf[self.phys(i)]
    }

    #[inline]
    pub fn head(&self) -> Point {
        self.buf[self.head as usize]
    }

    #[inline]
    pub fn tail(&self) -> Point {
        self.get(self.len() - 1)
    }

    /// Move: prepend `new_head` and drop the tail (length unchanged). This is
    /// the per-turn movement; growth is handled separately by [`Body::grow`].
    #[inline]
    pub fn advance(&mut self, new_head: Point) {
        let new_pos = (self.head as usize + MAX_BODY - 1) & MASK;
        self.buf[new_pos] = new_head;
        self.head = new_pos as u16;
        // len stays the same: one segment added at the head, tail logically dropped.
    }

    /// Grow by duplicating the current tail segment (the official `growSnake`).
    #[inline]
    pub fn grow(&mut self) {
        debug_assert!(self.len() < MAX_BODY);
        let tail = self.tail();
        let new_tail_pos = (self.head as usize + self.len as usize) & MASK;
        self.buf[new_tail_pos] = tail;
        self.len += 1;
    }

    /// True if `p` equals any body segment except the head (logical index 0).
    /// Used for both self- and other-collision checks.
    #[inline]
    pub fn collides_excluding_head(&self, p: Point) -> bool {
        for i in 1..self.len() {
            if self.get(i) == p {
                return true;
            }
        }
        false
    }

    /// Iterator over body segments head-first. Useful for encoding/serialization.
    pub fn iter(&self) -> impl Iterator<Item = Point> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }
}

impl Default for Body {
    fn default() -> Self {
        Self::new()
    }
}
