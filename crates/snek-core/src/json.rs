//! Parsing of the official Battlesnake `/move` request payload into a [`Board`].
//! The Battlesnake API uses the same coordinate convention as this engine:
//! (0, 0) is bottom-left and Up increases Y.

use crate::{Board, EliminatedCause, Point, Snake};
use serde_json::Value;

fn point(v: &Value) -> Option<Point> {
    Some(Point::new(
        v.get("x")?.as_i64()? as i8,
        v.get("y")?.as_i64()? as i8,
    ))
}

fn points(v: &Value) -> Vec<Point> {
    v.as_array()
        .map(|a| a.iter().filter_map(point).collect())
        .unwrap_or_default()
}

/// Parse a `/move` request body. Returns the board and the index of the snake
/// identified by `you` (the snake we are controlling).
pub fn parse_move_request(body: &str) -> Result<(Board, usize), String> {
    let v: Value = serde_json::from_str(body).map_err(|e| e.to_string())?;
    let board_v = v.get("board").ok_or("missing board")?;

    let width = board_v
        .get("width")
        .and_then(Value::as_i64)
        .ok_or("missing width")? as i8;
    let height = board_v
        .get("height")
        .and_then(Value::as_i64)
        .ok_or("missing height")? as i8;

    let mut board = Board::new(width, height);
    board.turn = v.get("turn").and_then(Value::as_u64).unwrap_or(0) as u32;
    board.food = points(board_v.get("food").unwrap_or(&Value::Null));
    board.hazards = points(board_v.get("hazards").unwrap_or(&Value::Null));

    let you_id = v
        .get("you")
        .and_then(|y| y.get("id"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let snakes_v = board_v
        .get("snakes")
        .and_then(Value::as_array)
        .ok_or("missing snakes")?;

    let mut me = 0usize;
    for (i, s) in snakes_v.iter().enumerate() {
        let body = points(s.get("body").unwrap_or(&Value::Null));
        if body.is_empty() {
            return Err("snake with empty body".into());
        }
        let health = s.get("health").and_then(Value::as_i64).unwrap_or(100) as i16;
        let mut snake_body = crate::Body::new();
        snake_body.init_from_head_first(&body);
        board.snakes.push(Snake {
            body: snake_body,
            health,
            eliminated: None::<EliminatedCause>,
        });
        if s.get("id").and_then(Value::as_str) == Some(you_id.as_str()) {
            me = i;
        }
    }

    Ok((board, me))
}
