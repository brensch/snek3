use crate::config::RunConfig;
use crate::trainer::{StartRequest, TrainerHandle};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use prost::Message;
use serde_json::json;
use std::convert::Infallible;
use std::time::Duration;
use tower_http::cors::CorsLayer;

pub fn router(trainer: TrainerHandle) -> Router {
    Router::new()
        .route("/api/stream/stats", get(stream_stats))
        .route("/api/stream/games", get(stream_games))
        .route("/api/state", get(state))
        .route("/api/config", get(get_config).post(set_config))
        .route("/api/control/start", post(start))
        .route("/api/control/stop", post(stop))
        .route("/api/runs", get(runs))
        .route("/api/metrics/history", get(history))
        .layer(CorsLayer::permissive())
        .with_state(trainer)
}

async fn stream_stats(
    State(trainer): State<TrainerHandle>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let mut rx = trainer.metrics().stats_rx();
    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(frame) => yield Ok(proto_event("stats", frame)),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::new().interval(Duration::from_secs(10)))
}

async fn stream_games(
    State(trainer): State<TrainerHandle>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let mut rx = trainer.metrics().games_rx();
    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(frame) => yield Ok(proto_event("games", frame)),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::new().interval(Duration::from_secs(10)))
}

async fn state(State(trainer): State<TrainerHandle>) -> Json<serde_json::Value> {
    let state = trainer.run_state();
    Json(json!({
        "phase": state.phase,
        "generation": state.generation,
        "run_id": state.run_id,
        "running": state.running,
        "device": trainer.device_label(),
    }))
}

async fn get_config(State(trainer): State<TrainerHandle>) -> Json<RunConfig> {
    Json(trainer.config())
}

async fn set_config(
    State(trainer): State<TrainerHandle>,
    Json(cfg): Json<RunConfig>,
) -> Json<RunConfig> {
    trainer.set_config(cfg.clone());
    Json(cfg)
}

async fn start(State(trainer): State<TrainerHandle>, Json(req): Json<StartRequest>) -> Response {
    match trainer.start(req) {
        Ok(run_id) => Json(json!({ "run_id": run_id })).into_response(),
        Err(err) => (
            StatusCode::CONFLICT,
            Json(json!({ "detail": err.to_string() })),
        )
            .into_response(),
    }
}

async fn stop(State(trainer): State<TrainerHandle>) -> Json<serde_json::Value> {
    trainer.stop();
    Json(json!({ "stopping": true }))
}

async fn runs(State(trainer): State<TrainerHandle>) -> Response {
    match trainer.runs() {
        Ok(runs) => Json(runs).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "detail": err.to_string() })),
        )
            .into_response(),
    }
}

async fn history() -> Json<serde_json::Value> {
    Json(json!({ "metrics": [] }))
}

fn proto_event<M: Message>(event: &'static str, msg: M) -> Event {
    let mut buf = Vec::new();
    msg.encode(&mut buf)
        .expect("encoding protobuf to Vec cannot fail");
    Event::default().event(event).data(STANDARD.encode(buf))
}
