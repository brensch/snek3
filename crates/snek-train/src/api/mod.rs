mod viewer;

use crate::config::RunConfig;
use crate::trainer::{StartRequest, TrainerHandle};
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use prost::Message;
use serde_json::json;
use std::convert::Infallible;
use std::path::Path as FsPath;
use std::time::Duration;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};

pub fn router(trainer: TrainerHandle, static_dir: Option<&FsPath>) -> Router {
    let mut router = Router::new()
        .route("/api/stream/stats", get(stream_stats))
        .route("/api/stream/logs", get(stream_logs))
        .route("/api/stream/eval", get(stream_eval))
        .route("/api/state", get(state))
        .route("/api/config", get(config))
        .route("/api/control/start", post(start))
        .route("/api/control/stop", post(stop))
        .route("/api/bench/stream", get(bench_stream))
        .route("/api/runs", get(runs))
        .route("/api/runs/:id", get(run_detail))
        .route("/api/runs/:id/config", post(set_run_config))
        .route("/api/runs/:id/games/:gen", get(run_game))
        .route("/api/runs/:id/eval/:seq/:gen/:opp", get(run_eval_game))
        .route("/api/metrics/history", get(history))
        .layer(CorsLayer::permissive())
        .with_state(trainer);
    if let Some(dir) = static_dir {
        // SPA fallback: unknown paths get index.html so client-side routes work.
        let serve = ServeDir::new(dir).fallback(ServeFile::new(dir.join("index.html")));
        router = router.fallback_service(serve);
    }
    router
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

async fn stream_logs(
    State(trainer): State<TrainerHandle>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let mut rx = trainer.metrics().log_rx();
    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(entry) => match serde_json::to_string(&entry) {
                    Ok(data) => yield Ok(Event::default().event("log").data(data)),
                    Err(_) => continue,
                },
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::new().interval(Duration::from_secs(10)))
}

/// Live view of the evaluation league's in-flight match (who is playing whom,
/// which turn each game is up to). Poll-based: the league updates a shared
/// snapshot as arena events stream in; we emit it once a second.
async fn stream_eval(
    State(_trainer): State<TrainerHandle>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        loop {
            if let Ok(data) = serde_json::to_string(&crate::eval::live()) {
                yield Ok(Event::default().event("eval").data(data));
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    };
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::new().interval(Duration::from_secs(10)))
}

/// Run a GPU batch-size throughput sweep, streaming progress as JSON events (same
/// SSE contract as the log stream). Refuses — via a single `error` event — if a
/// training run is active, since the sweep needs the GPU to itself. The sweep runs
/// on a dedicated thread; the channel closing ends the stream.
async fn bench_stream(
    State(trainer): State<TrainerHandle>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<crate::bench::BenchEvent>();
    match trainer.try_begin_bench() {
        Ok(()) => {
            let cfg = trainer.config();
            let handle = trainer.clone();
            std::thread::spawn(move || {
                crate::bench::run_sweep(&cfg, &tx);
                handle.end_bench();
            });
        }
        Err(err) => {
            // Emit one error event, then let the channel close to end the stream.
            let _ = tx.send(crate::bench::BenchEvent::Error {
                detail: err.to_string(),
            });
        }
    }
    let stream = async_stream::stream! {
        while let Some(ev) = rx.recv().await {
            match serde_json::to_string(&ev) {
                Ok(data) => yield Ok(Event::default().event("bench").data(data)),
                Err(_) => continue,
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

// The in-memory default config, used to seed the "start fresh run" knob form.
async fn config(State(trainer): State<TrainerHandle>) -> Json<RunConfig> {
    Json(trainer.config())
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
    let active = trainer.active_run_id();
    let reply = viewer::run_list(trainer.runs_dir(), active.as_deref(), trainer.is_running());
    Protobuf(reply).into_response()
}

async fn run_detail(State(trainer): State<TrainerHandle>, Path(id): Path<String>) -> Response {
    let Some(root) = viewer::resolve_run(trainer.runs_dir(), &id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let active = trainer.active_run_id();
    let detail = viewer::run_detail(&root, &id, active.as_deref(), trainer.is_running());
    Protobuf(detail).into_response()
}

/// Persist a run's config to its `config.json` on disk. Works for any run, live
/// or not; run_loop reloads config.json on resume. If the edited run is the
/// active one, the in-memory config is updated too so it applies at the next
/// generation boundary.
async fn set_run_config(
    State(trainer): State<TrainerHandle>,
    Path(id): Path<String>,
    Json(cfg): Json<RunConfig>,
) -> Response {
    let Some(root) = viewer::resolve_run(trainer.runs_dir(), &id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if let Err(err) = cfg.save_atomic(&root.join("config.json")) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "detail": err.to_string() })),
        )
            .into_response();
    }
    if trainer.active_run_id().as_deref() == Some(id.as_str()) {
        trainer.set_config(cfg.clone());
    }
    Json(cfg).into_response()
}

async fn run_game(
    State(trainer): State<TrainerHandle>,
    Path((id, gen)): Path<(String, u32)>,
) -> Response {
    let game =
        viewer::resolve_run(trainer.runs_dir(), &id).and_then(|root| viewer::game_file(&root, gen));
    match game {
        Some(file) => Protobuf(file).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn run_eval_game(
    State(trainer): State<TrainerHandle>,
    Path((id, seq, gen, opp)): Path<(String, u64, u32, u32)>,
) -> Response {
    let game = viewer::resolve_run(trainer.runs_dir(), &id)
        .and_then(|root| viewer::eval_game_file(&root, seq, gen, opp));
    match game {
        Some(file) => Protobuf(file).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Wraps a protobuf message as an `application/x-protobuf` response body.
struct Protobuf<M>(M);

impl<M: Message> IntoResponse for Protobuf<M> {
    fn into_response(self) -> Response {
        let mut buf = Vec::new();
        self.0
            .encode(&mut buf)
            .expect("encoding protobuf to Vec cannot fail");
        ([(header::CONTENT_TYPE, "application/x-protobuf")], buf).into_response()
    }
}

async fn history(State(trainer): State<TrainerHandle>) -> Response {
    match trainer.history() {
        Ok(metrics) => Json(json!({ "metrics": metrics })).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "detail": err.to_string() })),
        )
            .into_response(),
    }
}

fn proto_event<M: Message>(event: &'static str, msg: M) -> Event {
    let mut buf = Vec::new();
    msg.encode(&mut buf)
        .expect("encoding protobuf to Vec cannot fail");
    Event::default().event(event).data(STANDARD.encode(buf))
}
