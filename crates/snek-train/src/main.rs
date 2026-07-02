mod api;
mod config;
mod eval;
mod metrics;
mod proto;
mod replay;
mod sample;
mod selfplay;
mod session;
mod state;
mod train;
mod trainer;

use clap::Parser;
use config::RunConfig;
use metrics::Metrics;
use std::net::SocketAddr;
use std::path::PathBuf;
use trainer::TrainerHandle;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8050")]
    bind: SocketAddr,
    #[arg(long, default_value = "runs")]
    runs_dir: PathBuf,
    #[arg(long)]
    run_id: Option<String>,
    #[arg(long)]
    fresh: bool,
    #[arg(long, default_value_t = false)]
    start: bool,
    /// Directory of built frontend assets to serve at /. Skipped if missing.
    #[arg(long, default_value = "frontend/dist")]
    static_dir: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);
    tch::Cuda::cudnn_set_benchmark(std::env::var("SNEK_CUDNN_BENCH").as_deref() != Ok("0"));
    let args = Args::parse();
    let metrics = Metrics::new();
    let trainer = TrainerHandle::new(args.runs_dir, metrics.clone(), RunConfig::default());
    tokio::spawn(metrics.run_samplers());
    if args.start {
        trainer.start(trainer::StartRequest {
            run_id: args.run_id,
            fresh: Some(args.fresh),
            config: None,
        })?;
    }
    let static_dir = args.static_dir.join("index.html").is_file().then_some(args.static_dir.as_path());
    match static_dir {
        Some(dir) => tracing::info!(dir = %dir.display(), "serving frontend"),
        None => tracing::warn!(dir = %args.static_dir.display(), "no built frontend found; serving API only"),
    }
    let app = api::router(trainer, static_dir);
    tracing::info!(bind = %args.bind, "snek-train API listening");
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
