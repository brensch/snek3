mod api;
mod config;
mod metrics;
mod proto;
mod replay;
mod selfplay;
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);
    tch::Cuda::cudnn_set_benchmark(true);
    let args = Args::parse();
    let metrics = Metrics::new();
    let trainer = TrainerHandle::new(args.runs_dir, metrics.clone(), RunConfig::default());
    tokio::spawn(metrics.run_samplers());
    if args.start {
        trainer.start(trainer::StartRequest {
            run_id: args.run_id,
            fresh: Some(args.fresh),
        })?;
    }
    let app = api::router(trainer);
    tracing::info!(bind = %args.bind, "snek-train API listening");
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
