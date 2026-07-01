import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { control } from "../api/client";
import { ConfigPanel } from "../components/ConfigPanel";
import { GameViewer } from "../components/GameViewer";
import { LogPanel } from "../components/LogPanel";
import { RunControls } from "../components/RunControls";
import { SeriesChart } from "../components/SeriesChart";
import { TrainingProgress } from "../components/TrainingProgress";
import { useLiveStats } from "../hooks/useLiveStats";
import { useLogs } from "../hooks/useLogs";
import { useRunDetail } from "../hooks/useRunDetail";
import { Phase } from "../gen/snek_pb";
import type { MetricRow } from "../gen/viewer_pb";
import type { RunConfig } from "../types";

// Per-generation charts, driven off metrics.jsonl. Each pulls one field.
const GEN_CHARTS: { pick: (m: MetricRow) => number; label: string; color: string; digits?: number; fixedMax?: number }[] = [
  { pick: (m) => m.policyLoss, label: "Policy loss", color: "#38bdf8", digits: 3 },
  { pick: (m) => m.valueLoss, label: "Value loss", color: "#f59e0b", digits: 3 },
  { pick: (m) => m.targetEntropy, label: "Target entropy", color: "#e879f9", digits: 3 },
  { pick: (m) => m.completedGames, label: "Games completed", color: "#a855f7", digits: 0 },
  { pick: (m) => m.turns, label: "Self-play turns", color: "#34d399", digits: 0 },
  { pick: (m) => m.samples, label: "Samples", color: "#60a5fa", digits: 0 },
  { pick: (m) => Number(m.buffer), label: "Buffer size", color: "#94a3b8", digits: 0 },
  { pick: (m) => m.genSeconds, label: "Gen seconds", color: "#fbbf24", digits: 1 },
  { pick: (m) => m.trainSeconds, label: "Train seconds", color: "#f472b6", digits: 1 },
  { pick: (m) => m.gamesPerSec, label: "Games / sec", color: "#22c55e", digits: 1 },
  { pick: (m) => m.inferencesPerSec, label: "Inferences / sec", color: "#38bdf8", digits: 0 },
  { pick: (m) => m.gpuBusyPct, label: "GPU busy %", color: "#eab308", digits: 0, fixedMax: 100 },
  { pick: (m) => m.avgGameTurn, label: "Avg game turn (buffer)", color: "#2dd4bf", digits: 1 },
];

// A single run, focused: no run switcher — just this run's metrics, knobs, live
// controls (when it is the active run), and its recorded sample games.
export function RunView() {
  const { runId = "" } = useParams();
  const { detail, error, loading } = useRunDetail(runId);
  const summary = detail?.summary ?? null;
  const isLive = summary?.live ?? false;
  const live = useLiveStats(isLive);
  const logs = useLogs();

  // Derive running/stopping from the live phase (the SSE stream updates it ~4×/s)
  // so the Stop button reflects the loop draining in near real time, falling back
  // to the slower run-summary poll when the run isn't live.
  const livePhase = live.stats?.phase ?? live.state?.phase ?? null;
  const running =
    livePhase != null
      ? livePhase !== Phase.IDLE && livePhase !== Phase.STOPPED
      : summary?.running ?? false;
  const stopping = livePhase === Phase.STOPPING;

  // Single config path for every run, running or not: the on-disk config.json
  // (from detail) is the source of truth, and saves go through one endpoint that
  // writes that file (and also updates the in-memory config when the run is live).
  const diskConfig = useMemo<RunConfig | null>(() => {
    if (!detail?.configJson) return null;
    try {
      return JSON.parse(detail.configJson) as RunConfig;
    } catch {
      return null;
    }
  }, [detail?.configJson]);

  // Optimistic override so the panel reflects a save immediately, before the next
  // detail poll reports the new config.json back.
  const [savedConfig, setSavedConfig] = useState<RunConfig | null>(null);
  useEffect(() => {
    if (savedConfig && diskConfig && JSON.stringify(savedConfig) === JSON.stringify(diskConfig)) {
      setSavedConfig(null);
    }
  }, [diskConfig, savedConfig]);

  const saveConfig = async (next: RunConfig) => {
    await control.setRunConfig(runId, next);
    setSavedConfig(next);
  };

  const metrics = detail?.metrics ?? [];
  const genLeft = metrics.length ? `gen ${metrics[0].generation}` : "";
  const genRight = metrics.length ? `gen ${metrics[metrics.length - 1].generation}` : "";
  const gens = metrics.map((m) => m.generation);
  const winRates = metrics.filter((m) => m.hasWinRate);

  return (
    <div className="mx-auto max-w-6xl px-5 py-6">
      <LogPanel logs={logs} />
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <Link to="/" className="text-sm text-slate-400 hover:text-sky-300">
          ← runs
        </Link>
        <h1 className="font-mono text-lg font-semibold text-slate-100">{runId}</h1>
        {summary && (
          <span className="text-xs text-slate-500">
            gen {summary.generation} · {summary.board}² · {summary.numSnakes}p
          </span>
        )}
        <div className="ml-auto">
          <RunControls
            live={isLive}
            running={running}
            stopping={stopping}
            onResume={() => control.start(runId, false).then(() => undefined)}
            onStop={() => control.stop().then(() => undefined)}
          />
        </div>
      </div>

      {error && <div className="mb-4 rounded border border-red-900 bg-red-950 p-3 text-sm text-red-200">{error}</div>}
      {loading && !detail && <div className="text-sm text-slate-500">Loading run…</div>}

      {isLive && (
        <section className="mb-4 grid gap-3">
          <TrainingProgress stats={live.stats} state={live.state} />
          <div className="grid gap-3 lg:grid-cols-4">
            <SeriesChart values={live.history.map((r) => r.inferencesPerSec)} label="Inference rate" digits={0} xUnit="t" />
            <SeriesChart values={live.history.map((r) => r.gpuRowsPerSec)} label="GPU rows/s" color="#a855f7" digits={0} xUnit="t" />
            <SeriesChart values={live.history.map((r) => r.gamesPerSec)} label="Game rate" color="#22c55e" digits={1} xUnit="t" />
            <SeriesChart values={live.history.map((r) => r.gpuBusyPct)} label="GPU busy" color="#eab308" fixedMax={100} digits={0} xUnit="t" />
          </div>
        </section>
      )}

      {metrics.length > 0 && (
        <section className="mb-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {GEN_CHARTS.map((chart) => (
            <SeriesChart
              key={chart.label}
              values={metrics.map(chart.pick)}
              label={`${chart.label} / gen`}
              color={chart.color}
              digits={chart.digits}
              fixedMax={chart.fixedMax ?? null}
              xValues={gens}
              xLeft={genLeft}
              xRight={genRight}
            />
          ))}
          {winRates.length > 0 && (
            <SeriesChart values={winRates.map((m) => m.winRate)} label="Win rate / gen" color="#22c55e" fixedMax={1} digits={2} xValues={winRates.map((m) => m.generation)} />
          )}
        </section>
      )}

      <section className="mb-4">
        <ConfigPanel config={savedConfig ?? diskConfig} onSave={saveConfig} />
      </section>

      <section>
        <h2 className="section-title mb-2">Sample games</h2>
        <GameViewer runId={runId} gameGens={detail?.gameGens ?? []} metrics={metrics} />
      </section>
    </div>
  );
}
