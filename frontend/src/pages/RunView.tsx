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

// Per-generation charts, driven off metrics.jsonl. Related metrics that share a
// unit and scale are combined onto one chart (losses, timing, samples vs buffer)
// so the section reads as a handful of meaningful panels rather than a wall of
// single lines.
type GenSeries = { pick: (m: MetricRow) => number; name: string; color: string };
const GEN_CHARTS: { label: string; digits?: number; fixedMax?: number; series: GenSeries[] }[] = [
  {
    label: "Loss",
    digits: 3,
    series: [
      { pick: (m) => m.policyLoss, name: "policy", color: "#38bdf8" },
      { pick: (m) => m.valueLoss, name: "value", color: "#f59e0b" },
    ],
  },
  { label: "Target entropy", digits: 3, series: [{ pick: (m) => m.targetEntropy, name: "entropy", color: "#e879f9" }] },
  {
    label: "Phase timing (s)",
    digits: 1,
    series: [
      { pick: (m) => m.genSeconds, name: "gen", color: "#fbbf24" },
      { pick: (m) => m.trainSeconds, name: "train", color: "#f472b6" },
    ],
  },
  { label: "Games completed", digits: 0, series: [{ pick: (m) => m.completedGames, name: "games", color: "#a855f7" }] },
  { label: "Self-play turns", digits: 0, series: [{ pick: (m) => m.turns, name: "turns", color: "#34d399" }] },
  { label: "Avg game turn (buffer)", digits: 1, series: [{ pick: (m) => m.avgGameTurn, name: "avg", color: "#2dd4bf" }] },
  { label: "Samples", digits: 0, series: [{ pick: (m) => m.samples, name: "samples", color: "#60a5fa" }] },
  { label: "Buffer size", digits: 0, series: [{ pick: (m) => Number(m.buffer), name: "buffer", color: "#94a3b8" }] },
  { label: "Games / sec", digits: 1, series: [{ pick: (m) => m.gamesPerSec, name: "games/s", color: "#22c55e" }] },
  { label: "Inferences / sec", digits: 0, series: [{ pick: (m) => m.inferencesPerSec, name: "inf/s", color: "#38bdf8" }] },
  { label: "GPU busy %", digits: 0, fixedMax: 100, series: [{ pick: (m) => m.gpuBusyPct, name: "gpu", color: "#eab308" }] },
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

  const controls = (
    <RunControls
      live={isLive}
      running={running}
      stopping={stopping}
      onResume={() => control.start(runId, false).then(() => undefined)}
      onStop={() => control.stop().then(() => undefined)}
    />
  );

  return (
    <div className="mx-auto w-full max-w-[120rem] px-3 py-4 sm:px-5">
      <header className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-1">
        <Link to="/" className="text-sm text-slate-400 hover:text-sky-300">
          ← runs
        </Link>
        <h1 className="font-mono text-base font-semibold text-slate-100 sm:text-lg">{runId}</h1>
        {summary && (
          <span className="text-xs text-slate-500">
            gen {summary.generation} · {summary.board}² · {summary.numSnakes}p
          </span>
        )}
      </header>

      {error && <div className="mb-4 rounded border border-red-900 bg-red-950 p-3 text-sm text-red-200">{error}</div>}
      {loading && !detail && <div className="text-sm text-slate-500">Loading run…</div>}

      {/* Full-screen split: metrics/controls on the left, sample games on the
          right. Stacks into a single column below xl (tablet / mobile). */}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)] xl:items-start">
        <div className="grid gap-4">
          <TrainingProgress stats={live.stats} state={live.state} controls={controls} fallbackGen={summary?.generation ?? 0} />

          <LogPanel logs={logs} />

          {isLive && (
            <section>
              <h2 className="section-title mb-2">Realtime (this generation)</h2>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <SeriesChart values={live.history.map((r) => r.inferencesPerSec)} label="Inference rate" digits={0} xUnit="t" compact />
                <SeriesChart values={live.history.map((r) => r.gpuRowsPerSec)} label="GPU rows/s" color="#a855f7" digits={0} xUnit="t" compact />
                <SeriesChart values={live.history.map((r) => r.gamesPerSec)} label="Game rate" color="#22c55e" digits={1} xUnit="t" compact />
                <SeriesChart values={live.history.map((r) => r.gpuBusyPct)} label="GPU busy" color="#eab308" fixedMax={100} digits={0} xUnit="t" compact />
              </div>
            </section>
          )}

          {metrics.length > 0 && (
            <section>
              <h2 className="section-title mb-2">Per generation</h2>
              <div className="grid gap-3 sm:grid-cols-2">
                {GEN_CHARTS.map((chart) => (
                  <SeriesChart
                    key={chart.label}
                    series={chart.series.map((s) => ({ values: metrics.map(s.pick), color: s.color, name: s.name }))}
                    label={chart.label}
                    digits={chart.digits}
                    fixedMax={chart.fixedMax ?? null}
                    xValues={gens}
                    xLeft={genLeft}
                    xRight={genRight}
                  />
                ))}
                {winRates.length > 0 && (
                  <SeriesChart values={winRates.map((m) => m.winRate)} label="Win rate" color="#22c55e" fixedMax={1} digits={2} xValues={winRates.map((m) => m.generation)} />
                )}
              </div>
            </section>
          )}

          <section>
            <ConfigPanel config={savedConfig ?? diskConfig} onSave={saveConfig} />
          </section>
        </div>

        <section className="xl:sticky xl:top-4">
          <h2 className="section-title mb-2">Sample games</h2>
          <GameViewer runId={runId} gameGens={detail?.gameGens ?? []} metrics={metrics} />
        </section>
      </div>
    </div>
  );
}
