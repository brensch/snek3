import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { control } from "../api/client";
import { ConfigPanel } from "../components/ConfigPanel";
import { EvalViewer } from "../components/EvalViewer";
import { GameViewer } from "../components/GameViewer";
import { LogPanel } from "../components/LogPanel";
import { RunControls } from "../components/RunControls";
import { SeriesChart } from "../components/SeriesChart";
import { TrainingProgress } from "../components/TrainingProgress";
import { useLiveStats } from "../hooks/useLiveStats";
import { useLogs } from "../hooks/useLogs";
import { useRunDetail } from "../hooks/useRunDetail";
import { Phase } from "../gen/snek_pb";
import type { EvalPoint, MetricRow } from "../gen/viewer_pb";
import type { RunConfig } from "../types";

// Per-generation charts, driven off metrics.jsonl. Related metrics are grouped
// into a handful of categories rather than one line per field. Same-unit groups
// (losses, phase timing) share a real y-axis; mixed-scale groups (throughput,
// per-gen volume) set `normalize` so each line is scaled to its own range for
// shape comparison, with real values kept in the legend/tooltip.
type GenSeries = { pick: (m: MetricRow) => number; name: string; color: string };
const GEN_CHARTS: { label: string; digits?: number; fixedMax?: number; normalize?: boolean; series: GenSeries[] }[] = [
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
      { pick: (m) => m.genSeconds, name: "total", color: "#fbbf24" },
      { pick: (m) => m.playSeconds, name: "play", color: "#38bdf8" },
      { pick: (m) => m.trainSeconds, name: "train", color: "#f472b6" },
    ],
  },
  {
    label: "Throughput / sec",
    digits: 1,
    normalize: true,
    series: [
      { pick: (m) => m.gamesPerSec, name: "games", color: "#22c55e" },
      { pick: (m) => m.inferencesPerSec, name: "inf", color: "#38bdf8" },
    ],
  },
  {
    label: "Volume / gen",
    digits: 0,
    normalize: true,
    series: [
      { pick: (m) => m.samples, name: "samples", color: "#60a5fa" },
      { pick: (m) => m.turns, name: "turns", color: "#34d399" },
      { pick: (m) => m.completedGames, name: "games", color: "#a855f7" },
    ],
  },
  {
    label: "Game length & buffer",
    digits: 0,
    normalize: true,
    series: [
      { pick: (m) => m.avgGameTurn, name: "avg turn", color: "#2dd4bf" },
      { pick: (m) => Number(m.buffer), name: "buffer", color: "#94a3b8" },
    ],
  },
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
  const [showConfig, setShowConfig] = useState(false);
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
  const evalPoints = detail?.evalPoints ?? [];
  // Each eval point plays several past checkpoints at exponentially spaced
  // horizons (vs -5, -10, -20 gens…). One chart per horizon: short horizons
  // show whether the net is still improving, long ones show progress over time.
  const evalHorizons = useMemo(() => {
    const byHorizon = new Map<number, EvalPoint[]>();
    for (const p of evalPoints) {
      const h = p.gen - p.opponentGen;
      const arr = byHorizon.get(h);
      if (arr) arr.push(p);
      else byHorizon.set(h, [p]);
    }
    return [...byHorizon.entries()].sort((a, b) => a[0] - b[0]);
  }, [evalPoints]);
  const genLeft = metrics.length ? `gen ${metrics[0].generation}` : "";
  const genRight = metrics.length ? `gen ${metrics[metrics.length - 1].generation}` : "";
  const gens = metrics.map((m) => m.generation);
  const winRates = metrics.filter((m) => m.hasWinRate);
  // The LR schedule is code-owned (train.rs); rows predating it carry lr=0.
  const lrRows = metrics.filter((m) => m.lr > 0);

  const controls = (
    <div className="flex items-center gap-2">
      <button
        type="button"
        className="btn"
        aria-pressed={showConfig}
        onClick={() => setShowConfig((v) => !v)}
      >
        {showConfig ? "Hide config" : "Configure"}
      </button>
      <RunControls
        running={running}
        stopping={stopping}
        onResume={() => control.start(runId, false).then(() => undefined)}
        onStop={() => control.stop().then(() => undefined)}
      />
    </div>
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

      {/* Full-width status banner: generation progress + run controls, with the
          training knobs popping out below it on demand. */}
      <div className="mb-4 grid gap-4">
        <TrainingProgress
          stats={live.stats}
          state={live.state}
          controls={controls}
          fallbackGen={summary?.generation ?? 0}
          live={isLive && running && !stopping}
        />
        {showConfig && <ConfigPanel config={savedConfig ?? diskConfig} onSave={saveConfig} />}
      </div>

      {/* Full-screen split: graphs on the left, sample games on the right.
          Stacks into a single column below xl (tablet / mobile). */}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)] xl:items-start">
        <div className="grid gap-4">
          <LogPanel logs={logs} />

          {isLive && (
            <section>
              <h2 className="section-title mb-2">Realtime (this generation)</h2>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                <SeriesChart values={live.history.map((r) => r.inferencesPerSec)} label="Inference rate" digits={0} xUnit="t" compact />
                <SeriesChart values={live.history.map((r) => r.gpuRowsPerSec)} label="GPU rows/s" color="#a855f7" digits={0} xUnit="t" compact />
                <SeriesChart values={live.history.map((r) => r.gamesPerSec)} label="Game rate" color="#22c55e" digits={1} xUnit="t" compact />
                <SeriesChart values={live.history.map((r) => r.gpuBusyPct)} label="GPU busy" color="#eab308" fixedMax={100} digits={0} xUnit="t" compact />
                <SeriesChart values={live.history.map((r) => r.avgGameTurn)} label="Avg game turn" color="#2dd4bf" digits={1} xUnit="t" compact />
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
                    normalize={chart.normalize}
                    xValues={gens}
                    xLeft={genLeft}
                    xRight={genRight}
                  />
                ))}
                {winRates.length > 0 && (
                  <SeriesChart values={winRates.map((m) => m.winRate)} label="Win rate" color="#22c55e" fixedMax={1} digits={2} xValues={winRates.map((m) => m.generation)} />
                )}
                {/* Head-to-head evals, one chart per opponent horizon: >0 Elo
                    means the current net beats the checkpoint that many gens back. */}
                {evalHorizons.map(([h, pts], i) => (
                  <SeriesChart
                    key={`eval-${h}`}
                    values={pts.map((e) => e.elo)}
                    label={`Eval Elo vs -${h} gens`}
                    color={["#22c55e", "#38bdf8", "#f59e0b", "#e879f9", "#f87171"][i % 5]}
                    digits={0}
                    xValues={pts.map((e) => e.gen)}
                  />
                ))}
                {lrRows.length > 0 && (
                  <SeriesChart values={lrRows.map((m) => m.lr)} label="Learning rate" color="#f97316" digits={5} xValues={lrRows.map((m) => m.generation)} />
                )}
              </div>
            </section>
          )}
        </div>

        <section className="xl:sticky xl:top-4">
          <h2 className="section-title mb-2">Sample games</h2>
          <GameViewer runId={runId} gameGens={detail?.gameGens ?? []} metrics={metrics} />
        </section>
      </div>

      {/* Head-to-head eval matches (current vs older checkpoint), recorded by
          the arena and rendered with the same board primitives as self-play. */}
      <section className="mt-4">
        <h2 className="section-title mb-2">Evaluation games</h2>
        <EvalViewer runId={runId} evalPoints={evalPoints} />
      </section>
    </div>
  );
}
