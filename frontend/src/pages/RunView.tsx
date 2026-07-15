import { useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { control } from "../api/client";
import { ConfigPanel } from "../components/ConfigPanel";
import { LineChart } from "../components/charts/LineChart";
import { EloPanel } from "../components/run/EloPanel";
import { GamesPanels } from "../components/run/GamesPanels";
import { LiveThroughput } from "../components/run/LiveThroughput";
import { LogsPanel } from "../components/run/LogsPanel";
import { StrengthHero } from "../components/run/StrengthHero";
import { TopBar } from "../components/run/TopBar";
import { Phase } from "../gen/snek_pb";
import { useEvalLive } from "../hooks/useEvalLive";
import { useLiveStats } from "../hooks/useLiveStats";
import { useLogs } from "../hooks/useLogs";
import { useRunDetail } from "../hooks/useRunDetail";
import { series } from "../lib/palette";
import { registerPlayerNames } from "../lib/players";
import type { RunConfig } from "../types";

// One run, two tiers:
//   TIER 1 (always visible): the top bar answers "is it alive?" and the
//   Strength hero answers "is it getting better?" — held-out win rate vs
//   baselines in LE mode (the Elo replacement), League Elo for legacy AZ runs.
//   A small realtime throughput card sits beside it.
//   TIER 2 (tabs): Health (is learning sane), Throughput (is it fast), Games
//   (watch actual play). The tab lives in the URL (?tab=) so reload/share
//   keeps your place.
// Config and logs stay as on-demand panels toggled from the top bar.
const TABS = ["health", "throughput", "games"] as const;
type Tab = (typeof TABS)[number];

export function RunView() {
  const { runId = "" } = useParams();
  const { detail, error, loading } = useRunDetail(runId);
  const summary = detail?.summary ?? null;
  const isLive = summary?.live ?? false;
  const live = useLiveStats(isLive);
  const liveMatch = useEvalLive(isLive);
  const logs = useLogs();

  const [params, setParams] = useSearchParams();
  const tabParam = params.get("tab");
  const tab: Tab = (TABS as readonly string[]).includes(tabParam ?? "") ? (tabParam as Tab) : "health";
  const setTab = (t: Tab) =>
    setParams(
      (p) => {
        p.set("tab", t);
        return p;
      },
      { replace: true },
    );

  // `running` comes from the authoritative flag, not the phase: a resume
  // spends tens of seconds restoring the replay buffer before the first
  // generation starts, and the phase sits at Stopped the whole time. The
  // phase only drives the chip and the stopping state.
  const running = live.state?.running ?? summary?.running ?? false;
  const stopping = (live.stats?.phase ?? live.state?.phase) === Phase.STOPPING;

  // The on-disk config.json is the source of truth; saves go through one
  // endpoint that writes it (and updates the live trainer when applicable).
  const diskConfig = useMemo<RunConfig | null>(() => {
    if (!detail?.configJson) return null;
    try {
      return JSON.parse(detail.configJson) as RunConfig;
    } catch {
      return null;
    }
  }, [detail?.configJson]);
  const [savedConfig, setSavedConfig] = useState<RunConfig | null>(null);
  const [showConfig, setShowConfig] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  useEffect(() => {
    if (savedConfig && diskConfig && JSON.stringify(savedConfig) === JSON.stringify(diskConfig)) {
      setSavedConfig(null);
    }
  }, [diskConfig, savedConfig]);

  const metrics = detail?.metrics ?? [];
  const league = detail?.league ?? [];
  // Make external players (API snakes) nameable anywhere an id appears.
  registerPlayerNames(league);
  const gens = metrics.map((m) => m.generation);
  const isLE = useMemo(() => metrics.some((m) => m.hasLeEval), [metrics]);

  // Loss falls slowly and noisily, so "is it still trending down?" is hard to
  // read off the raw curve. The slope of a local least-squares fit answers it
  // directly: negative = still improving, hovering at zero = plateaued. The
  // window smooths out per-gen noise without the lag of an EMA.
  // The first few gens drop loss steeply, so their slopes dwarf everything
  // after and flatten the recent trend against the axis. Fit on the full
  // history (so the window has context at the boundary) but only plot the tail.
  const lossSlopes = useMemo(() => {
    const half = Math.min(15, Math.max(4, Math.round(metrics.length * 0.03)));
    const tail = Math.max(0, metrics.length - 50);
    return {
      gens: gens.slice(tail),
      policy: rollingSlope(metrics.map((m) => m.policyLoss), gens, half).slice(tail),
      value: rollingSlope(metrics.map((m) => m.valueLoss), gens, half).slice(tail),
    };
  }, [metrics, gens]);

  return (
    <div className="min-h-screen">
      <TopBar
        runId={runId}
        live={isLive}
        running={running}
        stopping={stopping}
        stats={live.stats}
        fallbackGen={summary?.generation ?? 0}
        configOpen={showConfig}
        onToggleConfig={() => setShowConfig((v) => !v)}
        logsOpen={showLogs}
        onToggleLogs={() => setShowLogs((v) => !v)}
        onStop={() => control.stop().then(live.refresh)}
        onResume={() => control.start(runId, false).then(live.refresh)}
      />

      <main className="space-y-3 px-3 py-3 sm:px-5">
        {error && <div className="card border-bad/40 p-3 text-sm text-bad">{error}</div>}
        {loading && !detail && <div className="text-sm text-ink-3">Loading run…</div>}

        {showConfig && (
          <ConfigPanel
            config={savedConfig ?? diskConfig}
            onSave={async (next) => {
              await control.setRunConfig(runId, next);
              setSavedConfig(next);
            }}
          />
        )}
        {showLogs && <LogsPanel logs={logs} />}

        {/* TIER 1 — the hero: held-out strength (LE) or League Elo (AZ), with
            the realtime throughput sidecard. */}
        <div className="grid gap-2.5 xl:grid-cols-[minmax(0,1fr)_13rem] xl:items-stretch">
          {isLE ? <StrengthHero metrics={metrics} /> : <EloPanel league={league} />}
          <LiveThroughput stats={live.stats} history={live.history} />
        </div>

        {/* TIER 2 — tabs. */}
        <div className="flex items-center gap-1 border-b border-white/10">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`-mb-px border-b-2 px-3 py-1.5 text-sm capitalize transition-colors ${
                tab === t
                  ? "border-accent font-medium text-ink"
                  : "border-transparent text-ink-3 hover:text-ink-2"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {tab === "health" && metrics.length > 0 && (
          <section className="grid gap-2.5 sm:grid-cols-2 2xl:grid-cols-4">
            <LineChart
              title="Entropy (collapse watch)"
              height={112}
              series={[
                { name: "target", color: series.violet, values: metrics.map((m) => m.targetEntropy) },
                { name: "net", color: series.blue, values: metrics.map((m) => m.netEntropy) },
              ]}
              xValues={gens}
            />
            <LineChart
              title="Loss"
              height={112}
              series={[
                { name: "policy", color: series.blue, values: metrics.map((m) => m.policyLoss) },
                { name: "value", color: series.aqua, values: metrics.map((m) => m.valueLoss) },
              ]}
              xValues={gens}
            />
            <LineChart
              title="Loss slope (Δ/gen)"
              height={112}
              series={[
                { name: "policy", color: series.blue, values: lossSlopes.policy },
                { name: "value", color: series.aqua, values: lossSlopes.value },
              ]}
              xValues={lossSlopes.gens}
              centerZero
              format={(v) => (v === 0 ? "0" : v.toExponential(1))}
            />
            <LineChart
              title="Game length (turns)"
              height={112}
              series={[{ name: "avg turns", color: series.aqua, values: metrics.map((m) => m.avgGameTurn) }]}
              xValues={gens}
            />
          </section>
        )}

        {tab === "throughput" && metrics.length > 0 && (
          <section className="grid grid-cols-2 gap-2.5 lg:grid-cols-3 2xl:grid-cols-5">
            <LineChart
              title="Inferences / s"
              height={96}
              series={[{ name: "inf/s", color: series.blue, values: metrics.map((m) => m.inferencesPerSec) }]}
              xValues={gens}
            />
            <LineChart
              title="Phase time (s)"
              height={96}
              series={[
                { name: "play", color: series.blue, values: metrics.map((m) => m.playSeconds) },
                { name: "train", color: series.aqua, values: metrics.map((m) => m.trainSeconds) },
              ]}
              xValues={gens}
            />
            <LineChart
              title="Games / gen"
              height={96}
              series={[{ name: "games", color: series.magenta, values: metrics.map((m) => m.completedGames) }]}
              xValues={gens}
            />
            <LineChart
              title="Samples / gen"
              height={96}
              series={[{ name: "samples", color: series.aqua, values: metrics.map((m) => m.samples) }]}
              xValues={gens}
            />
            <LineChart
              title="Replay buffer"
              height={96}
              series={[{ name: "buffer", color: series.violet, values: metrics.map((m) => Number(m.buffer)) }]}
              xValues={gens}
            />
          </section>
        )}

        {tab === "games" && (
          <GamesPanels
            runId={runId}
            matches={detail?.matches ?? []}
            gameGens={detail?.gameGens ?? []}
            metrics={metrics}
            liveMatch={liveMatch}
          />
        )}
      </main>
    </div>
  );
}

// Smoothed derivative: for each point, the slope of a least-squares line fit
// through the surrounding [-half, +half] window (x = generation, y = value).
// Fitting a window rather than differencing neighbours is what does the
// smoothing — one noisy point barely tilts the local line. Points without
// enough finite neighbours stay NaN and the chart just skips them.
function rollingSlope(values: number[], xs: number[], half: number): number[] {
  const n = values.length;
  const out = new Array<number>(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    let sx = 0,
      sy = 0,
      sxx = 0,
      sxy = 0,
      cnt = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(n - 1, i + half); j++) {
      const y = values[j];
      if (!Number.isFinite(y)) continue;
      const x = xs[j];
      sx += x;
      sy += y;
      sxx += x * x;
      sxy += x * y;
      cnt++;
    }
    const denom = cnt * sxx - sx * sx;
    if (cnt >= 2 && denom !== 0) out[i] = (cnt * sxy - sx * sy) / denom;
  }
  return out;
}
