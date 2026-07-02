import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { control } from "../api/client";
import { ConfigPanel } from "../components/ConfigPanel";
import { LineChart } from "../components/charts/LineChart";
import { EloPanel } from "../components/run/EloPanel";
import { GamesPanels } from "../components/run/GamesPanels";
import { LiveThroughput } from "../components/run/LiveThroughput";
import { LogsPanel } from "../components/run/LogsPanel";
import { TopBar } from "../components/run/TopBar";
import { Phase } from "../gen/snek_pb";
import { useEvalLive } from "../hooks/useEvalLive";
import { useLiveStats } from "../hooks/useLiveStats";
import { useLogs } from "../hooks/useLogs";
import { useRunDetail } from "../hooks/useRunDetail";
import { series } from "../lib/palette";
import type { RunConfig } from "../types";

// One run, laid out by what a monitor needs first:
//   1. the top bar answers "is it alive?" (phase, generation, progress);
//   2. the League Elo headline answers "is it getting better?", with a small
//      realtime throughput card beside it;
//   3. learning and throughput small-multiples diagnose why;
//   4. the games panels at the bottom — live league game, recorded league
//      games, self-play samples, all visible at once — are the qualitative
//      gut check.
// Config and logs are on-demand panels toggled from the top bar.
export function RunView() {
  const { runId = "" } = useParams();
  const { detail, error, loading } = useRunDetail(runId);
  const summary = detail?.summary ?? null;
  const isLive = summary?.live ?? false;
  const live = useLiveStats(isLive);
  const liveMatch = useEvalLive(isLive);
  const logs = useLogs();

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
  const gens = metrics.map((m) => m.generation);
  const lrRows = metrics.filter((m) => m.lr > 0);

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

        {/* Headline row: League Elo (hero, curve, leaderboard) plus the small
            realtime throughput card. */}
        <div className="grid gap-2.5 xl:grid-cols-[minmax(0,1fr)_13rem] xl:items-stretch">
          <EloPanel league={league} />
          <LiveThroughput stats={live.stats} history={live.history} />
        </div>

        {metrics.length > 0 && (
          <>
            <section>
              <h2 className="card-title mb-1.5">Learning</h2>
              <div className="grid gap-2.5 sm:grid-cols-2 2xl:grid-cols-4">
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
                  title="Target entropy"
                  height={112}
                  series={[{ name: "entropy", color: series.violet, values: metrics.map((m) => m.targetEntropy) }]}
                  xValues={gens}
                />
                <LineChart
                  title="Game length (turns)"
                  height={112}
                  series={[{ name: "avg turns", color: series.aqua, values: metrics.map((m) => m.avgGameTurn) }]}
                  xValues={gens}
                />
                {lrRows.length > 0 && (
                  <LineChart
                    title="Learning rate"
                    height={112}
                    series={[{ name: "lr", color: series.orange, values: lrRows.map((m) => m.lr) }]}
                    xValues={lrRows.map((m) => m.generation)}
                    format={(v) => v.toExponential(1)}
                  />
                )}
              </div>
            </section>

            <section>
              <h2 className="card-title mb-1.5">Throughput</h2>
              <div className="grid grid-cols-2 gap-2.5 lg:grid-cols-3 2xl:grid-cols-5">
                <LineChart
                  title="Inferences / s"
                  height={72}
                  series={[{ name: "inf/s", color: series.blue, values: metrics.map((m) => m.inferencesPerSec) }]}
                  xValues={gens}
                />
                <LineChart
                  title="Phase time (s)"
                  height={72}
                  series={[
                    { name: "play", color: series.blue, values: metrics.map((m) => m.playSeconds) },
                    { name: "train", color: series.aqua, values: metrics.map((m) => m.trainSeconds) },
                  ]}
                  xValues={gens}
                />
                <LineChart
                  title="Games / gen"
                  height={72}
                  series={[{ name: "games", color: series.magenta, values: metrics.map((m) => m.completedGames) }]}
                  xValues={gens}
                />
                <LineChart
                  title="Samples / gen"
                  height={72}
                  series={[{ name: "samples", color: series.aqua, values: metrics.map((m) => m.samples) }]}
                  xValues={gens}
                />
                <LineChart
                  title="Replay buffer"
                  height={72}
                  series={[{ name: "buffer", color: series.violet, values: metrics.map((m) => Number(m.buffer)) }]}
                  xValues={gens}
                />
              </div>
            </section>
          </>
        )}

        <GamesPanels
          runId={runId}
          matches={detail?.matches ?? []}
          gameGens={detail?.gameGens ?? []}
          metrics={metrics}
          liveMatch={liveMatch}
        />
      </main>
    </div>
  );
}
