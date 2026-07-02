import { useState } from "react";
import { Link } from "react-router-dom";
import { Phase } from "../../gen/snek_pb";
import type { StatsFrame } from "../../gen/snek_pb";
import { number } from "../../lib/format";
import { phaseLabel } from "../../lib/phase";

type Props = {
  runId: string;
  live: boolean;
  running: boolean;
  stopping: boolean;
  stats: StatsFrame | null;
  fallbackGen: number;
  configOpen: boolean;
  onToggleConfig: () => void;
  logsOpen: boolean;
  onToggleLogs: () => void;
  onStop: () => Promise<unknown>;
  onResume: () => Promise<unknown>;
};

const PHASE_DOT: Record<Phase, string> = {
  [Phase.IDLE]: "#898781",
  [Phase.PLAYING]: "#3987e5",
  [Phase.TRAINING]: "#c98500",
  [Phase.CHECKPOINT]: "#9085e9",
  [Phase.STOPPING]: "#ec835a",
  [Phase.STOPPED]: "#898781",
};

// The always-visible answer to "is it alive?", in two fixed rows so nothing
// wraps awkwardly at any width: identity + controls on top, then phase,
// generation and the phase's progress meter.
export function TopBar({
  runId,
  live,
  running,
  stopping,
  stats,
  fallbackGen,
  configOpen,
  onToggleConfig,
  logsOpen,
  onToggleLogs,
  onStop,
  onResume,
}: Props) {
  const [busy, setBusy] = useState(false);
  const phase = stats?.phase ?? (running ? Phase.PLAYING : Phase.STOPPED);
  const gen = stats?.generation ?? fallbackGen;

  let current = 0;
  let total = 0;
  let unit = "";
  if (phase === Phase.PLAYING && stats) {
    current = stats.samplesCollected;
    total = stats.samplesTarget;
    unit = "samples";
  } else if (phase === Phase.TRAINING && stats) {
    current = stats.trainStep;
    total = stats.trainStepsTotal;
    unit = "steps";
  }
  const pct = total > 0 ? Math.max(0, Math.min(100, (current / total) * 100)) : 0;

  const invoke = async (action: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await action();
    } finally {
      setBusy(false);
    }
  };

  return (
    <header className="sticky top-0 z-30 border-b border-white/10 bg-page/90 backdrop-blur">
      <div className="grid gap-y-1.5 px-3 py-2 sm:px-5">
        <div className="flex items-center gap-x-3">
          <Link to="/" className="shrink-0 text-sm text-ink-3 transition-colors hover:text-accent">
            ←
          </Link>
          <h1 className="truncate font-mono text-sm font-semibold text-ink">{runId}</h1>
          {live && running && (
            <span className="chip shrink-0 text-good">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-good" />
              live
            </span>
          )}
          <div className="ml-auto flex shrink-0 items-center gap-2">
            <button type="button" className="btn" aria-pressed={configOpen} onClick={onToggleConfig}>
              Config
            </button>
            <button type="button" className="btn" aria-pressed={logsOpen} onClick={onToggleLogs}>
              Logs
            </button>
            {running ? (
              <button className="btn-danger" disabled={busy || stopping} onClick={() => invoke(onStop)}>
                {stopping ? "Stopping…" : "Stop"}
              </button>
            ) : (
              <button className="btn-primary" disabled={busy} onClick={() => invoke(onResume)}>
                Resume
              </button>
            )}
          </div>
        </div>

        <div className="flex items-center gap-x-3">
          <span className="chip shrink-0">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: PHASE_DOT[phase] ?? "#898781" }} />
            {phaseLabel(phase)}
          </span>
          <span className="shrink-0 font-mono text-sm tabular-nums text-ink-2">
            gen <span className="font-semibold text-ink">{number(gen)}</span>
          </span>
          {/* Phase progress meter: fill in the accent, track a darker step of
              the same ramp. Present but empty when there is nothing to measure,
              so the row height never jumps. */}
          <div className="min-w-0 flex-1">
            <div className="relative h-4 overflow-hidden rounded-full" style={{ background: "#12304f" }}>
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-300 ease-out"
                style={{ width: `${pct}%` }}
              />
              {total > 0 && (
                <div className="absolute inset-0 flex items-center justify-center font-mono text-[10px] font-medium tabular-nums text-white/90">
                  <span className="truncate px-2">
                    {number(current)} / {number(total)} {unit} · {pct.toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
