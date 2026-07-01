import { Phase } from "../gen/snek_pb";
import type { StatsFrame } from "../gen/snek_pb";
import { number } from "../lib/format";
import { phaseLabel } from "../lib/phase";
import type { RunState } from "../types";

type Props = { stats: StatsFrame | null; state: RunState | null };

// Compact live status: the generation, the current phase, and a progress bar —
// self-play collects samples, training advances steps.
export function TrainingProgress({ stats, state }: Props) {
  const phase = stats?.phase ?? state?.phase ?? Phase.IDLE;
  const gen = stats?.generation ?? state?.generation ?? 0;

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

  const hasProgress = total > 0;
  const pct = hasProgress ? Math.max(0, Math.min(100, (current / total) * 100)) : 0;

  return (
    <div className="rounded border border-slate-800 bg-slate-900 p-4">
      <div className="mb-2 flex items-center gap-3">
        <span className="text-sm font-semibold text-slate-100">Generation {number(gen)}</span>
        <span className="rounded-full bg-slate-800 px-2 py-0.5 text-xs font-medium text-slate-300">
          {phaseLabel(phase)}
        </span>
        {hasProgress && (
          <span className="ml-auto font-mono text-xs text-slate-400">
            {number(current)} / {number(total)} {unit}
          </span>
        )}
      </div>
      <div className="h-2.5 overflow-hidden rounded bg-slate-950">
        <div
          className={`h-full transition-[width] duration-200 ${phase === Phase.TRAINING ? "bg-amber-500" : "bg-sky-500"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
