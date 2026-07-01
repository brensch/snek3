import type { ReactNode } from "react";
import { Phase } from "../gen/snek_pb";
import type { StatsFrame } from "../gen/snek_pb";
import { number } from "../lib/format";
import { phaseLabel } from "../lib/phase";
import type { RunState } from "../types";

type Props = {
  stats: StatsFrame | null;
  state: RunState | null;
  controls?: ReactNode;
  fallbackGen?: number;
  live?: boolean;
};

// Compact live status: the generation, a live chip, the current phase, a
// progress bar (with its count laid over it), and the run's start/stop controls
// — self-play collects samples, training advances steps.
export function TrainingProgress({ stats, state, controls, fallbackGen = 0, live = false }: Props) {
  const phase = stats?.phase ?? state?.phase ?? Phase.IDLE;
  const gen = stats?.generation ?? state?.generation ?? fallbackGen;

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
  const training = phase === Phase.TRAINING;

  return (
    <div className="rounded border border-slate-800 bg-slate-900 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-2">
        <span className="text-sm font-semibold text-slate-100">Generation {number(gen)}</span>
        {live && (
          <span className="flex items-center gap-1.5 rounded-full bg-green-500/15 px-2 py-0.5 text-xs font-medium text-green-400">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-green-400" />
            live
          </span>
        )}
        <span className="rounded-full bg-slate-800 px-2 py-0.5 text-xs font-medium text-slate-300">
          {phaseLabel(phase)}
        </span>
        {controls && <div className="ml-auto">{controls}</div>}
      </div>
      <div className="relative h-5 overflow-hidden rounded-full bg-slate-950 ring-1 ring-inset ring-slate-800">
        <div
          className={`h-full rounded-full bg-gradient-to-r transition-[width] duration-300 ease-out ${
            training ? "from-amber-600 to-amber-400" : "from-sky-600 to-sky-400"
          }`}
          style={{ width: `${pct}%` }}
        />
        {hasProgress ? (
          <div className="absolute inset-0 flex items-center justify-center gap-1.5 font-mono text-[11px] font-medium text-slate-100 mix-blend-luminosity">
            <span>
              {number(current)} / {number(total)} {unit}
            </span>
            <span className="text-slate-300">· {pct.toFixed(0)}%</span>
          </div>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center font-mono text-[11px] text-slate-500">
            {phaseLabel(phase)}
          </div>
        )}
      </div>
    </div>
  );
}
