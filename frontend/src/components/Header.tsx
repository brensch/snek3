import type { RunList, RunState } from "../types";

type Props = { runs: RunList; state: RunState | null };

export function Header({ runs, state }: Props) {
  return (
    <header className="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/95 px-5 py-3 backdrop-blur">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-sm font-semibold tracking-wide text-slate-100">snek3 training</h1>
        <span className="rounded border border-slate-800 px-2 py-1 text-xs text-slate-400">
          {runs.live ? `live: ${runs.live}` : "no live run"}
        </span>
        <span className="rounded border border-slate-800 px-2 py-1 text-xs text-slate-400">
          {state ? `${state.phase} · gen ${state.generation}` : "offline"}
        </span>
        {state?.device ? (
          <span className="rounded border border-slate-800 px-2 py-1 text-xs text-slate-400">
            device: {state.device}
          </span>
        ) : null}
      </div>
    </header>
  );
}
