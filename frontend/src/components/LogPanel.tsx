import type { LogEntry } from "../api/logs";

// Trainer events, newest first, shown at the top of the run view.
export function LogPanel({ logs }: { logs: LogEntry[] }) {
  if (!logs.length) return null;
  const recent = [...logs].reverse().slice(0, 8);
  return (
    <div className="rounded border border-slate-800 bg-slate-900 p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">Trainer log</div>
      <div className="grid max-h-32 gap-0.5 overflow-y-auto font-mono text-[11px]">
        {recent.map((entry, i) => (
          <div key={`${entry.t_unix_ms}-${i}`} className="flex gap-2">
            <span className="shrink-0 text-slate-600">{new Date(entry.t_unix_ms).toLocaleTimeString()}</span>
            <span className={i === 0 ? "text-slate-200" : "text-slate-400"}>{entry.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
