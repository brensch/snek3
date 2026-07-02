import type { LogEntry } from "../../api/events";

// Trainer events, newest first: generation summaries, league results, pauses.
// The narrative record when a chart looks odd.
export function LogsPanel({ logs }: { logs: LogEntry[] }) {
  const recent = [...logs].reverse();
  return (
    <div className="card flex min-h-0 flex-col">
      <div className="border-b border-white/10 px-3 py-1.5">
        <span className="card-title">Trainer log</span>
      </div>
      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto p-2 font-mono text-[11px] leading-snug" style={{ maxHeight: 220 }}>
        {recent.length === 0 && <div className="text-ink-3">no events yet — logs stream in from the live run</div>}
        {recent.map((entry, i) => (
          <div key={`${entry.t_unix_ms}-${i}`} className="flex gap-2">
            <span className="shrink-0 tabular-nums text-ink-3/70">
              {new Date(entry.t_unix_ms).toLocaleTimeString()}
            </span>
            <span className={i === 0 ? "text-ink" : "text-ink-3"}>{entry.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
