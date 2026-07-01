// Trainer log SSE. The server emits human-readable events (start/resume, per-gen
// summaries, stop) as JSON; we surface them at the top of the run view.
export type LogEntry = { t_unix_ms: number; message: string };

export function openLogStream(onEntry: (entry: LogEntry) => void): EventSource {
  const events = new EventSource("/api/stream/logs");
  events.addEventListener("log", (event) => {
    try {
      onEntry(JSON.parse((event as MessageEvent).data) as LogEntry);
    } catch {
      /* ignore malformed lines */
    }
  });
  return events;
}
