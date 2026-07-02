import { useEffect, useState } from "react";
import { subscribeEvents } from "../api/events";
import type { LogEntry } from "../api/events";

// Trainer log lines from the shared event stream, keeping the most recent `max`.
export function useLogs(max = 100) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  useEffect(
    () => subscribeEvents({ log: (entry) => setLogs((prev) => [...prev.slice(-(max - 1)), entry]) }),
    [max],
  );
  return logs;
}
