import { useEffect, useState } from "react";
import { openLogStream } from "../api/logs";
import type { LogEntry } from "../api/logs";

// Subscribe to the trainer log stream, keeping the most recent `max` entries.
export function useLogs(max = 100) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  useEffect(() => {
    const stream = openLogStream((entry) => setLogs((prev) => [...prev.slice(-(max - 1)), entry]));
    return () => stream.close();
  }, [max]);
  return logs;
}
