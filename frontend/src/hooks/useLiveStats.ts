import { useCallback, useEffect, useState } from "react";
import { control } from "../api/client";
import { subscribeEvents } from "../api/events";
import type { StatsFrame } from "../gen/snek_pb";
import type { RunState } from "../types";

// Live telemetry for the trainer's active run: stats frames from the shared
// event stream plus the polled run state. `refresh` lets actions (stop/resume)
// pull the authoritative state immediately instead of waiting for the poll.
export function useLiveStats(enabled: boolean) {
  const [stats, setStats] = useState<StatsFrame | null>(null);
  const [history, setHistory] = useState<StatsFrame[]>([]);
  const [state, setState] = useState<RunState | null>(null);

  useEffect(() => {
    if (!enabled) return;
    return subscribeEvents({
      stats: (frame) => {
        setStats(frame);
        setHistory((rows) => [...rows.slice(-239), frame]);
      },
    });
  }, [enabled]);

  const refresh = useCallback(async () => {
    const next = await control.state().catch(() => null);
    if (next) setState(next);
  }, []);

  useEffect(() => {
    if (!enabled) return;
    void refresh();
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => window.clearInterval(timer);
  }, [enabled, refresh]);

  return { stats, history, state, refresh };
}
