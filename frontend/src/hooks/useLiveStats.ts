import { useEffect, useState } from "react";
import { control } from "../api/client";
import { openStatsStream } from "../api/streams";
import type { StatsFrame } from "../gen/snek_pb";
import type { RunState } from "../types";

// Live telemetry for the trainer's active run: the stats SSE stream and the
// current run state. Only meaningful for the run that is currently live, so
// callers pass `enabled` accordingly. Config is handled separately (one path via
// the per-run config endpoint), so it is intentionally not fetched here.
export function useLiveStats(enabled: boolean) {
  const [stats, setStats] = useState<StatsFrame | null>(null);
  const [history, setHistory] = useState<StatsFrame[]>([]);
  const [state, setState] = useState<RunState | null>(null);

  useEffect(() => {
    if (!enabled) return;
    const stream = openStatsStream((frame) => {
      setStats(frame);
      setHistory((rows) => [...rows.slice(-239), frame]);
    });
    return () => stream.close();
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    const refresh = async () => {
      const next = await control.state().catch(() => null);
      if (alive && next) setState(next);
    };
    void refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [enabled]);

  return { stats, history, state };
}
