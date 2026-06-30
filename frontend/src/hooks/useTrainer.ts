import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { openGamesStream, openStatsStream } from "../api/streams";
import type { GamesSnapshot, RunConfig, RunList, RunState, StatsFrame } from "../types";

export function useTrainer() {
  const [config, setConfig] = useState<RunConfig | null>(null);
  const [runs, setRuns] = useState<RunList>({ runs: [], live: null });
  const [state, setState] = useState<RunState | null>(null);
  const [stats, setStats] = useState<StatsFrame | null>(null);
  const [history, setHistory] = useState<StatsFrame[]>([]);
  const [games, setGames] = useState<GamesSnapshot | null>(null);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const statsStream = openStatsStream((frame) => {
      setStats(frame);
      setHistory((rows) => [...rows.slice(-239), frame]);
      setState((old) => ({ phase: frame.phase, generation: frame.generation, run_id: old?.run_id ?? "", running: !["idle", "stopped"].includes(frame.phase), device: old?.device }));
    });
    const gamesStream = openGamesStream(setGames);
    return () => { statsStream.close(); gamesStream.close(); };
  }, []);

  const actions = useMemo(() => ({
    saveConfig: async (next: RunConfig) => setConfig(await api.setConfig(next)),
    start: async (runId: string, fresh: boolean) => {
      await api.start(runId, fresh);
      await refresh();
    },
    stop: async () => {
      await api.stop();
      await refresh();
    },
  }), []);

  async function refresh() {
    const [nextConfig, nextRuns, nextState] = await Promise.all([
      api.config().catch(() => null),
      api.runs().catch(() => ({ runs: [], live: null })),
      api.state().catch(() => null),
    ]);
    if (nextConfig) setConfig(nextConfig);
    setRuns(nextRuns);
    if (nextState) setState(nextState);
  }

  return { config, runs, state, stats, history, games, actions };
}
