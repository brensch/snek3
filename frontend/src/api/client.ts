import type { RunConfig, RunList, RunState } from "../types";

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  return res.json() as Promise<T>;
}

async function postJson<T>(url: string, body: unknown = {}): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `${url} returned ${res.status}`);
  return data as T;
}

const phaseName = (phase: number): RunState["phase"] =>
  ["idle", "playing", "training", "checkpoint", "stopping", "stopped"][phase] as RunState["phase"] ?? "idle";

export const api = {
  config: () => getJson<RunConfig>("/api/config"),
  setConfig: (config: RunConfig) => postJson<RunConfig>("/api/config", config),
  runs: () => getJson<RunList>("/api/runs").catch(() => ({ runs: [], live: null })),
  state: async (): Promise<RunState> => {
    const state = await getJson<Omit<RunState, "phase"> & { phase: number }>("/api/state");
    return { ...state, phase: phaseName(state.phase) };
  },
  start: (runId: string, fresh: boolean) => postJson<{ run_id: string }>("/api/control/start", { run_id: runId, fresh }),
  stop: () => postJson<{ stopping: boolean }>("/api/control/stop"),
};
