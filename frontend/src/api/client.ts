// JSON control-plane client. These endpoints act on the trainer's single active
// run (config knobs, run state, start/stop). Run/game browsing goes through the
// binary-protobuf client in ./proto.
import type { RunConfig, RunState } from "../types";

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
  if (!res.ok) throw new Error((data as { detail?: string }).detail ?? `${url} returned ${res.status}`);
  return data as T;
}

export const control = {
  // Single config-save path for every run, live or not: writes the run's
  // config.json (and updates the in-memory config server-side if it is active).
  setRunConfig: (runId: string, config: RunConfig) =>
    postJson<RunConfig>(`/api/runs/${encodeURIComponent(runId)}/config`, config),
  state: () => getJson<RunState>("/api/state"),
  // The trainer's default config, used to seed the "start fresh run" knob form.
  config: () => getJson<RunConfig>("/api/config"),
  start: (runId: string | null, fresh: boolean, config?: RunConfig) =>
    postJson<{ run_id: string }>("/api/control/start", { run_id: runId, fresh, config }),
  stop: () => postJson<{ stopping: boolean }>("/api/control/stop"),
};
