// Client over the trainer's API. Historical runs are read over REST; the live
// run is streamed over Server-Sent-Events (see api.stream).

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

async function postJSON(url, body, token) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const r = await fetch(url, { method: "POST", headers, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${url} -> ${r.status}`);
  return data;
}

export const api = {
  runs: () => getJSON("/api/runs").catch(() => ({ runs: [], live: null })),
  state: () => getJSON("/api/state").catch(() => null),
  status: (run) => getJSON(`/api/runs/${run}/status`).catch(() => ({})),
  meta: (run) => getJSON(`/api/runs/${run}/meta`).catch(() => ({})),
  metrics: (run) => getJSON(`/api/runs/${run}/metrics`).then((d) => d.metrics || []).catch(() => []),
  games: (run) => getJSON(`/api/runs/${run}/games`).then((d) => d.files || []).catch(() => []),
  gameFile: (run, file) => getJSON(`/api/runs/${run}/games/${file}`),

  // Live: returns the EventSource; caller handles onEvent({type, ...}).
  stream: (onEvent) => {
    const es = new EventSource("/api/stream");
    es.onmessage = (e) => {
      try { onEvent(JSON.parse(e.data)); } catch (_) { /* keepalive/comment */ }
    };
    return es;
  },
  setParams: (patch, token) => postJSON("/api/params", patch, token),
  control: (action, token) => postJSON("/api/control", { action }, token),
  createRun: (name, params, token) => postJSON("/api/runs", { name, params }, token),
  resumeRun: (name, token) => postJSON("/api/runs", { name, resume: true }, token),
};
