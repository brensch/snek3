// Client over the trainer's API. Historical runs are read over REST; the live
// run is streamed over Server-Sent-Events (see api.stream).

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

async function postJSON(url, body) {
  const headers = { "Content-Type": "application/json" };
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
  // Faithful eval (proxy ONNX + serve search vs the pool): win-rates + real games.
  evalIndex: (run) => getJSON(`/api/runs/${run}/eval`).then((d) => d.files || []).catch(() => []),
  evalFile: (run, file) => getJSON(`/api/runs/${run}/eval/${file}`),

  // Live: returns the EventSource; caller handles onEvent({type, ...}).
  stream: (onEvent) => {
    const es = new EventSource("/api/stream");
    es.onmessage = (e) => {
      try { onEvent(JSON.parse(e.data)); } catch (_) { /* keepalive/comment */ }
    };
    return es;
  },
  setParams: (patch) => postJSON("/api/params", patch),
  control: (action) => postJSON("/api/control", { action }),
  createRun: (name, params) => postJSON("/api/runs", { name, params }),
  resumeRun: (name) => postJSON("/api/runs", { name, resume: true }),
};
