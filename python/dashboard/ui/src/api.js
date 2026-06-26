// Thin client over the dashboard's read-only JSON API (file backend).

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

export const api = {
  runs: () => getJSON("/api/runs").then((d) => d.runs || []),
  status: (run) => getJSON(`/api/runs/${run}/status`).catch(() => ({})),
  meta: (run) => getJSON(`/api/runs/${run}/meta`).catch(() => ({})),
  metrics: (run) => getJSON(`/api/runs/${run}/metrics`).then((d) => d.metrics || []).catch(() => []),
  games: (run) => getJSON(`/api/runs/${run}/games`).then((d) => d.files || []).catch(() => []),
  gameFile: (run, file) => getJSON(`/api/runs/${run}/games/${file}`),
};
