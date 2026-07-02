import { useEffect, useMemo, useState } from "react";
import { getGameFile } from "../api/proto";
import type { GameFile, GameGen, MetricRow } from "../gen/viewer_pb";
import { GameTile } from "./GameTile";

type Props = { runId: string; gameGens: GameGen[]; metrics: MetricRow[] };

// Browse recorded self-play: pick a generation from the list on the left, and
// see all of its sample games at once as small tiles on the right. Grid-wide
// speed and size sliders control the tiles; each tile also has its own controls.
export function GameViewer({ runId, gameGens, metrics }: Props) {
  const [gen, setGen] = useState<number | null>(null);
  const [file, setFile] = useState<GameFile | null>(null);
  const [loading, setLoading] = useState(false);
  const [fps, setFps] = useState(12);
  const [cell, setCell] = useState(16);
  const [followLatest, setFollowLatest] = useState(true);
  const [showConfig, setShowConfig] = useState(false);

  // While following, keep the selection pinned to the newest generation as new
  // ones arrive (gameGens is newest-first). Manually picking a gen turns it off.
  useEffect(() => {
    if (followLatest && gameGens.length) setGen(gameGens[0].gen);
  }, [followLatest, gameGens]);

  // Load the selected generation's games.
  useEffect(() => {
    if (gen == null) return;
    let alive = true;
    setLoading(true);
    getGameFile(runId, gen)
      .then((next) => alive && setFile(next))
      .catch(() => alive && setFile(null))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [runId, gen]);

  const metricByGen = useMemo(() => new Map(metrics.map((m) => [m.generation, m])), [metrics]);
  const games = file?.games ?? [];
  const intervalMs = Math.round(1000 / fps);

  if (gameGens.length === 0) {
    return (
      <div className="rounded border border-slate-800 bg-slate-900 p-4 text-sm text-slate-400">
        No recorded sample games yet. The trainer writes them once a generation completes self-play.
      </div>
    );
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[12rem_minmax(0,1fr)] lg:items-start">
      <div className="rounded border border-slate-800 bg-slate-900">
        <div className="flex items-center justify-between gap-2 border-b border-slate-800 px-3 py-2">
          <span className="section-title">Generations</span>
          <label className="flex items-center gap-1 text-[10px] text-slate-400" title="Keep showing the newest generation as it arrives">
            <input
              type="checkbox"
              checked={followLatest}
              onChange={(e) => setFollowLatest(e.target.checked)}
              className="accent-sky-500"
            />
            follow latest
          </label>
        </div>
        <div className="max-h-[34rem] overflow-y-auto">
          {gameGens.map((g) => {
            const metric = metricByGen.get(g.gen);
            const active = g.gen === gen;
            return (
              <button
                key={g.gen}
                onClick={() => {
                  setFollowLatest(false);
                  setGen(g.gen);
                }}
                className={`flex w-full flex-col gap-0.5 border-b border-slate-800/60 px-3 py-2 text-left ${active ? "bg-sky-950/60" : "hover:bg-slate-800/50"}`}
              >
                <span className={`text-sm font-semibold ${active ? "text-sky-300" : "text-slate-200"}`}>
                  gen {g.gen}
                </span>
                {metric && (
                  <span className="font-mono text-[10px] text-slate-500">
                    π {metric.policyLoss.toFixed(3)} · v {metric.valueLoss.toFixed(3)} · {metric.completedGames} games
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <div className="mb-2 flex flex-wrap items-center gap-4">
          <span className="text-sm text-slate-300">
            {gen != null ? `gen ${gen}` : ""}
            {games.length ? ` · ${games.length} games` : ""}
          </span>
          <label className="flex items-center gap-2 text-xs text-slate-400">
            Speed
            <input type="range" min={1} max={20} value={fps} onChange={(e) => setFps(Number(e.target.value))} className="accent-sky-500" />
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-400">
            Size
            <input type="range" min={8} max={30} value={cell} onChange={(e) => setCell(Number(e.target.value))} className="accent-sky-500" />
          </label>
          {file?.configJson && (
            <button
              type="button"
              aria-pressed={showConfig}
              onClick={() => setShowConfig((v) => !v)}
              className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:border-sky-500"
            >
              {showConfig ? "Hide config" : "Config"}
            </button>
          )}
        </div>
        {showConfig && file?.configJson && (
          <div className="mb-3">
            <ConfigStrip json={file.configJson} gen={file.gen} />
          </div>
        )}
        {loading && games.length === 0 ? (
          <div className="text-sm text-slate-500">loading…</div>
        ) : games.length === 0 ? (
          <div className="text-sm text-slate-500">No games recorded for this generation.</div>
        ) : (
          <div
            className="grid gap-3"
            style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${cell * 12}px, 1fr))` }}
          >
            {games.map((game, i) => (
              <GameTile key={i} game={game} intervalMs={intervalMs} cell={cell} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// The training config in effect for the selected generation, historised in that
// generation's games file (config.json only holds the latest).
function ConfigStrip({ json, gen }: { json: string; gen: number }) {
  let cfg: Record<string, unknown>;
  try {
    cfg = JSON.parse(json) as Record<string, unknown>;
  } catch {
    return null;
  }
  return (
    <div className="rounded border border-slate-800 bg-slate-900 p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">Config used for gen {gen}</div>
      <div className="flex flex-wrap gap-1">
        {Object.entries(cfg).map(([k, v]) => (
          <span key={k} className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300">
            <span className="text-slate-500">{k}</span> <span className="font-mono">{String(v)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
