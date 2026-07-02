import { useEffect, useMemo, useState } from "react";
import { getEvalGameFile } from "../api/proto";
import type { EvalPoint, GameFile } from "../gen/viewer_pb";
import { GameTile } from "./GameTile";

type Props = { runId: string; evalPoints: EvalPoint[] };

// Per-game metadata the arena stores in the eval file's config slot: which
// seat parity the new net played and which side won, in game order.
type EvalGameMeta = { a_first?: boolean; winner?: "A" | "B" | null };

// Browse the league's recorded matches. A checkpoint joins the league every
// league_entrant_gens generations and game pairs between pool members run
// back-to-back on pinned CPU cores; all games are recorded in the same schema
// as self-play samples, so the tiles (board, scrubber, policy popover, value
// bars) are the same GameTile primitives.
export function EvalViewer({ runId, evalPoints }: Props) {
  // API order is oldest first; browse newest first like the sample games list.
  // Each entry is one league match, keyed by its match number.
  const points = useMemo(() => [...evalPoints].reverse(), [evalPoints]);
  const [sel, setSel] = useState<{ seq: bigint; gen: number; opp: number } | null>(null);
  const [file, setFile] = useState<GameFile | null>(null);
  const [loading, setLoading] = useState(false);
  const [fps, setFps] = useState(12);
  const [cell, setCell] = useState(16);
  const [followLatest, setFollowLatest] = useState(true);

  useEffect(() => {
    if (followLatest && points.length)
      setSel({ seq: points[0].seq, gen: points[0].gen, opp: points[0].opponentGen });
  }, [followLatest, points]);

  useEffect(() => {
    if (sel == null) return;
    let alive = true;
    setLoading(true);
    getEvalGameFile(runId, sel.seq, sel.gen, sel.opp)
      .then((next) => alive && setFile(next))
      .catch(() => alive && setFile(null))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [runId, sel]);

  const selected =
    points.find(
      (p) => p.seq === sel?.seq && p.gen === sel?.gen && p.opponentGen === sel?.opp,
    ) ?? null;
  const games = file?.games ?? [];
  // Per-game seat/winner metadata rides along in the file's config slot.
  const gameMeta = useMemo<EvalGameMeta[]>(() => {
    if (!file?.configJson) return [];
    try {
      const doc = JSON.parse(file.configJson) as { games?: EvalGameMeta[] };
      return doc.games ?? [];
    } catch {
      return [];
    }
  }, [file?.configJson]);
  const intervalMs = Math.round(1000 / fps);

  if (points.length === 0) {
    return (
      <div className="rounded border border-slate-800 bg-slate-900 p-4 text-sm text-slate-400">
        No completed league matches yet. The league needs two checkpoints in the pool (the first joins at
        gen <span className="font-mono">league_entrant_gens</span>), then plays game pairs back-to-back on CPU —
        the first result lands a few minutes after that. Set{" "}
        <span className="font-mono">league_entrant_gens</span> to 0 in the run config to disable it.
      </div>
    );
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[14rem_minmax(0,1fr)] lg:items-start">
      <div className="rounded border border-slate-800 bg-slate-900">
        <div className="flex items-center justify-between gap-2 border-b border-slate-800 px-3 py-2">
          <span className="section-title">Eval points</span>
          <label className="flex items-center gap-1 text-[10px] text-slate-400" title="Keep showing the newest eval as it arrives">
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
          {points.map((p) => {
            const active =
              p.seq === sel?.seq && p.gen === sel?.gen && p.opponentGen === sel?.opp;
            const decisive = p.wins + p.losses;
            return (
              <button
                key={`${p.seq}-${p.gen}-${p.opponentGen}`}
                onClick={() => {
                  setFollowLatest(false);
                  setSel({ seq: p.seq, gen: p.gen, opp: p.opponentGen });
                }}
                className={`flex w-full flex-col gap-0.5 border-b border-slate-800/60 px-3 py-2 text-left ${active ? "bg-sky-950/60" : "hover:bg-slate-800/50"}`}
              >
                <span className={`text-sm font-semibold ${active ? "text-sky-300" : "text-slate-200"}`}>
                  gen {p.gen} <span className="font-normal text-slate-500">vs {p.opponentGen}</span>
                </span>
                <span className="font-mono text-[10px] text-slate-500">
                  #{String(p.seq)} · {p.wins}-{p.losses}
                  {p.draws > 0 ? `-${p.draws}d` : ""}
                  {decisive === 0 ? " · all draws" : ""}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <div className="mb-2 flex flex-wrap items-center gap-4">
          <span className="text-sm text-slate-300">
            {selected
              ? `gen ${selected.gen} vs gen ${selected.opponentGen} · score ${selected.score.toFixed(2)} ± ${selected.scoreCi95.toFixed(2)} · ${selected.sims} sims`
              : ""}
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
        </div>
        {loading && games.length === 0 ? (
          <div className="text-sm text-slate-500">loading…</div>
        ) : games.length === 0 ? (
          <div className="text-sm text-slate-500">No games recorded for this eval point.</div>
        ) : (
          <div
            className="grid gap-3"
            style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${cell * 12}px, 1fr))` }}
          >
            {games.map((game, i) => (
              <div key={i}>
                <GameTile game={game} intervalMs={intervalMs} cell={cell} />
                {selected && <EvalGameCaption point={selected} meta={gameMeta[i]} />}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Which net won this game. Side A is always the newer net in trainer evals;
// seat parity (a_first) tells which snake colors it held.
function EvalGameCaption({ point, meta }: { point: EvalPoint; meta?: EvalGameMeta }) {
  if (!meta) return null;
  const label =
    meta.winner === "A" ? `gen ${point.gen} wins` : meta.winner === "B" ? `gen ${point.opponentGen} wins` : "draw";
  const tone =
    meta.winner === "A" ? "text-emerald-400" : meta.winner === "B" ? "text-red-400" : "text-slate-500";
  const seats = meta.a_first == null ? "" : meta.a_first ? " · new = even seats" : " · new = odd seats";
  return (
    <div className={`mt-1 text-center text-[10px] ${tone}`}>
      {label}
      <span className="text-slate-600">{seats}</span>
    </div>
  );
}
