import { useEffect, useMemo, useState } from "react";
import type { LiveEval, LiveEvalGame, LiveFrame } from "../api/eval";
import { getEvalGameFile } from "../api/proto";
import type { GameFile, LeagueMatch, LeagueRating } from "../gen/viewer_pb";
import { useEvalLive } from "../hooks/useEvalLive";
import { snakeColor } from "../lib/moves";
import { Board } from "./Board";
import type { Coord } from "./Board";
import { GameTile } from "./GameTile";

type Props = { runId: string; matches: LeagueMatch[]; league: LeagueRating[]; live: boolean };

// Metadata the arena stores in a recording's config slot: net names in --nets
// order, and each game's placements (net indexes that list).
type ArenaDoc = {
  config?: { nets?: { name?: string }[] };
  games?: { placements?: { seat: number; net: number; rank: number }[] }[];
};

const genName = (gen: number) => `gen_${String(gen).padStart(4, "0")}`;

// Browse the league's recorded games. Every game is a full multiplayer match —
// each snake driven by a different checkpoint — scored by elimination order
// and rated with Plackett–Luce. Games run back-to-back on pinned CPU cores
// while the run is active; all frames are recorded in the same schema as
// self-play samples, so the tiles are the same GameTile primitives.
export function EvalViewer({ runId, matches, league, live }: Props) {
  const liveMatch = useEvalLive(live);
  // API order is oldest first; browse newest first like the sample games list.
  const points = useMemo(() => [...matches].reverse(), [matches]);
  const [sel, setSel] = useState<bigint | null>(null);
  const [file, setFile] = useState<GameFile | null>(null);
  const [loading, setLoading] = useState(false);
  const [fps, setFps] = useState(12);
  const [cell, setCell] = useState(16);
  const [followLatest, setFollowLatest] = useState(true);

  useEffect(() => {
    if (followLatest && points.length) setSel(points[0].seq);
  }, [followLatest, points]);

  useEffect(() => {
    if (sel == null) return;
    let alive = true;
    setLoading(true);
    getEvalGameFile(runId, sel)
      .then((next) => alive && setFile(next))
      .catch(() => alive && setFile(null))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [runId, sel]);

  const selected = points.find((p) => p.seq === sel) ?? null;
  const games = file?.games ?? [];
  const doc = useMemo<ArenaDoc | null>(() => {
    if (!file?.configJson) return null;
    try {
      return JSON.parse(file.configJson) as ArenaDoc;
    } catch {
      return null;
    }
  }, [file?.configJson]);
  const intervalMs = Math.round(1000 / fps);

  if (points.length === 0) {
    return (
      <div className="grid gap-3">
        <LiveMatchBanner live={liveMatch} />
        <div className="rounded border border-slate-800 bg-slate-900 p-4 text-sm text-slate-400">
          No completed league games yet. The league needs two checkpoints in the pool (the first joins at
          gen <span className="font-mono">league_entrant_gens</span>), then plays games back-to-back on CPU —
          the first result lands a few minutes after that. Set{" "}
          <span className="font-mono">league_entrant_gens</span> to 0 in the run config to disable it.
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-3">
      <LiveMatchBanner live={liveMatch} />
      <div className="grid gap-4 lg:grid-cols-[16rem_minmax(0,1fr)] lg:items-start">
        <div className="grid gap-4">
          <Leaderboard league={league} />
          <div className="rounded border border-slate-800 bg-slate-900">
            <div className="flex items-center justify-between gap-2 border-b border-slate-800 px-3 py-2">
              <span className="section-title">Games</span>
              <label className="flex items-center gap-1 text-[10px] text-slate-400" title="Keep showing the newest game as it arrives">
                <input
                  type="checkbox"
                  checked={followLatest}
                  onChange={(e) => setFollowLatest(e.target.checked)}
                  className="accent-sky-500"
                />
                follow latest
              </label>
            </div>
            <div className="max-h-[24rem] overflow-y-auto">
              {points.map((p) => {
                const active = p.seq === sel;
                const order = [...p.placements].sort((a, b) => a.rank - b.rank);
                return (
                  <button
                    key={String(p.seq)}
                    onClick={() => {
                      setFollowLatest(false);
                      setSel(p.seq);
                    }}
                    className={`flex w-full flex-col gap-0.5 border-b border-slate-800/60 px-3 py-2 text-left ${active ? "bg-sky-950/60" : "hover:bg-slate-800/50"}`}
                  >
                    <span className={`font-mono text-xs ${active ? "text-sky-300" : "text-slate-200"}`}>
                      {order.map((pl) => `g${pl.gen}`).join(" › ")}
                    </span>
                    <span className="font-mono text-[10px] text-slate-500">
                      #{String(p.seq)} · {p.turns} turns
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <div>
          <div className="mb-2 flex flex-wrap items-center gap-4">
            <span className="text-sm text-slate-300">
              {selected
                ? `game #${selected.seq} · ${[...selected.placements]
                    .sort((a, b) => a.rank - b.rank)
                    .map((p) => genName(p.gen))
                    .join(" › ")} · ${selected.sims} sims`
                : ""}
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
            <div className="text-sm text-slate-500">No recording for this game (pruned).</div>
          ) : (
            <div
              className="grid gap-3"
              style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${cell * 12}px, 1fr))` }}
            >
              {games.map((game, i) => (
                <div key={i}>
                  <GameTile game={game} intervalMs={intervalMs} cell={cell} />
                  <SeatLegend doc={doc} gameIndex={i} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Every rated net ranked by fitted league Elo (Plackett–Luce over all games,
// anchored at the earliest generation = 0).
function Leaderboard({ league }: { league: LeagueRating[] }) {
  if (league.length === 0) return null;
  const rows = [...league].sort((a, b) => b.elo - a.elo);
  return (
    <div className="rounded border border-slate-800 bg-slate-900">
      <div className="border-b border-slate-800 px-3 py-2">
        <span className="section-title">Leaderboard</span>
      </div>
      <div className="max-h-[20rem] overflow-y-auto">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-slate-900">
            <tr className="text-[9px] uppercase text-slate-600">
              <th className="px-2 py-1 text-left">#</th>
              <th className="text-left">net</th>
              <th className="text-right">elo</th>
              <th className="text-right" title="rank-1 finishes / games">wins</th>
              <th className="pr-2 text-right" title="average finishing rank">rank</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.gen} className="border-t border-slate-800/60">
                <td className="px-2 py-1 text-slate-500">{i + 1}</td>
                <td className="font-mono text-slate-200">{genName(r.gen)}</td>
                <td className={`text-right font-mono ${r.elo >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {r.elo >= 0 ? "+" : ""}
                  {r.elo.toFixed(0)}
                </td>
                <td className="text-right font-mono text-slate-400">
                  {r.wins}/{r.games}
                </td>
                <td className="pr-2 text-right font-mono text-slate-500">{r.avgRank.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// The league game being played right now (streamed over SSE): live boards and
// every player's fitted Elo.
function LiveMatchBanner({ live }: { live: LiveEval | null }) {
  if (!live?.active) return null;
  return (
    <div className="rounded border border-emerald-900/60 bg-emerald-950/20 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-slate-300">
        <span className="flex items-center gap-1.5 font-semibold text-emerald-400">
          <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
          live game #{live.seq}
        </span>
        {live.players.map((p, i) => (
          <span key={i} className="flex items-baseline gap-1.5 rounded bg-slate-900/80 px-2 py-1">
            <span className="h-2 w-2 shrink-0 self-center rounded-full" style={{ background: snakeColor(i) }} />
            <span className="font-mono font-semibold text-slate-100">{genName(p.gen)}</span>
            <span className={`font-mono ${p.elo >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {p.elo >= 0 ? "+" : ""}
              {p.elo.toFixed(0)}
            </span>
            <span className="text-[10px] text-slate-500">{p.games}g</span>
          </span>
        ))}
      </div>
      {live.games.length > 0 && (
        <div className="flex flex-wrap gap-4">
          {live.games.map((g) => (
            <LiveGameBoard key={g.index} game={g} />
          ))}
        </div>
      )}
    </div>
  );
}

// A live board for one in-flight game. Seat s is played by player s % N, and
// the player chips above carry the matching seat-0..N-1 colors, so the dot
// colors line up board ↔ chips (league games have one seat per player).
function LiveGameBoard({ game }: { game: LiveEvalGame }) {
  const f: LiveFrame | null = game.frame;
  if (!f) {
    return (
      <div className="text-xs text-slate-500">
        game {game.index} · turn {game.turn}
      </div>
    );
  }
  const pt = ([x, y]: [number, number]): Coord => ({ x, y });
  return (
    <div className="w-56">
      <Board
        width={f.width}
        height={f.height}
        snakes={f.snakes.map((s) => ({ body: s.body.map(pt), alive: s.alive }))}
        food={f.food.map(pt)}
        hazards={f.hazards.map(pt)}
        cell={18}
      />
      <div className="mt-1 flex items-center justify-between text-[10px] text-slate-400">
        <span className="font-mono">turn {game.turn}</span>
        <span className="font-mono text-slate-600">{f.snakes.filter((s) => s.alive).length} alive</span>
      </div>
    </div>
  );
}

// Which net held each snake color in a recorded game, from the arena metadata
// stored alongside the recording.
function SeatLegend({ doc, gameIndex }: { doc: ArenaDoc | null; gameIndex: number }) {
  const placements = doc?.games?.[gameIndex]?.placements;
  const nets = doc?.config?.nets;
  if (!placements || !nets) return null;
  const bySeat = [...placements].sort((a, b) => a.seat - b.seat);
  return (
    <div className="mt-1 flex flex-wrap justify-center gap-x-2 gap-y-0.5 text-[10px] text-slate-500">
      {bySeat.map((p) => (
        <span key={p.seat} className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full" style={{ background: snakeColor(p.seat) }} />
          <span className={p.rank === 1 ? "text-emerald-400" : ""}>
            {nets[p.net]?.name ?? `net ${p.net}`}
            {p.rank === 1 ? " ♛" : ""}
          </span>
        </span>
      ))}
    </div>
  );
}
