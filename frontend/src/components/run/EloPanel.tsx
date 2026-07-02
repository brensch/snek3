import { useState } from "react";
import type { LeagueRating } from "../../gen/viewer_pb";
import { isHeuristic, playerName, playerNameLong } from "../../lib/players";
import { series } from "../../lib/palette";
import { LineChart } from "../charts/LineChart";

const signed = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(0)}`;

// The run's headline: fitted league Elo (Plackett–Luce over every game,
// anchored at the earliest checkpoint = 0). A hero figure for the newest
// checkpoint next to the fixed flood-fill baseline's Elo (the one player that
// never learns — the run works iff the gap keeps widening), the
// Elo-by-checkpoint curve, and the leaderboard.
export function EloPanel({ league }: { league: LeagueRating[] }) {
  // On phones the leaderboard is behind a toggle under the chart; on md+ it is
  // always visible.
  const [showBoard, setShowBoard] = useState(false);
  const nets = league.filter((r) => !isHeuristic(r.gen));
  const heuristic = league.find((r) => isHeuristic(r.gen)) ?? null;
  if (nets.length === 0 || league.length < 2) {
    return (
      <div className="card p-3 text-sm text-ink-3">
        League Elo appears once the first checkpoint has played the flood-fill baseline (entrants every{" "}
        <span className="font-mono">league_entrant_gens</span> gens).
      </div>
    );
  }
  const latest = nets[nets.length - 1];
  const prev = nets.length > 1 ? nets[nets.length - 2] : null;
  const best = [...nets].sort((a, b) => b.elo - a.elo)[0];

  return (
    <div className="grid gap-2.5 md:grid-cols-[10.5rem_minmax(0,1fr)] xl:grid-cols-[10.5rem_minmax(0,1fr)_13rem]">
      <div className="card flex flex-row items-center justify-between gap-2 p-3 md:flex-col md:items-stretch">
        <span className="card-title">League Elo</span>
        <div>
          <div className="text-4xl font-semibold leading-none text-ink">{signed(latest.elo)}</div>
          <div className="mt-1.5 text-xs text-ink-3">gen_{String(latest.gen).padStart(4, "0")}</div>
          {prev && (
            <div className={`mt-1 text-xs font-medium ${latest.elo - prev.elo >= 0 ? "text-good" : "text-bad"}`}>
              {signed(latest.elo - prev.elo)} vs g{prev.gen}
            </div>
          )}
        </div>
        <div className="text-right text-[11px] leading-tight text-ink-3 md:text-left">
          {heuristic && (
            <>
              floodfill <span className="font-mono text-ink-2">{signed(heuristic.elo)}</span>
              <br />
            </>
          )}
          best {signed(best.elo)} at g{best.gen}
        </div>
      </div>

      <LineChart
        title="Elo by checkpoint"
        series={[
          { name: "Elo", color: series.blue, values: nets.map((r) => r.elo) },
          ...(heuristic
            ? [{ name: "floodfill", color: series.yellow, values: nets.map(() => heuristic.elo) }]
            : []),
        ]}
        xValues={nets.map((r) => r.gen)}
        height={168}
        area
        format={signed}
      />

      <button
        type="button"
        className="btn w-full text-xs md:hidden"
        aria-pressed={showBoard}
        onClick={() => setShowBoard((v) => !v)}
      >
        {showBoard ? "Hide leaderboard" : `Leaderboard (${league.length} players)`}
      </button>
      <div className={`${showBoard ? "" : "hidden"} md:block md:col-span-2 xl:col-span-1`}>
        <Leaderboard league={league} />
      </div>
    </div>
  );
}

// Every rated player ranked by fitted Elo — wins are rank-1 finishes.
function Leaderboard({ league }: { league: LeagueRating[] }) {
  const rows = [...league].sort((a, b) => b.elo - a.elo);
  return (
    <div className="card flex h-full min-h-0 flex-col">
      <div className="border-b border-white/10 px-3 py-1.5">
        <span className="card-title">Leaderboard</span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto" style={{ maxHeight: 200 }}>
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-surface">
            <tr className="text-[9px] uppercase text-ink-3/70">
              <th className="px-2 py-1 text-left font-medium">#</th>
              <th className="text-left font-medium">player</th>
              <th className="text-right font-medium">elo</th>
              <th className="pr-2 text-right font-medium" title="rank-1 finishes / games">
                wins
              </th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {rows.map((r, i) => (
              <tr key={r.gen} className="border-t border-white/5">
                <td className="px-2 py-0.5 text-ink-3">{i + 1}</td>
                <td className="text-ink-2" title={playerNameLong(r.gen)}>
                  {playerName(r.gen)}
                </td>
                <td className="text-right text-ink">{signed(r.elo)}</td>
                <td className="pr-2 text-right text-ink-3" title={`avg rank ${r.avgRank.toFixed(2)}`}>
                  {r.wins}/{r.games}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
