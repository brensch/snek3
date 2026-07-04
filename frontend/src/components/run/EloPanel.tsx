import { useState } from "react";
import type { LeagueRating } from "../../gen/viewer_pb";
import { isExternal, playerName, playerNameLong } from "../../lib/players";
import { series } from "../../lib/palette";
import { LineChart } from "../charts/LineChart";

const signed = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(0)}`;

// Flat reference lines for external players, in a stable color order.
const EXTERNAL_COLORS = [series.yellow, series.orange, series.magenta, series.aqua, series.violet];

const displayName = (r: LeagueRating) => r.name || playerName(r.gen);

// The run's headline: fitted league Elo (Plackett–Luce over every game,
// anchored at the earliest checkpoint = 0). The hero card shows the latest
// net's margin over the strongest fixed baseline (green ahead / red behind),
// its absolute Elo and improvement trend, and the three fixed competitors
// (heuristics and API snakes that never learn — the run works iff the gap
// keeps widening). Beside it: the Elo-by-checkpoint curve and the leaderboard.
export function EloPanel({ league }: { league: LeagueRating[] }) {
  // On phones the leaderboard is behind a toggle under the chart; on md+ it is
  // always visible.
  const [showBoard, setShowBoard] = useState(false);
  const nets = league.filter((r) => !isExternal(r.gen));
  const externals = league.filter((r) => isExternal(r.gen));
  if (nets.length === 0 || league.length < 2) {
    return (
      <div className="card p-3 text-sm text-ink-3">
        League Elo appears once the first checkpoint has played an external baseline (entrants every{" "}
        <span className="font-mono">league_entrant_gens</span> gens).
      </div>
    );
  }
  const latest = nets[nets.length - 1];
  const prev = nets.length > 1 ? nets[nets.length - 2] : null;
  // The strongest fixed baseline — the bar the net has to clear. The headline
  // figure is our latest Elo relative to it.
  const bestExt = externals.length ? [...externals].sort((a, b) => b.elo - a.elo)[0] : null;
  const margin = bestExt ? latest.elo - bestExt.elo : null;
  const trend = prev ? latest.elo - prev.elo : null;

  return (
    <div className="grid gap-2.5 md:grid-cols-[10.5rem_minmax(0,1fr)] xl:grid-cols-[10.5rem_minmax(0,1fr)_20rem]">
      <div className="card flex flex-col justify-between gap-3 p-3">
        <span className="card-title">League Elo</span>

        {/* Headline: how far the latest net sits above (green) or below (red)
            the strongest non-learning competitor. */}
        <div>
          {margin != null ? (
            <>
              <div className={`text-4xl font-semibold leading-none ${margin >= 0 ? "text-good" : "text-bad"}`}>
                {signed(margin)}
              </div>
              <div className="mt-1 text-xs text-ink-3">
                {margin >= 0 ? "ahead of" : "behind"} {displayName(bestExt!)}
              </div>
            </>
          ) : (
            <div className="text-4xl font-semibold leading-none text-ink">{signed(latest.elo)}</div>
          )}
          <div className="mt-2 flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-xs">
            <span className="font-mono text-ink-2">{signed(latest.elo)} elo</span>
            <span className="text-ink-3">gen_{String(latest.gen).padStart(4, "0")}</span>
            {trend != null && (
              <span className={`font-medium ${trend >= 0 ? "text-good" : "text-bad"}`}>
                {trend >= 0 ? "▲" : "▼"} {signed(trend)}
              </span>
            )}
          </div>
        </div>

        {/* The three fixed competitors (Elo, anchored at the first checkpoint). */}
        <div className="space-y-0.5 text-[11px] leading-tight">
          {externals.map((r) => (
            <div key={r.gen} className="flex justify-between gap-2 text-ink-3">
              <span className="truncate">{displayName(r)}</span>
              <span className="font-mono text-ink-2">{signed(r.elo)}</span>
            </div>
          ))}
        </div>
      </div>

      <LineChart
        title="Elo by checkpoint"
        series={[
          { name: "Elo", color: series.blue, values: nets.map((r) => r.elo) },
          ...externals.map((r, i) => ({
            name: displayName(r),
            color: EXTERNAL_COLORS[i % EXTERNAL_COLORS.length],
            values: nets.map(() => r.elo),
          })),
        ]}
        xValues={nets.map((r) => r.gen)}
        height={288}
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

// Ordinal header for a 1-based rank column ("1st", "2nd", "3rd", "4th", ...).
const ordinal = (n: number) => {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
};

// Every rated player ranked by fitted Elo, with each player's finishes broken
// out by place (1sts, 2nds, 3rds, ...) instead of just rank-1 wins.
function Leaderboard({ league }: { league: LeagueRating[] }) {
  const rows = [...league].sort((a, b) => b.elo - a.elo);
  // Widest placement histogram across all rows = number of rank columns.
  const maxRanks = rows.reduce((m, r) => Math.max(m, r.placements.length), 0);
  const ranks = Array.from({ length: maxRanks }, (_, i) => i);
  return (
    <div className="card flex h-full min-h-0 flex-col">
      <div className="border-b border-white/10 px-3 py-1.5">
        <span className="card-title">Leaderboard</span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto" style={{ maxHeight: 320 }}>
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-surface">
            <tr className="text-[9px] uppercase text-ink-3/70">
              <th className="px-2 py-1 text-left font-medium">#</th>
              <th className="text-left font-medium">player</th>
              <th className="text-right font-medium">elo</th>
              {ranks.map((i) => (
                <th key={i} className="text-right font-medium" title={`${ordinal(i + 1)}-place finishes`}>
                  {ordinal(i + 1)}
                </th>
              ))}
              <th className="pr-2 text-right font-medium" title="games played">
                gp
              </th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {rows.map((r, i) => (
              <tr key={r.gen} className="border-t border-white/5">
                <td className="px-2 py-0.5 text-ink-3">{i + 1}</td>
                <td className="text-ink-2" title={r.name || playerNameLong(r.gen)}>
                  {displayName(r)}
                </td>
                <td className="text-right text-ink">{signed(r.elo)}</td>
                {ranks.map((idx) => (
                  <td key={idx} className={`text-right ${idx === 0 ? "text-ink" : "text-ink-3"}`}>
                    {r.placements[idx] ?? 0}
                  </td>
                ))}
                <td className="pr-2 text-right text-ink-3" title={`avg rank ${r.avgRank.toFixed(2)}`}>
                  {r.games}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
