import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getEvalGameFile, getGameFile } from "../../api/proto";
import type { GameFile, GameGen, LeagueMatch, MatchPlacement, MetricRow } from "../../gen/viewer_pb";
import { playerColor, snakeColor } from "../../lib/palette";
import { playerName, playerNameLong } from "../../lib/players";
import { GameTile } from "../GameTile";

type Props = {
  runId: string;
  matches: LeagueMatch[];
  gameGens: GameGen[];
  metrics: MetricRow[];
  /** Which game collection this instance shows (each has its own tab). */
  mode: "selfplay" | "arena";
};

/** A league player's stable color, hashed from its display name. */
const genColor = (gen: number) => playerColor(playerNameLong(gen));

/** Below this tile size the league history renders as a results table. */
const TABLE_CELL_THRESHOLD = 11;

const PAGE_SIZES = [12, 24, 48, 96];

// Watch actual play — the qualitative gut check the charts can't give. Two
// tabs, each a GamesPanels instance: "self-play" (the games training learns
// from) and "arena" (recorded held-out/league games). One control row scopes
// the tab; the Size slider doubles as the arena history's zoom (small enough
// and it becomes a pure results table).
export function GamesPanels({ runId, matches, gameGens, metrics, mode }: Props) {
  const [fps, setFps] = useState(12);
  const [cell, setCell] = useState(14);
  const [followLatest, setFollowLatest] = useState(true);
  const intervalMs = Math.round(1000 / fps);
  // LE runs have no checkpoint league; the "league" files are the held-out
  // baseline eval games, so name the panel for what it shows.
  const isLE = metrics.some((m) => m.hasLeEval);

  return (
    <section>
      <div className="mb-1.5 flex flex-wrap items-center gap-x-4 gap-y-1">
        <label className="flex items-center gap-2 text-xs text-ink-3">
          Speed
          <input type="range" min={1} max={20} value={fps} onChange={(e) => setFps(Number(e.target.value))} className="accent-accent" />
        </label>
        <label className="flex items-center gap-2 text-xs text-ink-3" title="Tile size; small enough turns the arena history into a table">
          Size
          <input type="range" min={8} max={30} value={cell} onChange={(e) => setCell(Number(e.target.value))} className="accent-accent" />
        </label>
        <label className="ml-auto flex items-center gap-1.5 text-xs text-ink-3" title="Keep showing the newest games as they arrive">
          <input
            type="checkbox"
            checked={followLatest}
            onChange={(e) => setFollowLatest(e.target.checked)}
            className="accent-accent"
          />
          follow latest
        </label>
      </div>

      {mode === "selfplay" ? (
        <SelfPlayPanel
          runId={runId}
          gameGens={gameGens}
          metrics={metrics}
          followLatest={followLatest}
          setFollow={setFollowLatest}
          intervalMs={intervalMs}
          cell={cell}
        />
      ) : (
        <LeaguePanel
          runId={runId}
          matches={matches}
          title={isLE ? "Held-out games" : "League game history"}
          followLatest={followLatest}
          setFollow={setFollowLatest}
          intervalMs={intervalMs}
          cell={cell}
        />
      )}
    </section>
  );
}

// The one header used by all three games panels, so their top bars are
// pixel-identical: the live dot lives inside the fixed-height row instead of
// wrapping the title in its own flex container.
function PanelShell({
  title,
  children,
  accent,
  liveDot,
}: {
  title: React.ReactNode;
  children: React.ReactNode;
  accent?: boolean;
  liveDot?: boolean;
}) {
  return (
    <div className={`card ${accent ? "border-good/30" : ""}`}>
      <div className="flex items-center gap-1.5 border-b border-white/10 px-3 py-1.5">
        {liveDot && <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-good" />}
        <span className={`card-title ${accent ? "text-good" : ""}`}>{title}</span>
      </div>
      <div className="p-2.5">{children}</div>
    </div>
  );
}

/** A game's finishing order as colored player names, best first. */
function RankingLine({ placements }: { placements: MatchPlacement[] }) {
  const order = useMemo(() => [...placements].sort((a, b) => a.rank - b.rank), [placements]);
  return (
    <span className="font-mono">
      {order.map((p, i) => (
        <span key={`${p.seat}`}>
          {i > 0 && <span className="text-ink-3"> › </span>}
          <span style={{ color: genColor(p.gen) }}>{playerName(p.gen)}</span>
        </span>
      ))}
    </span>
  );
}

type ViewerProps = {
  runId: string;
  followLatest: boolean;
  setFollow: (v: boolean) => void;
  intervalMs: number;
  cell: number;
};

// The recorded league history, newest first, paginated. Zoomed in it's a grid
// of replay tiles (one per game, colored by player); zoomed out (Size slider
// below the threshold) it's a pure results table over the same page.
function LeaguePanel({ runId, matches, title, followLatest, setFollow, intervalMs, cell }: ViewerProps & { matches: LeagueMatch[]; title: string }) {
  const items = useMemo(() => [...matches].reverse(), [matches]);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(PAGE_SIZES[0]);
  const pages = Math.max(1, Math.ceil(items.length / pageSize));

  useEffect(() => {
    if (followLatest) setPage(0);
  }, [followLatest, items.length]);
  useEffect(() => {
    setPage((p) => Math.min(p, pages - 1));
  }, [pages]);

  const pageItems = items.slice(page * pageSize, (page + 1) * pageSize);
  const asTable = cell < TABLE_CELL_THRESHOLD;
  const go = (next: number) => {
    setFollow(false);
    setPage(Math.max(0, Math.min(pages - 1, next)));
  };

  return (
    <PanelShell title={`${title} (${matches.length})`}>
      {items.length === 0 ? (
        <div className="text-xs text-ink-3">No recorded games yet.</div>
      ) : (
        <>
          <div className="mb-2 flex items-center gap-2 text-[11px] text-ink-3">
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setPage(0);
              }}
              className="rounded border border-white/10 bg-inset px-1 py-0.5 text-ink-2"
              title="Games per page"
            >
              {PAGE_SIZES.map((n) => (
                <option key={n} value={n}>
                  {n} / page
                </option>
              ))}
            </select>
            <button onClick={() => go(page - 1)} disabled={page === 0} className="rounded border border-white/10 px-1.5 py-0.5 text-ink-2 disabled:opacity-30">
              ‹
            </button>
            <span className="font-mono tabular-nums">
              {page + 1}/{pages}
            </span>
            <button
              onClick={() => go(page + 1)}
              disabled={page >= pages - 1}
              className="rounded border border-white/10 px-1.5 py-0.5 text-ink-2 disabled:opacity-30"
            >
              ›
            </button>
            <span className="ml-auto">newest first</span>
          </div>
          {asTable ? (
            <div className="max-h-[28rem] overflow-y-auto rounded-md border border-white/5 bg-inset/50">
              {pageItems.map((m) => (
                <div
                  key={String(m.seq)}
                  className="flex items-baseline justify-between gap-2 border-b border-white/5 px-2.5 py-1 text-xs last:border-0"
                >
                  <span className="shrink-0 font-mono tabular-nums text-ink-3">#{String(m.seq)}</span>
                  <span className="min-w-0 flex-1 truncate">
                    <RankingLine placements={m.placements} />
                  </span>
                  <span className="shrink-0 font-mono text-[10px] tabular-nums text-ink-3">{m.turns}t</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="grid gap-2.5" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${cell * 12}px, 1fr))` }}>
              {pageItems.map((m) => (
                <LeagueGameTile key={String(m.seq)} runId={runId} match={m} intervalMs={intervalMs} cell={cell} />
              ))}
            </div>
          )}
        </>
      )}
    </PanelShell>
  );
}

// One recorded league/held-out game: finishing order on top, the replay tile
// below (snakes colored by player). Games are stored forever; the ⛶ opens the
// fullscreen permalink view. Old runs may still have pruned recordings, which
// fall back to the result line alone.
function LeagueGameTile({ runId, match, intervalMs, cell }: { runId: string; match: LeagueMatch; intervalMs: number; cell: number }) {
  const file = useGameFile(() => getEvalGameFile(runId, match.seq), [runId, match.seq]);
  const game = file?.games[0];
  const colors = useMemo(() => {
    const bySeat = new Map(match.placements.map((p) => [p.seat, genColor(p.gen)]));
    const seats = Math.max(match.placements.length, 1);
    return Array.from({ length: seats }, (_, s) => bySeat.get(s) ?? snakeColor(s));
  }, [match.placements]);
  return (
    <div>
      <div className="mb-0.5 flex items-baseline gap-1 truncate text-[10px]" title={`game #${match.seq} · ${match.sims} sims/move`}>
        <span className="font-mono tabular-nums text-ink-3">#{String(match.seq)}</span>
        <span className="min-w-0 flex-1 truncate">
          <RankingLine placements={match.placements} />
        </span>
        <Link
          to={`/runs/${encodeURIComponent(runId)}/eval/${match.seq}`}
          className="shrink-0 text-ink-3 hover:text-accent"
          title="Open fullscreen (permalink)"
        >
          ⛶
        </Link>
      </div>
      {game ? (
        <GameTile game={game} intervalMs={intervalMs} cell={cell} colors={colors} />
      ) : (
        <div className="card flex h-20 items-center justify-center p-2 text-[10px] text-ink-3">
          {file === undefined ? "loading…" : "recording pruned"}
        </div>
      )}
    </div>
  );
}

// Self-play sample games: the games training actually learns from.
function SelfPlayPanel({
  runId,
  gameGens,
  metrics,
  followLatest,
  setFollow,
  intervalMs,
  cell,
}: ViewerProps & { gameGens: GameGen[]; metrics: MetricRow[] }) {
  const [gen, setGen] = useState<number | null>(null);
  const file = useGameFile(gen == null ? null : () => getGameFile(runId, gen), [runId, gen]);
  const metricByGen = useMemo(() => new Map(metrics.map((m) => [m.generation, m])), [metrics]);

  useEffect(() => {
    if (followLatest && gameGens.length) setGen(gameGens[0].gen);
  }, [followLatest, gameGens]);

  return (
    <PanelShell title={`Self-play samples (${gameGens.length} gens)`}>
      {gameGens.length === 0 ? (
        <div className="text-xs text-ink-3">No recorded sample games yet — the trainer writes a few per generation.</div>
      ) : (
        <>
          <div className="mb-2 max-h-36 overflow-y-auto rounded-md border border-white/5 bg-inset/50">
            {gameGens.map((g) => {
              const metric = metricByGen.get(g.gen);
              return (
                <ListRow
                  key={g.gen}
                  active={g.gen === gen}
                  onClick={() => {
                    setFollow(false);
                    setGen(g.gen);
                  }}
                  title={`gen ${g.gen}`}
                  sub={metric ? `π ${metric.policyLoss.toFixed(3)} · v ${metric.valueLoss.toFixed(3)}` : ""}
                />
              );
            })}
          </div>
          {file == null ? (
            <div className="text-xs text-ink-3">No recording for this selection.</div>
          ) : (
            <div className="grid gap-2.5" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${cell * 12}px, 1fr))` }}>
              {file.games.map((game, i) => (
                <div key={i}>
                  <div className="mb-0.5 flex items-baseline gap-1.5 text-[10px] text-ink-3">
                    <span className="font-mono tabular-nums">game {i + 1}</span>
                    {game.scenario && (
                      <span
                        className="rounded bg-warn/15 px-1 font-mono text-warn"
                        title={`seeded from the "${game.scenario}" curriculum scenario`}
                      >
                        {game.scenario}
                      </span>
                    )}
                    <Link
                      to={`/runs/${encodeURIComponent(runId)}/games/${gen}/${i}`}
                      className="ml-auto hover:text-accent"
                      title="Open fullscreen (permalink)"
                    >
                      ⛶
                    </Link>
                  </div>
                  <GameTile game={game} intervalMs={intervalMs} cell={cell} />
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </PanelShell>
  );
}

// Load a recording whenever the selection changes; `undefined` while loading,
// `null` when the fetch failed (e.g. the recording was pruned).
function useGameFile(load: (() => Promise<GameFile>) | null, deps: unknown[]) {
  const [file, setFile] = useState<GameFile | null | undefined>(undefined);
  useEffect(() => {
    if (!load) return;
    let alive = true;
    setFile(undefined);
    load()
      .then((next) => alive && setFile(next))
      .catch(() => alive && setFile(null));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return file;
}

function ListRow({ active, onClick, title, sub }: { active: boolean; onClick: () => void; title: string; sub: string }) {
  return (
    <button
      onClick={onClick}
      className={`flex w-full items-baseline justify-between gap-2 border-b border-white/5 px-2.5 py-1.5 text-left last:border-0 ${
        active ? "bg-accent/10" : "hover:bg-white/5"
      }`}
    >
      <span className={`truncate font-mono text-xs ${active ? "text-accent" : "text-ink-2"}`}>{title}</span>
      {sub && <span className="shrink-0 font-mono text-[10px] tabular-nums text-ink-3">{sub}</span>}
    </button>
  );
}
