import { useEffect, useMemo, useRef, useState } from "react";
import type { LiveEval, LiveFrame } from "../../api/eval";
import { getEvalGameFile, getGameFile } from "../../api/proto";
import type { GameFile, GameGen, LeagueMatch, MetricRow } from "../../gen/viewer_pb";
import { snakeColor } from "../../lib/palette";
import { playerName, playerNameLong } from "../../lib/players";
import { Board } from "../Board";
import type { Coord } from "../Board";
import { GameTile } from "../GameTile";

type Props = {
  runId: string;
  matches: LeagueMatch[];
  gameGens: GameGen[];
  metrics: MetricRow[];
  liveMatch: LiveEval | null;
};

// Metadata the arena stores in a recording's config slot: net names in --nets
// order, and each game's placements (net indexes that list).
type ArenaDoc = {
  config?: { nets?: { name?: string }[] };
  games?: { placements?: { seat: number; net: number; rank: number }[] }[];
};

const genName = (gen: number) => `gen_${String(gen).padStart(4, "0")}`; // self-play gens only; league players use playerName*

// Watch actual play — the qualitative gut check the charts can't give. Three
// panels side by side, all live at once: the league game in flight, recorded
// league games, and self-play samples. One control row scopes all three.
export function GamesPanels({ runId, matches, gameGens, metrics, liveMatch }: Props) {
  const [fps, setFps] = useState(12);
  const [cell, setCell] = useState(14);
  const [followLatest, setFollowLatest] = useState(true);
  const intervalMs = Math.round(1000 / fps);

  return (
    <section>
      <div className="mb-1.5 flex flex-wrap items-center gap-x-4 gap-y-1">
        <h2 className="card-title">Games</h2>
        <label className="flex items-center gap-2 text-xs text-ink-3">
          Speed
          <input type="range" min={1} max={20} value={fps} onChange={(e) => setFps(Number(e.target.value))} className="accent-accent" />
        </label>
        <label className="flex items-center gap-2 text-xs text-ink-3">
          Size
          <input type="range" min={8} max={30} value={cell} onChange={(e) => setCell(Number(e.target.value))} className="accent-accent" />
        </label>
        <label className="ml-auto flex items-center gap-1.5 text-xs text-ink-3" title="Keep showing the newest game as it arrives">
          <input
            type="checkbox"
            checked={followLatest}
            onChange={(e) => setFollowLatest(e.target.checked)}
            className="accent-accent"
          />
          follow latest
        </label>
      </div>

      <div className="grid gap-2.5 lg:grid-cols-3 lg:items-start">
        <LivePanel live={liveMatch} />
        <LeaguePanel runId={runId} matches={matches} followLatest={followLatest} setFollow={setFollowLatest} intervalMs={intervalMs} cell={cell} />
        <SelfPlayPanel runId={runId} gameGens={gameGens} metrics={metrics} followLatest={followLatest} setFollow={setFollowLatest} intervalMs={intervalMs} cell={cell} />
      </div>
    </section>
  );
}

function PanelShell({ title, children, accent }: { title: React.ReactNode; children: React.ReactNode; accent?: boolean }) {
  return (
    <div className={`card ${accent ? "border-good/30" : ""}`}>
      <div className="border-b border-white/10 px-3 py-1.5">
        <span className={`card-title ${accent ? "text-good" : ""}`}>{title}</span>
      </div>
      <div className="p-2.5">{children}</div>
    </div>
  );
}

// The league game being played right now (streamed over SSE): every player's
// fitted Elo and a live board per in-flight game. Seat s is played by player
// s % N, so the chip dots match the snakes on the board. Between games the
// panel keeps its size — last players dimmed, an empty board with a "starting
// next game" note — so the layout never jumps when a game ends.
function LivePanel({ live }: { live: LiveEval | null }) {
  const lastDims = useRef({ width: 11, height: 11 });
  const lastPlayers = useRef<LiveEval["players"]>([]);
  const frame = live?.active ? live.games.find((g) => g.frame)?.frame : null;
  if (frame) lastDims.current = { width: frame.width, height: frame.height };
  if (live?.active && live.players.length) lastPlayers.current = live.players;

  if (!live?.active || live.games.length === 0 || !frame) {
    // Mirror the active state's geometry exactly — same chip grid (placeholder
    // chips when no players are known yet), same board size, same footer line —
    // so ending or starting a game never shifts the layout.
    const players = live?.active ? live.players : lastPlayers.current;
    const chips = players.length > 0 ? players : Array.from({ length: 4 }, () => null);
    return (
      <PanelShell title="Live league game">
        <div className={`mb-2 grid grid-cols-2 gap-1.5 ${live?.active ? "" : "opacity-50"}`}>
          {chips.map((p, i) => (
            <span key={i} className="flex items-baseline gap-1.5 rounded-md bg-inset px-2 py-1 text-[11px]">
              <span className="h-2 w-2 shrink-0 self-center rounded-full" style={{ background: snakeColor(i) }} />
              <span className="font-mono font-semibold text-ink">{p ? playerName(p.gen) : "–"}</span>
              <span className="ml-auto font-mono tabular-nums text-ink-2">
                {p ? `${p.elo >= 0 ? "+" : ""}${p.elo.toFixed(0)}` : "–"}
              </span>
            </span>
          ))}
        </div>
        <div className="relative">
          <Board width={lastDims.current.width} height={lastDims.current.height} snakes={[]} cell={22} />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="rounded-md border border-white/10 bg-page/85 px-3 py-1.5 text-xs text-ink-2">
              starting next game…
            </span>
          </div>
        </div>
        <div className="mt-1 flex items-center justify-between font-mono text-[10px] tabular-nums text-ink-3">
          <span>turn –</span>
          <span>– alive</span>
        </div>
      </PanelShell>
    );
  }
  const pt = ([x, y]: [number, number]): Coord => ({ x, y });
  return (
    <PanelShell
      accent
      title={
        <span className="flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-good" />
          Live league game #{String(live.seq)}
        </span>
      }
    >
      <div className="mb-2 grid grid-cols-2 gap-1.5">
        {live.players.map((p, i) => (
          <span key={i} className="flex items-baseline gap-1.5 rounded-md bg-inset px-2 py-1 text-[11px]">
            <span className="h-2 w-2 shrink-0 self-center rounded-full" style={{ background: snakeColor(i) }} />
            <span className="font-mono font-semibold text-ink">{playerName(p.gen)}</span>
            <span className="ml-auto font-mono tabular-nums text-ink-2">
              {p.elo >= 0 ? "+" : ""}
              {p.elo.toFixed(0)}
            </span>
          </span>
        ))}
      </div>
      <div className="grid gap-3">
        {live.games.map((g) => {
          const f: LiveFrame | null = g.frame;
          if (!f) return null;
          return (
            <div key={g.index}>
              <Board
                width={f.width}
                height={f.height}
                snakes={f.snakes.map((s) => ({ body: s.body.map(pt), alive: s.alive }))}
                food={f.food.map(pt)}
                hazards={f.hazards.map(pt)}
                cell={22}
              />
              <div className="mt-1 flex items-center justify-between font-mono text-[10px] tabular-nums text-ink-3">
                <span>turn {g.turn}</span>
                <span>{f.snakes.filter((s) => s.alive).length} alive</span>
              </div>
            </div>
          );
        })}
      </div>
    </PanelShell>
  );
}

type ViewerProps = {
  runId: string;
  followLatest: boolean;
  setFollow: (v: boolean) => void;
  intervalMs: number;
  cell: number;
};

// Recorded league games: finishing-order list on top, the selected recording's
// tiles below (with a seat legend naming the nets).
function LeaguePanel({ runId, matches, followLatest, setFollow, intervalMs, cell }: ViewerProps & { matches: LeagueMatch[] }) {
  const items = useMemo(() => [...matches].reverse(), [matches]);
  const [seq, setSeq] = useState<bigint | null>(null);
  const file = useGameFile(seq == null ? null : () => getEvalGameFile(runId, seq), [runId, seq]);

  useEffect(() => {
    if (followLatest && items.length) setSeq(items[0].seq);
  }, [followLatest, items]);

  const doc = useMemo<ArenaDoc | null>(() => {
    if (!file?.configJson) return null;
    try {
      return JSON.parse(file.configJson) as ArenaDoc;
    } catch {
      return null;
    }
  }, [file?.configJson]);
  const selected = items.find((m) => m.seq === seq) ?? null;

  return (
    <PanelShell title={`League games (${matches.length})`}>
      {items.length === 0 ? (
        <div className="text-xs text-ink-3">No league games yet — they start once two checkpoints exist.</div>
      ) : (
        <>
          <div className="mb-2 max-h-36 overflow-y-auto rounded-md border border-white/5 bg-inset/50">
            {items.map((m) => {
              const order = [...m.placements].sort((a, b) => a.rank - b.rank);
              return (
                <ListRow
                  key={String(m.seq)}
                  active={m.seq === seq}
                  onClick={() => {
                    setFollow(false);
                    setSeq(m.seq);
                  }}
                  title={order.map((p) => playerName(p.gen)).join(" › ")}
                  sub={`#${String(m.seq)} · ${m.turns} turns`}
                />
              );
            })}
          </div>
          {selected && (
            <div className="mb-1.5 truncate text-[11px] text-ink-3" title={`${selected.sims} sims/move`}>
              <span className="font-mono text-ink-2">
                {[...selected.placements]
                  .sort((a, b) => a.rank - b.rank)
                  .map((p) => playerNameLong(p.gen))
                  .join(" › ")}
              </span>
            </div>
          )}
          <Tiles file={file} intervalMs={intervalMs} cell={cell} legend={(i) => <SeatLegend doc={doc} gameIndex={i} />} />
        </>
      )}
    </PanelShell>
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
          <Tiles file={file} intervalMs={intervalMs} cell={cell} />
        </>
      )}
    </PanelShell>
  );
}

// Load a recording whenever the selection changes; hold the old one meanwhile.
function useGameFile(load: (() => Promise<GameFile>) | null, deps: unknown[]) {
  const [file, setFile] = useState<GameFile | null>(null);
  useEffect(() => {
    if (!load) return;
    let alive = true;
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

function Tiles({
  file,
  intervalMs,
  cell,
  legend,
}: {
  file: GameFile | null;
  intervalMs: number;
  cell: number;
  legend?: (gameIndex: number) => React.ReactNode;
}) {
  const games = file?.games ?? [];
  if (games.length === 0) {
    return <div className="text-xs text-ink-3">No recording for this selection.</div>;
  }
  return (
    <div className="grid gap-2.5" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${cell * 12}px, 1fr))` }}>
      {games.map((game, i) => (
        <div key={i}>
          <GameTile game={game} intervalMs={intervalMs} cell={cell} />
          {legend?.(i)}
        </div>
      ))}
    </div>
  );
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

// Which net held each snake color in a recorded league game, from the arena
// metadata stored alongside the recording. Winner gets the crown.
function SeatLegend({ doc, gameIndex }: { doc: ArenaDoc | null; gameIndex: number }) {
  const placements = doc?.games?.[gameIndex]?.placements;
  const nets = doc?.config?.nets;
  if (!placements || !nets) return null;
  const bySeat = [...placements].sort((a, b) => a.seat - b.seat);
  return (
    <div className="mt-1 flex flex-wrap justify-center gap-x-2 gap-y-0.5 text-[10px] text-ink-3">
      {bySeat.map((p) => (
        <span key={p.seat} className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full" style={{ background: snakeColor(p.seat) }} />
          <span className={p.rank === 1 ? "text-good" : ""}>
            {nets[p.net]?.name ?? `net ${p.net}`}
            {p.rank === 1 ? " ♛" : ""}
          </span>
        </span>
      ))}
    </div>
  );
}
