import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { getEvalGameFile, getGameFile } from "../api/proto";
import { GameTile } from "../components/GameTile";
import type { Frame, Game, GameFile } from "../gen/viewer_pb";
import { MOVE_ARROW } from "../lib/moves";
import { playerColor, snakeColor } from "../lib/palette";
import { playerName, playerNameLong } from "../lib/players";

// Fullscreen replay of one recorded game with a permalink — reached by
// clicking a tile in the run view. Two flavours share the page:
//   /runs/:runId/games/:gen/:idx   one of a generation's self-play samples
//   /runs/:runId/eval/:seq         one held-out / league game
// The replay engine is the same GameTile the grids use, just scaled to the
// viewport; games are stored forever, so these links stay live.
export function GameView({ kind }: { kind: "selfplay" | "eval" }) {
  const { runId = "", gen = "", idx = "", seq = "" } = useParams();
  const navigate = useNavigate();
  const genNum = Number(gen);
  const idxNum = Number(idx);
  const seqNum = seq === "" ? 0n : BigInt(seq);

  const [file, setFile] = useState<GameFile | null | undefined>(undefined);
  useEffect(() => {
    let alive = true;
    setFile(undefined);
    const load = kind === "selfplay" ? getGameFile(runId, genNum) : getEvalGameFile(runId, seqNum);
    load.then((f) => alive && setFile(f)).catch(() => alive && setFile(null));
    return () => {
      alive = false;
    };
  }, [kind, runId, genNum, seq]); // eslint-disable-line react-hooks/exhaustive-deps

  const game = kind === "selfplay" ? file?.games[idxNum] : file?.games[0];

  // Eval recordings embed their players/placements in config_json — enough to
  // color seats by player and show the finishing order without extra fetches.
  const evalMeta = useMemo(() => {
    if (kind !== "eval" || !file?.configJson) return null;
    try {
      return JSON.parse(file.configJson) as {
        players?: { gen: number; name: string }[];
        placements?: { gen: number; seat: number; rank: number }[];
        sims?: number;
      };
    } catch {
      return null;
    }
  }, [kind, file?.configJson]);
  const colors = useMemo(() => {
    if (!evalMeta?.players) return undefined;
    return evalMeta.players.map((p, s) =>
      p ? playerColor(playerNameLong(p.gen)) : snakeColor(s),
    );
  }, [evalMeta]);

  // Fit the board to the viewport: the tile carries ~150px of chrome below the
  // board and the page ~120px above it; the odds panel takes ~250px on the
  // right when there's room. Clamped so tiny boards don't balloon.
  const [viewport, setViewport] = useState({ w: window.innerWidth, h: window.innerHeight });
  useEffect(() => {
    const onResize = () => setViewport({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const sideBySide = viewport.w >= 880;
  const frame0 = game?.frames[0];
  const cell = frame0
    ? Math.max(
        14,
        Math.min(
          52,
          Math.floor(
            Math.min((viewport.w - 64 - (sideBySide ? 250 : 0)) / frame0.width, (viewport.h - 290) / frame0.height),
          ),
        ),
      )
    : 32;

  // Default to 70% playback speed so there's time to read the odds live.
  const [fps, setFps] = useState(14);
  const [frameIdx, setFrameIdx] = useState(0);
  const onFrame = useCallback((i: number) => setFrameIdx(i), []);
  const [copied, setCopied] = useState(false);
  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard unavailable */
    }
  };

  // Prev/next within the same collection: sample index inside the gen file,
  // or the neighbouring eval seq (files are contiguous and kept forever).
  const nav = (d: number) => {
    if (kind === "selfplay") {
      const n = idxNum + d;
      if (file && n >= 0 && n < file.games.length) navigate(`/runs/${runId}/games/${genNum}/${n}`);
    } else {
      const n = seqNum + BigInt(d);
      if (n >= 0n) navigate(`/runs/${runId}/eval/${n}`);
    }
  };
  const prevDisabled = kind === "selfplay" ? idxNum <= 0 : seqNum <= 0n;
  const nextDisabled = kind === "selfplay" ? !file || idxNum >= file.games.length - 1 : false;

  const title =
    kind === "selfplay" ? `self-play · gen ${genNum} · game ${idxNum + 1}${file ? `/${file.games.length}` : ""}` : `held-out game #${seq}`;

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-white/10 bg-surface/95 px-3 py-2 backdrop-blur sm:px-5">
        <Link
          to={`/runs/${encodeURIComponent(runId)}?tab=${kind === "selfplay" ? "self-play" : "arena"}`}
          className="text-sm text-accent hover:underline"
        >
          ‹ {runId}
        </Link>
        <span className="font-mono text-sm text-ink">{title}</span>
        {game?.scenario && (
          <span
            className="rounded bg-warn/15 px-1.5 py-0.5 font-mono text-xs text-warn"
            title={`seeded from the "${game.scenario}" curriculum scenario`}
          >
            {game.scenario}
          </span>
        )}
        {evalMeta?.placements && (
          <span className="font-mono text-xs">
            {[...evalMeta.placements]
              .sort((a, b) => a.rank - b.rank)
              .map((p, i) => (
                <span key={p.seat}>
                  {i > 0 && <span className="text-ink-3"> › </span>}
                  <span style={{ color: playerColor(playerNameLong(p.gen)) }}>{playerName(p.gen)}</span>
                </span>
              ))}
          </span>
        )}
        <div className="ml-auto flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-ink-3">
            Speed
            <input type="range" min={1} max={20} value={fps} onChange={(e) => setFps(Number(e.target.value))} className="accent-accent" />
          </label>
          <button onClick={() => nav(-1)} disabled={prevDisabled} className="rounded border border-white/10 px-2 py-0.5 text-sm text-ink-2 disabled:opacity-30">
            ‹
          </button>
          <button onClick={() => nav(1)} disabled={nextDisabled} className="rounded border border-white/10 px-2 py-0.5 text-sm text-ink-2 disabled:opacity-30">
            ›
          </button>
          <button
            onClick={copyLink}
            className={`rounded border border-white/10 px-2 py-0.5 text-xs ${copied ? "border-good/40 text-good" : "text-ink-2 hover:bg-white/5"}`}
            title="Copy permalink to this game"
          >
            {copied ? "✓ copied" : "⧉ permalink"}
          </button>
        </div>
      </header>

      <main className="flex justify-center px-3 py-4">
        {file === undefined && <div className="py-16 text-sm text-ink-3">Loading game…</div>}
        {file === null && (
          <div className="py-16 text-sm text-ink-3">
            No recording found for this link{kind === "eval" ? " (older runs pruned eval games; new ones are kept forever)" : ""}.
          </div>
        )}
        {file && !game && <div className="py-16 text-sm text-ink-3">This file has no game at index {idxNum}.</div>}
        {game && (
          <div className={`flex max-w-full gap-4 ${sideBySide ? "flex-row items-start" : "flex-col items-center"}`}>
            <div style={{ width: frame0 ? frame0.width * cell + 20 : undefined }} className="max-w-full">
              <GameTile game={game} intervalMs={Math.round(1000 / fps)} cell={cell} colors={colors} onFrame={onFrame} />
            </div>
            <OddsPanel game={game} frame={game.frames[Math.min(frameIdx, game.frames.length - 1)]} colors={colors} />
          </div>
        )}
      </main>
    </div>
  );
}

// Live per-direction odds for every snake at the displayed turn: the search's
// equilibrium policy as horizontal bars, the played move marked, plus the
// search value. Follows the replay frame-by-frame — the "what was it
// thinking" readout the hover popover can't give while the game plays.
function OddsPanel({ game, frame, colors }: { game: Game; frame: Frame; colors?: string[] }) {
  const isHeuristic = (i: number): boolean => ((game.heurMask >> i) & 1) === 1;
  return (
    <div className="w-full shrink-0 space-y-2 sm:w-[15rem]">
      {frame.snakes.map((s, i) => {
        const color = colors?.[i] ?? snakeColor(i);
        const best = Math.max(...s.policy.map((p) => p ?? 0), 0);
        return (
          <div key={i} className={`card p-2 ${s.alive ? "" : "opacity-40"}`}>
            <div className="mb-1 flex items-center gap-1.5 text-[11px]">
              <span
                className="h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: color, boxShadow: isHeuristic(i) ? "0 0 0 1.5px #fff" : undefined }}
              />
              <span className="font-mono text-ink-2">
                snake {i}
                {isHeuristic(i) && <span className="text-ink-3"> · voronoi</span>}
              </span>
              <span
                className={`ml-auto font-mono tabular-nums ${s.value >= 0 ? "text-good" : "text-bad"}`}
                title="search value for this snake"
              >
                {s.value >= 0 ? "+" : ""}
                {s.value.toFixed(2)}
              </span>
            </div>
            {s.alive ? (
              [0, 1, 2, 3].map((m) => {
                const p = s.policy[m] ?? 0;
                const played = s.chosenMove === m;
                return (
                  <div key={m} className="flex items-center gap-1.5 py-px text-[11px]">
                    <span className={`w-3 text-center ${played ? "font-semibold text-ink" : "text-ink-3"}`}>
                      {MOVE_ARROW[m]}
                    </span>
                    <span className="relative h-2 min-w-0 flex-1 overflow-hidden rounded bg-inset">
                      <span
                        className="absolute inset-y-0 left-0 rounded"
                        style={{
                          width: `${Math.round(Math.max(0, Math.min(1, p)) * 100)}%`,
                          background: color,
                          opacity: played ? 1 : p === best && best > 0 ? 0.75 : 0.35,
                        }}
                      />
                    </span>
                    <span
                      className={`w-9 text-right font-mono tabular-nums ${played ? "text-ink" : "text-ink-3"}`}
                    >
                      {(p * 100).toFixed(0)}%
                    </span>
                  </div>
                );
              })
            ) : (
              <div className="text-[10px] text-ink-3">dead</div>
            )}
          </div>
        );
      })}
    </div>
  );
}
