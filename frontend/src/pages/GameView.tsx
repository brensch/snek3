import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { getEvalGameFile, getGameFile } from "../api/proto";
import { GameTile } from "../components/GameTile";
import type { GameFile } from "../gen/viewer_pb";
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
  // board and the page ~120px above it. Clamped so tiny boards don't balloon.
  const [viewport, setViewport] = useState({ w: window.innerWidth, h: window.innerHeight });
  useEffect(() => {
    const onResize = () => setViewport({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const frame0 = game?.frames[0];
  const cell = frame0
    ? Math.max(14, Math.min(52, Math.floor(Math.min((viewport.w - 64) / frame0.width, (viewport.h - 290) / frame0.height))))
    : 32;

  const [fps, setFps] = useState(8);
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
          <div style={{ width: frame0 ? frame0.width * cell + 20 : undefined }} className="max-w-full">
            <GameTile game={game} intervalMs={Math.round(1000 / fps)} cell={cell} colors={colors} />
          </div>
        )}
      </main>
    </div>
  );
}
