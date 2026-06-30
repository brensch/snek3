import { BoardCanvas } from "./BoardCanvas";
import type { GamesSnapshot } from "../types";

type Props = { snapshot: GamesSnapshot | null };

export function LiveGames({ snapshot }: Props) {
  const games = snapshot?.games ?? [];
  return (
    <section className="panel">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="section-title">Live games</h2>
        <span className="text-xs text-slate-500">{games.length} active boards</span>
      </div>
      {games.length ? (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5 2xl:grid-cols-6">
          {games.map((game, i) => (
            <div key={`${i}-${game.turn}`} className="rounded border border-slate-800 bg-slate-950 p-2">
              <BoardCanvas game={game} />
              <div className="mt-2 font-mono text-xs text-slate-500">turn {game.turn}</div>
            </div>
          ))}
        </div>
      ) : <p className="text-sm text-slate-500">No game snapshots yet.</p>}
    </section>
  );
}
