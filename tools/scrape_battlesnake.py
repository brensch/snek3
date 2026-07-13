#!/usr/bin/env python3
"""Scrape completed Battlesnake ladder games from the top of the leaderboard.

Battlesnake only retains games for ~1 week, so this mines them into a durable,
compact local archive we can later convert into training samples.

Pipeline:
  leaderboard (ranked snakes) -> each snake's /stats page (recent game IDs)
  -> engine.battlesnake.com/games/<id>/frames (full trajectory)
  -> quality filter (drop games where snakes just crash early)
  -> strip to essentials + zstd -> data/scraped-games/<id>.json.zst

Idempotent: a manifest of seen IDs means re-runs only fetch new games. Run it
on a loop (default) to keep catching fresh games before they expire.

Storage schema (per game, zstd-compressed JSON):
  {id,width,height,ruleset,source,turns,
   snakes:[{id,name,author}],                       # roster, index-aligned
   frames:[{food:[[x,y]..], snakes:[{body:[[x,y]..],health,alive}..]}..]}
Everything a trainer needs to re-encode boards, infer each move (head delta
between frames), and derive outcomes (final alive mask). Cosmetic fields
(color/head/tail/latency/…) are dropped; bodies overlap heavily frame-to-frame
so zstd shrinks them hard.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import zstandard as zstd

PLAY = "https://play.battlesnake.com"
ENGINE = "https://engine.battlesnake.com"
UA = "snek3-game-miner/1.0 (personal training data collection)"


def get(url, tries=4, timeout=30):
    """GET with retries and a polite UA; returns bytes or None."""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code == 404:
                return None  # gone / expired — don't retry
            time.sleep(1.5 * (i + 1))
    return None


def leaderboard_snakes(board="standard", limit=None):
    """Ranked snake slugs from the leaderboard, best first."""
    html = get(f"{PLAY}/leaderboard/{board}")
    if not html:
        return []
    slugs = re.findall(
        rf"/leaderboard/{board}/([A-Za-z0-9_-]+)/stats", html.decode("utf-8", "ignore")
    )
    seen, out = set(), []
    for s in slugs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:limit] if limit else out


def snake_game_ids(slug, board="standard"):
    """Recent game IDs from a snake's leaderboard stats page."""
    html = get(f"{PLAY}/leaderboard/{board}/{slug}/stats")
    if not html:
        return []
    ids = re.findall(
        r"/game/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        html.decode("utf-8", "ignore"),
    )
    return list(dict.fromkeys(ids))


def fetch_frames(gid, page=100):
    """All frames for a game. The engine caps a response at 100 frames
    regardless of `limit`, so advance by the actual count returned and stop only
    on an empty batch (never on batch < requested — that truncates long games)."""
    frames, offset = [], 0
    while True:
        raw = get(f"{ENGINE}/games/{gid}/frames?offset={offset}&limit={page}")
        if raw is None:
            return None
        batch = json.loads(raw).get("Frames", [])
        if not batch:
            break
        frames.extend(batch)
        offset += len(batch)
    return frames


def sensible(meta, frames, min_turns, min_survivors_at):
    """Drop broken/degenerate games: too short, wrong ruleset/size, or almost
    everyone crashing in the opening (the 'straight into the wall' case)."""
    g = meta["Game"]
    if g.get("Status") != "complete":
        return False
    if g.get("Width") != 11 or g.get("Height") != 11:
        return False
    if g.get("Ruleset", {}).get("name") != "standard":
        return False
    if len(frames) < min_turns:
        return False
    # How many snakes are still alive at `min_survivors_at`? A real game has a
    # contest; a lobby of wall-crashers is empty by then.
    idx = min(min_survivors_at, len(frames) - 1)
    alive = sum(1 for s in frames[idx].get("Snakes", []) if s.get("Death") is None)
    return alive >= 2


def strip(meta, frames):
    """Compact, training-ready form: roster + per-turn food and snake bodies."""
    g = meta["Game"]
    roster = [
        {"id": s.get("ID"), "name": s.get("Name"), "author": s.get("Author")}
        for s in frames[0].get("Snakes", [])
    ]
    out_frames = []
    for f in frames:
        out_frames.append(
            {
                "food": [[p["X"], p["Y"]] for p in f.get("Food", [])],
                "snakes": [
                    {
                        "body": [[p["X"], p["Y"]] for p in s.get("Body", [])],
                        "health": s.get("Health", 0),
                        "alive": s.get("Death") is None,
                    }
                    for s in f.get("Snakes", [])
                ],
            }
        )
    return {
        "id": g.get("ID"),
        "width": g.get("Width"),
        "height": g.get("Height"),
        "ruleset": g.get("Ruleset", {}).get("name"),
        "source": g.get("Source"),
        "turns": len(frames),
        "snakes": roster,
        "frames": out_frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "snek3/data/scraped-games"))
    ap.add_argument("--board", default="standard")
    ap.add_argument("--top-snakes", type=int, default=0, help="0 = whole leaderboard")
    ap.add_argument("--min-turns", type=int, default=20)
    ap.add_argument("--min-survivors-at", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=0.25, help="seconds between game fetches")
    ap.add_argument("--loop-mins", type=float, default=60, help="re-scan every N min; 0 = once")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "seen.txt"
    seen = set(manifest.read_text().split()) if manifest.exists() else set()
    cctx = zstd.ZstdCompressor(level=19)

    while True:
        snakes = leaderboard_snakes(args.board, args.top_snakes or None)
        print(f"[{time.strftime('%H:%M:%S')}] leaderboard: {len(snakes)} snakes", flush=True)
        game_ids = []
        for slug in snakes:
            for gid in snake_game_ids(slug, args.board):
                if gid not in seen:
                    game_ids.append(gid)
            time.sleep(args.sleep)
        game_ids = list(dict.fromkeys(game_ids))
        print(f"  {len(game_ids)} new candidate games", flush=True)

        kept = dropped = failed = 0
        for gid in game_ids:
            if gid in seen:
                continue
            meta_raw = get(f"{ENGINE}/games/{gid}")
            frames = fetch_frames(gid) if meta_raw else None
            if not meta_raw or not frames:
                failed += 1
                # mark expired/broken so we don't retry forever
                seen.add(gid)
                continue
            meta = json.loads(meta_raw)
            if not sensible(meta, frames, args.min_turns, args.min_survivors_at):
                dropped += 1
                seen.add(gid)
                continue
            blob = json.dumps(strip(meta, frames), separators=(",", ":")).encode()
            (out / f"{gid}.json.zst").write_bytes(cctx.compress(blob))
            seen.add(gid)
            kept += 1
            if kept % 25 == 0:
                manifest.write_text("\n".join(sorted(seen)))
                print(f"  ...kept {kept} dropped {dropped} failed {failed}", flush=True)
            time.sleep(args.sleep)

        manifest.write_text("\n".join(sorted(seen)))
        total = len(list(out.glob("*.json.zst")))
        size_mb = sum(p.stat().st_size for p in out.glob("*.json.zst")) / 1e6
        print(
            f"  round done: +{kept} kept, {dropped} dropped, {failed} gone | "
            f"archive {total} games, {size_mb:.1f} MB",
            flush=True,
        )
        if args.loop_mins <= 0:
            break
        time.sleep(args.loop_mins * 60)


if __name__ == "__main__":
    main()
