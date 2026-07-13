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

Output: each game is stored as the trainer's own **GameFile JSON** (zstd),
directly materialisable into training samples with no adapter — obs re-encoded
from the bodies, policy = HARD one-hot on the expert's inferred move (head delta;
AlphaGo's supervised scheme, no label smoothing — data diversity is the
regulariser), value derived by the trainer from `winner` (last snake alive).
Deployed long-term on the always-on box (192.168.1.8) via nohup + an @reboot
cron; compresses through the python `zstandard` module or the `zstd` CLI,
whichever is present.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# zstd compression, pluggable: prefer the python module, fall back to the `zstd`
# CLI (so it runs on hosts without the wheel, e.g. Python 3.14 + no pip).
try:
    import zstandard as _zstd

    _CCTX = _zstd.ZstdCompressor(level=19)

    def zcompress(b: bytes) -> bytes:
        return _CCTX.compress(b)

except ImportError:
    import subprocess

    def zcompress(b: bytes) -> bytes:
        return subprocess.run(
            ["zstd", "-19", "-q", "-c"], input=b, stdout=subprocess.PIPE, check=True
        ).stdout

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


# Move index <-> head delta, matching snek-core Move (Up=0:+y Down=1:-y Left=2:-x Right=3:+x).
_DELTA_TO_MOVE = {(0, 1): 0, (0, -1): 1, (-1, 0): 2, (1, 0): 3}
_UNIFORM = [0.25, 0.25, 0.25, 0.25]


def _infer_move(head, nxt):
    """Move index from head position `head` at turn T to `nxt` at T+1, or None
    if it isn't a single orthogonal step (data anomaly / off-board fatal move)."""
    return _DELTA_TO_MOVE.get((nxt[0] - head[0], nxt[1] - head[1]))


def to_gamefile(meta, frames):
    """Convert an engine game to the trainer's own GameFile JSON — directly
    materialisable into training samples, no adapter needed.

    Policy targets are HARD one-hot on the expert's actual move (AlphaGo's
    supervised scheme — the diversity of hundreds of snakes across thousands of
    games is the regulariser, not label smoothing). The value target is derived
    by the trainer from `winner` (last snake alive) via its usual terminal_value.
    Dead seats and the terminal frame are skipped at materialise time, so their
    (unknowable) moves don't matter; genuinely un-inferrable live moves get a
    uniform target (max-entropy, harmless).
    """
    g = meta["Game"]
    w, h = g.get("Width"), g.get("Height")
    bodies = [
        [[[p["X"], p["Y"]] for p in s.get("Body", [])] for s in f.get("Snakes", [])]
        for f in frames
    ]
    out_frames = []
    for i, f in enumerate(frames):
        snakes = []
        for si, s in enumerate(f.get("Snakes", [])):
            alive = s.get("Death") is None
            body = bodies[i][si]
            mv, pol = 0, _UNIFORM
            # infer this seat's move from where its head is next turn
            if alive and i + 1 < len(frames) and body:
                nxt = bodies[i + 1][si]
                inferred = _infer_move(body[0], nxt[0]) if nxt else None
                if inferred is not None:
                    mv = inferred
                    pol = [1.0 if k == inferred else 0.0 for k in range(4)]
            snakes.append(
                {
                    "alive": alive,
                    "body": body,
                    "health": s.get("Health", 0),
                    "chosen_move": mv,
                    "policy": pol,
                    "play_policy": pol,
                    "value": 0.0,
                }
            )
        out_frames.append(
            {
                "turn": f.get("Turn", i),
                "width": w,
                "height": h,
                "food": [[p["X"], p["Y"]] for p in f.get("Food", [])],
                "hazards": [[p["X"], p["Y"]] for p in f.get("Hazards", [])],
                "snakes": snakes,
            }
        )
    # winner = the single snake still alive in the terminal frame (else a draw)
    last_alive = [si for si, s in enumerate(frames[-1].get("Snakes", [])) if s.get("Death") is None]
    winner = last_alive[0] if len(last_alive) == 1 else None
    game = {
        "frames": out_frames,
        "winner": winner,
        "num_turns": len(frames),
        "heur_mask": 0,
    }
    # Wrap in the GameFile envelope the trainer's viewer/ingest already parse.
    return {
        "gen": 0,
        "config": {"source": "battlesnake_ladder", "game_id": g.get("ID")},
        "games": [game],
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
            blob = json.dumps(to_gamefile(meta, frames), separators=(",", ":")).encode()
            (out / f"{gid}.json.zst").write_bytes(zcompress(blob))
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
