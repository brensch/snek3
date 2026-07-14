#!/usr/bin/env python3
"""Critical health probe for an LE run: is the value learning to DISCRIMINATE
winners from losers (the thing that drives winning policy), or just regressing to
the mean? Usage: value-health.py <run_id>  (default snek3-le-4)"""
import json, zstandard, io, glob, sys, statistics as st

run = sys.argv[1] if len(sys.argv) > 1 else "snek3-le-4"
base = f"/home/brensch/snek3/runs/{run}"

def load(fn):
    return json.load(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(fn, "rb"))))

files = sorted(glob.glob(f"{base}/games/gen_*.json.zst"))[-4:]
if not files:
    print(f"no games for {run}"); sys.exit(0)

buckets = {"early": [[], []], "mid": [[], []], "late": [[], []]}
polmax = []
for fn in files:
    for g in load(fn)["games"]:
        w = g.get("winner"); frames = g["frames"]; N = len(frames)
        if w is None: continue
        for t, fr in enumerate(frames):
            frac = t / max(N - 1, 1)
            key = "early" if frac < 0.33 else "mid" if frac < 0.66 else "late"
            for s, sn in enumerate(fr["snakes"]):
                if not sn["alive"]: continue
                buckets[key][0 if s == w else 1].append(sn["value"])
                p = sn.get("policy", [])
                if p and sum(p) > 0.5: polmax.append(max(p))

print(f"=== {run} value discrimination (winner vs loser by phase) ===")
late_gap = 0
for k in ("early", "mid", "late"):
    wv, lv = buckets[k]
    if wv and lv:
        gap = st.mean(wv) - st.mean(lv)
        if k == "late": late_gap = gap
        print(f"  {k:5}: winner {st.mean(wv):+.3f}  loser {st.mean(lv):+.3f}  GAP {gap:+.3f}")
print(f"  mean max-LE-policy {st.mean(polmax):.3f} (0.25=uniform, 1=argmax)")

# eval win-rate + survival
ev = sorted(glob.glob(f"{base}/eval/game_*.json"))
if ev:
    ranks = []; won = 0
    for fn in ev[-40:]:
        pls = json.load(open(fn))["config"]["placements"]
        net = [p for p in pls if p["gen"] < 1_000_000]
        if net:
            ranks.append(net[0]["rank"])
            if net[0]["rank"] == 1: won += 1
    print(f"=== eval: net wins {won}/{len(ranks)}  rank-dist {dict(sorted(__import__('collections').Counter(ranks).items()))} ===")

# metrics tail
rows = [json.loads(l) for l in open(f"{base}/metrics.jsonl")]
last = rows[-1]
evr = [r for r in rows if r.get("le_ff_winrate") is not None][-4:]
print(f"=== gen {last['generation']}: vloss {last['value_loss']:.3f} ploss {last['policy_loss']:.3f} tgtH {last.get('target_entropy',0):.3f} avgturn {last.get('avg_game_turn',0):.1f} ===")
for r in evr:
    print(f"  eval gen {r['generation']}: ff {r['le_ff_winrate']*100:.0f}%  vor {r['le_vor_winrate']*100:.0f}%")
print(f">>> VERDICT: late-game winner-loser gap = {late_gap:+.3f} "
      f"({'HEALTHY, discriminating' if late_gap > 0.35 else 'IMPROVING' if late_gap > 0.15 else 'STILL CRUSHED — value not learning to win'})")
