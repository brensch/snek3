# Training on RunPod

`deploy/runpod.sh` launches the trainer image on a RunPod GPU, joined to your
tailnet exactly like a local run. Keys come from the repo `.env`
(`RUNPOD_APIKEY`, `TS_AUTHKEY`).

```sh
deploy/runpod.sh launch 4090        # create pod (community cloud, on-demand)
deploy/runpod.sh sync main          # push runs/main to it over Tailscale SSH
deploy/runpod.sh status             # id, GPU, $/hr, uptime
deploy/runpod.sh pull main          # fetch checkpoints/games back
deploy/runpod.sh stop <pod-id>      # terminate
```

The pod pulls `ghcr.io/brensch/snek3-trainer:latest` (public), gets
`TS_AUTHKEY`/`TS_HOSTNAME`/`RUN_ID` as env, and mounts the **snek3-runs
network volume** (`xvsrc62zau`, 20 GB, US-NC-1, ~$0.70/mo) at `/runs`. The
volume outlives pods: terminate freely, relaunch on whatever GPU is
available, and the runs are still there — upload a run once, reuse it
across pods. The trade-offs: secure-cloud pricing only, and pods are pinned
to the volume's datacenter (`NETVOL=` launches with a throwaway pod-local
volume on community cloud instead).

Training does **not** start at boot — seed `/runs` first if you want to
resume something, then start from the dashboard (or launch with `START=1`
to begin immediately). Dashboard:
`https://snek-train-runpod.<tailnet>.ts.net`.

## Which GPU

The workload is ~99% small-CNN forward passes (fp32 weights, TF32 convs via
cuDNN default). That shape is clock- and latency-sensitive, not
bandwidth-bound, so consumer cards beat datacenter cards per dollar by a wide
margin. Community-cloud on-demand prices, checked 2026-07-02:

| GPU | VRAM | $/hr | vs local 5080 (est.) | verdict |
|---|---|---|---|---|
| RTX 3090 | 24 GB | 0.22 | ~0.6–0.7x | cheapest, but slower than staying local |
| **RTX 4090** | 24 GB | **0.34** | ~1.3–1.5x | **best value; default** |
| RTX 5090 | 32 GB | 0.69 | ~1.8–2x | fastest sane single-GPU option |
| L40S | 48 GB | 0.79 | ~1.4x | only if VRAM-bound (low stock) |
| A100 SXM 80GB | 80 GB | 1.39 | ~1–1.5x | poor fit: 19.5 TFLOPS raw fp32, low clocks; its TF32 tensor peak needs big-model batches this net can't fill |
| H100 SXM | 80 GB | 2.69 | ~2–3x | only worth it after scaling the net up |

The local 5080 (~65k inf/s measured) costs only ~$0.10–0.15/hr of
electricity — unbeatable per-flop, but it blocks gaming. A 4090 pod is
~$8/day for ~1.4x the local throughput.

**On-demand only.** Spot is ~free discount here but preemption is expensive:
the replay buffer isn't checkpointed, so every restart costs ~20–30
generations of flat win-rate while it refills.

## Copying runs

The pod's tailscaled runs with Tailscale SSH enabled (`TS_SSH=1` default in
`tailscale-dashboard.sh`), so any tailnet device can `ssh root@<hostname>`
with no key management. `sync`/`pull` stream a tar of `runs/<run-id>` over
that. The first connection may print a check-mode URL to approve in the
browser (default tailnet SSH policy).

Requires a trainer image built after Tailscale SSH was enabled — rebuild
`deploy/trainer.Dockerfile` (or let CI push `:latest`) before the first sync.
