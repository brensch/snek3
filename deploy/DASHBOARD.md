# Remote training dashboard (Tailscale)

Check on training from your phone without exposing anything to the public
internet. The trainer serves the built frontend itself, and a userspace
Tailscale node proxies it onto your tailnet with HTTPS. Works in unprivileged
containers (no TUN device, no port forwarding, any NAT).

## One-time tailnet setup

1. Install the Tailscale app on your phone and log in (creates your tailnet).
2. In the [admin console](https://login.tailscale.com/admin):
   - **DNS** → enable MagicDNS and HTTPS certificates.
   - **Settings → Keys** → generate an auth key that is **reusable** and
     **ephemeral** (ephemeral nodes vanish from the tailnet when the pod dies).

## On each pod (Docker image)

CI builds `ghcr.io/brensch/snek3-trainer:latest` from `deploy/trainer.Dockerfile`
(the `/move` API for battlesnake.com is the separate `snek3-api` image). The
entrypoint brings up the tunnel and execs the trainer, so a pod only needs:

```sh
docker run --gpus all -v /your/volume:/runs \
  -v snek3-ts-state:/root/.snek-tailscale \
  -e TS_AUTHKEY=tskey-auth-... -e START=1 -e RUN_ID=main \
  ghcr.io/brensch/snek3-trainer:latest
```

The state volume keeps the tailscale identity across container restarts, so
the auth key is only consumed on first boot (still make it **reusable** —
fresh pods have fresh volumes).

`RUN_ID` fixed + `START=1` means restarts resume the same run instead of
minting a new one. Without `TS_AUTHKEY` the container still trains; the
dashboard is just unreachable (or set `BIND=0.0.0.0:8050` and map the port).

## From a source checkout

```sh
make frontend-build            # once, produces frontend/dist (needs node)
make train START=1             # dashboard on 127.0.0.1:8050
TS_AUTHKEY=tskey-auth-... make tunnel
```

`make tunnel` runs `deploy/tailscale-dashboard.sh`, which downloads static
Tailscale binaries if needed, starts `tailscaled --tun=userspace-networking`,
joins the tailnet, and runs `tailscale serve` in front of the dashboard port.
It prints the URL, e.g. `https://snek-train.<tailnet>.ts.net` — open that on
your phone (Tailscale app connected). The first HTTPS request can take a few
seconds while the cert is provisioned.

Env knobs: `TS_HOSTNAME` (node name, default `snek-train` — set per pod if you
run several), `DASH_PORT`/`PORT` (default 8050), `TS_DIR` (state + binaries,
default `~/.snek-tailscale`; keep the path short — the tailscaled Unix socket
lives here and long paths break `bind`).

Notes:
- The trainer still binds `127.0.0.1` only; nothing listens publicly. Only
  devices on your tailnet can reach the dashboard.
- The dashboard has no auth of its own and includes start/stop controls —
  keep it tailnet-only (do not `tailscale funnel` it).
- If `frontend/dist` is missing the trainer logs a warning and serves the API
  only; `--static-dir` overrides the location (e.g. a prebuilt dist copied
  onto the pod).
