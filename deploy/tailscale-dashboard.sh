#!/usr/bin/env bash
# Expose the snek-train dashboard over your tailnet from any pod/container.
#
# Requires no root network privileges: tailscaled runs in userspace-networking
# mode, so it works in unprivileged containers behind arbitrary NAT. Nothing
# listens on a public interface; the dashboard is reachable only from devices
# in your tailnet at https://$TS_HOSTNAME.<tailnet>.ts.net.
#
# Env:
#   TS_AUTHKEY   (required on first run) tailnet auth key; use a reusable,
#                ephemeral key so pods self-register and vanish when they die.
#   TS_HOSTNAME  node name on the tailnet (default: snek-train)
#   DASH_PORT    local dashboard port to expose (default: 8050)
#   TS_DIR       where to keep binaries + node state (default: ~/.snek-tailscale)
set -euo pipefail

TS_HOSTNAME="${TS_HOSTNAME:-snek-train}"
DASH_PORT="${DASH_PORT:-8050}"
TS_DIR="${TS_DIR:-$HOME/.snek-tailscale}"
STATE_DIR="$TS_DIR/state"
BIN_DIR="$TS_DIR/bin"
SOCKET="$TS_DIR/tailscaled.sock"
mkdir -p "$STATE_DIR" "$BIN_DIR"

# --- locate or download tailscale binaries -----------------------------------
if command -v tailscale >/dev/null && command -v tailscaled >/dev/null; then
    TAILSCALE=tailscale
    TAILSCALED=tailscaled
else
    TAILSCALE="$BIN_DIR/tailscale"
    TAILSCALED="$BIN_DIR/tailscaled"
    if [[ ! -x "$TAILSCALE" || ! -x "$TAILSCALED" ]]; then
        case "$(uname -m)" in
            x86_64) arch=amd64 ;;
            aarch64 | arm64) arch=arm64 ;;
            *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;;
        esac
        echo "downloading static tailscale ($arch)..."
        tarball=$(curl -fsSL "https://pkgs.tailscale.com/stable/?mode=json" |
            grep -o "\"tailscale_[0-9.]*_${arch}.tgz\"" | head -1 | tr -d '"')
        [[ -n "$tarball" ]] || { echo "could not resolve tailscale tarball" >&2; exit 1; }
        curl -fsSL "https://pkgs.tailscale.com/stable/$tarball" | tar -xz -C "$BIN_DIR" --strip-components=1
    fi
fi

# --- start tailscaled (userspace networking, no TUN device needed) ------------
if ! "$TAILSCALE" --socket="$SOCKET" status >/dev/null 2>&1; then
    echo "starting tailscaled..."
    nohup "$TAILSCALED" \
        --tun=userspace-networking \
        --statedir="$STATE_DIR" \
        --socket="$SOCKET" \
        >"$TS_DIR/tailscaled.log" 2>&1 &
    for _ in $(seq 1 50); do
        "$TAILSCALE" --socket="$SOCKET" status >/dev/null 2>&1 && break
        [[ "$("$TAILSCALE" --socket="$SOCKET" status 2>&1)" == *"Logged out"* ]] && break
        sleep 0.2
    done
fi

# --- join the tailnet ----------------------------------------------------------
up_args=(--hostname="$TS_HOSTNAME")
if [[ -n "${TS_AUTHKEY:-}" ]]; then
    up_args+=(--authkey="$TS_AUTHKEY")
fi
"$TAILSCALE" --socket="$SOCKET" up "${up_args[@]}"

# --- HTTPS-proxy the dashboard onto the tailnet --------------------------------
"$TAILSCALE" --socket="$SOCKET" serve --bg "$DASH_PORT"

fqdn=$("$TAILSCALE" --socket="$SOCKET" status --peers=false --json |
    sed -n 's/.*"DNSName": *"\([^"]*\)".*/\1/p' | head -1)
fqdn="${fqdn%.}"
echo
echo "dashboard is live on your tailnet: https://${fqdn}"
echo "(open the Tailscale app on your phone, then browse to that URL)"
