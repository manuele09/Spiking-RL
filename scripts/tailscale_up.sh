#!/usr/bin/env bash
# Entra nella tailnet in modalita' userspace (container senza TUN) usando CLAUDE_TS_AUTHKEY.
# Dopo: SSH verso gli IP 100.x passa dal proxy SOCKS5 localhost:1055 (gpu_ssh_check.sh lo usa da solo).
# Non stampa mai la chiave. Da lanciare con run_in_background il demone, vedi sotto.
set -eu
[ -n "${CLAUDE_TS_AUTHKEY:-}" ] || { echo "MANCA CLAUDE_TS_AUTHKEY"; exit 1; }
W=${TS_DIR:-/tmp/claude-ts}; mkdir -p "$W/state"; cd "$W"
if [ ! -x tailscaled ]; then
  curl -sSL -m 120 -o ts.tgz https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz
  tar xzf ts.tgz --strip-components=1
fi
S="$W/ts.sock"
if ! ./tailscale --socket="$S" status >/dev/null 2>&1; then
  (./tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --state="$W/state/ts.state" --socket="$S" >"$W/tailscaled.log" 2>&1 &)
  sleep 4
fi
./tailscale --socket="$S" up --authkey="$CLAUDE_TS_AUTHKEY" --hostname=claude-cloud --accept-dns=false 2>&1 | sed -E 's/tskey-[A-Za-z0-9-]+/<redatto>/g'
./tailscale --socket="$S" status | head
