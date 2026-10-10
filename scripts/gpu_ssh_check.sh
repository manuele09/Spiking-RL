#!/usr/bin/env bash
# Verifica la connessione SSH al computer con GPU usando le variabili d'ambiente CLAUDE_GPU_*.
# Non stampa mai la chiave privata. Uso: scripts/gpu_ssh_check.sh
set -u
for v in CLAUDE_GPU_SSH_KEY_B64 CLAUDE_GPU_HOST CLAUDE_GPU_PORT CLAUDE_GPU_USER; do
  [ -n "${!v:-}" ] || { echo "MANCA la variabile $v (impostala nelle impostazioni dell'ambiente e apri una sessione nuova)"; exit 1; }
done
command -v ssh >/dev/null || { echo "installo openssh-client..."; apt-get install -y -q openssh-client >/dev/null 2>&1 || { echo "installazione ssh fallita"; exit 1; }; }
D=$(mktemp -d); chmod 700 "$D"; KEY="$D/key"
printf '%s' "$CLAUDE_GPU_SSH_KEY_B64" | base64 -d > "$KEY" 2>/dev/null || { echo "CLAUDE_GPU_SSH_KEY_B64 non e' base64 valido"; exit 1; }
chmod 600 "$KEY"; [ "$(tail -c1 "$KEY" | wc -c)" = 1 ] && [ -n "$(tail -c1 "$KEY")" ] && echo >> "$KEY"
trap 'rm -rf "$D"' EXIT
if [ -n "${CLAUDE_GPU_HOSTKEY:-}" ]; then
  echo "[$CLAUDE_GPU_HOST]:$CLAUDE_GPU_PORT $CLAUDE_GPU_HOSTKEY" | awk '{print $1,$2,$3}' > "$D/known_hosts"
  OPTS="-o StrictHostKeyChecking=yes -o UserKnownHostsFile=$D/known_hosts"
else
  echo "ATTENZIONE: CLAUDE_GPU_HOSTKEY non impostata, l'identita' del server NON e' verificata"
  OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$D/known_hosts"
fi
echo "1) raggiungibilita' porta:"; timeout 10 nc -vz "$CLAUDE_GPU_HOST" "$CLAUDE_GPU_PORT" 2>&1 | tail -1
echo "2) ssh + nvidia-smi:"
timeout 30 ssh -i "$KEY" -p "$CLAUDE_GPU_PORT" $OPTS -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 \
  "$CLAUDE_GPU_USER@$CLAUDE_GPU_HOST" 'hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; python3 --version; df -h ~ | tail -1' 2>&1
echo "exit ssh: $?"
