#!/usr/bin/env bash
# L1 con la configurazione D (reward rilassato). Uso: scripts/run_l1.sh <ann|snn> <T> <seed> [steps]
cd "$(dirname "$0")/.." && . .venv/bin/activate
actor=$1; T=$2; seed=$3; steps=${4:-2500000}
name=go2_l1D_${actor}$([ "$actor" = snn ] && echo _T$T)_s$seed
[ -f runs/$name/push_curve.json ] && { echo "$name gia' fatto"; exit 0; }
python -m spiking_rl.train --env Go2Balance-v0 --actor $actor --T $T --seed $seed --steps $steps --num-envs 8 --async-envs \
  --init-log-std -1.2 --env-kwargs '{"push_vel": [0.3, 1.2], "push_every": [1.0, 2.0], "w_pose": 0.05, "w_action_rate": 0.002, "term_penalty": 1.0}' --out runs/$name
python -m spiking_rl.eval.push_curve runs/$name
