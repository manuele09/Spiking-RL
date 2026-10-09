#!/usr/bin/env bash
# L1 ANN, variante C: spinte piu' frequenti (1-2 s) per densificare il segnale di recupero; std iniziale 0.3
cd "$(dirname "$0")/.." && . .venv/bin/activate
seed=${1:-0}; steps=${2:-2500000}
python -m spiking_rl.train --env Go2Balance-v0 --actor ann --seed $seed --steps $steps --num-envs 8 --async-envs \
  --init-log-std -1.2 --env-kwargs '{"push_vel": [0.3, 1.5], "push_every": [1.0, 2.0]}' --out runs/go2_l1_annC_s$seed
python -m spiking_rl.eval.push_curve runs/go2_l1_annC_s$seed
