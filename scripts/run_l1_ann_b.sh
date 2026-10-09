#!/usr/bin/env bash
# L1 ANN, variante B (dopo il fallimento del run 0): std iniziale 0.3, spinte in training fino a 1.5 m/s
cd "$(dirname "$0")/.." && . .venv/bin/activate
seed=${1:-0}; steps=${2:-1500000}
python -m spiking_rl.train --env Go2Balance-v0 --actor ann --seed $seed --steps $steps --num-envs 8 --async-envs \
  --init-log-std -1.2 --env-kwargs '{"push_vel": [0.3, 1.5]}' --out runs/go2_l1_annB_s$seed
python -m spiking_rl.eval.push_curve runs/go2_l1_annB_s$seed
