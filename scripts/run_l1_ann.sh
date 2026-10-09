#!/usr/bin/env bash
# L1 (stand & balance con spinte): ANN baseline su Go2 in MuJoCo CPU. Uso: scripts/run_l1_ann.sh [seed] [steps]
cd "$(dirname "$0")/.." && . .venv/bin/activate
seed=${1:-0}; steps=${2:-2000000}
python -m spiking_rl.train --env Go2Balance-v0 --actor ann --seed $seed --steps $steps --num-envs 8 --async-envs \
  --env-kwargs '{"push_vel": [0.3, 2.0]}' --out runs/go2_l1_ann_s$seed
python -m spiking_rl.eval.push_curve runs/go2_l1_ann_s$seed
