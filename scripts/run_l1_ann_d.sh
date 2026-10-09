#!/usr/bin/env bash
# L1 ANN, variante D: reward rilassato per permettere il passo di recupero (pose 0.5->0.05, action-rate 0.01->0.002),
# penalita' di caduta -1, spinte 0.3-1.2 m/s ogni 1-2 s, std iniziale 0.3
cd "$(dirname "$0")/.." && . .venv/bin/activate
seed=${1:-0}; steps=${2:-2500000}
python -m spiking_rl.train --env Go2Balance-v0 --actor ann --seed $seed --steps $steps --num-envs 8 --async-envs \
  --init-log-std -1.2 --env-kwargs '{"push_vel": [0.3, 1.2], "push_every": [1.0, 2.0], "w_pose": 0.05, "w_action_rate": 0.002, "term_penalty": 1.0}' --out runs/go2_l1_annD_s$seed
python -m spiking_rl.eval.push_curve runs/go2_l1_annD_s$seed
