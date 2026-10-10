#!/usr/bin/env bash
# L1 (config D) su GPU/MJX. Uso (sul PC GPU): scripts/run_l1_gpu.sh <ann|snn> <T> <seed> [steps]
cd "$(dirname "$0")/.." && . ~/venv-gpu/bin/activate && export PYTHONPATH=.
actor=$1; T=$2; seed=$3; steps=${4:-50000000}
name=gpu_l1D_${actor}$([ "$actor" = snn ] && echo _T$T)_s$seed
[ -f runs/$name/push_curve_gpu.json ] && { echo "$name gia' fatto"; exit 0; }
python -W ignore -m spiking_rl.train_gpu --actor $actor --T $T --seed $seed --steps $steps --out runs/$name \
  --env-kwargs '{"push_vel": [0.3, 1.2], "push_every": [1.0, 2.0], "w_pose": 0.05, "w_action_rate": 0.002, "term_penalty": 1.0}' \
  && python -W ignore -m spiking_rl.eval.push_curve_gpu runs/$name --envs 1024
