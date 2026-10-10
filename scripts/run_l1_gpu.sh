#!/usr/bin/env bash
# L1 (config D) su GPU/MJX con cono ELLITTICO (obbligatorio, vedi CLAUDE.md), poi valutazione MJX e sim2sim su CPU.
# Uso (sul PC GPU): scripts/run_l1_gpu.sh <ann|snn> <T> <seed> [steps]
cd "$(dirname "$0")/.." && . ~/venv-gpu/bin/activate && export PYTHONPATH=.
actor=$1; T=$2; seed=$3; steps=${4:-10000000}
name=gpu_ell_${actor}$([ "$actor" = snn ] && echo _T$T)_s$seed
[ -f runs/$name/push_curve_cpu.txt ] && { echo "$name gia' fatto"; exit 0; }
EK='"cone":"elliptic","push_vel":[0.3,1.2],"push_every":[1.0,2.0],"w_pose":0.05,"w_action_rate":0.002,"term_penalty":1.0'
[ -f runs/$name/result.json ] || python -W ignore -m spiking_rl.train_gpu --actor $actor --T $T --seed $seed --steps $steps \
  --num-envs 1024 --rollout 32 --minibatches 8 --epochs 8 --out runs/$name --env-kwargs "{$EK}"
python -W ignore -c "
from spiking_rl.eval.push_curve_gpu import evaluate
import json
json.dump(evaluate('runs/$name',[0.5,1.0,1.5],1024,cone='elliptic'),open('runs/$name/push_curve_gpu_ell.json','w'))"
python -W ignore -m spiking_rl.eval.push_curve runs/$name --episodes 30 --mags 0.5 1.0 1.5 | tee runs/$name/push_curve_cpu.txt
