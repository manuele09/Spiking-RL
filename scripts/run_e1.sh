#!/usr/bin/env bash
# E1: SNN-PPO vs ANN-PPO su CartPole/Pendulum, 3 seed. Uso: scripts/run_e1.sh  (2 job in parallelo)
cd "$(dirname "$0")/.." && . .venv/bin/activate
jobs_list=()
for env in CartPole-v1 Pendulum-v1; do
  steps=100000; [ "$env" = Pendulum-v1 ] && steps=300000
  for seed in 0 1 2; do
    jobs_list+=("--env $env --actor ann --seed $seed --steps $steps --out runs/e1/${env}_ann_s$seed")
    for T in 1 4; do
      jobs_list+=("--env $env --actor snn --T $T --seed $seed --steps $steps --out runs/e1/${env}_snn_T${T}_s$seed")
    done
  done
done
printf '%s\n' "${jobs_list[@]}" | xargs -P 2 -I{} sh -c 'python -m spiking_rl.train {} > /dev/null 2>&1'
echo E1 done
