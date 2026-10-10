"""CLI GPU: python -m spiking_rl.train_gpu --actor ann --seed 0 --steps 50000000 --out runs/x"""
import argparse
import json
import os

from spiking_rl.algos.ppo import PPOConfig
from spiking_rl.algos.ppo_gpu import train_gpu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", default="ann", choices=["ann", "snn"])
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--neuron", default="lif", choices=["lif", "alif"])
    p.add_argument("--encoding", default="direct", choices=["direct", "pop"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50_000_000)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--init-log-std", type=float, default=-1.2)
    p.add_argument("--env-kwargs", default="{}", help="JSON per Go2BalanceMJX")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = PPOConfig(env_id="Go2Balance-v0", actor=a.actor, T=a.T, neuron=a.neuron, encoding=a.encoding, seed=a.seed,
                    total_steps=a.steps, num_envs=a.num_envs, rollout=a.rollout, epochs=a.epochs,
                    minibatches=a.minibatches, lr=a.lr, out_dir=a.out, init_log_std=a.init_log_std,
                    env_kwargs=json.loads(a.env_kwargs))
    res = train_gpu(cfg)
    json.dump({**vars(a), **res}, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
