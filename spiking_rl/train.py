"""CLI minimale: python -m spiking_rl.train --env Pendulum-v1 --actor snn --T 4 --seed 0"""
import argparse
import json
import os

from spiking_rl.algos import PPOConfig, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--actor", default="ann", choices=["ann", "snn"])
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--neuron", default="lif", choices=["lif", "alif"])
    p.add_argument("--encoding", default="direct", choices=["direct", "pop"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--init-log-std", type=float, default=0.0)
    p.add_argument("--async-envs", action="store_true")
    p.add_argument("--env-kwargs", default="{}", help="JSON, es. '{\"push_vel\": [1.0, 2.0]}'")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or f"runs/{a.env}_{a.actor}{'_T%d' % a.T if a.actor == 'snn' else ''}_s{a.seed}"
    cfg = PPOConfig(env_id=a.env, actor=a.actor, T=a.T, neuron=a.neuron, encoding=a.encoding, seed=a.seed,
                    total_steps=a.steps, num_envs=a.num_envs, lr=a.lr, out_dir=out,
                    async_envs=a.async_envs, init_log_std=a.init_log_std, env_kwargs=json.loads(a.env_kwargs))
    res = train(cfg)
    os.makedirs(out, exist_ok=True)
    json.dump({**vars(a), **res}, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
