import gymnasium as gym

from .mujoco_go2 import Go2BalanceEnv

if "Go2Balance-v0" not in gym.registry:
    gym.register("Go2Balance-v0", entry_point="spiking_rl.envs.mujoco_go2:Go2BalanceEnv")
