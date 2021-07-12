import sys

import pandas as pd

sys.path.append('./')
import gym
import numpy as np
import env
from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines import TRPO


def con2float64(filepath):
    str_command = f'np.savez("{filepath}",'
    zf = np.load(filepath)

    for _i in zf.files:
        if _i != "episode_returns" or "episode_starts":
            str_command += f'{_i}=zf["{_i}"].astype(np.float64),'
        else:
            str_command = f'{_i}=zf["{_i}"],'[:-1]

    str_command = str_command[:-1] + ")"

    exec(str_command)


# Test
env = gym.make('myenv-v0')
n_steps = 10
for _ in range(n_steps):
    # Random action
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)

# Generate expert trajectories (train expert)
env = gym.make('myenv-v0')
model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log="./trpo_myenv_tensorboard/")
model.learn(total_timesteps=25000)
generate_expert_traj(model, 'expert_trpo', n_episodes=100)

# Load the expert dataset
con2float64("./expert_trpo.npz")
dataset = ExpertDataset(expert_path='expert_trpo.npz', traj_limitation=100, verbose=1)

model = GAIL('MlpPolicy', env, dataset, verbose=1, tensorboard_log="./gail_myenv_tensorboard/")
model.learn(total_timesteps=1000)
model.save("gail_trpo")

model = GAIL.load("gail_trpo")

env = gym.make('myenv-v0')
obs = env.reset()
s = 0
stave = np.zeros([12, env.L])
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        stave[:, s] = obs
        s += 1
        env.reset()
    if s > (stave.shape[1] - 1):
        break

pd.DataFrame(stave).astype(np.uint8)
