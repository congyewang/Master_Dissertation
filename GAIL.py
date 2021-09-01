import sys

sys.path.append('./')
import gym
import env
from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines import TRPO
import tensorflow as tf


# Initialize Environment
env = gym.make('myenv-v0')

# Test
n_steps = 2
for _ in range(n_steps):
    # Random action
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)

# Generate expert trajectories (train expert)
g_TRPO = tf.Graph()
sess_TRPO = tf.Session(graph=g_TRPO)
with sess_TRPO.as_default():
    with g_TRPO.as_default():
        model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log="./Model/Temp/GAIL/Logs/TRPO/Expert_TRPO")
        # model.learn(total_timesteps=25000)
        generate_expert_traj(model, './Model/Temp/GAIL/Expert_TRPO', n_timesteps=100, n_episodes=10)

# Load the expert dataset

dataset = ExpertDataset(expert_path='./Model/Temp/GAIL/Expert_TRPO/Expert_TRPO.npz', traj_limitation=10, verbose=1)

g_GAIL = tf.Graph()
sess_GAIL = tf.Session(graph=g_GAIL)
with sess_GAIL.as_default():
    with g_GAIL.as_default():
        model = GAIL('MlpPolicy', env, dataset, verbose=1, tensorboard_log="./Model/Temp/GAIL/Logs/GAIL")
        # Note: in practice, you need to train for 1M steps to have a working policy
        model.learn(total_timesteps=1000)
        model.save("./Model/Temp/GAIL/Expert_TRPO/")

del model # remove to demonstrate saving and loading

g_GAIL_prediction=tf.Graph()
sess_GAIL_prediction = tf.Session(graph=g_GAIL)
with sess_GAIL.as_default():
    with g_GAIL.as_default():
        model = GAIL.load("gail_pendulum")

        env = gym.make('Pendulum-v0')
        obs = env.reset()
        s = 0
        note_list = []
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            note_list.append(obs[-1])
            env.render()
            s += 1
            if s > 500:
                break
