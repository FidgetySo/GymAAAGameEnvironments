import gym
import gym_game

import os

import numpy as np

from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy



env_id = "BattleFront2-v0"

env = gym.make(env_id)
env = DummyVecEnv([lambda: env])
model = PPO2(CnnLstmPolicy, env, verbose=1, nminibatches=1, tensorboard_log="D:/Hmm/tensorboard", learning_rate=0.0005, gamma=0.9997, n_steps=128, noptepochs=4, cliprange_vf=0.3, n_cpu_tf_sess=3)
#model.load("best_model")
model.learn(total_timesteps=1500000)
model.save("agent")
# Enjoy trained agent
obs = env.reset()
state = Non
while True:
    action, state = model.predict(obs, state=state)
    obs, rewards, dones, info = env.step(action)
