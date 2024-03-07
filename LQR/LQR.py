from torch.autograd import Variable
import torch

import numpy as np

import argparse

import math
import random

import gym
from gym import spaces

env_name = "AV_Kinematics_Env-v2"
env = gym.make(env_name).unwrapped

parser = argparse.ArgumentParser(description='xxx')

parser.add_argument('--seed', type=int, default=2024, help='random seed (default: 10)')
parser.add_argument('--step_each_episode', type=int, default=600, help='the number of time steps within each episode')
parser.add_argument('--state_dim', type=int, default=env.observation_space.shape[0], help='state_dimension')
parser.add_argument('--action_dim', type=int, default=env.action_space.shape[0], help='action_dimension')

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

class LQR():
    def __init__(self):
        self.Q = env.Q
        self.R = env.R
        self.initial_state = np.array([[0.0],
                                       [5.0],
                                       [0.0]])

    def plot_tracking_trajectory(self):
        state1_store = np.zeros((1, args.step_each_episode))
        state2_store = np.zeros((1, args.step_each_episode))
        state3_store = np.zeros((1, args.step_each_episode))
        error_action1_store = np.zeros((1, args.step_each_episode))
        error_action2_store = np.zeros((1, args.step_each_episode))
        real_action1_store = np.zeros((1, args.step_each_episode))
        real_action2_store = np.zeros((1, args.step_each_episode))

        state_transfer = self.initial_state
        for i in range(args.step_each_episode):
            s = state_transfer
            s_next, error_a, real_a = env.linear_model(s,i)
            state_transfer = s_next

            state1_store[0, i] = s[0]
            state2_store[0, i] = s[1]
            state3_store[0, i] = s[2]
            real_action1_store[0, i] = real_a[0]
            real_action2_store[0, i] = real_a[1]

        import matplotlib.pyplot as plt
        x = np.linspace(0, env.state_high[0].item(), env.num_points)
        y = env.amplitude * np.sin(2 * np.pi * x / env.T)
        print('x', x.tolist())
        print('y', y.tolist())
        print('state1_store', state1_store.tolist())
        print('state2_store', state2_store.tolist())
        print('state3_store', state3_store.tolist())
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label='expected trajectory', color='b')
        # plt.plot(state1_store.squeeze(0), state2_store.squeeze(0), label='vehicle', color='r')
        plt.scatter(state1_store, state2_store, label='vehicle', color='r')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('sin(x)')
        plt.title('Sin Function')

        plt.show()


if __name__ == '__main__':
    LQR = LQR()
    LQR.plot_tracking_trajectory()



