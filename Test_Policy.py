import torch
import numpy as np

import gym

import argparse

from RL.utils import Normalization


env_name = "AV_Kinematics_Env-v2"
env = gym.make(env_name).unwrapped
parser = argparse.ArgumentParser(description='Test the xxxx with xxx')
parser.add_argument('--render', default=True, help='render the environment')
parser.add_argument('--state_dim', type=int, default=env.observation_space.shape[0], help='state_dimension')
parser.add_argument('--action_dim', type=int, default=env.action_space.shape[0], help='action_dimension')
parser.add_argument('--costate_dim', type=int, default=env.observation_space.shape[0], help='costate_dimension')
parser.add_argument('--disturbance_dim', type=int, default=1, help='disturbance_dimension')
parser.add_argument('--NN_width', type=int, default=64, help='the number of neurons')

parser.add_argument("--num_episodes_for_test", type=int, default=1, help="the number of episodes during evaluating policy")
parser.add_argument('--step_each_episode', type=int, default=600, help='the number of time steps within each episode')
parser.add_argument("--max_action", type=float, default=env.action_space.high, help="the value of max_action")
parser.add_argument("--min_action", type=float, default=env.action_space.low, help="the value of min_action")
parser.add_argument("--disturbance_limit", type=float, default=0.05, help="the limitation of disturbance")

parser.add_argument('--policy_dist', type=str, default='Beta', help='the distribution way for choosing action')
parser.add_argument('--policy_style', type=str, default='Stochastic', help='Deterministic or Stochastic or CoSADP')

parser.add_argument('--plot_rect_freq', type=int, default=4, help='the interval of plotting rectangle(vehicle)')
parser.add_argument('--plot_convergence_curve', type=bool, default=True, help='if plot the loss/state/action/costate/performance_function')

parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")

args = parser.parse_args()


class Test_Policy():
    def __init__(self, env):
        self.step_each_episode = args.step_each_episode
        self.scale_test = env.scale
        self.fig_width = 40
        self.fig_height = 40
        self.vehicle_width = env.vehicle_width
        self.vehicle_length = env.vehicle_length

        self.Test_Initial_State_Set = [[0, 1, 0],
                                       [1, 5, 0],
                                       [0, 10, -0.2],
                                       [0, -3, 0.2]]

        self.loss_A = []
        self.loss_C = []
        self.loss_Co = []

    def evaluate_policy_stochastic(self):
        from RL.PPO import Actor_Beta, Actor_Normal, Critic

        if args.policy_dist == 'Beta':
            actor_test = Actor_Beta(args.state_dim, args.action_dim, args.NN_width)
        if args.policy_dist == 'Normal':
            actor_test = Actor_Normal(args.state_dim, args.action_dim, args.NN_width)
        critic_test = Critic(args.state_dim, args.NN_width)

        actor_test.load_state_dict(torch.load('./RL/param_PPO/save_net_param/actor_net_PPO16000.pkl'))
        critic_test.load_state_dict(torch.load('./RL/param_PPO/save_net_param/critic_net_PPO16000.pkl'))

        self.plot_tracking_trajectory(actor_test, critic_test)

        env.close()



    def evaluate_policy_deterministic(self):
        from RL.DDPG import Actor, Critic_Q

        actor_test = Actor(args.state_dim, args.action_dim, args.NN_width, args.max_action, args.min_action)
        critic_test = Critic_Q(args.state_dim, args.action_dim, args.NN_width)
        actor_test.load_state_dict(torch.load('./DDPG/param/save_net_param/actor_net_DDPG1500.pkl'))
        critic_test.load_state_dict(torch.load('./DDPG/param/save_net_param/critic_net_DDPG1500.pkl'))

        evaluate_utility = 0
        for _ in range(args.num_episodes_for_test):
            s = env.reset()
            if s.ndim == 2: s = s.squeeze(1)

            done = False
            episode_utility = 0
            for t_evaluate in range(args.step_each_episode):
                a = actor_test(torch.from_numpy(s).float().T).detach().numpy().T
                s_, U, done, _ = env.step(a)
                episode_utility += U

                if args.render: env.render()

                if done:
                    break
                s = s_
            evaluate_utility += episode_utility

        env.close()

        self.plot_trajectory(self.Test_Initial_State_Set, actor_test)
        self.plot_performance_function(actor_test, critic_test, self.Test_Initial_State_Set)


    def plot_tracking_trajectory(self, actor_test, critic_test):
        if args.use_state_norm:
            state_norm = Normalization(shape=args.state_dim)
            state_norm.load('./RL/param_PPO/save_norm_param/state_normalization_params16000.pkl')
        state1_store = np.zeros((1, args.step_each_episode))
        state2_store = np.zeros((1, args.step_each_episode))
        state3_store = np.zeros((1, args.step_each_episode))
        action1_store = np.zeros((1, args.step_each_episode))
        action2_store = np.zeros((1, args.step_each_episode))

        s = env.reset()
        if s.ndim == 2: s = s.squeeze(1)
        if args.use_state_norm: s = state_norm(s, update=False)
        for i in range(args.step_each_episode):
            # s = state_transfer
            if args.policy_style == 'Stochastic':
                action = actor_test.deterministic_act(torch.from_numpy(s).float().T).detach().numpy().T
                if args.policy_dist == "Beta":
                    action = action * (args.max_action - args.min_action) + args.min_action
                if args.policy_dist == "Normal":
                    action = action
            if args.policy_style == 'Deterministic':
                action = actor_test(torch.from_numpy(s).float().T).detach().numpy().T

            s_next, _, _, _ = env.step(action)

            state1_store[0, i] = s_next[0]
            state2_store[0, i] = s_next[1]
            state3_store[0, i] = s_next[2]
            action1_store[0, i] = action[0]
            action2_store[0, i] = action[1]

            if args.use_state_norm: s_next = state_norm(s_next, update=False)
            s = s_next

        import matplotlib.pyplot as plt
        print('state1_store', state1_store.tolist())
        print('state2_store', state2_store.tolist())
        x = np.linspace(0, env.state_high[0].item(), env.num_points)
        y = env.amplitude * np.sin(2 * np.pi * x / env.T)
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label='expected trajectory', color='b')
        plt.scatter(state1_store, state2_store, label='vehicle', color='r')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('sin(x)')
        plt.title('Sin Function')

        plt.show()


if __name__ == '__main__':
    Test = Test_Policy(env)
    if args.policy_style == 'Deterministic':
        Test.evaluate_policy_deterministic()
    if args.policy_style == 'Stochastic':
        Test.evaluate_policy_stochastic()