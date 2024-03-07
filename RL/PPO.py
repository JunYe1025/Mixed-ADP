import argparse
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical, Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

from utils import Normalization

# Parameters
env_name = "AV_Kinematics_Env-v2"
env = gym.make(env_name).unwrapped
parser = argparse.ArgumentParser(description='Solve the xxxx with PPO')
parser.add_argument('--seed', type=int, default=500, metavar='N', help='random seed (default: PPO: 229/500/1226)')
parser.add_argument('--state_dim', type=int, default=env.observation_space.shape[0], help='state_dimension')
parser.add_argument('--action_dim', type=int, default=env.action_space.shape[0], help='action_dimension')
parser.add_argument('--NN_width', type=int, default=64, help='the number of neurons')
parser.add_argument('--policy_dist', type=str, default='Beta', help='the distribution way for choosing action')

parser.add_argument('--gamma', type=float, default=0.995, help='discount factor (default: 0.99)')
parser.add_argument('--lambda_', type=float, default=0.97, help='Truncated GAE (default: 0.97)')
parser.add_argument('--a_lr', type=float, default=1e-3, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic')

parser.add_argument('--A_target_selection', type=int, default=2, help='Actor: total utility for the trajectory; TD error; truncated advatage function')
parser.add_argument('--C_target_selection', type=int, default=0, help='Critic: total utility for the trajectory; TD error; truncated advatage function')

parser.add_argument('--clip_param', type=int, default=0.2, help='clip parameter for PPO target function')
parser.add_argument('--max_grad_norm', type=int, default=1.0, help='the maximal threshold for gradient update')
parser.add_argument('--ppo_update_time', type=int, default=10, help='the number of updating NN based on each buffer')
parser.add_argument('--buffer_capacity', type=int, default=2048, help='the capacity of buffer')
parser.add_argument('--mini_batch_size', type=int, default=128, metavar='N', help='batch_size for each NN update, but the total buffer can be used to update')
parser.add_argument('--num_episodes', type=int, default=40001, help='the number of episodes')
parser.add_argument('--step_each_episode', type=int, default=600, help='the number of time steps within each episode')

parser.add_argument('--render', default=True, help='render the environment')
parser.add_argument('--updating_interval', type=int, default=1000, metavar='N', help='displayed/printed interval during updating (default: 1000)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument("--save_net_param_freq", type=int, default=4000, help="frequency (episodes) of saving net_parameters result")

parser.add_argument("--num_episodes_for_evaluate", type=int, default=20, help="the number of episodes during evaluating policy")
parser.add_argument("--evaluate_freq", type=int, default=10000, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--max_action", type=float, default=env.action_space.high, help="the value of max_action")
parser.add_argument("--min_action", type=float, default=env.action_space.low, help="the value of min_action")
parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
parser.add_argument("--use_utility_norm", type=bool, default=False, help="Trick 3:utility normalization")
parser.add_argument("--use_utility_scaling", type=bool, default=False, help="Trick 4:utility scaling")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")

args = parser.parse_args()
torch.manual_seed(args.seed)

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'utility', 'mask', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'utility'])


class Actor_Normal(nn.Module):
    def __init__(self, state_dim, action_dim, NN_width):
        super(Actor_Normal, self).__init__()

        self.l1 = nn.Linear(state_dim, NN_width)
        # self.l2 = nn.Linear(NN_width, NN_width)
        self.mu_head = nn.Linear(NN_width, action_dim)
        self.sigma_head = nn.Linear(NN_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        # a = torch.tanh(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        sigma = F.softplus(self.sigma_head(a))
        return mu, sigma

    def get_dist(self, state):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        return dist

    def deterministic_act(self, state):
        mu, sigma = self.forward(state)
        return mu


class Actor_Beta(nn.Module):
    def __init__(self, state_dim, action_dim, NN_width):
        super(Actor_Beta, self).__init__()

        self.l1 = nn.Linear(state_dim, NN_width)
        # self.l2 = nn.Linear(NN_width, NN_width)
        self.alpha_head = nn.Linear(NN_width, action_dim)
        self.beta_head = nn.Linear(NN_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        # a = torch.tanh(self.l2(a))
        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0
        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def deterministic_act(self, state):
        alpha, beta = self.forward(state)
        mode = alpha / (alpha + beta)
        return mode


class Critic(nn.Module):
    def __init__(self, state_dim, NN_width):
        super(Critic, self).__init__()
        self.c1 = nn.Linear(state_dim, NN_width)
        # self.c2 = nn.Linear(NN_width, NN_width)
        self.state_value = nn.Linear(NN_width, 1)

    def forward(self, state):
        x = torch.tanh(self.c1(state))
        # x = torch.tanh(self.c2(x))
        value = self.state_value(x)

        return value


class PPO():
    def __init__(self):
        super(PPO, self).__init__()
        if args.policy_dist == "Beta":
            self.actor_net = Actor_Beta(args.state_dim, args.action_dim, args.NN_width)
        if args.policy_dist == "Normal":
            self.actor_net = Actor_Normal(args.state_dim, args.action_dim, args.NN_width)
        self.critic_net = Critic(args.state_dim, args.NN_width)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.writer_1 = SummaryWriter('./exp_PPO_AV_Tracking_v1')
        # self.writer_2 = SummaryWriter('./exp_Vehicle_2')
        self.writer_1.add_graph(self.actor_net, input_to_model=torch.rand(1, args.state_dim))
        # self.writer_2.add_graph(self.critic_net, input_to_model=torch.rand(1,args.state_dim))

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.a_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), args.c_lr)
        # print('param_group:', self.actor_optimizer.param_groups)

        if not os.path.exists('param_PPO'):
            os.makedirs('param_PPO/save_net_param')
            os.makedirs('param_PPO/save_norm_param')

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            action_dist = self.actor_net.get_dist(state.T)
        action = action_dist.sample()
        logprob_action = action_dist.log_prob(action)
        # action and logprob_action are tensor with size(1,2)
        return action.numpy().T, logprob_action.numpy().T

    def select_action_evaluate(self, state):
        state = torch.from_numpy(state).float()
        action_deterministic = self.actor_net.deterministic_act(state.T)
        return action_deterministic.detach().numpy().T

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state.T)
        return value

    def save_param(self, num_episodes):
        torch.save(self.actor_net.state_dict(), './param_PPO/save_net_param/actor_net_PPO' + str(num_episodes) + '.pkl')
        torch.save(self.critic_net.state_dict(), './param_PPO/save_net_param/critic_net_PPO' + str(num_episodes) + '.pkl')


    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def learning(self, i_ep, step_t, total_steps):
        states = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        actions = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float)
        old_action_log_probs = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float)
        utilities = torch.tensor(np.array([t.utility for t in self.buffer]), dtype=torch.float)
        masks = torch.tensor(np.array([t.mask for t in self.buffer]), dtype=torch.float)
        next_states = torch.tensor(np.array([t.next_state for t in self.buffer]), dtype=torch.float)

        current_values_net = self.critic_net(states)

        buffer_length = len(self.buffer)
        Total_Utility = torch.Tensor(buffer_length, 1)
        TD_error = torch.Tensor(buffer_length, 1)
        advantages = torch.Tensor(buffer_length, 1)

        later_total_utility = 0
        next_value_net = 0
        later_advantage = 0
        for i in reversed(range(buffer_length)):
            Total_Utility[i] = utilities[i] + args.gamma * later_total_utility * masks[i]
            TD_error[i] = utilities[i] + args.gamma * next_value_net * masks[i] - current_values_net.data[i]
            advantages[i] = TD_error[i] + args.gamma * args.lambda_ * later_advantage * masks[i]

            later_total_utility = Total_Utility[i, 0]
            next_value_net = current_values_net.data[i, 0]
            later_advantage = advantages[i, 0]

        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)

        target_pool = [Total_Utility.data, TD_error.data, advantages.data]
        actor_target = target_pool[args.A_target_selection]
        critic_target = target_pool[args.C_target_selection]

        for i in range(args.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(args.buffer_capacity)), args.mini_batch_size, False):
                action_dist = self.actor_net.get_dist(states[index])
                action_dist_entropy = action_dist.entropy().sum(1, keepdim=True)
                action_prob = action_dist.log_prob(actions[index])
                ratio = torch.exp(action_prob - old_action_log_probs[index])
                # if self.training_step % args.updating_interval == 0:
                #     print('I_ep {} , train {} times ||| step_t {} ||| total_steps {} ||| ratio.mean() {}'.format(i_ep, self.training_step, step_t, total_steps, ratio.mean()))
                # print('ratio', ratio.T)
                surr1 = ratio * actor_target[index]
                # print('clamp_rati0', torch.clamp(ratio, 1 - args.clip_param, 1 + args.clip_param))
                surr2 = torch.clamp(ratio, 1 - args.clip_param, 1 + args.clip_param) * actor_target[index]

                # update actor network
                # action_loss = - torch.min(surr1, surr2).mean()  # MAX->MIN desent
                action_loss = (- torch.min(surr1, surr2) - args.entropy_coef * action_dist_entropy).mean()
                self.writer_1.add_scalar('loss/action_loss', action_loss, global_step=total_steps)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), args.max_grad_norm)
                self.actor_optimizer.step()
                # if self.training_step % args.updating_interval == 0:
                #     print('I_ep {} , train {} times ||| step_t {} ||| total_steps {} ||| action_loss.mean {}'.format(i_ep, self.training_step, step_t, total_steps, action_loss.mean()))
                # print('ACTOR_ADP.hidden1.weight.grad_PEV', self.actor_net.l1.weight.grad)
                A_global_grads = [A_layer_para for A_layer_name, A_layer_para in self.actor_net.named_parameters() if A_layer_para.requires_grad]
                # print('A_global_grads', torch.norm(A_global_grads[1].grad))
                A_global_grads_norm = torch.sqrt(torch.sum(torch.tensor([torch.norm(A_layer_para_.grad) ** 2 for A_layer_para_ in A_global_grads])))
                self.writer_1.add_scalar('A_global_gradients_norm', A_global_grads_norm, global_step=total_steps)

                #update critic network
                value_loss = F.smooth_l1_loss(self.critic_net(states[index]), critic_target[index])
                self.writer_1.add_scalar('loss/value_loss', value_loss, global_step=total_steps)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                # print('critic_net.c1.weight.grad.norm_before', self.critic_net.c1.weight.grad.norm())
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), args.max_grad_norm)
                self.critic_optimizer.step()
                # print('critic_net.state_value.weight.grad.norm_after', self.critic_net.state_value.weight.grad.norm())
                C_global_grads = [C_layer_para for C_layer_name, C_layer_para in self.critic_net.named_parameters() if C_layer_para.requires_grad]
                C_global_grads_norm = torch.sqrt(torch.sum(torch.tensor([C_layer_para.grad.norm() ** 2 for C_layer_para in C_global_grads])))
                # print('C_global_grads_norm', C_global_grads_norm)
                self.writer_1.add_scalar('C_global_gradients_norm', C_global_grads_norm, global_step=total_steps)

                self.training_step += 1

        del self.buffer[:]  # clear experience


def evaluate_policy_state_norm(args, env, agent, state_norm):
    evaluate_utility = 0
    for _ in range(args.num_episodes_for_evaluate):
        s = env.reset()
        if s.ndim == 2: s = s.squeeze(1)
        if args.use_state_norm: s = state_norm(s, update=False)  # During the evaluating,update=False

        done = False
        episode_utility = 0
        for t_evaluate in range(args.step_each_episode):
        # while not done:
            a = agent.select_action_evaluate(s)  # We use the deterministic policy (mean) during the evaluating
            if args.policy_dist == "Beta":
                a_execute = a * (args.max_action - args.min_action) + args.min_action
                # a_execute = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                a_execute = a
            s_, U, done, _ = env.step(a_execute)
            if args.use_state_norm: s_ = state_norm(s_, update=False)
            # if args.use_utility_norm: U = utility_norm(U, update=False)
            episode_utility += U
            if done:
                break
            s = s_
        evaluate_utility += episode_utility

    return evaluate_utility / args.num_episodes_for_evaluate

def evaluate_policy(args, env, agent):
    evaluate_utility = 0
    for _ in range(args.num_episodes_for_evaluate):
        s = env.reset()
        if s.ndim == 2:
            s = s.squeeze(1)
        done = False
        episode_utility = 0
        for t_evaluate in range(args.step_each_episode):
            a = agent.select_action_evaluate(s)  # We use the deterministic policy (mean) during the evaluating
            if args.policy_dist == "Beta":
                a_execute = a * (args.max_action - args.min_action) + args.min_action
                # a_execute = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                a_execute = a
            s_, U, done, _ = env.step(a_execute)
            # if args.use_utility_norm: U = utility_norm(U, update=False)
            episode_utility += U
            if done:
                break
            s = s_
        evaluate_utility += episode_utility

    return evaluate_utility / args.num_episodes_for_evaluate

def main():
    agent = PPO()
    if args.use_state_norm:
        state_norm = Normalization(shape=args.state_dim)
    if args.use_adv_norm:
        adv_norm = Normalization(shape=1)
    if args.use_utility_norm:
        utility_norm = Normalization(shape=1)
    evaluate_num = 0
    evaluate_rewards = []
    total_steps = 0

    for i_epoch in range(args.num_episodes):
        state = env.reset()
        if state.ndim == 2: state = state.squeeze(1)
        if args.use_state_norm: state = state_norm(state)
        if args.render: env.render()

        done = False
        for step_t in range(args.step_each_episode):
            action, action_prob = agent.select_action(state)
            if args.policy_dist == "Beta":
                a_execute = action * (args.max_action - args.min_action) + args.min_action
                # a_execute = 2 * (action - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                a_execute = action
            next_state, utility, done, _ = env.step(a_execute)
            if not isinstance(utility, np.ndarray): utility = np.array([utility])
            if args.use_state_norm: next_state = state_norm(next_state)
            if args.use_utility_norm: utility = utility_norm(utility)

            mask = 1
            if done or step_t == args.step_each_episode - 1:
                mask = 0
            trans = Transition(state.T, action.T, action_prob.T, utility, mask, next_state.T)

            if args.render: env.render()
            agent.store_transition(trans)

            state = next_state
            # step_t += 1
            if len(agent.buffer) % args.buffer_capacity == 0:
                agent.learning(i_epoch, step_t, total_steps)

            total_steps += 1
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                if args.use_state_norm:
                    evaluate_reward = evaluate_policy_state_norm(args, env, agent, state_norm)
                else:
                    evaluate_reward = evaluate_policy(args, env, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t total_steps:{} \t evaluate_reward:{} \t".format(evaluate_num, total_steps, evaluate_rewards[-1]))
                agent.writer_1.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1],
                                          global_step=total_steps)
                break

            if done:
                break

        if i_epoch % args.save_net_param_freq == 0:
            agent.save_param(i_epoch)
            if args.use_state_norm:
                state_norm.save('./param_PPO/save_norm_param/state_normalization_params'+ str(i_epoch) + '.pkl')

    env.close()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("end ||| time-consuming {}".format((end_time - start_time)))