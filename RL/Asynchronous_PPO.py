import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Normal, Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import numpy as np

import argparse
import time
import os
from tensorboardX import SummaryWriter
from collections import namedtuple

from utils import Normalization

import gym

env_name = 'AV_Kinematics_Env-v2'     # "Kinematics_Rear_Axle_Env-v0"
# env_name = 'AV_Kinematics_Env-v2'  # "Pendulum-v0" "AV_Kinematics_Env-v2"
env = gym.make(env_name)

parser = argparse.ArgumentParser(description='Solve the xxxx with Asynchronous_PPO')

parser.add_argument('--seed', type=int, default=100, help='random seed (default: 500)')
parser.add_argument('--num_processes', type=int, default=4, help='how many training processes to use (default: 4)')
parser.add_argument('--render', default=False, help='render the environment')
parser.add_argument('--render_test', default=False, help='render the environment in the policy_test')

parser.add_argument('--state_dim', type=int, default=env.observation_space.shape[0], help='state_dimension')
parser.add_argument('--action_dim', type=int, default=env.action_space.shape[0], help='action_dimension')
parser.add_argument('--NN_width', type=int, default=64, help='the number of neurons')
parser.add_argument('--policy_dist', type=str, default='Beta', help='the distribution way for choosing action')
parser.add_argument("--max_action", type=float, default=env.action_space.high, help="the value of max_action")
parser.add_argument("--min_action", type=float, default=env.action_space.low, help="the value of min_action")

parser.add_argument('--A_target_selection', type=int, default=2, help='Actor: total utility for the trajectory; TD error; truncated advatage function')
parser.add_argument('--C_target_selection', type=int, default=0, help='Critic: total utility for the trajectory; TD error; truncated advatage function')

parser.add_argument('--a_lr', type=float, default=1e-3, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic')
parser.add_argument('--gamma', type=float, default=0.995, help='discount factor for rewards (default: 0.99)')
parser.add_argument('--lambda_', type=float, default=0.97, help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')

parser.add_argument('--clip_param', type=int, default=0.2, help='clip parameter for PPO target function')
parser.add_argument('--max_grad_norm', type=int, default=1.0, help='the maximal threshold for gradient update')
parser.add_argument('--ppo_update_time', type=int, default=10, help='the number of updating NN based on each buffer')
parser.add_argument('--buffer_capacity', type=int, default=2048, help='the capacity of buffer')
parser.add_argument('--mini_batch_size', type=int, default=128, metavar='N', help='batch_size for each NN update, but the total buffer can be used to update')

# parser.add_argument('--local_net_update_freq', type=int, default=2048, help='the frequency for updating parameters of local net')
parser.add_argument("--evaluate_freq", type=int, default=40000, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--save_freq", type=int, default=20, help="the freq of saving the policy")
parser.add_argument("--num_episodes_for_evaluate", type=int, default=20, help="the number of episodes during evaluating policy")
parser.add_argument('--num_episodes', type=int, default=400001, help='the number of episodes')
parser.add_argument('--step_each_episode', type=int, default=600, help='the number of time steps within each episode')

parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")

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

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            action_dist = self.get_dist(state.T)
        action = action_dist.sample()
        logprob_action = action_dist.log_prob(action)
        # action and logprob_action are tensor with size(1,2)
        return action.numpy().T, logprob_action.numpy().T

    def select_action_evaluate(self, state):
        state = torch.from_numpy(state).float()
        action_deterministic = self.deterministic_act(state.T)
        return action_deterministic.detach().numpy().T


class Critic(nn.Module):
    def __init__(self, state_dim, NN_width):
        super(Critic, self).__init__()
        self.c1 = nn.Linear(state_dim, NN_width)
        # self.c2 = nn.Linear(NN_width, NN_width)
        self.state_value = nn.Linear(NN_width, 1)

    def forward(self, state):
        x = torch.tanh(self.c1(state))
        # x = F.leaky_relu(self.c2(x))
        value = self.state_value(x)
        return value


class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states."""
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    # def share_memory(self):
    #     for group in self.param_groups:
    #         for p in group['params']:
    #             state = self.state[p]
    #             state['step'].share_memory_()
    #             state['exp_avg'].share_memory_()
    #             state['exp_avg_sq'].share_memory_()


def ensure_shared_grads(local_model, global_model):
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        # print('global_net.parameters().data', global_param.data)
        # print('global_net.parameters().grad', global_param.grad)
        if global_param.grad is not None:
            # print('failure')
            return 'global_param.grad is not None'
        global_param._grad = local_param.grad
        # print('success')

def learning(index_process, args, lock1, counter1, global_Cnet_optimizer, global_Anet_optimizer, global_Cnet, global_Anet, shared_state_norm):
    '''This function <learning> aims to each process.
    In other words, each <learning> is for each agent in their own independent env.
    Therefore, for this <learning> function, it always runs constantly (but possibly restricted by the max_num_episodes and max_step_each_episode).'''

    env = gym.make(env_name)

    torch.manual_seed(args.seed + index_process)

    if args.policy_dist == "Beta":
        local_Anet = Actor_Beta(args.state_dim, args.action_dim, args.NN_width)
    if args.policy_dist == "Normal":
        local_Anet = Actor_Normal(args.state_dim, args.action_dim, args.NN_width)
    local_Cnet = Critic(args.state_dim, args.NN_width)

    # local_Anet.train()
    # local_Cnet.train()

    done = False
    # while not done:
    total_step_local = 0
    local_net_update_num = 0
    buffer = []
    for num_epi in range(args.num_episodes):
        state = env.reset()
        if state.ndim == 2: state = state.squeeze(1)
        if args.use_state_norm: state = shared_state_norm(state)
        if args.render: env.render()

        # local_Anet.load_state_dict(global_Anet.state_dict())
        # local_Cnet.load_state_dict(global_Cnet.state_dict())

        for step_t in range(args.step_each_episode):
            # print('Process: {}||| Episodes {} ||| step_t {} ||| counter {}'.format(index_process, num_epi, step_t, counter.value))

            action, logprob_action = local_Anet.select_action(state)

            if args.policy_dist == "Beta":
                a_execute = action * (args.max_action - args.min_action) + args.min_action
            else:
                a_execute = action
            next_state, utility, done, _ = env.step(a_execute)
            if not isinstance(utility, np.ndarray): utility = np.array([utility])
            if args.use_state_norm: next_state = shared_state_norm(next_state)

            mask = 1
            if done or step_t == args.step_each_episode - 1:
                mask = 0
            trans = Transition(state.T, action.T, logprob_action.T, utility, mask, next_state.T)

            if args.render: env.render()
            buffer.append(trans)

            with lock1:
                counter1.value = counter1.value + 1

            total_step_local += 1

            if total_step_local % args.buffer_capacity == 0:
                # print('Process {} ||| local_net_update_num {} ||| step {}'.format(index_process, local_net_update_num, counter1.value))
                states = torch.tensor(np.array([t.state for t in buffer]), dtype=torch.float)
                actions = torch.tensor(np.array([t.action for t in buffer]), dtype=torch.float)
                old_action_log_probs = torch.tensor(np.array([t.a_log_prob for t in buffer]), dtype=torch.float)
                utilities = torch.tensor(np.array([t.utility for t in buffer]), dtype=torch.float)
                masks = torch.tensor(np.array([t.mask for t in buffer]), dtype=torch.float)
                next_states = torch.tensor(np.array([t.next_state for t in buffer]), dtype=torch.float)

                current_values_net = local_Cnet(states)

                buffer_length = len(buffer)
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
                        # update actor network
                        action_dist = local_Anet.get_dist(states[index])
                        action_dist_entropy = action_dist.entropy().sum(1, keepdim=True)
                        action_prob = action_dist.log_prob(actions[index])
                        ratio = torch.exp(action_prob - old_action_log_probs[index])
                        surr1 = ratio * actor_target[index]
                        surr2 = torch.clamp(ratio, 1 - args.clip_param, 1 + args.clip_param) * actor_target[index]
                        action_loss = (- torch.min(surr1, surr2) - args.entropy_coef * action_dist_entropy).mean()

                        global_Anet_optimizer.zero_grad()
                        action_loss.backward()
                        nn.utils.clip_grad_norm_(local_Anet.parameters(), args.max_grad_norm)
                        # print('process',index_process,'i',i,'local_actor_net.l1.weight.norm_before', local_Anet.l1.weight.norm())
                        # print('process',index_process,'i',i,'global_actor_net.l1.weight.norm_before', global_Anet.l1.weight.norm())
                        ensure_shared_grads(local_Anet, global_Anet)
                        global_Anet_optimizer.step()
                        # print('process',index_process,'i',i,'local_actor_net.l1.weight.norm_after', local_Anet.l1.weight.norm())
                        # print('process',index_process,'i',i,'global_actor_net.l1.weight.norm_after', global_Anet.l1.weight.norm())

                        # update critic network
                        value_loss = F.smooth_l1_loss(local_Cnet(states[index]), critic_target[index])
                        global_Cnet_optimizer.zero_grad()
                        value_loss.backward()
                        nn.utils.clip_grad_norm_(local_Cnet.parameters(), args.max_grad_norm)
                        # print('critic_net.c1.weight.grad.norm_before', local_Cnet.c1.weight.grad.norm())
                        ensure_shared_grads(local_Cnet, global_Cnet)
                        global_Cnet_optimizer.step()

                        # pull global parameters
                        local_Anet.load_state_dict(global_Anet.state_dict())
                        local_Cnet.load_state_dict(global_Cnet.state_dict())

                local_net_update_num += 1

                del buffer[:]

            state = next_state

            if done:
                break

def policy_test(args, counter1, lock2, counter2, global_Anet, global_Cnet, writer_1, evaluate_rewards, shared_state_norm):

    env = gym.make(env_name)

    if args.policy_dist == "Beta":
        Test_Anet = Actor_Beta(args.state_dim, args.action_dim, args.NN_width)
    if args.policy_dist == "Normal":
        Test_Anet = Actor_Normal(args.state_dim, args.action_dim, args.NN_width)
    Test_Cnet = Critic(args.state_dim, args.NN_width)

    save_counter = 0
    start_time =time.time()
    while True:
        # time.sleep(1)
        # print('counter1.value', counter1.value)
        if counter1.value % args.evaluate_freq == 0:
            record_num_X = counter1.value
            save_counter += 1

            Test_Anet.load_state_dict(global_Anet.state_dict())
            Test_Cnet.load_state_dict(global_Cnet.state_dict())
            with lock2:
                counter2.value = counter2.value + 1

            evaluate_utility = 0
            for _ in range(args.num_episodes_for_evaluate):
                s = env.reset()
                if s.ndim == 2: s = s.squeeze(1)
                if args.use_state_norm: s = shared_state_norm(s, update=False)
                if args.render_test: env.render()

                done = False
                episode_utility = 0
                for t_evaluate in range(args.step_each_episode):
                    a = Test_Anet.select_action_evaluate(s)

                    if args.policy_dist == "Beta":
                        a_execute = a * (args.max_action - args.min_action) + args.min_action
                        # a_execute = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                    else:
                        a_execute = a

                    s_, U, done, _ = env.step(a_execute)
                    if args.use_state_norm: s_ = shared_state_norm(s_, update=False)
                    # if args.use_utility_norm: U = utility_norm(U, update=False)
                    if args.render_test: env.render()

                    episode_utility += U
                    if done:
                        break
                    s = s_

                evaluate_utility += episode_utility

            evaluate_reward = evaluate_utility / args.num_episodes_for_evaluate

            # print('****global_Anet.l1.weight.norm_test****', global_Anet.l1.weight.norm(), '****Test_Anet.l1.weight.norm_test****', Test_Anet.l1.weight.norm())
            phased_time = time.time()
            consumed_time = phased_time - start_time

            evaluate_rewards.append(evaluate_reward)
            print("evaluate_num:{} \t total_steps:{} \t consumed_time:{} \t evaluate_reward:{} \t".format(counter2.value, record_num_X, consumed_time, evaluate_rewards[-1]))
            writer_1.add_scalar('episode_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=record_num_X)

            if save_counter % args.save_freq == 0:
                torch.save(Test_Anet.state_dict(), './param_Asynchronous_PPO/save_net_param/actor_net_Asynchronous_PPO' + str(record_num_X) + '.pkl')
                torch.save(Test_Cnet.state_dict(), './param_Asynchronous_PPO/save_net_param/critic_net_Asynchronous_PPO' + str(record_num_X) + '.pkl')
                if args.use_state_norm:
                    shared_state_norm.save('./param_Asynchronous_PPO/save_norm_param/shared_state_normalization_params' + str(record_num_X) + '.pkl')


def main():
    if not os.path.exists('param_Asynchronous_PPO'):
        os.makedirs('param_Asynchronous_PPO/save_net_param')
        os.makedirs('param_Asynchronous_PPO/save_norm_param')

    os.environ['OMP_NUM_THREADS'] = '1'  # 每个进程只允许用一个线程
    os.environ['CUDA_VISIBLE_DEVICES'] = ""  # CUDA 为空，表示只用CPU

    writer_1 = SummaryWriter('./exp_Asynchronous_PPO_AV_Tracking')

    '''global information'''
    if args.policy_dist == "Beta":
        global_Anet = Actor_Beta(args.state_dim, args.action_dim, args.NN_width)
    if args.policy_dist == "Normal":
        global_Anet = Actor_Normal(args.state_dim, args.action_dim, args.NN_width)
    global_Cnet = Critic(args.state_dim, args.NN_width)
    global_Anet.share_memory()
    global_Cnet.share_memory()

    global_Anet_optimizer = SharedAdam(global_Anet.parameters(), lr=args.a_lr)
    global_Cnet_optimizer = SharedAdam(global_Cnet.parameters(), lr=args.c_lr)

    # writer_1.add_graph(global_Anet, input_to_model=torch.rand(1, args.state_dim))

    if args.use_state_norm: shared_state_norm = Normalization(shape=args.state_dim)
    else: shared_state_norm = None
    evaluate_rewards = []
    '''create several processes'''
    counter1 = mp.Value('i', 0)
    lock1 = mp.Lock()

    counter2 = mp.Value('i', 0)
    lock2 = mp.Lock()

    processes = []  # store each process
    process = mp.Process(target=policy_test, args=(args, counter1, lock2, counter2, global_Anet, global_Cnet, writer_1, evaluate_rewards, shared_state_norm))
    process.start()
    processes.append(process)
    ## execute on each process
    for index_process in range(args.num_processes):
        process = mp.Process(target=learning, args=(index_process, args, lock1, counter1, global_Cnet_optimizer, global_Anet_optimizer, global_Cnet, global_Anet, shared_state_norm))
        process.start()
        processes.append(process)


    for process in processes:
        process.join()




if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("end ||| time-consuming {}".format((end_time - start_time)))