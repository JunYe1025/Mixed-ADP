import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Normal, Beta

import numpy as np

import argparse
import time
import os
from tensorboardX import SummaryWriter
from collections import namedtuple

import gym

env_name = 'Pendulum-v0'     # "Kinematics_Rear_Axle_Env-v0"
# env_name = 'Kinematics_Rear_Axle_Env-v0'  # "Pendulum-v0" "AV_Kinematics_Env-v2"
env = gym.make(env_name).unwrapped

parser = argparse.ArgumentParser(description='Solve the xxxx with A3C')

parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--num_processes', type=int, default=4, help='how many training processes to use (default: 4)')
parser.add_argument('--render', default=True, help='render the environment')

parser.add_argument('--state_dim', type=int, default=env.observation_space.shape[0], help='state_dimension')
parser.add_argument('--action_dim', type=int, default=env.action_space.shape[0], help='action_dimension')
parser.add_argument('--NN_width', type=int, default=64, help='the number of neurons')
parser.add_argument('--policy_dist', type=str, default='Beta', help='the distribution way for choosing action')
parser.add_argument("--max_action", type=float, default=env.action_space.high, help="the value of max_action")
parser.add_argument("--min_action", type=float, default=env.action_space.low, help="the value of min_action")

parser.add_argument('--A_target_selection', type=int, default=2, help='Actor: total utility for the trajectory; TD error; truncated advatage function')
parser.add_argument('--C_target_selection', type=int, default=0, help='Critic: total utility for the trajectory; TD error; truncated advatage function')

parser.add_argument('--a_lr', type=float, default=2e-3, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-3, help='Learning rate of critic')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
parser.add_argument('--lambda_', type=float, default=0.97, help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
parser.add_argument('--max_grad_norm', type=float, default=50, help='value loss coefficient (default: 50)')

parser.add_argument('--local_net_update_freq', type=int, default=80, help='the frequency for updating parameters of local net')
parser.add_argument("--evaluate_freq", type=int, default=10000, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--num_episodes_for_evaluate", type=int, default=20, help="the number of episodes during evaluating policy")
parser.add_argument('--num_episodes', type=int, default=20000, help='maximum length of an episode (default: 1000000)')
parser.add_argument('--step_each_episode', type=int, default=200, help='number of forward steps in A3C (default: 20)')

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
        self.l2 = nn.Linear(NN_width, NN_width)
        self.alpha_head = nn.Linear(NN_width, action_dim)
        self.beta_head = nn.Linear(NN_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
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
        # print('global_net.parameters(),grad', global_param.grad)
        if global_param.grad is not None:
            # print('failure')
            return 'global_param.grad is not None'
        global_param._grad = local_param.grad
        # print('success')

def learning(index_process, args, lock, counter, global_Cnet_optimizer, global_Anet_optimizer, global_Cnet, global_Anet):
    '''This function <learning> aims to each process.
    In other words, each <learning> is for each agent in their own independent env.
    Therefore, for this <learning> function, it always runs constantly (but possibly restricted by the max_num_episodes and max_step_each_episode).'''
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

            mask = 1
            if done or step_t == args.step_each_episode - 1:
                mask = 0
            trans = Transition(state.T, action.T, logprob_action.T, utility, mask, next_state.T)

            if args.render: env.render()
            buffer.append(trans)

            with lock:
                counter.value = counter.value + 1

            total_step_local += 1

            if total_step_local % args.local_net_update_freq == 0:
                # print('Process {} ||| local_net_update_num {}'.format(index_process, local_net_update_num))
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

                # advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)

                target_pool = [Total_Utility.data, TD_error.data, advantages.data]
                actor_target = target_pool[args.A_target_selection]
                critic_target = target_pool[args.C_target_selection]


                # update actor network
                action_dist = local_Anet.get_dist(states)
                logprob_action = action_dist.log_prob(actions)
                action_dist_entropy = action_dist.entropy().sum(1, keepdim=True)
                action_loss = (- logprob_action * advantages - args.entropy_coef * action_dist_entropy).mean()

                global_Anet_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(local_Anet.parameters(), args.max_grad_norm)
                # print('global_Anet.a1.weight.grad.norm()',global_Anet.l1.weight.grad)
                ensure_shared_grads(local_Anet, global_Anet)
                global_Anet_optimizer.step()


                #update critic network
                value_loss = F.smooth_l1_loss(local_Cnet(states), critic_target)
                # writer_1.add_scalar('loss/value_loss', value_loss, global_step=total_steps)

                global_Cnet_optimizer.zero_grad()
                '''local --> global [gradient information]'''
                value_loss.backward()
                # print('critic_net.c1.weight.grad.norm_before', self.critic_net.c1.weight.grad.norm())
                nn.utils.clip_grad_norm_(local_Cnet.parameters(), args.max_grad_norm)
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

def policy_test(args, counter1, lock2, counter2, global_Anet, writer_1, evaluate_rewards):
    if args.policy_dist == "Beta":
        Test_Anet = Actor_Beta(args.state_dim, args.action_dim, args.NN_width)
    if args.policy_dist == "Normal":
        Test_Anet = Actor_Normal(args.state_dim, args.action_dim, args.NN_width)
    Test_Cnet = Critic(args.state_dim, args.NN_width)

    while True:
        if counter1.value % args.evaluate_freq == 0:
            Test_Anet.load_state_dict(global_Anet.state_dict())
            with lock2:
                counter2.value = counter2.value + 1

            evaluate_utility = 0
            for _ in range(args.num_episodes_for_evaluate):
                s = env.reset()
                if s.ndim == 2: s = s.squeeze(1)
                # if args.use_state_norm: s = shared_state_norm(s, update=False)

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
                    # if args.use_state_norm: s_ = shared_state_norm(s_, update=False)
                    # if args.use_utility_norm: U = utility_norm(U, update=False)
                    episode_utility += U
                    if done:
                        break
                    s = s_

                evaluate_utility += episode_utility

            evaluate_reward = evaluate_utility / args.num_episodes_for_evaluate

            print('****global_Anet.l1.weight.norm_test****', global_Anet.l1.weight.norm(), '****Test_Anet.l1.weight.norm_test****', Test_Anet.l1.weight.norm())

            evaluate_rewards.append(evaluate_reward)
            print("evaluate_num:{} \t evaluate_reward:{} \t".format(counter2.value, evaluate_rewards[-1]))
            writer_1.add_scalar('episode_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=counter1.value)



def main():
    if not os.path.exists('./param_A3C'):
        os.makedirs('./param_A3C/save_net_param')

    os.environ['OMP_NUM_THREADS'] = '1'  # 每个进程只允许用一个线程
    os.environ['CUDA_VISIBLE_DEVICES'] = ""  # CUDA 为空，表示只用CPU

    writer_1 = SummaryWriter('./exp_A3C_KinematicsRearAxle_v1')

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

    '''create several processes'''
    counter1 = mp.Value('i', 0)
    lock1 = mp.Lock()
    counter2 = mp.Value('i', 0)
    lock2 = mp.Lock()

    processes = []  # store each process

    ## execute on each process
    evaluate_rewards = []
    evaluate_num = 0

    process = mp.Process(target=policy_test, args=(args, counter1, lock2, counter2, global_Anet, writer_1, evaluate_rewards))
    process.start()
    processes.append(process)
    for index_process in range(args.num_processes):
        process = mp.Process(target=learning, args=(index_process, args, lock1, counter1, global_Cnet_optimizer, global_Anet_optimizer, global_Cnet, global_Anet))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


    torch.save(global_Anet.state_dict(), './param_A3C/save_net_param/global_actor_net_A3C' + env_name + '.pkl')
    torch.save(global_Cnet.state_dict(), './param_A3C/save_net_param/global_critic_net_A3C' + env_name + '.pkl')




if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("end ||| time-consuming {}".format((end_time - start_time)))