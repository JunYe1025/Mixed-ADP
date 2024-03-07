import argparse
import time
import os
import sys
import gym

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal, Categorical, Beta
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from collections import namedtuple

import numpy as np

env_name = "Kinematics_Rear_Axle_Env-v0"
env = gym.make(env_name).unwrapped
parser = argparse.ArgumentParser(description='Solve the xxxx with Distributed_PPO')
parser.add_argument('--seed', type=int, default=2024, metavar='N', help='random seed')
parser.add_argument('--num_processes', type=int, default=4, help='how many training processes to use (default: 4)')
parser.add_argument("--update_NN_threshold", type=int, default=4, help='The interval for update global NN')

parser.add_argument('--state_dim', type=int, default=env.observation_space.shape[0], help='state_dimension')
parser.add_argument('--action_dim', type=int, default=env.action_space.shape[0], help='action_dimension')
parser.add_argument("--max_action", type=float, default=env.action_space.high, help="the value of max_action")
parser.add_argument("--min_action", type=float, default=env.action_space.low, help="the value of min_action")
parser.add_argument('--NN_width', type=int, default=64, help='the number of neurons')
parser.add_argument('--policy_dist', type=str, default='Beta', help='the distribution way for choosing action')

parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--lambda_', type=float, default=0.97, help='Truncated GAE (default: 0.97)')
parser.add_argument('--a_lr', type=float, default=1e-3, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic')
parser.add_argument("--entropy_coef", type=float, default=0.01, help="policy entropy")

parser.add_argument('--A_target_selection', type=int, default=2, help='Actor: total utility for the trajectory; TD error; truncated advatage function')
parser.add_argument('--C_target_selection', type=int, default=0, help='Critic: total utility for the trajectory; TD error; truncated advatage function')

parser.add_argument('--clip_param', type=int, default=0.2, help='clip parameter for PPO target function')
parser.add_argument('--max_grad_norm', type=int, default=0.8, help='the maximal threshold for gradient update')
parser.add_argument('--ppo_update_time', type=int, default=2, help='the number of updating NN based on each buffer')
parser.add_argument('--buffer_capacity', type=int, default=2048, help='the capacity of buffer')
parser.add_argument('--mini_batch_size', type=int, default=128, metavar='N', help='batch_size for each NN update, but the total buffer can be used to update')
parser.add_argument('--num_episodes', type=int, default=10001, help='the number of episodes')
parser.add_argument('--step_each_episode', type=int, default=600, help='the number of time steps within each episode')

parser.add_argument('--render', default=True, help='render the environment')
parser.add_argument("--save_net_param_freq", type=int, default=4000, help="frequency (episodes) of saving net_parameters result")

parser.add_argument("--num_episodes_for_evaluate", type=int, default=20, help="the number of episodes during evaluating policy")
parser.add_argument("--evaluate_freq", type=int, default=800, help="Evaluate the policy every 'evaluate_freq' steps")

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
        # x = torch.tanh(self.c2(x))
        value = self.state_value(x)

        return value


class Shared_grad_buffers():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] = torch.ones(p.size()).share_memory_()

    def add_gradient(self, model):
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] += p.grad.data

    def reset(self):
        for name,grad in self.grads.items():
            self.grads[name].fill_(0)

def evaluate_policy(args, env, global_Anet, evaluate_num):
    # print('----------------evaluating----------------Num: {}'.format(evaluate_num))
    evaluate_utility = 0
    for _ in range(args.num_episodes_for_evaluate):
        s = env.reset()
        if s.ndim == 2:
            s = s.squeeze(1)
        # if args.use_state_norm:
        #     s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_utility = 0
        for t_evaluate in range(args.step_each_episode):
            a = global_Anet.select_action_evaluate(s)

            if args.policy_dist == "Beta":
                a_execute = a * (args.max_action - args.min_action) + args.min_action
                # a_execute = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                a_execute = a

            s_, U, done, _ = env.step(a_execute)
            # if args.use_state_norm: s_ = state_norm(s_, update=False)
            # if args.use_utility_norm: U = utility_norm(U, update=False)
            episode_utility += U
            if done:
                break
            s = s_
        evaluate_utility += episode_utility

    return evaluate_utility / args.num_episodes_for_evaluate


def ensure_shared_grads(local_model, global_model):
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        if global_param.grad is not None:
            return 'global_param.grad is not None'
        global_param._grad = local_param.grad

def chief(args, traffic_light, counter, global_Anet, global_Cnet, shared_A_grad_buffers, shared_C_grad_buffers, global_Anet_optimizer, global_Cnet_optimizer):
    while True:
        time.sleep(1)
        # workers will wait after last loss computation
        if counter.get() >= args.update_NN_threshold:
            #print(shared_grad_buffers.grads['.weight_grad'])
            for n_A,p_A in global_Anet.named_parameters():
                p_A.grad = shared_A_grad_buffers.grads[n_A+'_grad']
            for n_C,p_C in global_Cnet.named_parameters():
                p_C.grad = shared_C_grad_buffers.grads[n_C+'_grad']
            global_Anet_optimizer.step()
            global_Cnet_optimizer.step()
            counter.reset()
            shared_A_grad_buffers.reset()
            shared_C_grad_buffers.reset()
            traffic_light.switch() # workers start new loss computation
            # print('update')

def learning(index_process, args, traffic_light, counter, global_Anet, global_Cnet, shared_A_grad_buffers, shared_C_grad_buffers, test_n):
    '''This function <learning> aims to each process.
    In other words, each <learning> is for each agent in their own independent env.
    Therefore, for this <learning> function, it always runs constantly (but possibly restricted by the max_num_episodes and max_step_each_episode).'''
    import ctypes
    x11 = ctypes.cdll.LoadLibrary('libX11.so')
    x11.XInitThreads()

    torch.manual_seed(args.seed + index_process)

    if args.policy_dist == "Beta":
        local_Anet = Actor_Beta(args.state_dim, args.action_dim, args.NN_width)
    if args.policy_dist == "Normal":
        local_Anet = Actor_Normal(args.state_dim, args.action_dim, args.NN_width)
    local_Cnet = Critic(args.state_dim, args.NN_width)

    test_number = 0
    done = False
    total_step_local = 0
    local_net_update_num = 0
    buffer = []
    for num_epi in range(args.num_episodes):
        state = env.reset()
        if state.ndim == 2: state = state.squeeze(1)
        if args.render: env.render()
        for step_t in range(args.step_each_episode):
            action, logprob_action = local_Anet.select_action(state)
            if args.policy_dist == "Beta": a_execute = action * (args.max_action - args.min_action) + args.min_action
            else: a_execute = action
            next_state, utility, done, _ = env.step(a_execute)
            if not isinstance(utility, np.ndarray): utility = np.array([utility])

            mask = 1
            if done or step_t == args.step_each_episode - 1:
                mask = 0
            trans = Transition(state.T, action.T, logprob_action.T, utility, mask, next_state.T)

            if args.render: env.render()
            buffer.append(trans)

            if args.policy_dist == "Beta":
                local_Anet_old = Actor_Beta(args.state_dim, args.action_dim, args.NN_width)
            if args.policy_dist == "Normal":
                local_Anet_old = Actor_Normal(args.state_dim, args.action_dim, args.NN_width)
            local_Cnet_old = Critic(args.state_dim, args.NN_width)
            local_Anet_old.load_state_dict(local_Anet.state_dict())
            local_Cnet_old.load_state_dict(local_Cnet.state_dict())
            total_step_local += 1
            if total_step_local % args.buffer_capacity == 0:
                states = torch.tensor(np.array([t.state for t in buffer]), dtype=torch.float)
                actions = torch.tensor(np.array([t.action for t in buffer]), dtype=torch.float)
                old_action_log_probs = torch.tensor(np.array([t.a_log_prob for t in buffer]), dtype=torch.float)
                utilities = torch.tensor(np.array([t.utility for t in buffer]), dtype=torch.float)
                masks = torch.tensor(np.array([t.mask for t in buffer]), dtype=torch.float)
                next_states = torch.tensor(np.array([t.next_state for t in buffer]), dtype=torch.float)

                current_values_net = local_Cnet_old(states)

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
                        local_Anet.load_state_dict(global_Anet.state_dict())
                        local_Cnet.load_state_dict(global_Cnet.state_dict())
                        local_Anet.zero_grad()
                        local_Cnet.zero_grad()
                        signal_initial = traffic_light.get()

                        action_dist = local_Anet.get_dist(states[index])
                        action_dist_entropy = action_dist.entropy().sum(1, keepdim=True)
                        action_prob = action_dist.log_prob(actions[index])
                        ratio = torch.exp(action_prob - old_action_log_probs[index])
                        surr1 = ratio * actor_target[index]
                        surr2 = torch.clamp(ratio, 1 - args.clip_param, 1 + args.clip_param) * actor_target[index]

                        # update actor network
                        # action_loss = - torch.min(surr1, surr2).mean()  # MAX->MIN desent
                        action_loss = (- torch.min(surr1, surr2) - args.entropy_coef * action_dist_entropy).mean()
                        action_loss.backward()
                        shared_A_grad_buffers.add_gradient(local_Anet)

                        # update critic network
                        value_loss = F.smooth_l1_loss(local_Cnet(states[index]), critic_target[index])
                        value_loss.backward()
                        shared_C_grad_buffers.add_gradient(local_Cnet)

                        counter.increment()
                        # print('index_process',index_process, 'counter.get() in each process:', counter.get())
                        while traffic_light.get() == signal_initial:
                            # print('index_process',index_process, 'traffic_light.get()', traffic_light.get(), 'signal_initial', signal_initial,'test_number', test_number)
                            pass

                del buffer[:]  # clear experience
                test_n =+ 1




            state = next_state

            if done:
                break



class TrafficLight:
    """used by chief to allow workers to run or not"""

    def __init__(self, val=True):
        self.val = mp.Value("b", False)
        self.lock = mp.Lock()

    def get(self):
        with self.lock:
            return self.val.value

    def switch(self):
        with self.lock:
            self.val.value = (not self.val.value)

class Counter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self, val=True):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self):
        # used by workers
        with self.lock:
            self.val.value += 1

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0



def main():
    if not os.path.exists('Distributed_PPO_param'):
        os.makedirs('Distributed_PPO_param/save_net_param')

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    writer_1 = SummaryWriter('./exp_Distributed_PPO_AV_Tracking')

    traffic_light = TrafficLight()
    counter = Counter()

    if args.policy_dist == "Beta":
        global_Anet = Actor_Beta(args.state_dim, args.action_dim, args.NN_width)
    if args.policy_dist == "Normal":
        global_Anet = Actor_Normal(args.state_dim, args.action_dim, args.NN_width)
    global_Cnet = Critic(args.state_dim, args.NN_width)

    global_Anet.share_memory()
    global_Cnet.share_memory()

    shared_A_grad_buffers = Shared_grad_buffers(global_Anet)
    shared_C_grad_buffers = Shared_grad_buffers(global_Cnet)

    global_Anet_optimizer = optim.Adam(global_Anet.parameters(), lr=args.a_lr)
    global_Cnet_optimizer = optim.Adam(global_Cnet.parameters(), lr=args.c_lr)

    test_n = torch.Tensor([0])
    test_n.share_memory_()

    processes = []
    process = mp.Process(target=chief, args=(args, traffic_light, counter, global_Anet, global_Cnet, shared_A_grad_buffers, shared_C_grad_buffers, global_Anet_optimizer, global_Cnet_optimizer))
    process.start()
    processes.append(process)

    evaluate_rewards = []
    evaluate_num = 0

    for index_process in range(args.num_processes):
        process = mp.Process(target=learning, args=(index_process, args, traffic_light, counter, global_Anet, global_Cnet, shared_A_grad_buffers, shared_C_grad_buffers, test_n))
        process.start()
        processes.append(process)

        if int(test_n.item()) % args.evaluate_freq == 0:
            evaluate_num += 1
            evaluate_reward = evaluate_policy(args, env, global_Anet, evaluate_num)

            evaluate_rewards.append(evaluate_reward)
            print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_rewards[-1]))
            writer_1.add_scalar('episode_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=counter.val.value)

        if int(test_n.item()) % args.save_net_param_freq == 0:
            torch.save(global_Anet.state_dict(),
                       './Distributed_PPO_param/save_net_param/actor_net_Distributed_PPO' + str(test_n) + '.pkl')
            torch.save(global_Cnet.state_dict(),
                       './Distributed_PPO_param/save_net_param/critic_net_Distributed_PPO' + str(test_n) + '.pkl')

    for process in processes:
        process.join()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("end ||| time-consuming {}".format((end_time - start_time)))