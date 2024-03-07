import argparse
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import numpy as np

import os
from tensorboardX import SummaryWriter

import time
import math


env_name = "AV_Kinematics_Env-v3"
env = gym.make(env_name).unwrapped

parser = argparse.ArgumentParser(description='Solve the xxxx with CoSADP')

parser.add_argument('--seed', type=int, default=2190, help='random seed (default: 10)')
parser.add_argument('--render', type=bool, default=True, help='render the environment')

parser.add_argument('--state_dim', type=int, default=env.observation_space.shape[0], help='state_dimension')
parser.add_argument('--action_dim', type=int, default=env.action_space.shape[0], help='action_dimension')
parser.add_argument('--disturbance_dim', type=int, default=1, help='disturbance_dimension')
parser.add_argument('--NN_width', type=int, default=128, help='the number of neurons')

parser.add_argument("--max_state", type=float, default=env.observation_space.high, help="the value of max_state")
parser.add_argument("--min_state", type=float, default=env.observation_space.low, help="the value of min_state")
parser.add_argument("--max_action", type=float, default=env.action_space.high, help="the value of max_action")
parser.add_argument("--min_action", type=float, default=env.action_space.low, help="the value of min_action")
parser.add_argument("--disturbance_limit", type=float, default=env.disturbance_limit, help="the limitation of disturbance")

parser.add_argument('--buffer_capacity_MADP', type=int, default=512000, help='the capacity of dataset (for the whole state-action-disturbance space)')
parser.add_argument('--batch_size_MADP', type=int, default=512, metavar='N', help='batch_size for each update, but the total dataset should be updated')
parser.add_argument('--iter_c_log_display', type=int, default=1, help='the frequency of displaying Critic_NN log information')
parser.add_argument('--iter_a_log_display', type=int, default=1, help='the frequency of displaying Actor_NN log information')
parser.add_argument('--iter_d_log_display', type=int, default=1, help='the frequency of displaying Disturbance_NN log information')

parser.add_argument('--a_lr', type=float, default=5e-4, help='Learning rate of actor net')
parser.add_argument('--c_lr', type=float, default=5e-4, help='Learning rate of critic net')
parser.add_argument('--d_lr', type=float, default=5e-4, help='Learning rate of disturbance net')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for rewards (default: 0.99)')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='the maximal threshold for gradient update')

parser.add_argument('--VI_num', type=int, default=3000, help='the number of PI (including PIM and PEV)')
parser.add_argument('--Q_NN_Approximation_Num', type=int, default=1000, help='the number of approximating Q via NN technique')
parser.add_argument('--u_NN_Approximation_Num', type=int, default=1, help='the number of approximating Q via NN technique')
parser.add_argument('--u_NN_Approximation_Num_pre_A', type=int, default=2000, help='the number of approximating u via NN technique in the pre_A process')
parser.add_argument('--PIM_step', type=int, default=11, metavar='N', help='the number of gradient descent for PIM')

parser.add_argument('--num_episode', type=int, default=200000, help='the number of episodes for collecting data')
parser.add_argument('--step_each_episode', type=int, default=600, help='the step in one episode')

parser.add_argument('--save_net_param_freq', type=int, default=10, help="frequency of saving net_parameters result")

parser.add_argument('--a', type=float, default=1.0, help="hyper-parameter used in MADP")
parser.add_argument('--b', type=float, default=0.5, help="hyper-parameter used in MADP")
parser.add_argument('--c', type=float, default=0.01, help="hyper-parameter used in MADP")

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

class Actor_NN(nn.Module):
    def __init__(self, state_dim, action_dim, NN_width, max_action, min_action):
        super(Actor_NN, self).__init__()
        self.hidden1 = nn.Linear(state_dim, NN_width)
        self.output_action = nn.Linear(NN_width, action_dim)

        self.max_action = torch.tensor(max_action)
        self.min_action = torch.tensor(min_action)

    def forward(self, state):
        self.hidden1_output = self.hidden1(state)
        self.excitation_output1 = torch.tanh(self.hidden1_output)
        self.action = self.output_action(self.excitation_output1)
        # self.action = ((torch.tanh(self.action) + torch.ones_like(self.action)) / 2) * (
        #             self.max_action - self.min_action) + self.min_action

        return self.action

class Disturbance_NN(nn.Module):
    def __init__(self, state_dim, disturbance_dim, NN_width, disturbance_limit):
        super(Disturbance_NN, self).__init__()

        self.hidden1 = nn.Linear(state_dim, NN_width)
        self.output_Dis = nn.Linear(NN_width, disturbance_dim)
        self.disturbance_limit = torch.tensor(disturbance_limit)

    def forward(self, state):
        self.hidden1_output = self.hidden1(state)
        self.excitation_output1 = torch.tanh(self.hidden1_output)
        self.Dis = self.output_Dis(self.excitation_output1)
        self.Dis = self.disturbance_limit * torch.tanh(self.Dis)

        return self.Dis

class Critic_NN(nn.Module):
    def __init__(self, state_dim, action_dim, disturbance_dim, NN_width):
        super(Critic_NN, self).__init__()

        self.hidden1 = nn.Linear(state_dim + action_dim + disturbance_dim, NN_width)
        # self.hidden1.weight.data.normal_(0.5, 0.1)  # initialization
        # self.hidden1.bias.data.fill_(0)
        self.output_J = nn.Linear(NN_width, 1)
        self.output_J.weight.data.normal_(0.5, 0.1)  # initialization
        self.output_J.bias.data.fill_(0)

    def forward(self, state, action, disturbance):
        self.critic_input = torch.hstack((state, action, disturbance))
        self.hidden1_output = self.hidden1(self.critic_input)
        self.excitation_output1 = F.relu(self.hidden1_output)
        self.J = self.output_J(self.excitation_output1)
        self.J = F.relu(self.J)

        return self.J


'''iterating'''
class MADP():
    def __init__(self):
        self.memory_counter = 0
        self.buffer = np.zeros((args.buffer_capacity_MADP, args.state_dim*4 + args.action_dim*2 + args.disturbance_dim + 1 + 1), dtype=np.float32)

        self.actor_NN = Actor_NN(args.state_dim, args.action_dim, args.NN_width, args.max_action, args.min_action)
        self.critic_1_NN = Critic_NN(args.state_dim, args.action_dim, args.disturbance_dim, args.NN_width)
        self.critic_2_NN = Critic_NN(args.state_dim, args.action_dim, args.disturbance_dim, args.NN_width)
        self.disturbance_NN = Disturbance_NN(args.state_dim, args.disturbance_dim, args.NN_width, args.disturbance_limit)

        self.actor_optimizer = torch.optim.Adam(self.actor_NN.parameters(), args.a_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1_NN.parameters(), args.c_lr)
        self.disturbance_optimizer = torch.optim.Adam(self.disturbance_NN.parameters(), args.d_lr)

        self.initial_state = np.array([[0.0],
                                       [1.0],
                                       [0.0]])
        self.K = torch.tensor([[0.72597898,-0.61444139, 0.07141239],
                               [0.63068898, 0.74901528, 1.24531936]])
        self.writer_1 = SummaryWriter('./exp_MADP_AV_Tracking_v1')
        if not os.path.exists('./MADP_param'):
            os.makedirs('./MADP_param/save_net_param')

    def iterating(self):
        self.Initial_buffer_V2()
        for i in range(args.VI_num):
            '''critic'''
            indices_ADP = np.random.randint(0, args.buffer_capacity_MADP, size=args.batch_size_MADP)
            bt = self.buffer[indices_ADP, :]
            bes = torch.FloatTensor(bt[:, :args.state_dim])
            bs = torch.FloatTensor(bt[:, args.state_dim: 2 * args.state_dim])
            blank = torch.FloatTensor(bt[:, 2 * args.state_dim: 2 * args.state_dim + args.action_dim])
            ba = torch.FloatTensor(bt[:, 2 * args.state_dim + args.action_dim: 2 * args.state_dim + 2 * args.action_dim])
            bd = torch.FloatTensor(bt[:, 2 * args.state_dim + 2 * args.action_dim: 2 * args.state_dim + 2 * args.action_dim + args.disturbance_dim])
            bU = torch.FloatTensor(bt[:, -2 * args.state_dim - 1 - 1: -2 * args.state_dim - 1])
            bes_ = torch.FloatTensor(bt[:, -2 * args.state_dim - 1: -args.state_dim - 1])
            bs_ = torch.FloatTensor(bt[:, -args.state_dim - 1: -1])
            b_t_index = bt[:, -1:].astype(int)

            ba_ = self.actor_NN(bes_)
            bd_ = self.disturbance_NN(bes_)
            self.critic_2_NN.load_state_dict(self.critic_1_NN.state_dict())
            Q_next = self.critic_2_NN(bes_, ba_, bd_)
            Q_target = bU + args.gamma * Q_next - 1 * self.critic_1_NN(torch.zeros(1, args.state_dim), torch.zeros(1, args.action_dim), torch.zeros(1, args.disturbance_dim))

            for i_equal in range(args.Q_NN_Approximation_Num):
                loss_critic_C = F.smooth_l1_loss(self.critic_1_NN(bes, ba, bd), Q_target.data)

                self.critic_1_optimizer.zero_grad()
                loss_critic_C.backward()
                self.critic_1_optimizer.step()

                # print('CRITIC_1_ADP.hidden1.weight.grad', CRITIC_1_ADP.hidden1.weight.grad)
                # print('第', i, '次VI迭代', 'loss_critic_C', loss_critic_C.data)
            if i % args.iter_c_log_display == 0:
                print('第', i, '次VI迭代', 'loss_critic_C_Bellman', loss_critic_C.data)

            '''actor'''
            u_LQR = torch.tensor(env.linear_model_MADP(bs, b_t_index, args.batch_size_MADP), dtype=torch.float)
            # eu_LQR = (- self.K @ bes.T).T
            # u_LQR = np.zeros((args.batch_size_MADP, args.action_dim))
            # delta_ref = np.zeros((args.batch_size_MADP, 1))
            # for i_row in range(args.batch_size_MADP):
            #     x_ref, y_ref, yaw_ref, kappa_ref = env.find_reference_point(b_t_index[i_row,:].item())
            #     delta_ref[i_row,:] = np.arctan((env.Lf + env.Lr) / (1 / (kappa_ref + 1e-8)))
            #     u_LQR[i_row, :] = np.array([eu_LQR[i_row,0] + env.v_ref, eu_LQR[i_row,1] + delta_ref[i_row,:]])
            # u_LQR = torch.tensor(u_LQR, dtype=torch.float)

            for i_equal in range(args.u_NN_Approximation_Num):
                u_NN = self.actor_NN(bs)
                A_loss_function1 = nn.MSELoss()
                A_loss1 = A_loss_function1(u_NN, u_LQR.data)
                A_loss2 = 1 * torch.mean(self.critic_1_NN(bes, self.actor_NN(bes), self.disturbance_NN(bes))) ##############+-号确定下‘’‘应该都是+号吧’‘’

                Lambda_PG = math.tanh(args.b*math.log10(args.a + args.c * i))
                Lambda_BC = 1 - Lambda_PG
                A_loss = Lambda_BC * A_loss1 + Lambda_PG * A_loss2

                self.actor_optimizer.zero_grad()
                A_loss.backward()
                self.actor_optimizer.step()

                # print('self.actor_NN.hidden1.weight.grad', self.actor_NN.hidden1.weight.grad)
            if i % args.iter_a_log_display == 0:
                print('第', i, '次VI迭代', 'Lambda_PG', Lambda_PG,
                      'A_loss', A_loss.data, 'A_loss_BC', A_loss1.data, 'A_loss_PG', A_loss2.data)

            '''disturbance'''
            # D_loss = - torch.mean(self.critic_1_NN(bes, self.actor_NN(bes), self.disturbance_NN(bes))) ##############+-号确定下‘’‘应该都是+号吧’‘’
            # self.disturbance_optimizer.zero_grad()
            # D_loss.backward()
            # self.disturbance_optimizer.step()
            #
            # # print('disturbance_NN.hidden1.weight.grad', self.disturbance_NN.hidden1.weight.grad)
            # # print('disturbance_NN.output_Dis.weight.grad', self.disturbance_NN.output_Dis.weight.grad)
            # if i % args.iter_d_log_display == 0:
            #     print('第', i, '次VI迭代', 'D_loss', D_loss.data)

    def Initial_buffer_V1(self):
        for i_Ep in range(args.num_episode):
            state_transfer = env.reset()
            if state_transfer.ndim == 2: state_transfer = state_transfer.squeeze(1)
            for i in range(args.step_each_episode):
                s = state_transfer
                a_NN = (self.actor_NN(torch.tensor(s.T ,dtype=torch.float))).detach().numpy()
                d_NN = (self.disturbance_NN(torch.tensor(s.T, dtype=torch.float))).detach().numpy()
                s_next_uD, _, a_uD = env.linear_model(s, i)
                s_next_NN, U, done, _ = env.step_ADP(a_NN)

                transition = np.hstack((s.T, a_uD.T, a_NN, d_NN, U, s_next_NN, i))
                index = self.memory_counter % args.buffer_capacity_MADP
                self.buffer[index, :] = transition
                self.memory_counter += 1

                state_transfer = s_next_NN

                if self.memory_counter == args.buffer_capacity_MADP or done:
                    break
            if self.memory_counter == args.buffer_capacity_MADP:
                break


    def Initial_buffer_V2(self):
        for i_Ep in range(args.num_episode):
            state_transfer = env.reset_in_whole_region()
            if state_transfer.ndim == 2: state_transfer = state_transfer.squeeze(1)
            for i in range(args.step_each_episode):
                s = state_transfer
                x_ref, y_ref, yaw_ref, kappa_ref = env.find_reference_point(i)
                error_s = np.array([s[0] - x_ref, s[1] - y_ref, s[2] - yaw_ref])
                a_NN = (self.actor_NN(torch.tensor(s.T ,dtype=torch.float))).detach().numpy()
                d_NN = (self.disturbance_NN(torch.tensor(s.T, dtype=torch.float))).detach().numpy()
                blank = np.array([0, 0])
                s_next_NN, U, done, _ = env.step_ADP(a_NN)
                if i==args.step_each_episode - 1:
                    i = args.step_each_episode - 2
                x_next_ref, y_next_ref, yaw_next_ref, kappa_next_ref = env.find_reference_point(i+1)
                error_s_next = np.array([s_next_NN[0] - x_next_ref, s_next_NN[1] - y_next_ref, s_next_NN[2] - yaw_next_ref])

                transition = np.hstack((error_s.T, s.T, blank.T, a_NN, d_NN, U, error_s_next, s_next_NN, i))
                index = self.memory_counter % args.buffer_capacity_MADP
                self.buffer[index, :] = transition
                self.memory_counter += 1

                state_transfer = s_next_NN

                if self.memory_counter == args.buffer_capacity_MADP or done:
                    break
            if self.memory_counter == args.buffer_capacity_MADP:
                break


    def Initial_buffer_V3(self):
        state_lib = np.random.uniform(low=env.state_low, high=env.state_high, size=(args.buffer_capacity_MADP, args.state_dim))
        action_lib = np.random.uniform(low=env.action_low, high=env.action_high, size=(args.buffer_capacity_MADP, args.action_dim))
        disturbance_lib = np.random.uniform(low=-args.disturbance_limit, high=args.disturbance_limit, size=(args.buffer_capacity_MADP, args.disturbance_dim))


        return state_lib

    def pre_train_A(self):
        state_lib = self.Initial_buffer_V3()
        error_s_lib = np.zeros((args.buffer_capacity_MADP, args.state_dim))
        real_a_lib = np.zeros((args.buffer_capacity_MADP, args.action_dim))
        for i in range(args.buffer_capacity_MADP):
            current_position = (state_lib[i, 0].item(), state_lib[i, 1].item())
            x_closest, y_closest, yaw_ref, kappa = env.find_closest_point_on_continuous_function(current_position)

            error_s_lib[i, :] = np.array([state_lib[i, 0] - x_closest, state_lib[i, 1] - y_closest, state_lib[i, 2] - yaw_ref])

            s_next, error_a , real_a = env.linear_model_path_tracking(error_s_lib[i, :])

            real_a_lib[i, :] = real_a

        state_lib = torch.tensor(state_lib, dtype=torch.float)
        real_a_lib = torch.tensor(real_a_lib, dtype=torch.float)
        for i_equal in range(args.u_NN_Approximation_Num_pre_A):
            u_NN = self.actor_NN(state_lib)
            A_loss_function1 = nn.MSELoss()
            A_loss = A_loss_function1(u_NN, real_a_lib.data)

            self.actor_optimizer.zero_grad()
            A_loss.backward()
            self.actor_optimizer.step()

            print('第', i_equal, '次近似', 'A_loss', A_loss.data)
    # def store_data(self, s, a_uD, a_NN, d_NN, U, s_):
    #     transition = np.hstack((s, a_uD, a_NN, d_NN, U, s_))
    #     index = self.memory_counter % args.buffer_capacity_MADP
    #     self.buffer[index, :] = transition
    #     self.memory_counter += 1

    def save_NN(self):
        torch.save(self.critic_1_NN, './MADP_param/save_net_param/' + 'critic_1_NN_MADP' + '.pth')
        torch.save(self.actor_NN, './MADP_param/save_net_param/'+ 'actor_NN_MADP' + '.pth')
        torch.save(self.disturbance_NN, './MADP_param/save_net_param/' + 'disturbance_NN_MADP' + '.pth')

    def policy_test(self):
        state1_store = np.zeros((1, args.step_each_episode))
        state2_store = np.zeros((1, args.step_each_episode))
        state3_store = np.zeros((1, args.step_each_episode))
        action1_store = np.zeros((1, args.step_each_episode))
        action2_store = np.zeros((1, args.step_each_episode))

        state_transfer = env.reset()
        if state_transfer.ndim == 2: state_transfer = state_transfer.squeeze(1)
        for i in range(args.step_each_episode):
            s = state_transfer

            x_ref, y_ref, yaw_ref, kappa_ref = env.find_reference_point(i)
            error_s = np.array([s[0] - x_ref, s[1] - y_ref, s[2] - yaw_ref])

            # current_position = (s[0].item(), s[1].item())
            # x_closest, y_closest, yaw_ref, kappa = env.find_closest_point_on_continuous_function(current_position)
            # error_s = np.array([s[0] - x_closest, s[1] - y_closest, s[2] - yaw_ref])

            a = (self.actor_NN(torch.tensor(error_s.T ,dtype=torch.float))).detach().numpy()
            s_next, _, _, _ = env.step_ADP(a)
            state_transfer = s_next

            state1_store[0, i] = s[0]
            state2_store[0, i] = s[1]
            state3_store[0, i] = s[2]
            action1_store[0, i] = a[0]
            action2_store[0, i] = a[1]

        import matplotlib.pyplot as plt
        x = np.linspace(0, env.state_high[0], args.step_each_episode)
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
    MADP = MADP()
    MADP.iterating()
    # MADP.pre_train_A()
    MADP.save_NN()
    MADP.policy_test()
