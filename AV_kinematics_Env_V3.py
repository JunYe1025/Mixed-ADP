'''
This environment is kinematics models of vehicle, including linear model extended at expected trajectory point and augmented model, applying for trajectory tracking control
The feasible region of this Env scaled (scale=0.1)
'''

from torch.autograd import Variable
import torch

import numpy as np

import scipy.linalg
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

import math
import random

import gym
from gym import spaces

import control

from gym.envs.classic_control import rendering

np.random.seed(random.randint(1, 2000))

class AV_Kinematics_Env_V3(gym.Env):
    def __init__(self):
        self.dt = 0.1

        self.state_low = np.array([0, -1.5, -3.14])
        self.state_high = np.array([10, 1.5, 3.14])
        self.reset_state_low = np.array([0., 0, -0.])
        self.reset_state_high = np.array([0., 0.1, 0.])
        self.action_low = np.array([0., -1.04])
        self.action_high = np.array([0.5, 1.04])
        self.observation_space = spaces.Box(low=self.state_low, high=self.state_high, dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)  # 0.2rad=11.46°

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        # self.Q = np.array([[0.05, 0, 0],
        #                    [0, 0.05, 0],
        #                    [0, 0, 0.05]])  # LQR
        # self.R = np.array([[0.05, 0],
        #                    [0, 0.05]])  # LQR
        self.Q = np.array([[0.03, 0, 0],
                           [0, 0.03, 0],
                           [0, 0, 0.03]]) # PPO
        self.R = np.array([[0.01, 0],
                           [0, 0.01]]) # PPO
        self.Augmented_Q = scipy.linalg.block_diag(self.Q, np.zeros((self.state_dim,self.state_dim)))
        self.Augmented_R = scipy.linalg.block_diag(self.R, np.zeros((self.action_dim,self.action_dim)))

        self.disturbance_limit = 0.13

        self.Lf = 0.15
        self.Lr = 0.15
        self.amplitude = 1.
        self.T = 2 * np.pi
        self.w = (2 * np.pi)/self.T

        self.num_points = 600
        self.least_step = 500
        self.v_ref = 0.1

        self.viewer = None
        self.scale = 100
        self.screen_width = 1000
        self.screen_length = 300
        self.vehicle_width = 0.16
        self.vehicle_length = 0.3

    def step(self, action):
        '''next state'''
        action = action.reshape(self.action_dim, -1)

        x_ref, y_ref, yaw_ref, kappa_ref = self.find_reference_point(self.t)
        error_s = np.array([[self.state[0, 0] - x_ref],
                            [self.state[1, 0] - y_ref],
                            [self.state[2, 0] - yaw_ref]])

        # delta_ref = np.arctan((self.Lf+self.Lr) / (1 / (kappa_ref + 1e-8)))
        # error_a = np.array([[action[0, 0] - self.v_ref],
        #                     [action[1, 0] - delta_ref]])

        s_next = np.zeros((3, 1))
        s_next[0, 0] = self.state[0, 0] + action[0, 0] * np.cos(self.state[2, 0]) * self.dt
        s_next[1, 0] = self.state[1, 0] + action[0, 0] * np.sin(self.state[2, 0]) * self.dt
        s_next[2, 0] = self.state[2, 0] + (action[0, 0] / (self.Lf+self.Lr)) * np.tan(action[1, 0]) * self.dt

        self.state = s_next
        self.t = self.t + 1

        '''reward'''
        reward = - (error_s.T @ self.Q @ error_s + action.T @ self.R @ action)
        if np.any(self.state.squeeze(1) <= self.state_low) or np.any(self.state_high[1:self.state_dim] <= self.state.squeeze(1)[1:self.state_dim]):
            reward = - (error_s.T @ self.Q @ error_s + action.T @ self.R @ action) - 2000.0
        if np.any(self.state.squeeze(1) <= self.state_low) or np.any(self.state_high <= self.state.squeeze(1)):
            reward = - (error_s.T @ self.Q @ error_s + action.T @ self.R @ action) - 300.0
        if self.t >= self.least_step:
            reward = reward + 3.0

        '''if done'''
        done = False
        if np.any(self.state.squeeze(1) <= self.state_low) or np.any(self.state_high <= self.state.squeeze(1)):
            done = True

        '''Extra INfO for Debugging'''
        info = {}

        return s_next.squeeze(1), reward.squeeze(1), done, info

    def step_ADP(self, action):
        '''next state'''
        action = action.reshape(self.action_dim, -1)

        x_ref, y_ref, yaw_ref, kappa_ref = self.find_reference_point(self.t)
        error_s = np.array([[self.state[0, 0] - x_ref],
                            [self.state[1, 0] - y_ref],
                            [self.state[2, 0] - yaw_ref]])

        # delta_ref = np.arctan((self.Lf+self.Lr) / (1 / (kappa_ref + 1e-8)))
        # error_a = np.array([[action[0, 0] - self.v_ref],
        #                     [action[1, 0] - delta_ref]])

        s_next = np.zeros((3, 1))
        s_next[0, 0] = self.state[0, 0] + action[0, 0] * np.cos(self.state[2, 0]) * self.dt
        s_next[1, 0] = self.state[1, 0] + action[0, 0] * np.sin(self.state[2, 0]) * self.dt
        s_next[2, 0] = self.state[2, 0] + (action[0, 0] / (self.Lf+self.Lr)) * np.tan(action[1, 0]) * self.dt

        self.state = s_next
        self.t = self.t + 1

        '''reward'''
        reward =  (error_s.T @ self.Q @ error_s + action.T @ self.R @ action)
        # if np.any(self.state.squeeze(1) <= self.state_low) or np.any(self.state_high[1:self.state_dim] <= self.state.squeeze(1)[1:self.state_dim]):
        #     reward = - (error_s.T @ self.Q @ error_s + action.T @ self.R @ action) - 2000.0
        # if np.any(self.state.squeeze(1) <= self.state_low) or np.any(self.state_high <= self.state.squeeze(1)):
        #     reward = - (error_s.T @ self.Q @ error_s + action.T @ self.R @ action) - 1000.0

        '''if done'''
        done = False
        if np.any(self.state.squeeze(1) <= self.state_low) or np.any(self.state_high <= self.state.squeeze(1)):
            done = True

        '''Extra INfO for Debugging'''
        info = {}

        return s_next.squeeze(1), reward.squeeze(1), done, info

    def linear_model_path_tracking(self, state):
        state = state.reshape(self.state_dim, 1)
        '''This model is linear at the expected trajectory point, used for LQR'''
        current_position = (state[0, 0].item(), state[1, 0].item())
        # print('current_position', current_position)
        x_closest, y_closest, yaw_ref, kappa = self.find_closest_point_on_continuous_function(current_position)
        # x_closest, y_closest, yaw_ref, kappa = self.find_closest_point_on_discrete_function(current_position)
        # print('sin_point',x_closest,y_closest,yaw_ref)
        v_ref = 3.6  # customized value
        delta_ref = np.arctan((self.Lf+self.Lr) / (1 / kappa))
        # delta_ref = 0

        A_dt = np.array([[1, 0, -self.dt * v_ref * np.sin(yaw_ref)],
                         [0, 1,  self.dt * v_ref * np.cos(yaw_ref)],
                         [0, 0, 1]])
        B_dt = np.array([[self.dt * np.cos(yaw_ref), 0],
                         [self.dt * np.sin(yaw_ref), 0],
                         [self.dt * np.tan(delta_ref)/(self.Lf+self.Lr), (self.dt * v_ref)/((self.Lf+self.Lr) * np.cos(delta_ref)**2)]])

        error_s = np.array([[state[0, 0] - x_closest],
                            [state[1, 0] - y_closest],
                            [state[2, 0] - yaw_ref]])

        P, eigvals, K = control.dare(A_dt, B_dt, self.Q, self.R)
        error_action = -K @ error_s

        # error_s_next = A_dt @ error_s + B_dt @ action
        # s_next = np.array([[error_s_next[0, 0] + x_closest],
        #                    [error_s_next[1, 0] + y_closest],
        #                    [error_s_next[2, 0] + yaw_ref]])
        #
        s_next = np.zeros((3, 1))
        v_real = error_action[0, 0] + v_ref
        delta_real = error_action[1, 0] + delta_ref
        action_real = np.array([v_real, delta_real])
        s_next[0, 0] = state[0, 0] + v_real * np.cos(state[2, 0]) * self.dt
        s_next[1, 0] = state[1, 0] + v_real * np.sin(state[2, 0]) * self.dt
        s_next[2, 0] = state[2, 0] + (v_real / (self.Lf+self.Lr)) * np.tan(delta_real) * self.dt

        return s_next.squeeze(1), error_action.squeeze(1), action_real

    def linear_model(self, state, t):
        state = state.reshape(self.state_dim, 1)
        '''This model is linear at the expected trajectory point, used for LQR'''
        x_ref, y_ref, yaw_ref, kappa_ref = self.find_reference_point(t)

        delta_ref = np.arctan((self.Lf+self.Lr) / (1 / (kappa_ref + 1e-8)))

        A_dt = np.array([[1, 0, -self.dt * self.v_ref * np.sin(yaw_ref)],
                         [0, 1,  self.dt * self.v_ref * np.cos(yaw_ref)],
                         [0, 0, 1]])
        B_dt = np.array([[self.dt * np.cos(yaw_ref), 0],
                         [self.dt * np.sin(yaw_ref), 0],
                         [self.dt * np.tan(delta_ref)/(self.Lf+self.Lr), (self.dt * self.v_ref)/((self.Lf+self.Lr) * np.cos(delta_ref)**2)]])

        error_s = np.array([[state[0, 0] - x_ref],
                            [state[1, 0] - y_ref],
                            [state[2, 0] - yaw_ref]])

        P, eigvals, K = control.dare(A_dt, B_dt, self.Q, self.R)

        # K = np.array([[0.72597898,-0.61444139, 0.07141239],
        #               [0.63068898, 0.74901528, 1.24531936]])
        error_a = -K @ error_s

        # error_s_next = A_dt @ error_s + B_dt @ action
        # s_next = np.array([[error_s_next[0, 0] + x_ref],
        #                    [error_s_next[1, 0] + y_ref],
        #                    [error_s_next[2, 0] + yaw_ref]])

        v_real = error_a[0, 0] + self.v_ref
        delta_real = error_a[1, 0] + delta_ref
        real_a = np.array([[v_real],
                           [delta_real]])

        # disturbance = np.random.uniform(-self.disturbance_limit, self.disturbance_limit)
        disturbance = (2 * self.disturbance_limit * np.random.rand(1, 1).squeeze(1) - self.disturbance_limit)

        s_next = np.zeros((3, 1))
        s_next[0, 0] = state[0, 0] + v_real * np.cos(state[2, 0]) * self.dt
        s_next[1, 0] = state[1, 0] + v_real * np.sin(state[2, 0]) * self.dt
        s_next[2, 0] = state[2, 0] + (v_real / (self.Lf+self.Lr)) * np.tan(delta_real) * self.dt + disturbance

        return s_next.squeeze(1), error_a.squeeze(1), real_a.squeeze(1)

    def linear_model_LQR(self, state, t):
        state = state.reshape(self.state_dim, 1)
        '''This model is linear at the expected trajectory point, used for LQR'''
        x_ref, y_ref, yaw_ref, kappa_ref = self.find_reference_point(t)

        delta_ref = np.arctan((self.Lf+self.Lr) / (1 / (kappa_ref + 1e-8)))

        A_dt = np.array([[1, 0, -self.dt * self.v_ref * np.sin(yaw_ref)],
                         [0, 1,  self.dt * self.v_ref * np.cos(yaw_ref)],
                         [0, 0, 1]])
        B_dt = np.array([[self.dt * np.cos(yaw_ref), 0],
                         [self.dt * np.sin(yaw_ref), 0],
                         [self.dt * np.tan(delta_ref)/(self.Lf+self.Lr), (self.dt * self.v_ref)/((self.Lf+self.Lr) * np.cos(delta_ref)**2)]])

        error_s = np.array([[state[0, 0] - x_ref],
                            [state[1, 0] - y_ref],
                            [state[2, 0] - yaw_ref]])

        P, eigvals, K = control.dare(A_dt, B_dt, self.Q, self.R)
        error_a = -K @ error_s

        # error_s_next = A_dt @ error_s + B_dt @ action
        # s_next = np.array([[error_s_next[0, 0] + x_ref],
        #                    [error_s_next[1, 0] + y_ref],
        #                    [error_s_next[2, 0] + yaw_ref]])

        v_real = error_a[0, 0] + self.v_ref
        delta_real = error_a[1, 0] + delta_ref
        real_a = np.array([[v_real],
                           [delta_real]])

        # disturbance = np.random.uniform(-self.disturbance_limit, self.disturbance_limit)
        disturbance = (2 * self.disturbance_limit * np.random.rand(1, 1).squeeze(1) - self.disturbance_limit)

        s_next = np.zeros((3, 1))
        s_next[0, 0] = state[0, 0] + v_real * np.cos(state[2, 0]) * self.dt
        s_next[1, 0] = state[1, 0] + v_real * np.sin(state[2, 0]) * self.dt
        s_next[2, 0] = state[2, 0] + (v_real / (self.Lf+self.Lr)) * np.tan(delta_real) * self.dt + disturbance

        return s_next.squeeze(1), error_a.squeeze(1), real_a.squeeze(1)

    def linear_model_MADP(self, state, t, N):
        real_a = np.zeros((N, self.action_dim))
        '''This model is linear at the expected trajectory point, used for LQR'''
        for i in range(N):
            x_ref, y_ref, yaw_ref, kappa_ref = self.find_reference_point(t[i, 0])

            delta_ref = np.arctan((self.Lf+self.Lr) / (1 / (kappa_ref + 1e-8)))

            A_dt = np.array([[1, 0, -self.dt * self.v_ref * np.sin(yaw_ref)],
                             [0, 1,  self.dt * self.v_ref * np.cos(yaw_ref)],
                             [0, 0, 1]])
            B_dt = np.array([[self.dt * np.cos(yaw_ref), 0],
                             [self.dt * np.sin(yaw_ref), 0],
                             [self.dt * np.tan(delta_ref)/(self.Lf+self.Lr), (self.dt * self.v_ref)/((self.Lf+self.Lr) * np.cos(delta_ref)**2)]])

            error_s = np.array([[state[i, 0] - x_ref],
                                [state[i, 1] - y_ref],
                                [state[i, 2] - yaw_ref]])

            P, eigvals, K = control.dare(A_dt, B_dt, self.Q, self.R)
            error_a = - K @ error_s

            v_real = error_a[0, 0] + self.v_ref
            delta_real = error_a[1, 0] + delta_ref
            real_a[i, :] = np.array([v_real, delta_real])

        return real_a

    def next_state(self, state, action, disturbance, N):
        error_s = np.zeros((N, self.state_dim))
        Utility = np.zeros((N, 1))
        for i in range(N):
            current_position = (state[i, 0].item(), state[i, 1].item())
            x_closest, y_closest, yaw_ref, kappa = self.find_closest_point_on_continuous_function(current_position)
            error_s[i, :] = np.array([state[i, 0] - x_closest], [state[i, 1] - y_closest], [state[i, 2] - yaw_ref])
            Utility[i, :] = (error_s[i, :].reshape(self.state_dim, 1).T @ self.Q @ error_s[i, :].reshape(self.state_dim, 1) + action.T @ self.R @ action)

        s_next_lib = np.zeros((N, self.state_dim))
        for i in range(N):
            s_next_lib[i, 0] = state[i, 0] + action[i, 0] * np.cos(state[i, 2]) * self.dt
            s_next_lib[i, 1] = state[i, 1] + action[i, 0] * np.sin(state[i, 2]) * self.dt
            s_next_lib[i, 2] = state[i, 2] + (action[i, 0] / (self.Lf + self.Lr)) * np.tan(action[i, 1]) * self.dt + disturbance[i, 0]
        return s_next_lib, Utility

    def find_reference_point(self, t):
        x = np.linspace(0, self.state_high[0].item(), self.num_points)
        y = self.amplitude * np.sin(2 * np.pi * x / self.T)
        indexes = np.arange(len(x), dtype=int)
        points = np.vstack((indexes, x, y)).T

        ref_point = points[t, :]
        x_ref = ref_point[1]
        y_ref = ref_point[2]

        y_prime = (2 * np.pi / self.T) * self.amplitude * np.cos(2 * np.pi * x_ref / self.T)
        y_double_prime = -(2 * np.pi / self.T) ** 2 * self.amplitude * np.sin(2 * np.pi * x_ref / self.T)
        kappa_ref = abs(y_double_prime) / (1 + y_prime ** 2) ** (3 / 2)
        yaw_ref = np.arctan(y_prime)

        return x_ref, y_ref, yaw_ref, kappa_ref

    def find_closest_point_on_continuous_function(self, current_p):
        def distance_squared(x, a, b):
            return (x - a) ** 2 + (self.amplitude * np.sin(2 * np.pi * x / self.T) - b) ** 2

        # Initial guess for x
        x0 = 0
        bounds = [(current_p[0] - self.T, current_p[0] + self.T)]
        # Minimize the distance squared
        # res = minimize(distance_squared, x0, args=current_position, method='SLSQP')
        res = differential_evolution(distance_squared, bounds, args=current_p)

        x_closest = res.x[0]
        y_closest = self.amplitude * np.sin(2 * np.pi * x_closest / self.T)

        # Curvature calculation
        y_prime = (2 * np.pi / self.T) * self.amplitude * np.cos(2 * np.pi * x_closest / self.T)
        y_double_prime = -(2 * np.pi / self.T) ** 2 * self.amplitude * np.sin(2 * np.pi * x_closest / self.T)
        kappa = abs(y_double_prime) / (1 + y_prime ** 2) ** (3 / 2)

        # Tangent angle calculation (in radians)
        yaw_ref = np.arctan(y_prime)

        return x_closest, y_closest, yaw_ref, kappa

    def reset(self):
        # self.state = self.observation_space.sample().reshape(self.state_dim,-1)
        self.state = np.random.uniform(self.reset_state_low, self.reset_state_high).reshape(self.state_dim, -1)
        self.t = 0
        return self.state

    def reset_in_whole_region(self):
        # self.state = self.observation_space.sample().reshape(self.state_dim,-1)
        self.state = np.random.uniform(self.state_low, self.state_high).reshape(self.state_dim, -1)
        self.t = 0
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_length)

            '''generate a sine wave'''
            x = np.linspace(0, self.state_high[0].item(), self.num_points)
            y = self.scale * self.amplitude * np.sin((2 * np.pi / self.T) * x)
            x = self.scale * x
            y = y + self.screen_length / 2
            sine_wave_points = list(zip(x, y))
            sine_wave = rendering.make_polyline(sine_wave_points)
            sine_wave.set_color(0, 0, 1)  # 蓝色
            self.viewer.add_geom(sine_wave)

            '''generate a vehicle'''
            '''generate a vehicle (using a rectangle, the vehicle is generated by four verteces)'''
            # Here can represent the vehicle model is established based on the rear axle.
            a, b, c = -(self.vehicle_width / 2) * self.scale, (self.vehicle_width / 2) * self.scale, (
                self.vehicle_length) * self.scale
            vehicle = rendering.make_polyline([(a, 0), (a, c), (b, c), (b, 0), (a, 0)])
            vehicle.set_color(0, 0, 0)
            self.vehicle_transform = rendering.Transform()
            vehicle.add_attr(self.vehicle_transform)
            self.viewer.add_geom(vehicle)

            '''generate a arrow (actually it's a pentagon)'''
            arrow = rendering.make_polygon(
                [(-0.02 * self.scale, 0), (-0.02 * self.scale, 0.2 * self.scale), (0, 0.25 * self.scale),
                 (0.02 * self.scale, 0.2 * self.scale), (0.02 * self.scale, 0)])
            arrow.set_color(0, 0, 0)
            self.arrow_transform = rendering.Transform()
            arrow.add_attr(self.arrow_transform)
            self.viewer.add_geom(arrow)

        # 更新车辆和箭头的位置和旋转等
        state_inrendering_x = self.state[0][0] * self.scale
        state_inrendering_y = self.state[1][0] * self.scale + self.screen_length / 2
        self.vehicle_transform.set_translation(state_inrendering_x, state_inrendering_y)
        self.arrow_transform.set_translation(state_inrendering_x, state_inrendering_y)

        state_inrendering_phi = np.radians(np.degrees(self.state[2][0]) - 90.)
        self.vehicle_transform.set_rotation(state_inrendering_phi)
        self.arrow_transform.set_rotation(state_inrendering_phi)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        self.viewer.close()
