import numpy as np
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    '''
    Change the buffer to array and delete for loop.
    '''
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def get(self, idx):
        return self.storage[idx]

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def add_final_reward(self, final_reward, steps, delay=0):
        len_buffer = len(self.storage)
        for i in range(len_buffer - steps - delay, len_buffer - delay):
            item = list(self.storage[i])
            item[3] += final_reward
            self.storage[i] = tuple(item)

    def add_specific_reward(self, reward_vec, idx_vec):
        for i in range(len(idx_vec)):
            time_step_num = int(idx_vec[i])
            item = list(self.storage[time_step_num])
            item[3] += reward_vec[i]
            self.storage[time_step_num] = tuple(item)

    def sample_on_policy(self, batch_size, option_buffer_size):
        return self.sample_from_storage(batch_size, self.storage[-option_buffer_size:])

    def sample(self, batch_size):
        return self.sample_from_storage(batch_size, self.storage)

    @staticmethod
    def sample_from_storage(batch_size, storage):
        ind = np.random.randint(0, len(storage), size=batch_size)
        x, y, u, r, d, p = [], [], [], [], [], []
        for i in ind:
            X, Y, U, R, D, P = storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            p.append(np.array(P, copy=False))
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), \
               np.array(d).reshape(-1, 1), np.array(p).reshape(-1, 1)


# Expects tuples of (state, next_state, action, reward, done)
class ReplayBufferMat(object):
    '''
    Change the buffer to array and delete for loop.
    '''
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.data_size = 0

    def add(self, data):
        data = list(data)
        if 0 == len(self.storage):
            for item in data:
                self.storage.append(np.asarray(item).reshape((1, -1)))
        else:
            if self.storage[0].shape[0] < int(self.max_size):
                for i in range(len(data)):
                    self.storage[i] = np.r_[self.storage[i], np.asarray(data[i]).reshape((1, -1))]
            else:
                for i in range(len(data)):
                    self.storage[i][int(self.ptr)] = np.asarray(data[i]).reshape((1, -1))
                self.ptr = (self.ptr + 1) % self.max_size
        self.data_size = len(self.storage[0])

    def sample_on_policy(self, batch_size, option_buffer_size):
        return self.sample_from_storage(
            batch_size, start_idx = self.storage[0].shape[0] - option_buffer_size)

    def sample(self, batch_size):
        return self.sample_from_storage(batch_size)

    def sample_from_storage(self, batch_size, start_idx = 0):
        buffer_len = self.storage[0].shape[0]
        ind = np.random.randint(start_idx, buffer_len, size=batch_size)
        data_list = []
        # if buffer_len > 9998:
        #     print(buffer_len, ind)
        for i in range(len(self.storage)):
            # if buffer_len > 9998:
            #     print('{},shape:{}'.format(i, self.storage[i].shape))
            data_list.append(self.storage[i][ind])
        return tuple(data_list)

    def add_final_reward(self, final_reward, steps):
        self.storage[3][-steps:] += final_reward


def fifo_data(data_mat, data):
    data_mat[:-1] = data_mat[1:]
    data_mat[-1] = data
    return data_mat


def softmax(x):
    # This function is different from the Eq. 17, but it does not matter because
    # both the nominator and denominator are divided by the same value.
    # Equation 17: pi(o|s) = ext(Q^pi - max(Q^pi))/sum(ext(Q^pi - max(Q^pi))
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    e_x_sum = np.sum(e_x, axis=-1, keepdims=True)
    out = e_x / e_x_sum
    return out


def write_table(file_name, data):
    df = pd.DataFrame(data)
    df.to_excel(file_name + '.xls', index=False)



import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import glob
if torch.cuda.is_available():
	torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_layer=1,
				 l1_hidden_dim = 400, l2_hidden_dim = 300):
		super().__init__()
		self.gru = nn.Linear(state_dim, l1_hidden_dim)
		self.l2 = nn.Linear(l1_hidden_dim, l2_hidden_dim)
		self.l3 = nn.Linear(l2_hidden_dim, action_dim)
		self.max_action = max_action

	def forward(self, x):
		x = self.gru(x)
		x = x[:, -1, :]
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x))
		return x


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, l1_hidden_dim = 300, hidden_layer=1):
		super(Critic, self).__init__()

		# Q1 architecture
		self.gru1 = nn.Linear(state_dim, l1_hidden_dim)
		self.l1 = nn.Linear(action_dim, 100)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# Q2 architecture
		self.gru2 = nn.Linear(state_dim, l1_hidden_dim)
		self.l4 = nn.Linear(action_dim, 100)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, x, u):
		xg1 = self.gru1(x)
		xg1 = xg1[:, -1, :]
		u1 = F.relu(self.l1(u))
		xu1 = torch.cat([xg1, u1], 1)

		x1 = F.relu(self.l2(xu1))
		x1 = self.l3(x1)

		xg2 = self.gru2(x)
		xg2 = xg2[:, -1, :]
		u2 = F.relu(self.l4(u))
		xu2 = torch.cat([xg2, u2], 1)

		x2 = F.relu(self.l5(xu2))
		x2 = self.l6(x2)

		return x1, x2


class ATD3(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.max_action = max_action
		self.it = 0
		self.state_dim = state_dim


	def select_action(self, state):
		# (batch_size, seq_len, input_len)
		state = state.reshape(-1, state.shape[0], state.shape[1])
		state = torch.FloatTensor(state).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def cal_estimate_value(self, replay_buffer, eval_states=10000):
		x, _, u, _, _, _ = replay_buffer.sample(eval_states)
		x = x.reshape(x.shape[0], -1, self.state_dim)
		state = torch.FloatTensor(x).to(device) # (batch_size, seq_len * input_len) ->(batch_size, seq_len, input_len)
		action = torch.FloatTensor(u).to(device)
		Q1, Q2 = self.critic(state, action)
		# target_Q = torch.mean(torch.min(Q1, Q2))
		Q_val = 0.5 * (torch.mean(Q1) + torch.mean(Q2))
		return Q_val.detach().cpu().numpy(), \
			   torch.mean(Q1).detach().cpu().numpy(), torch.mean(Q2).detach().cpu().numpy()


	def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005,
			  policy_noise=0.2, noise_clip=0.5, policy_freq=2):
		self.it += 1
		# Sample replay buffer
		x, y, u, r, d, _ = replay_buffer.sample(batch_size)
		x = x.reshape(x.shape[0],  -1, self.state_dim)
		y = y.reshape(y.shape[0],  -1, self.state_dim)
		state = torch.FloatTensor(x).to(device)
		action = torch.FloatTensor(u).to(device)
		next_state = torch.FloatTensor(y).to(device)
		done = torch.FloatTensor(1 - d).to(device)
		reward = torch.FloatTensor(r).to(device)

		# Select action according to policy and add clipped noise
		noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
		noise = noise.clamp(-noise_clip, noise_clip)
		next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

		# Compute the target Q value
		target_Q1, target_Q2 = self.critic_target(next_state, next_action)
		# if torch.rand(1) > 0.5:
		# 	target_Q = target_Q1
		# else:
		# 	target_Q = target_Q2

		target_Q = torch.min(target_Q1, target_Q2)
		target_Q = reward + (done * discount * target_Q).detach()

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) - \
					  0.1 * F.mse_loss(current_Q1, current_Q2)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.it % policy_freq == 0:

			# Compute actor loss
			current_Q1, current_Q2 = self.critic(state, self.actor(state))
			actor_loss = -0.5 * (current_Q1 + current_Q2).mean()

			# actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

	def load(self, filename, directory):
		actor_path = '%s/%s_actor.pth' % (directory, filename)
		self.actor.load_state_dict(torch.load(actor_path))
		critic_path = '%s/%s_critic.pth' % (directory, filename)
		print('actor path: {}, critic path: {}'.format(actor_path, critic_path))
		self.critic.load_state_dict(torch.load(critic_path))

    
import numpy as np
import os
import datetime
import cv2
import torch
import glob
import shutil
from tqdm import tqdm
from scipy.stats import multivariate_normal
from tensorboardX import SummaryWriter
# import ATD3_RNN
# from .utils import ReplayBufferMat

class Solver(object):
    def __init__(self, args, env):
        print(args)
        self.args = args
        self.env = env
        self.file_name = ''
        self.result_path = "results"

        self.evaluations = []
        self.estimate_Q_vals = []
        self.Q1_vec = []
        self.Q2_vec = []
        self.true_Q_vals = []
        self.Q_ae_mean_vec = []
        self.Q_ae_std_vec = []


        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        # Initialize policy
        policy = ATD3(state_dim, action_dim, max_action)
        self.policy = policy
        print('-------Current policy: {} --------------'.format(self.policy.__class__.__name__))
        self.replay_buffer = ReplayBufferMat(max_size=args.max_timesteps)
        self.total_timesteps = 0
        self.pre_num_steps = self.total_timesteps
        self.timesteps_since_eval = 0
        self.timesteps_calc_Q_vale = 0
        self.best_reward = 0.0

        self.env_timeStep = 4

    def train_once(self):
        if self.total_timesteps != 0:
            self.policy.train(self.replay_buffer, self.args.batch_size, self.args.discount,
                              self.args.tau, self.args.policy_noise, self.args.noise_clip,
                              self.args.policy_freq)

    def eval_once(self):
        self.pbar.update(self.total_timesteps - self.pre_num_steps)
        self.pre_num_steps = self.total_timesteps

        # Evaluate episode
        if self.timesteps_since_eval >= self.args.eval_freq:
            self.timesteps_since_eval %= self.args.eval_freq
            avg_reward = evaluate_policy(self.env, self.policy, self.args)
            self.evaluations.append(avg_reward)
            self.writer_test.add_scalar('ave_reward', avg_reward, self.total_timesteps)

            if self.args.save_all_policy:
                self.policy.save(
                    self.file_name + str(int(int(self.total_timesteps/self.args.eval_freq)* self.args.eval_freq)),
                    directory=self.log_dir)

            np.save(self.log_dir + "/test_accuracy", self.evaluations)
            write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))
            if self.best_reward < avg_reward:
                self.best_reward = avg_reward
                print("Best reward! Total T: %d Episode T: %d Reward: %f" %
                      (self.total_timesteps, self.episode_timesteps, avg_reward))
                self.policy.save(self.file_name, directory=self.log_dir)

    def reset(self):
        # Reset environment
        self.obs = self.env.reset()
        self.obs_vec = np.dot(np.ones((self.args.seq_len, 1)), self.obs.reshape((1, -1)))
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.still_steps = 0

    def train(self):
        # Evaluate untrained policy
        self.evaluations = [evaluate_policy(self.env, self.policy, self.args)]
        self.log_dir = '{}/{}'.format(self.result_path, self.args.log_path)
        print("---------------------------------------")
        print("Settings: %s" % self.log_dir)
        print("---------------------------------------")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # TesnorboardX
        self.writer_test = SummaryWriter(logdir=self.log_dir)
        self.pbar = tqdm(total=self.args.max_timesteps, initial=self.total_timesteps, position=0, leave=True)
        done = True
        while self.total_timesteps < self.args.max_timesteps:
            # ================ Train =============================================#
            self.train_once()
            # ====================================================================#
            if done:
                self.eval_once()
                self.reset()
                done = False
            # Select action randomly or according to policy
            if self.total_timesteps < self.args.start_timesteps:
                action = self.env.action_space.sample()
                p = 1
            else:
                if 'RNN' in self.args.policy_name:
                    action = self.policy.select_action(np.array(self.obs_vec))
                elif 'SAC' in self.args.policy_name or 'HRL' in self.args.policy_name:
                    action = self.policy.select_action(np.array(self.obs), eval=False)
                else:
                    action = self.policy.select_action(np.array(self.obs))

                noise = np.random.normal(0, self.args.expl_noise,
                                         size=self.env.action_space.shape[0])
                if self.args.expl_noise != 0:
                    action = (action + noise).clip(
                        self.env.action_space.low, self.env.action_space.high)

                if 'HRL' in self.args.policy_name:
                    p_noise = multivariate_normal.pdf(
                        noise, np.zeros(shape=self.env.action_space.shape[0]),
                        self.args.expl_noise * self.args.expl_noise * np.identity(noise.shape[0]))
                    if 'SHRL' in self.args.policy_name:
                        p = (p_noise * softmax(self.policy.option_prob))[0]
                    else:
                        p = (p_noise * softmax(self.policy.q_predict)[self.policy.option_val])[0]

            state_id = 0
            # Perform action
            new_obs, reward, done, _ = self.env.step(action)

            self.episode_reward += reward

            done_bool = 0 if self.episode_timesteps + 1 == self.env._max_episode_steps else float(done)

            if 'RNN' in self.args.policy_name:
                # Store data in replay buffer
                new_obs_vec = fifo_data(np.copy(self.obs_vec), new_obs)
                self.replay_buffer.add((np.copy(self.obs_vec), new_obs_vec, action, reward, done_bool, state_id))
                self.obs_vec = fifo_data(self.obs_vec, new_obs)
            else:
                self.replay_buffer.add((self.obs, new_obs, action, reward, done_bool, state_id))

            self.obs = new_obs
            self.episode_timesteps += 1
            self.total_timesteps += 1
            self.timesteps_since_eval += 1
            self.timesteps_calc_Q_vale += 1

        # Final evaluation
        self.eval_once()
        self.env.reset()

    def eval_only(self, is_reset = True):
        video_dir = '{}/video_all'.format(self.result_path)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        files = './' + self.result_path + '/' + self.args.log_path
        for model_path in os.listdir(files):
            if (model_path == ".ipynb_checkpoints"):
                continue
            self.policy.load("%s" % (self.file_name + self.args.load_policy_idx), directory=files + '/' + model_path)
            for _ in range(1):
                if self.args.save_video:
                    print("check")
                    video_name = video_dir + '/{}_{}_{}.mp4'.format(
                        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        self.file_name, self.args.load_policy_idx)
                obs = self.env.reset()
                if 'RNN' in self.args.policy_name:
                    obs_vec = np.dot(np.ones((self.args.seq_len, 1)), obs.reshape((1, -1)))

                obs_mat = np.asarray(obs)
                done = False

                if self.args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                    img = self.env.render(mode='rgb_array')
                    out_video = cv2.VideoWriter(video_name, fourcc, 50.0, (img.shape[1], img.shape[0]))

                while not done:
                    if 'RNN' in self.args.policy_name:
                        action = self.policy.select_action(np.array(obs_vec))
                    else:
                        action = self.policy.select_action(np.array(obs))

                    obs, reward, done, _ = self.env.step(action)

                    if 'RNN' in self.args.policy_name:
                        obs_vec = fifo_data(obs_vec, obs)

                    if 0 != self.args.state_noise:
                        obs[8:20] += np.random.normal(0, self.args.state_noise, size=obs[8:20].shape[0]).clip(
                            -1, 1)

                    obs_mat = np.c_[obs_mat, np.asarray(obs)]

                    if self.args.save_video:
                        img = self.env.render(mode='rgb_array')
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out_video.write(img)
                    elif self.args.render:
                        self.env.render(mode='rgb_array')

                if self.args.save_video:
                    out_video.release()
        if is_reset:
            self.env.reset()

# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, args, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        if 'RNN' in args.policy_name:
            obs_vec = np.dot(np.ones((args.seq_len, 1)), obs.reshape((1, -1)))
        done = False
        while not done:
            if 'RNN' in args.policy_name:
                action = policy.select_action(np.array(obs_vec))
            else:
                action = policy.select_action(np.array(obs))

            obs, reward, done, _ = env.step(action)
            if 'RNN' in args.policy_name:
                obs_vec = fifo_data(obs_vec, obs)
            avg_reward += reward
    avg_reward /= eval_episodes
    # print ("---------------------------------------"                      )
    # print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    # print ("---------------------------------------"                      )
    return avg_reward

import os
import sys
import pybullet_envs, gym
import argparse
import numpy as np

sys.argv=['']
del sys

def test_env(env):
    env.reset()
    state = np.random.rand(22)
    print(env.set_robot(state) - state)
    while True:
        env.render()

def convert_args_to_bool(args):
    args.eval_only = (args.eval_only in ['True', True])
    args.render = (args.render in ['True', True])
    args.save_video = (args.save_video in ['True', True])
    args.save_all_policy = (args.save_all_policy in ['True', True])
    return args

def main(args):
    env = gym.make(args.env_name)
    env.reset()
    if args.render:
        env.render('human')
    sol = Solver(args, env)
    a = int(input())
    if a:
        args.buffer_size = 2e5
        args.max_timesteps = 2e5
        sol.train()
    else:
        args.save_video = True
        sol.eval_only()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default='ATD3_RNN')  # Policy name
    parser.add_argument("--env_name", default="Walker2DBulletEnv-v0")  # OpenAI gym environment name
    parser.add_argument("--log_path", default='runs')

    parser.add_argument("--eval_only", default=True)
    parser.add_argument("--render", default=True)
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--save_all_policy", default=False)
    parser.add_argument("--load_policy_idx", default='')
    parser.add_argument("--reward_name", default='')
    parser.add_argument("--seq_len", default=2, type=int)
    parser.add_argument("--option_num", default=4, type=int)
    parser.add_argument("--buffer_size", default=3e5, type=int)
    parser.add_argument("--lr", default=1e-3, type=int)
    parser.add_argument("--seed", default=1, type=int) # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate.
    # 5e3 for not evaluating Q value; 3e4 for evaluating Q value
    parser.add_argument("--max_timesteps", default=3e5, type=int)  # Max time steps to run environment for

    parser.add_argument("--disc_ratio", default=0.2, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--state_noise", default=0, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates

    args = parser.parse_args()
    args = convert_args_to_bool(args)
    main(args)

  
