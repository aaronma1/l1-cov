import numpy as np
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

import gymnasium as gym
from gymnasium import wrappers
import copy

import gc

from lib.policies import AggregatingActionCoder, DiscreteActionCoder
import lib.environments as environments


class ReinforcePolicy(nn.Module):
    def __init__(self, gamma, lr, obs_dim, act_coder):
        super(ReinforcePolicy, self).__init__()

        self.device = "cpu"
        hidden_dim=64

        self.affine1 = nn.Linear(obs_dim, hidden_dim, device=self.device)
        self.middle = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.affine2 = nn.Linear(hidden_dim, act_coder.num_actions(), device=self.device)

        torch.nn.init.xavier_uniform_(self.affine1.weight)
        torch.nn.init.xavier_uniform_(self.middle.weight)
        torch.nn.init.xavier_uniform_(self.affine2.weight)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()

        self.gamma = gamma
        self.obs_dim = obs_dim
        self.action_dim = act_coder.num_actions()
        self.atc = act_coder


    def init(self, init_policy):
        print("init to policy")
        self.load_state_dict(init_policy.state_dict())

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.affine1(x))
        x = F.relu(self.middle(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores/2, dim=1)

    def get_probs(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        return probs

    def select_action(self, state, epsilon=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return self.atc.idx_to_act(action)


    def _select_action(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return self.atc.idx_to_act(action)


    def update_policy(self):
        R = 0
        policy_loss = [] #
        rewards = []

        #Get discounted rewards from the episode.
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward.float())

        self.optimizer.zero_grad(set_to_none=True)
        policy_loss = torch.cat(policy_loss).sum() 
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

        return policy_loss


    def learn_policy_internal(self, env, reward_fn, T=1000):
        ep_reward = 0
        last_state, info = env.reset()
        for t in range(T):
            action = self._select_action(last_state)
            state, reward, terminated, truncated, info = env.step(action)
            if reward_fn != None:
                reward = reward_fn(last_state, action, state, terminated)
            self.rewards.append(reward)
            ep_reward += reward

            last_state = state

            if terminated or truncated:
                return ep_reward
        
        return ep_reward



    def learn_policy(self, env, T, epochs, reward_fn = None, verbose=True, print_every=100, update_every=100):

        running_reward = 0
        running_loss = 0
        loss = 0
        for i_episode in range(epochs):
            ep_reward = self.learn_policy_internal(env, reward_fn, T)

            running_reward = running_reward * (1-0.05) + ep_reward * 0.05
            if (i_episode == 0):
                running_reward = ep_reward
            

            if i_episode % 100 == 0 and i_episode != 0:
                loss = self.update_policy()
            running_loss = running_loss * (1-.005) + loss*0.05

            gc.collect()

            if (i_episode) % print_every == 0 and verbose:
                print('Episode {}\tEpisode reward {:.2f}\tRunning reward: {:.2f}\tLoss: {:.2f}'.format(
                    i_episode, ep_reward, running_reward, running_loss))
    
    def learn_offline_policy(self, rollouts, reward_fn, offline_epochs, verbose=False):
        pass



def get_reinforce_agent(env_name, gamma, lr, a_bins=None):
    if env_name == "MountainCarContinuous-v0":

        _, _, a_low, a_high = environments.mountaincar_bounds()
        if a_bins == None:
            a_bins = [3]
        mcac = AggregatingActionCoder(a_low, a_high, num_bins=a_bins[0])
        return ReinforcePolicy(gamma, lr, 2, mcac)

    if env_name == "Pendulum-v1":
        _, _, a_low, a_high = environments.mountaincar_bounds()
        if a_bins == None:
            a_bins = [11]
        pdac = AggregatingActionCoder(a_low, a_high, num_bins=a_bins[0])
        return ReinforcePolicy(gamma, lr, 3, pdac )

    if env_name == "CartPole-v1":
        cpac = DiscreteActionCoder(2)
        return ReinforcePolicy(gamma, lr, 4, cpac)
    
    if env_name == "AcroBot-v1":
        acac = DiscreteActionCoder(3)
        return ReinforcePolicy(gamma, lr, 6, acac)

    assert False, f"unknown env name {env_name}"


from lib.trajectories import collect_rollouts


if __name__ == "__main__":
    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100)
    np.set_printoptions(precision=3, suppress=True)

    env_name = "Pendulum-v1"
    # Make environment.
    env = gym.make(env_name, render_mode="rgb_array")
    mtac = AggregatingActionCoder(-2.0, 2.0, num_bins=11)
    policy = ReinforcePolicy( 0.99, 0.001, 3, mtac)
    env1 = gym.wrappers.RecordVideo(env, video_folder="videos/reinforce_test", episode_trigger=lambda e: True)
    policy.learn_policy(env, None, 10000, 200)
    collect_rollouts(env1, policy, 200, 10, None, 0)
    policy.learn_policy(env, None, 10000, 200)
    collect_rollouts(env1, policy, 200, 10, None, 0)
    policy.learn_policy(env, None, 10000, 200)
    collect_rollouts(env1, policy, 200, 10, None, 0)
    policy.learn_policy(env, None, 10000, 200)
    collect_rollouts(env1, policy, 200, 10, None, 0)
