import numpy as np
from plotting import plot_heatmap
from numba import jit, njit
import lib.tiles3 as tc
import lib.environments as environments

from lib.policies import AggregatingActionCoder, DiscreteActionCoder
import random
# Q-learning agent
class TileCoder:
    def __init__(self, low, high, num_tilings, num_tiles):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        high -- (n) upper value of state
        low -- (n) lower value of state
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht_size =  4 * num_tilings * (num_tiles ** len(high))
        self.state_upper = np.array(high)
        self.state_lower = np.array(low)

        self.iht = tc.IHT(self.iht_size)
    
        self.n_tilings = num_tilings
        self.n_tiles = num_tiles

    def tile_state(self, state):
        """
        Takes in a position and velocity from the mountaincar environment
        and returns a numpy array of active tiles.
        
        Arguments:
        position -- float, the position of the agent between -1.2 and 0.5
        velocity -- float, the velocity of the agent between -0.07 and 0.07
        returns:
        tiles - np.array, active tiles
        """
        state_normalized = self.n_tiles  * np.divide(state - self.state_lower, self.state_upper - self.state_lower)
        tiles = tc.tiles(self.iht, self.n_tilings, list(state_normalized))
        return np.array(tiles)    

    def feature_shape(self):
        return [self.iht_size]
    
    def num_tilings(self):
        return self.n_tilings

class QLearningAgent:

    def __init__(self, tilecoder, action_coder, gamma, lr, max_rew=1):
        self.stc =  tilecoder
        self.atc = action_coder
        self.Q_w = np.ones(shape=action_coder.feature_shape() + tilecoder.feature_shape() ) * max_rew * 1/(1-gamma)
        self.gamma = gamma
        self.lr = lr/(self.stc.num_tilings())
        # all possible actions
        self.actions = self.atc.enumerate_actions()

        self.epsilon=1.0

    def Q_values(self, state_features):
        return np.sum((self.Q_w)[:,state_features], axis=-1) 

    def select_action(self, state, epsilon=0.0):
       return self._select_action(self.stc.tile_state(state), epsilon)

    def _select_action(self, state_features, epsilon=0.0):
        if np.random.rand() < epsilon:
            return self.atc.idx_to_act(np.random.choice(self.actions))
        else:
            Q = self.Q_values(state_features)
            return self.atc.idx_to_act(np.random.choice(np.where(Q == Q.max())[0]))


    def Q_update(self, last_state_features, action_features, reward, state_features, terminated):
        td_target = reward
        if not terminated:
            td_target += self.gamma * np.max(self.Q_values(state_features))

        delta = td_target - self.Q_values(last_state_features)[action_features]
        self.Q_w[action_features][last_state_features] += self.lr * delta

    def learn_policy_internal(self, env, T, reward_fn):
        ep_reward = 0
        last_state, info = env.reset()

        last_state_features = self.stc.tile_state(last_state)
        action = self._select_action(last_state_features, epsilon=self.epsilon)
        action_features = self.atc.idx_from_act(action)
        
        for _ in range(T):
            # take a step
            state, reward, terminated, truncated, info = env.step(action)
            if reward_fn != None:
                reward = reward_fn(last_state, action, state, terminated)
            ep_reward += reward
            state_features = self.stc.tile_state(state)
            # compute td target and update Q
            self.Q_update(
                last_state_features,
                action_features,
                reward,
                state_features,
                terminated
            )
            # update state and select action
            last_state_features = state_features
            action = self._select_action(state_features, self.epsilon)
            action_features = self.atc.idx_from_act(action)

            if terminated or truncated:
                return ep_reward, terminated

        return ep_reward, False

    def learn_policy(self,env, T, episodes,  reward_fn = None, 
                     verbose=True, print_every=200, epsilon_decay=0.999, decay_every=1, epsilon_start = 1.0
                     ):
        self.epsilon = epsilon_start
        running_reward = 0
        for i in range(episodes):
            ep_reward, terminated = self.learn_policy_internal(env, T, reward_fn)

            if i == 0:
                running_reward = ep_reward
            else:
                running_reward = running_reward * 0.95 + ep_reward * 0.05

            # decay epsilon
            if i % decay_every == 0:
                self.epsilon *= epsilon_decay
            # print if debuging
            if verbose and (i+1) % print_every == 0:
                # print("ep reward", ep_reward, terminated, self.epsilon, np.max(self.Q_w), np.min(self.Q_w))
                print(f"epoch {i} running reward: {running_reward}, ep_reward: {ep_reward} terminated: {terminated}, epsilon: {self.epsilon} max_q: {np.max(self.Q_w)}, min_q: {np.min(self.Q_w)}")

        

    # learn offline
    def learn_offline_policy(self, rollouts, offline_epochs, reward_fn = None, verbose=False):

        trajectory_tiles = []
        for trajectory in rollouts:
            s,a,r,s_prime = trajectory.dump()
            n = r.size
            s_tiles = np.zeros(shape = (n, self.stc.num_tilings()), dtype=np.int32)
            s_prime_tiles = np.zeros(shape = (n, self.stc.num_tilings()), dtype=np.int32)
            a_idx = np.zeros(shape = n, dtype=np.int32)

            for t in range(n):
                s_tiles[t] = self.stc.tile_state(s[t])
                s_prime_tiles[t] = self.stc.tile_state(s_prime[t])
                a_idx[t] = self.atc.idx_from_act(a[t])
                if reward_fn != None:
                    r[t] = reward_fn(s[t], a[t], s_prime[t], trajectory.terminated and (t == s.shape[0]-1))
            
            trajectory_tiles.append((s_tiles, a_idx, r, s_prime_tiles, trajectory.terminated))
        random.shuffle(trajectory_tiles)

        @njit
        def _internal(Q, trajectory_tiles, offline_epochs, lr, gamma):
            avg_reward = 0
            for _ in range(offline_epochs):
                for s,a,r,s_prime,terminated in trajectory_tiles:

                    for t in range(s.shape[0]):
                        avg_reward += r[t]

                        td_target = r[t]
                        if not (t == s.shape[0]-1 and terminated):
                            # td_target += self.gamma * np.max(self.Q_values(state_features))
                            td_target += gamma * np.max(np.sum(Q[:, s_prime[t]], axis=-1))
                        #delta = td_target - self.Q_values(last_state_features)[action_features]
                        delta = td_target - np.sum(Q[:, s[t]], axis=-1)[a[t]]
                        Q[a[t]][s[t]] += lr * delta
            return Q, avg_reward/(len(trajectory_tiles) *offline_epochs)

        self.Q_w, avg_reward = _internal(self.Q_w, trajectory_tiles, offline_epochs, self.lr, self.gamma )

        if verbose:
            print(f"average reward for offline learning {avg_reward}")
        

def get_qlearning_agent(env_name, gamma, lr, a_bins = None, max_rew=1):
    if env_name == "MountainCarContinuous-v0":
        state_low, state_high, act_low, act_high = environments.mountaincar_bounds()
        mctc = TileCoder(state_low,state_high, num_tilings=16, num_tiles=8)
        if a_bins == None:
            a_bins = [3]
        mcac = AggregatingActionCoder(act_low, act_high, num_bins=a_bins)
        return QLearningAgent(mctc,mcac, gamma,lr ,max_rew=max_rew)

    if env_name == "Pendulum-v1":
        state_low, state_high, act_low, act_high = environments.pendulum_bounds()
        pdtc = TileCoder(low=state_low, high=state_high, num_tilings=32, num_tiles=16)
        if a_bins == None:
            a_bins = [3]
        pdac = AggregatingActionCoder(act_low, act_high, num_bins=a_bins)
        return QLearningAgent(pdtc, pdac, gamma=gamma, lr=lr, max_rew=max_rew)

    if env_name == "CartPole-v1":
        state_low, state_high = environments.cartpole_bounds()
        cptc = TileCoder(low=state_low, high=state_high,num_tilings=32, num_tiles=4) 
        cpac = DiscreteActionCoder(2)
        return QLearningAgent(cptc, cpac, gamma=gamma, lr=lr, max_rew=max_rew)
    
    if env_name == "Acrobot-v1":
        state_low, state_high = environments.acrobot_bounds()
        actc = TileCoder(state_low, state_high, 64, 8) 
        acac = DiscreteActionCoder(3)
        return QLearningAgent(actc, acac, gamma=gamma, lr=lr, max_rew=max_rew)

    assert False, f"unknown env name {env_name}"

import itertools
import gymnasium as gym
from lib.trajectories import collect_rollouts
if __name__ == "__main__":
    # env_name = "MountainCarContinuous-v0"
    # env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    np.set_printoptions(precision=4)

    env_name = "MountainCarContinuous-v0"
    env_name = "Acrobot-v1"

    env = gym.make(env_name, render_mode="rgb_array") 
    env_rec = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)

    agent = get_qlearning_agent(env_name, 0.999, 0.01)

    agent.learn_policy(env, 200, 20000, epsilon_start=1.0, epsilon_decay=0.999, decay_every=20, print_every=100)
    collect_rollouts(env_rec, agent, 200, 5, epsilon=0.0)
    agent.learn_policy(env, 200, 20000, epsilon_start=0.300, epsilon_decay=0.999, decay_every=20, print_every=100)
    collect_rollouts(env_rec, agent,  200, 5, epsilon=0.1)
    agent.learn_policy(env, 200, 20000, epsilon_start=0.1, epsilon_decay=0.999, decay_every=20, print_every=100)
    collect_rollouts(env_rec, agent, 200, 5, epsilon=0.1)
    env_rec.close()
 