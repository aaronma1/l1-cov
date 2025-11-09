import tiles3 as tc
import numpy as np



# Q-learning agent

class MountainCarTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.state_upper = np.array([0.5, 0.07])
        self.state_lower = np.array([-1.2, -0.07])

        self.act_upper = np.array(1)
        self.act_lower = np.array(-1)

        self.iht_size = iht_size
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

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
        state_normalized = self.num_tiles  * np.divide(state - self.state_lower, self.state_upper - self.state_lower)
        tiles = tc.tiles(self.iht, self.num_tilings, list(state_normalized))
        return np.array(tiles)    

    def state_feature_shape(self):
        return [self.iht_size]
    
    def sa_feature_shape(self):
        return [3, self.iht_size]
    
    def enumerate_actions(self):
        return [0, 1, 2]

    def idx_from_action(self, action):
        if action[0] < -0.333:
            return 0 
        if action[0] > 0.333:
            return 2
        return 1
    
    def action_from_idx(self,act_idx):
        return [act_idx - 1]


class RandomAgent:

    def __init__(self, actions):
        self.actions = actions

    def select_action(self, state):
        return np.ranodm.choice(self.actions)




class QLearningAgent:

    def __init__(self,tilecoder, gamma, lr):
        self.tilecoder =  tilecoder
        self.Q_w = np.zeros(shape=tilecoder.sa_feature_shape()) 
        self.gamma = gamma
        self.lr = lr/self.tilecoder.num_tilings
        print(self.lr)
        self.epsilon = 1.0
    

    def select_action(self, state):
        return self.tilecoder.action_from_idx(
            np.argmax(
                self.Q_values(self.tilecoder.tile_state(state))
                ))

    def Q_values(self, state_features):
        return np.sum((self.Q_w)[:,state_features], axis=1) 

    def select_action_epsilon_greedy(self, state_features, epsilon=0.0):
        if np.random.rand() < epsilon:
            return self.tilecoder.action_from_idx(np.random.choice(self.tilecoder.enumerate_actions()))
        else:
            return self.tilecoder.action_from_idx(np.argmax(self.Q_values(state_features)))

    def Q_update(self, last_state_features, action_features, reward, state_features, terminated):
        td_target = reward
        if not terminated:
            td_target += self.gamma * np.max(self.Q_values(state_features))
        delta = td_target - self.Q_values(last_state_features)[action_features]
        self.Q_w[action_features, last_state_features] += self.lr * delta 
    
    # def learn_policy_from_transitions(transitions):
    #     for trajectory in transitions:


    def learn_policy_internal(self, env, T, reward_fn):
        ep_reward = 0
        last_state, info = env.reset()
        last_state_features = self.tilecoder.tile_state(last_state)
        action = self.select_action_epsilon_greedy(last_state_features)
        action_features = self.tilecoder.idx_from_action(action)
        for j in range(T):
            # take a step
            state, reward, terminated, truncated, info = env.step(action)
            if reward_fn != None:
                reward = reward_fn(state, action)
            ep_reward += reward
            state_features = self.tilecoder.tile_state(state)

            # compute td target and update Q
            self.Q_update(
                last_state_features,
                action_features,
                reward, state_features,
                terminated)

            last_state_features = state_features
            action = self.select_action_epsilon_greedy(state_features, self.epsilon)
            action_features = self.tilecoder.idx_from_action(action)

            if terminated or truncated:
                return ep_reward, terminated
        return ep_reward, False

    def learn_policy(self,env, T, episodes,  reward_fn = None, verbose=True):
        # train q learning agent by interacting with the environment

        for i in range(episodes):
            ep_reward = self.learn_policy_internal(env, T, reward_fn)

            if i % 25 == 0:
                self.epsilon *= 0.99
                print(ep_reward, self.epsilon)
                print(np.min(self.Q_w),np.max(self.Q_w))


class Trajectory:
    def __init__(self):
        self.last_state = []
        self.reward = []
        self.action = []
        self.next_state = []
        self.terminated = False
    
    def add_transition(self, state, action, reward, next_state, terminated = False):
        if self.terminated:
            return 

        self.last_state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)

        self.terminated = self.terminated | terminated
    

    def dump(self):
        return np.array(self.last_state), np.array(self.action), np.array(self.reward), np.array(self.next_state)

def collect_rollouts(env, agent, T, num_rollouts, reward_fn = None):

    def _collect_rollout(env, agent, T, reward_fn):
        trajectory = Trajectory()
        last_state, _ = env.reset()
        for i in range(T):
            action = agent.select_action(last_state)
            state, reward, terminated, truncated, info = env.step(action)
            if reward_fn != None:
                reward = reward_fn(state, action)
            
            trajectory.add_transition(last_state, action, reward, state, terminated)
            last_state = state

            if terminated or truncated:
                return trajectory

        return trajectory
          
    rollouts = []
    for i in range(num_rollouts):
        rollouts.append(_collect_rollout(env, agent, T, reward_fn))

    return rollouts



import itertools
import gymnasium as gym
from tilecoding import MountainCarTileCoder
if __name__ == "__main__":

    mctc = MountainCarTileCoder(iht_size=4096, num_tilings=16, num_tiles=16)
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)

    agent = QLearningAgent(mctc, gamma=0.99,lr=0.1)
    agent.learn_policy(env, 1000, 1000)
    

    rollouts = collect_rollouts(env, agent, 1000, 100)
 