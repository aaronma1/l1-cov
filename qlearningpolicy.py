import tiles3 as tc
import numpy as np



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




class MTCCActionCoder:
    
    def num_actions(self):
        return 3

    def feature_shape(self):
        return [3]
    
    def idx_from_act(self, action):
        if action[0] < -0.333:
            return 0
        if action[0] > 0.333:
            return 2
        return 1

    def idx_to_act(self, idx):
        if idx == 0:
            return [-1]
        if idx == 1:
            return [0]
        return [1]
    
    def enumerate_actions(self):
        return [0,1,2]



class PendulumActionCoder:
    def __init__(self, num_bins=5, action_low=-2.0, action_high=2.0):
        """
        Discretizes Pendulum actions into `num_bins` evenly spaced torques.

        Args:
            num_bins (int): number of discrete actions
            action_low (float): min torque
            action_high (float): max torque
        """
        self.num_bins = num_bins
        self.action_low = action_low
        self.action_high = action_high

        # Create evenly spaced discrete actions
        self.actions = np.linspace(action_low, action_high, num_bins)

    def num_actions(self):
        return self.num_bins

    def feature_shape(self):
        return [self.num_bins]

    def idx_from_act(self, action):
        """
        Map a continuous action to discrete index.
        """
        action = np.clip(action, self.action_low, self.action_high)
        idx = np.argmin(np.abs(self.actions - action))
        return idx

    def idx_to_act(self, idx):
        """
        Map discrete index back to continuous action.
        """
        return [self.actions[idx]]

    def enumerate_actions(self):
        """
        Return all discrete action indices.
        """
        return list(range(self.num_bins))


class DiscreteActionCoder:
    def __init__(self, num_bins=2):
        self.num_bins = num_bins
        pass

    def idx_from_act(self, action):
        return action
    
    def idx_to_act(self, idx):
        return idx
    
    def feature_shape(self):
        return [self.num_bins]

    def enumerate_actions(self):
        return list(range(self.num_bins)) 



    





class QLearningAgent:

    def __init__(self, tilecoder, action_coder, gamma, lr):
        self.stc =  tilecoder
        self.atc = action_coder

        self.Q_w = np.zeros(shape=action_coder.feature_shape() + tilecoder.feature_shape() ) 
        print(self.Q_w.shape)
        self.gamma = gamma
        self.lr = lr/(self.stc.num_tilings())
        print(self.lr)

        # all possible actions
        self.actions = self.atc.enumerate_actions()

        self.epsilon = 1.0
    


    def Q_values(self, state_features):
        return np.sum((self.Q_w)[:,state_features], axis=-1) 


    def select_action(self, state, epsilon=0.0):
       return self._select_action(self.stc.tile_state(state), epsilon)

    def _select_action(self, state_features, epsilon=0.0):
        if np.random.rand() < epsilon:
            return self.atc.idx_to_act(np.random.choice(self.actions))
        else:
            return self.atc.idx_to_act(np.argmax(self.Q_values(state_features)))

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
                reward = reward_fn(last_state, action)

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

            last_state_features = state_features
            action = self._select_action(state_features, self.epsilon)
            action_features = self.atc.idx_from_act(action)

            if terminated or truncated:
                return ep_reward, terminated
            
        return ep_reward, False

    def learn_policy(self,env, T, episodes,  reward_fn = None, 
                     verbose=True, print_every=200, epsilon_decay=0.999, decay_every=1
                     ):
        # train q learning agent by interacting with the environment

        self.epsilon = 1.0

        for i in range(episodes):
            ep_reward = self.learn_policy_internal(env, T, reward_fn)

            if i % decay_every == 0:
                self.epsilon *= epsilon_decay


            if verbose and i % print_every == 0:
                print(ep_reward, self.epsilon)
                print(np.min(self.Q_w),np.max(self.Q_w))
        

    # learn offline
    def learn_offline_policy():
        pass




def get_qlearning_agent(env_name, gamma, lr):
    
    if env_name == "MountainCarContinuous-v0":
        print("mountain car q learning")
        mctc = TileCoder([-1.2, -0.07],[0.6, 0.07], num_tilings=8, num_tiles=8)
        mcac = MTCCActionCoder()
        return QLearningAgent(mctc,mcac, gamma,lr )

    if env_name == "CartPole-v1":
        cptc = TileCoder(low=[-2.5, -3.5, -0.3, -4.0], high=[2.5,3.5, 0.3, 4.0],num_tilings=32, num_tiles=16) 
        cpac = DiscreteActionCoder(2)
        return QLearningAgent(cptc, cpac, gamma=gamma, lr=lr)

    if env_name == "Pendulum-v1":
        pdtc = TileCoder(low=[-1.0, -1.0, -8.0], high=[1.0,1.0,8.0], num_tilings=32, num_tiles=16 )
        pdac = PendulumActionCoder()
        return QLearningAgent(pdtc, pdac, gamma=gamma, lr=lr)
    



import itertools
import gymnasium as gym
if __name__ == "__main__":
    # env_name = "MountainCarContinuous-v0"
    # env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")

    env_name = "MountainCarContinuous-v0"
    env = gym.make(env_name, render_mode="rgb_array") 

    agent = get_qlearning_agent(env_name, 0.999, 0.1)
    agent.learn_policy(env, 200, 10000)
    
    env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)
 