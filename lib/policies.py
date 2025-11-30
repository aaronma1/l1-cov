from lib.aggregation import get_aggregator
import numpy as np
from scipy.sparse import lil_matrix, eye


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

class RandomAgent:
    def __init__(self, atc):
        self.idx = np.arange(len(atc))
        self.atc = atc

    def select_action(self, state, epsilon=0.0):
        return self.atc[np.random.choice(self.idx)]


# TODO change random agent to use action coders
def get_random_agent(env_name):
    if env_name == "MountainCarContinuous-v0":
        return RandomAgent(np.array([np.array([-1.0]), np.array([0.0]), np.array([1.0])]))

    if env_name == "CartPole-v1":
        return RandomAgent(np.array([0,1]))

    if env_name == "Pendulum-v1":
        return RandomAgent(
            np.array([
                np.array([-2.0]),
                np.array([-1.3]),
                np.array([-0.66]),
                np.array([-0.0]),
                np.array([0.66]),
                np.array([1.3]),
                np.array([2.0]),
            ])
        )
    
    assert False, f"Unknown env name {env_name}"






