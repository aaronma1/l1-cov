from lib.aggregation import get_aggregator
import numpy as np

class AggregatingActionCoder:
    def __init__(self, low, high, num_bins):
        """
        Discretizes Pendulum actions into `num_bins` evenly spaced torques.

        Args:
            num_bins (int): number of discrete actions
            action_low (float): min torque
            action_high (float): max torque
        """
        self.num_bins = num_bins
        self.action_low = low
        self.action_high = high

        # Create evenly spaced discrete actions
        self.actions = np.linspace(low, high, num_bins)

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
        return np.array(range(self.num_bins))


class DiscreteActionCoder:
    def __init__(self, num_bins=2):
        self.num_bins = num_bins

    def idx_from_act(self, action):
        return action
    
    def idx_to_act(self, idx):
        return idx
    
    def feature_shape(self):
        return [self.num_bins]

    def enumerate_actions(self):
        return np.array(range(self.num_bins)) 

class RandomAgent:
    def __init__(self, ac):
        self.actions = ac.enumerate_actions()
        self.ac = ac

    def select_action(self, state, epsilon=0.0):
        return self.ac.idx_to_act(np.random.choice(self.actions))


# TODO change random agent to use action coders
def get_random_agent(env_name):
    if env_name == "MountainCarContinuous-v0":
        mcac = AggregatingActionCoder(-1.0, 1.0, num_bins=7)
        return RandomAgent(pdac)

    if env_name == "CartPole-v1":
        return RandomAgent(DiscreteActionCoder(2))

    if env_name == "Acrobot-v1":
        return RandomAgent(DiscreteActionCoder(3))

    if env_name == "Pendulum-v1":
        pdac = AggregatingActionCoder(-2.0, 2.0, num_bins=7)
        return RandomAgent(pdac)


    
    assert False, f"Unknown env name {env_name}"

if __name__ == "__main__":
    pdac = AggregatingActionCoder(low=-2.0, high=2.0, num_bins=7)
    print([pdac.idx_to_act(i) for i in pdac.enumerate_actions()])


