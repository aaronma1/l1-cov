from ctypes.util import test
import numpy as np

class Aggregator:
    def __init__(self, low, high, bins):
        """
        Vectorized state aggregator / tiling.

        Args:
            low: array-like, min values per dimension
            high: array-like, max values per dimension
            bins: array-like, number of bins per dimension
        """
        self.low = np.array(low, dtype=float)
        self.high = np.array(high, dtype=float)
        self.bins = np.array(bins, dtype=int)
        self.state_shape = tuple(self.bins)
        self.dims = len(low)
        # Precompute bin edges per dimension
        self.edges = [np.linspace(l, h, b+1) for l, h, b in zip(self.low, self.high, self.bins)]


    # ---------------------------
    # Features → flat index
    # ---------------------------
    def features_to_idx(self, features):
        """
        Convert discrete features → flat index.
        Vectorized: features shape = (N, dims) or (dims,)
        """
        return_array = len(features.shape) > 1

        features = np.atleast_2d(features)
        print("features", features)
        idx = np.ravel_multi_index(features.T, self.state_shape, order='F')
        print("idx", idx)

        if return_array:
            return idx
        else:
            return idx[0]

    # ---------------------------
    # Flat index → features
    # ---------------------------
    def idx_to_features(self, idx):
        """
        Convert flat index → discrete features.
        Vectorized: idx can be scalar or array
        Returns: shape (N, dims) if idx is array
        """
        idx = np.atleast_1d(idx)
        features = np.array(np.unravel_index(idx, self.state_shape, order='F')).T
        return features if features.shape[0] > 1 else features[0]

    # ---------------------------
    # State → flat index
    # ---------------------------
    def s_to_idx(self, states):
        features = self.s_to_features(states)
        return self.features_to_idx(features)
    # ---------------------------
    # State → discrete features
    # ---------------------------
    def s_to_features(self, states):
        """
        Convert continuous states → discrete features (per-dimension bin indices).
        Vectorized: states shape = (N, dims) or (dims,)
        Returns: array of shape (N, dims)
        """
        states = np.atleast_2d(states)
        N = states.shape[0]

        # Vectorized discretization
        features = np.zeros_like(states, dtype=int)
        for d, edge in enumerate(self.edges):
            features[:, d] = np.digitize(states[:, d], edge) - 1
            features[:, d] = np.clip(features[:, d], 0, len(edge)-2)
        return features if N > 1 else features[0]

    # ---------------------------
    # Utilities
    # ---------------------------
    def shape(self):
        return self.state_shape

    def num_states(self):
        return np.prod(self.state_shape)


    def flatten_s_table(self,state_table):
        """
        convert a table of shape self.shape() to a list of shape (self.total_states(), )
        """
        return np.ravel(state_table, order='F')



    def unflatten_s_table(self, state_table):
        """
        convert a list of shape (self.total_states(), ) to a table of shape self.shape()
        """
        return state_table.reshape(self.shape(), order='F')
            
            



# SA aggregator for discrete actions
class SA_Aggregator_Disc:
    def __init__(self, state_low, state_high, state_bins, num_actions):
        self.state_space = state_bins
        self.agg_internal = Aggregator(state_low, state_high, state_bins)
        self.num_actions = num_actions

    def sa_to_idx(self, states, actions):
        """
        inputs: 
            States: np.array of shape (n, state_shape) or (state_shape, )
            Actions: np.array of shape (n, ) or int
    
        outputs:
        """
        return_int = type(actions) == int

        states = np.atleast_2d(states)
        actions = np.atleast_1d(actions)

        state_features = self.agg_internal.state_to_idx(states)

        if return_int:
            return (state_features + actions * self.agg_internal.num_states())[0]
        return state_features + actions * self.agg_internal.num_states()

    def sa_to_features(self, states, actions):

        return_int = type(actions) == int

        states = np.atleast_2d(states)
        actions = np.atleast_1d(actions)

        state_features = self.agg_internal.s_to_features(states)
        return np.concatenate([state_features, actions], axis=-1)
        


    def idx_to_features(self, idxs):
        idxs = np.atleast_1d(idxs)

        states = idxs % self.agg_internal.num_states()
        actions = idxs // self.agg_internal.num_states()

        return np.concatenate([self.agg_internal.idx_to_features(states), actions], axis=-1)

    def features_to_idx(self, features):
        pass

    def flatten_sa_table(self, sa_table):
        pass
        
    def unflatten_sa_table(self, sa_table):
        pass
    
        """
        transforms SA table for plotting.
        state_table: (state_dims, act_dims) -> (state_dims[0] * prod(act_dims), state_dims[1:])
        """

    

class SA_Aggregator:

    def __init__(self, state_low, state_high, state_bins, act_low, act_high, act_bins):
        self.state_space = state_bins
        self.act_space = act_bins

        self.agg_internal = Aggregator(state_low + act_low, state_high+act_high, state_bins + act_bins)

    def sa_to_idx(self, states, actions):
        states = np.atleast_2d(states)
        actions = np.atleast_2d(actions)
        return self.agg_internal.s_to_idx(np.concatenate([states, actions], axis=-1))

    def sa_to_features(self, states, actions):
        states = np.atleast_2d(states)
        actions = np.atleast_2d(actions)
        return self.agg_internal.s_to_features(np.concatenate([states, actions], axis=-1))
    

    def idx_to_features(self, idxs):
        feat = self.agg_internal.idx_to_features(idxs)
        return feat[:, : len(self.state_space)], feat[:, len(self.state_space):]

    def shape(self):
        return self.state_space+ self.act_space



    def flatten_sa_table(self, sa_table):
        pass
        
    def unflatten_sa_table(self, sa_table):
        """
        transforms SA table for plotting.
        state_table: (state_dims, act_dims) -> (state_dims[0] * prod(act_dims), state_dims[1:])
        """


class S_Reward:

    def __init__(self, agg, reward_table):
        self.agg = agg
        self.reward_table = reward_table


    def __call__(self, state, action):
        return self.reward_table[self.agg.s_to_features(state)]
        
class SA_Reward:

    def __init__(self, agg, reward_table):
        self.agg = agg
        self.reward_table = reward_table

    def __call__(self, state, action):
        return self.reward_table[self.agg.sa_to_features(state, action)]









def get_aggregator(env_name, bin_res=1):
    if env_name == "MountainCarContinuous-v0":
        s_low = [-1.2, -0.07]
        s_high = [0.6, 0.07]
        s_bins = [12*bin_res,11*bin_res]
        # s_bins = [4, 4]
        a_low = [-1.0]
        a_high= [1.0]
        a_bins = [3*bin_res]
        return Aggregator(s_low, s_high, s_bins), SA_Aggregator(s_low, s_high, s_bins, a_low, a_high, a_bins)
    
    # if env_name == "CartPole-v0":
    #     s_low = []
    #     s_high =[]
    #     s_bins = []
    #     num_actions = 2

    #     return Aggregator(s_low, s_high, s_bins), SA_Aggregator_Disc(s_low, s_high, s_bins, num_actions)

    
    



import plotting

if __name__ == "__main__":

# quick self-consistency test
    agg = Aggregator([-1, -1], [1, 1], [5, 6])
    test_states = np.random.uniform(-1, 1, (200, 2))
    idx = agg.s_to_idx(test_states)
    features = agg.idx_to_features(idx)
    assert np.array_equal(agg.features_to_idx(features), idx)
    assert np.array_equal(agg.idx_to_features(idx),features)


    test_state_table = np.arange(30)

    test_state_table1 = np.zeros((5,6))
    for i in range(30):
        test_state_table1[i%5, i//5] = i

        assert agg.features_to_idx(np.array([i%5, i//5])) == i
        assert np.all(agg.idx_to_features(i) == np.array([i%5, i//5]))

    print(agg.flatten_s_table(test_state_table1))
    assert np.array_equal(agg.flatten_s_table(test_state_table1), test_state_table)
