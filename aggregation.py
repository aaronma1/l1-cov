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
        idx = np.ravel_multi_index(features.T, self.state_shape)

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
        features = np.array(np.unravel_index(idx, self.state_shape)).T
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


    def flatten_state_table(self,state_table):
        """
        convert a table of shape self.shape() to a list of shape (self.total_states(), )
        """
        return np.ravel(state_table)



    def unflatten_state_table(self, state_table):
        """
        convert a list of shape (self.total_states(), ) to a table of shape self.shape()
        """
        return state_table.reshape(self.shape())


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
        s_bins = [12,11]
        a_low = [-1.0]
        a_high= [1.0]
        a_bins = [3]
        return Aggregator(s_low, s_high, s_bins), SA_Aggregator(s_low, s_high, s_bins, a_low, a_high, a_bins)
    
    # if env_name == "CartPole-v0":
    #     s_low = []
    #     s_high =[]
    #     s_bins = []
    #     num_actions = 2

    #     return Aggregator(s_low, s_high, s_bins), SA_Aggregator_Disc(s_low, s_high, s_bins, num_actions)

    
    





if __name__ == "__main__":


    sa = SA_Aggregator([-5, -5], [5, 5], [1], [-1], [1], [3])

    print(sa.sa_to_idx(np.array([2,2]), np.array([1])))

    

    def spatial_tests():
        # Example 2D aggregator
        low = [0.0, 0.0]
        high = [1.0, 1.0]
        bins = [4, 5]  # 4 rows x 5 columns

        agg = Aggregator(low, high, bins)

        # Generate all discrete feature combinations in order
        features_list = []
        for i in range(bins[0]):      # first dim (row)
            for j in range(bins[1]):  # second dim (col)
                features_list.append([i, j])
        features_list = np.array(features_list)

        # Convert features to indices
        indices = agg.features_to_idx(features_list)

        # Print grid of indices
        print("Features -> Indices mapping:")
        for i in range(bins[0]):
            row_indices = indices[i * bins[1]:(i+1) * bins[1]]
            print(row_indices)

        # Also check top-left and bottom-right
        print("\nTop-left index:", indices[0])
        print("Bottom-right index:", indices[-1])
        print("Total indices:", agg.total_states())


        #test reshaping

        state_table = np.random.rand(*agg.shape())
        assert np.all(agg.unflatten_state_table(agg.flatten_state_table(state_table)) == state_table) 
        print("passed spatial tests")


    def random_state_tests():
        np.random.seed(42)  # for reproducibility

        # Aggregator setup
        low = [-1.0, -5.0, 0.0]
        high = [1.0, 5.0, 10.0]
        bins = [4, 5, 6]

        agg = Aggregator(low, high, bins)

        # Generate random states (including some out-of-bounds)
        N = 10000
        states = np.random.uniform(
            low=[l for l in low],   # deliberately extend below min
            high=[h for h in high], # deliberately extend above max
            size=(N, len(low))
        )

        # ---- Step 1: State → features ----
        features = agg.state_to_features(states)

        # Check all features are within valid range
        for d, b in enumerate(bins):
            assert np.all(features[:, d] >= 0), f"Dimension {d}: feature < 0"
            assert np.all(features[:, d] < b), f"Dimension {d}: feature >= bin count"

        # ---- Step 2: Features → index ----
        indices = agg.features_to_idx(features)

        # Check indices are within range
        total_states = agg.total_states()
        assert np.all(indices >= 0) and np.all(indices < total_states), "Index out of range"

        # ---- Step 3: Index → features (round-trip) ----
        recovered_features = agg.idx_to_features(indices)
        assert np.all(features == recovered_features), "Round-trip feature mismatch"

        # ---- Step 4: State → index → features → index round-trip ----
        recovered_indices = agg.features_to_idx(recovered_features)
        assert np.all(indices == recovered_indices), "Round-trip index mismatch"

        print(f"Randomized test passed for {N} states with {len(low)} dimensions!")

    # Run the random test

    random_state_tests()

    spatial_tests()