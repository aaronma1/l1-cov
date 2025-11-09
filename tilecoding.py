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
    # State → discrete features
    # ---------------------------
    def state_to_features(self, states):
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
    # Features → flat index
    # ---------------------------
    def features_to_idx(self, features):
        """
        Convert discrete features → flat index.
        Vectorized: features shape = (N, dims) or (dims,)
        """
        features = np.atleast_2d(features)
        idx = np.ravel_multi_index(features.T, self.state_shape)
        return idx if idx.size > 1 else idx[0]

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
    def state_to_idx(self, states):
        features = self.state_to_features(states)
        return self.features_to_idx(features)

    # ---------------------------
    # Utilities
    # ---------------------------
    def shape(self):
        return self.state_shape

    def total_states(self):
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


def get_aggregator(env_name, bin_res=1):

    if env_name == "MountainCarContinuous-v0":
        return Aggregator([-1.2, -0.07], [0.6, 0.07] [12*bin_res, 11*bin_res])
    



if __name__ == "__main__":

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