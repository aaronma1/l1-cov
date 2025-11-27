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
        self.edges = [
            np.linspace(l, h, b + 1) for l, h, b in zip(self.low, self.high, self.bins)
        ]

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
        idx = np.ravel_multi_index(features.T, self.state_shape, order="F")

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
        features = np.array(np.unravel_index(idx, self.state_shape, order="F")).T
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
            features[:, d] = np.clip(features[:, d], 0, len(edge) - 2)
        return features if N > 1 else features[0]

    # ---------------------------
    # Utilities
    # ---------------------------
    def shape(self):
        return self.state_shape

    def num_states(self):
        return np.prod(self.state_shape)

    def flatten_s_table(self, state_table):
        """
        convert a table of shape self.shape() to a list of shape (self.total_states(), )
        """
        return np.ravel(state_table, order="F")

    def unflatten_s_table(self, state_table):
        """
        convert a list of shape (self.total_states(), ) to a table of shape self.shape()
        """
        return state_table.reshape(self.shape(), order="F")


# SA aggregator for discrete actions
class SA_Aggregator_Disc:
    def __init__(self, state_low, state_high, state_bins, num_actions):
        self.state_bins = state_bins
        self.agg_internal = Aggregator(state_low, state_high, state_bins)
        self.num_actions = num_actions

    def sa_to_idx(self, states, actions):
        """
        inputs:
            States: np.array of shape (n, state_shape) or (state_shape, )
            Actions: np.array of shape (n, ) or int

        outputs:
        """
        return_int = np.isscalar(actions)

        states = np.atleast_2d(states)
        actions = np.atleast_1d(actions)

        state_features = self.agg_internal.state_to_idx(states)

        if return_int:
            return (state_features + actions * self.agg_internal.num_states())[0]
        return state_features + actions * self.agg_internal.num_states()

    def sa_to_features(self, states, actions):
        return_int = np.isscalar(actions)

        actions = np.atleast_2d(actions).T
        state_features = np.atleast_2d(self.agg_internal.s_to_features(states))
        ret = np.concatenate([state_features, actions], axis=-1)

        if return_int:
            return ret[0]
        else:
            return ret

    def idx_to_features(self, idxs):
        idxs = np.atleast_1d(idxs)

        states = idxs % self.agg_internal.num_states()
        actions = idxs // self.agg_internal.num_states()

        return np.concatenate(
            [self.agg_internal.idx_to_features(states), actions], axis=-1
        )

    def shape(self):
        return self.state_bins + [self.num_actions]

    def num_sa(self):
        return np.prod(self.shape())

    def flatten_sa_table(self, sa_table):
        return sa_table.ravel(order="F")

    def unflatten_sa_table(self, sa_table):
        return sa_table.reshape(self.shape(), order="F")


class SA_Aggregator:
    def __init__(self, state_low, state_high, state_bins, act_low, act_high, act_bins):
        self.state_space = state_bins
        self.act_space = act_bins

        self.agg_internal = Aggregator(
            state_low + act_low, state_high + act_high, state_bins + act_bins
        )

    def sa_to_idx(self, states, actions):
        states = np.atleast_2d(states)
        actions = np.atleast_2d(actions)
        return self.agg_internal.s_to_idx(np.concatenate([states, actions], axis=-1))

    def sa_to_features(self, states, actions):
        states = np.atleast_2d(states)
        actions = np.atleast_2d(actions)
        return self.agg_internal.s_to_features(
            np.concatenate([states, actions], axis=-1)
        )

    def idx_to_features(self, idxs):
        return self.agg_internal.idx_to_features(idxs)

    def features_to_idx(self, features):
        return self.agg_internal.features_to_idx(features)

    def shape(self):
        return self.state_space + self.act_space

    def num_sa(self):
        return np.prod(self.shape())

    def flatten_sa_table(self, table):
        return self.agg_internal.flatten_s_table(table)

    def unflatten_sa_table(self, table):
        return self.agg_internal.unflatten_s_table(table)


class S_Reward:
    def __init__(self, agg, reward_table):
        self.agg = agg
        self.reward_table = reward_table

    def __call__(self, state, action):
        return self.reward_table[tuple(self.agg.s_to_features(state))]


class SA_Reward:
    def __init__(self, agg, reward_table):
        self.agg = agg
        self.reward_table = agg.flatten_sa_table(reward_table)

    def __call__(self, state, action):
        rew = self.reward_table[self.agg.sa_to_idx(state, action)]
        return rew


def get_aggregator(env_name, bin_res=1):
    if env_name == "MountainCarContinuous-v0":
        s_low = [-1.2, -0.07]
        s_high = [0.6, 0.07]
        s_bins = [12 * bin_res, 11 * bin_res]
        # s_bins = [4, 4]
        a_low = [-1.0]
        a_high = [1.0]
        a_bins = [3 * bin_res]
        return Aggregator(s_low, s_high, s_bins), SA_Aggregator(
            s_low, s_high, s_bins, a_low, a_high, a_bins
        )

    if env_name == "CartPole-v1":
        s_low = [-2.5, -3.5, -0.3, -4.0]
        s_high = [2.5, 3.5, 0.3, 4.0]
        s_bins = [12, 12, 5, 12]
        num_actions = 2

        return Aggregator(s_low, s_high, s_bins), SA_Aggregator_Disc(
            s_low, s_high, s_bins, num_actions
        )

    if env_name == "Pendulum-v1":
        s_low = [-1.0, -1.0, -8.0]
        s_high = [1.0, 1.0, 8.0]
        a_low = [-2.0]
        a_high= [2.0]
        s_bins = [8,8, 16]
        a_bins = [5]
        return Aggregator(s_low, s_high, s_bins), SA_Aggregator(
            s_low, s_high, s_bins, num_actions
        )

    if env_name == "AcroBot-v1":
        s_low = [-1, -1, -1, -1, 12.6, 28.6]
        s_high = [1, 1, 1, 1, -12.6, -28.6]
        s_bins = [12, 12, 5, 12]
        num_actions = 3
        return Aggregator(s_low, s_high, s_bins), SA_Aggregator_Disc(s_low, s_high, s_bins, num_actions)


def flatten_idx(x, v):
    return x + 12 * v


def unflatten_idx(i):
    return (i % 12, i // 12)


def unflatten_state(state):
    a, b = (12, 11)

    mat = np.zeros(shape=(a, b))

    for i in range(a * b):
        mat[unflatten_idx(i)] = state[i]

    return mat


def flatten_state(state):
    a, b = (12, 11)
    mat = np.zeros(shape=(a * b))

    for i in range(a * b):
        mat[i] = state[unflatten_idx(i)]

    return mat


if __name__ == "__main__":
    # quick self-consistency test

    def s_test():
        agg = Aggregator([-1, -1], [1, 1], [12, 11])
        test_states = np.random.uniform(-1, 1, (200, 2))
        idx = agg.s_to_idx(test_states)
        features = agg.s_to_features(test_states)
        assert np.array_equal(agg.features_to_idx(features), idx)
        assert np.array_equal(agg.idx_to_features(idx), features)

        test_state_table = np.arange(12 * 11)
        test_state_table1 = np.zeros((12, 11))

        for i in range(12 * 11):
            test_state_table1[i % 12, i // 12] = i

            assert np.all(np.array(unflatten_idx(i)) == agg.idx_to_features(i))
            assert np.all(
                np.array(flatten_idx(i % 12, i // 12))
                == agg.features_to_idx(np.array([i % 12, i // 12]))
            )

            assert agg.features_to_idx(np.array([i % 12, i // 12])) == i
            assert np.all(agg.idx_to_features(i) == np.array([i % 12, i // 12]))

        print(agg.flatten_s_table(test_state_table1))
        assert np.array_equal(agg.flatten_s_table(test_state_table1), test_state_table)
        assert np.array_equal(
            unflatten_state(test_state_table), agg.unflatten_s_table(test_state_table)
        )
        assert np.array_equal(
            flatten_state(test_state_table1), agg.flatten_s_table(test_state_table1)
        )

    def sa_test():
        sa_agg = SA_Aggregator([-1, -1], [1, 1], [12, 11], [-1], [1], [3])

        test_states = np.random.uniform(-1, 1, (200, 2))
        test_actions = np.random.uniform(-1, 1, size=(200, 1))
        idx = sa_agg.sa_to_idx(test_states, test_actions)
        features = sa_agg.sa_to_features(test_states, test_actions)
        print(features.shape, idx.shape)
        assert np.array_equal(sa_agg.features_to_idx(features), idx)
        assert np.array_equal(sa_agg.idx_to_features(idx), features)

        test_sa_table = np.arange(sa_agg.num_sa())
        test_sa_table1 = sa_agg.unflatten_sa_table(test_sa_table)
        for i in range(test_sa_table.size):
            assert test_sa_table1[tuple(sa_agg.idx_to_features(i))] == test_sa_table[i]

    s_test()
    sa_test()
