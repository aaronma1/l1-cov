import numpy as np

class Trajectory:
    def __init__(self, T):
        self.last_state = None
        self.reward = None
        self.action = None
        self.next_state = None
        self.terminated = False
        self.t = 0
        self.T = T


    #classmethod 
    @classmethod
    def from_container(cls, last_state, action,reward,next_state, terminated, t):
        traj = Trajectory(np.size(reward))
        traj.T = np.size(reward)
        traj.last_state = last_state
        traj.action = action
        traj.reward = reward
        traj.next_state = next_state
        traj.terminated = terminated
        traj.t = t

        return traj
    
    def add_transition(self, state, action, reward, next_state, terminated = False):
        if self.t == 0:
            self.last_state = np.zeros( (self.T,) + tuple(np.shape(state)))
            self.action = np.zeros( (self.T,) + tuple(np.shape(action)))
            self.reward = np.zeros(self.T)
            self.next_state = np.zeros( (self.T,) + tuple(state.shape))
        

        if self.terminated:
            return 
        self.last_state[self.t] = state
        self.action[self.t] = action
        self.reward[self.t] = reward
        self.next_state[self.t] = next_state
        self.terminated = self.terminated | terminated

        self.t += 1

    def dump(self):
        return np.array(self.last_state[:self.t]), np.array(self.action[:self.t]), np.array(self.reward[:self.t]), np.array(self.next_state[:self.t])

    def _dump_raw(self):
        return self.last_state, self.action, self.reward,self.next_state, self.terminated, self.t


import numpy as np

class TrajectoryContainer:
    def __init__(self, T, init_capacity=16):
        self.T = T
        self.capacity = init_capacity
        self.current_idx = 0

        self.last_state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.terminated = None
        self.ts = None

        self.state_shape = None
        self.action_shape = None

    def _init_arrays(self, ls, a, r, ns):
        self.state_shape = ls.shape[1:]
        self.action_shape = a.shape[1:]

        self.states = np.zeros((self.capacity, self.T+1) + self.state_shape, dtype=ls.dtype)
        self.action = np.zeros((self.capacity, self.T) + self.action_shape, dtype=a.dtype)
        self.reward = np.zeros((self.capacity, self.T), dtype=r.dtype)
        self.terminated = np.zeros(self.capacity, dtype=bool)
        self.ts = np.zeros(self.capacity, dtype=int)

    def _grow(self):
        new_capacity = self.capacity * 2
        def grow_array(arr):
            new_arr = np.zeros((new_capacity,) + arr.shape[1:], dtype=arr.dtype)
            new_arr[:self.capacity] = arr
            return new_arr

        self.last_state = grow_array(self.last_state)
        self.action = grow_array(self.action)
        self.reward = grow_array(self.reward)
        self.next_state = grow_array(self.next_state)
        self.terminated = grow_array(self.terminated)
        self.ts = grow_array(self.ts)

        self.capacity = new_capacity

    def add_trajectory(self, trajectory):
        last_state, action, r, ns, terminated, t = trajectory._dump_raw()

        if self.current_idx == 0:
            self._init_arrays(last_state, action, r, ns)

        if self.current_idx >= self.capacity:
            self._grow()

        idx = self.current_idx
        self.last_state[idx][:t] = last_state
        self.states[idx][t] = ns[-1]
        self.action[idx] = action
        self.reward[idx] = r
        self.terminated[idx] = terminated
        self.ts[idx] = t
        self.current_idx += 1

    def get_trajectory(self, i):
        if i >= self.current_idx:
            raise IndexError("Trajectory index out of range")
        return Trajectory.from_container(
            self.last_state[i],
            self.action[i],
            self.reward[i],
            self.next_state[i],
            self.terminated[i],
            self.ts[i]
        )

    def trim(self):
        self.last_state = self.last_state[:self.current_idx]
        self.action = self.action[:self.current_idx]
        self.reward = self.reward[:self.current_idx]
        self.next_state = self.next_state[:self.current_idx]
        self.terminated = self.terminated[:self.current_idx]
        self.ts = self.ts[:self.current_idx]
        self.capacity = self.current_idx

    # ----------------------
    # Iterators
    # ----------------------

    def __len__(self):
        return self.current_idx

    def __iter__(self):
        """Iterator over single trajectories"""
        self._iter_idx = 0
        return self

    def __next__(self):
        if self._iter_idx >= self.current_idx:
            raise StopIteration
        traj = self.get_trajectory(self._iter_idx)
        self._iter_idx += 1
        return traj

    def batches(self, batch_size):
        """Generator yielding batches of trajectories as arrays"""
        for start in range(0, self.current_idx, batch_size):
            end = min(start + batch_size, self.current_idx)
            yield (self.last_state[start:end],
                   self.action[start:end],
                   self.reward[start:end],
                   self.next_state[start:end],
                   self.terminated[start:end],
                   self.ts[start:end])