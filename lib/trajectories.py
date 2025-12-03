import numpy as np
from scipy.sparse import lil_matrix, eye

class Trajectory:
    def __init__(self, T):
        self.last_state = None
        self.reward = None
        self.action = None
        self.next_state = None
        self.terminated = False
        self.t = 0
        self.T = T

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

class TrajectoryContainer:
    def __init__(self, T, init_capacity=200):
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

        self.last_state = np.zeros((self.capacity, self.T) + self.state_shape, dtype=ls.dtype)
        self.action = np.zeros((self.capacity, self.T) + self.action_shape, dtype=a.dtype)
        self.reward = np.zeros((self.capacity, self.T), dtype=r.dtype)
        self.next_state = np.zeros((self.capacity, self.T) + self.state_shape, dtype=ns.dtype)
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
        ls, a, r, ns, terminated, t = trajectory._dump_raw()

        if self.current_idx == 0:
            self._init_arrays(ls, a, r, ns)

        if self.current_idx >= self.capacity:
            self._grow()

        idx = self.current_idx
        self.last_state[idx] = ls
        self.action[idx] = a
        self.reward[idx] = r
        self.next_state[idx] = ns
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

    def dump(self):
        return self.last_state[:self.current_idx], self.action[:self.current_idx], self.reward[:self.current_idx], self.next_state[:self.current_idx], self.ts[:self.current_idx], self.terminated[:self.current_idx],

    def extend(self, other):
        self.trim()
        last_state, action, reward, next_state , ts, terminated= other.dump()

        self.last_state = np.concat([self.last_state, last_state], axis=0)
        self.action = np.concat([self.action, action], axis=0)
        self.reward = np.concat([self.reward, reward], axis=0)
        self.next_state = np.concat([self.next_state, next_state], axis=0)
        self.terminated = np.concat([self.terminated, terminated], axis=0)
        self.ts = np.concat([self.ts, ts], axis=0)


        self.current_idx += other.current_idx

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

def p_s_from_rollouts(rollouts, s_agg):
    p = np.zeros(s_agg.shape())
    for trajectory in rollouts:
        s, _, _, s_prime = trajectory.dump()
        s_f = s_agg.s_to_features(s)
        s_prime_f = s_agg.s_to_features(s_prime)
        p[tuple(s_f[0])] += 1
        np.add.at(p, tuple(s_prime_f.T), 1)

    p /= p.sum()
    return p

def p_sa_from_rollouts(rollouts, sa_agg):
    p = np.zeros(sa_agg.shape())
    num_sa =0
    for trajectory in rollouts:

        s, a, _, _ = trajectory.dump()
        sa = sa_agg.sa_to_features(s, a)
        np.add.at(p, tuple(sa.T), 1)
        num_sa += s.shape[0]

    p /= num_sa
    return p

def sr_from_rollouts(rollouts, s_agg, gamma=0.99, step_size = 0.01):
    n = s_agg.num_states()
    sr = lil_matrix((n,n), dtype=np.float32)


    for trajectory in rollouts:
        s, _, _, s_prime = trajectory.dump()
        s_idx = s_agg.s_to_idx(s)
        s_prime_idx = s_agg.s_to_idx(s_prime)

        prev = sr[s_idx[0], :].toarray().ravel()
        for t in range(s.shape[0]):
            next = sr[s_prime_idx[t], :].toarray().ravel()
            delta = gamma*next - prev 
            delta[s_idx[t]] += 1
            sr[s_idx[t], :] += step_size * delta
    return sr.tocsr() + 1e-9 * eye(n, format="csr")

def sa_sr_from_rollouts(rollouts, sa_agg, gamma=0.99, step_size=0.01):
    n = sa_agg.num_sa()
    sr = lil_matrix((n, n), dtype=np.float32)

    # dense work buffers
    curr = np.zeros(n, dtype=np.float32)
    nxt  = np.zeros(n, dtype=np.float32)
    delta = np.zeros(n, dtype=np.float32)

    for trajectory in rollouts:
        s, a, _, sp = trajectory.dump()

        # Correct alignment
        s_idx        = sa_agg.sa_to_idx(s[:-1], a[:-1])
        s_prime_idx  = sa_agg.sa_to_idx(sp[:-1], a[1:])

        for t in range(len(s_idx)):
            i  = s_idx[t]
            ip = s_prime_idx[t]

            # load current SR row into dense buffer
            curr[:] = 0.0
            if sr.rows[i]:
                curr[sr.rows[i]] = sr.data[i]

            # load successor SR row into buffer
            nxt[:] = 0.0
            if sr.rows[ip]:
                nxt[sr.rows[ip]] = sr.data[ip]

            # TD update:  delta = 1 + gamma * nxt - curr
            delta[:] = gamma * nxt - curr
            delta[i] += 1.0

            # new row = curr + step_size * delta
            curr[:] = curr + step_size * delta

            # write dense row back to sparse
            nz = curr.nonzero()[0]
            sr.rows[i] = nz.tolist()
            sr.data[i] = curr[nz].tolist()
    return sr.tocsr() + 1e-9 * eye(n, format="csr")

def average_reward_from_rollouts(rollouts, reward_fn = None):
    avg_reward = 0
    for trajectory in rollouts:
        s, a , r, _ = trajectory.dump()
        if reward_fn == None:
            avg_reward +=  r.sum()
        else: 
            for t in range(s.shape[0]):
                avg_reward += reward_fn(s[t],a[t])
    return avg_reward/len(rollouts)

def collect_rollouts(env, agent, T, num_rollouts, reward_fn = None, epsilon=0.0):
    def _collect_rollout(env, agent, T, reward_fn, epsilon):
        trajectory = Trajectory(T)
        last_state, _ = env.reset()
        for i in range(T):
            action = agent.select_action(last_state, epsilon)
            state, reward, terminated, truncated, info = env.step(action)
            if reward_fn != None:
                reward = reward_fn(last_state, action, state, terminated)
            trajectory.add_transition(last_state, action, reward, state, terminated)
            last_state = state
            if terminated or truncated:
                return trajectory
        return trajectory
    rollouts = TrajectoryContainer(T, init_capacity=num_rollouts)

    for i in range(num_rollouts):
        rollouts.add_trajectory(_collect_rollout(env, agent, T, reward_fn, epsilon))
    return rollouts

import itertools
import gymnasium as gym
from lib.policies import get_random_agent
from lib.aggregation import get_aggregator
if __name__ == "__main__":

    # env_name = "CartPole-v1"
    env_name = "MountainCarContinuous-v0"
    # env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    env = gym.make(env_name, render_mode="rgb_array") 
    agent = get_random_agent(env_name)

    s_agg, sa_agg = get_aggregator(env_name, bin_res=1)
    rollouts = collect_rollouts(env, agent, 1000, 100)

    p = p_s_from_rollouts(rollouts, s_agg)
    p_sa = p_sa_from_rollouts(rollouts, sa_agg)
    sr = sr_from_rollouts(rollouts, s_agg)

    p1 = s_agg.flatten_state_table(p)
    p2 = s_agg.shape()
