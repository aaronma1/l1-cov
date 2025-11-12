from aggregation import get_aggregator
import numpy as np

class RandomAgent:
    def __init__(self, atc):
        self.idx = np.arange(len(atc))
        self.atc = atc

    def select_action(self, state, epsilon=0.0):
        return self.atc[np.random.choice(self.idx)]

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


def collect_rollouts(env, agent, T, num_rollouts, reward_fn = None, epsilon=0.0):

    def _collect_rollout(env, agent, T, reward_fn, epsilon):
        trajectory = Trajectory()
        last_state, _ = env.reset()
        for i in range(T):
            action = agent.select_action(last_state, epsilon)
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
        rollouts.append(_collect_rollout(env, agent, T, reward_fn, epsilon))

    return rollouts


def get_random_agent(env_name):
    if env_name == "MountainCarContinuous-v0":
        return RandomAgent(np.array([np.array([-1.0]), np.array([0.0]), np.array([1.0])]))

    if env_name == "CartPole-v1":
        return RandomAgent(np.array([0,1]))


def p_s_from_rollouts(rollouts, s_agg):

    p = np.zeros(s_agg.shape())
    num_s =0
    for trajectory in rollouts:

        s, _, _, s_prime = trajectory.dump()
        num_s += s.shape[0] + 1

        s_f = s_agg.s_to_features(s)
        s_prime_f = s_agg.s_to_features(s_prime)

        p[tuple(s_f[0])] += 1

        for i in range(s.shape[0]):
            
            p[s_prime_f[i][0], s_prime_f[i][1]] += 1

    p /= num_s
    return p
         

def p_s_from_rollouts_1(rollouts, s_agg):

    p = np.zeros(s_agg.num_states())
    num_s =0
    for trajectory in rollouts:
        s, _, _, s_prime = trajectory.dump()
        num_s += s.shape[0] + 1
        s = s_agg.s_to_idx(s)
        s_prime = s_agg.s_to_idx(s_prime)
        p[s[0]] += 1

        for idx in s_prime:
            p[idx] += 1

    p /= num_s
    return s_agg.unflatten_s_table(p)
        


def p_sa_from_rollouts(rollouts, sa_agg):
    
    p = np.zeros(sa_agg.shape())
    num_sa =0
    for trajectory in rollouts:

        s, a, _, _ = trajectory.dump()
        sa = sa_agg.sa_to_features(s, a)
        print(tuple(sa.T))
        np.add.at(p, tuple(sa.T), 1)
        num_sa += s.shape[0]

    p /= num_sa
    return p

    


def sr_from_rollouts(rollouts, s_agg, gamma=0.99, step_size = 0.01):
    
    n = s_agg.num_states()
    sr = np.zeros((n,n))

    for trajectory in rollouts:

        s, _, _, s_prime = trajectory.dump()
        s_idx = s_agg.s_to_idx(s)
        s_prime_idx = s_agg.s_to_idx(s_prime)

        for t in range(s.shape[0]):
            delta = gamma*sr[s_prime_idx[t], :] - sr[s_idx[t], :]
            delta[s_idx[t]] += 1
            sr[s_idx[t], :] += step_size * delta


    return sr




    





import itertools
import gymnasium as gym
import plotting
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
    p2 = s_agg.unflatten_state_table(p)


    plotting.plot_heatmap(p, save_path="test.png")
    plotting.plot_heatmap(p_sa[..., 0], save_path="test1.png")
    plotting.plot_heatmap(p_sa[..., 1], save_path="test2.png")
    plotting.plot_heatmap(p_sa[..., 2], save_path="test3.png")





