import enum
from dataclasses import dataclass
from tkinter import W

import gymnasium as gym
from lib.policies import (
    average_reward_from_rollouts,
    get_random_agent,
    sr_from_rollouts,
    p_s_from_rollouts,
    p_sa_from_rollouts,
    sa_sr_from_rollouts,
    collect_rollouts,
)
from lib.qlearningpolicy import get_qlearning_agent
from lib.aggregation import get_aggregator, S_Reward, SA_Reward, SAS_Reward, SS_Reward

from scipy.sparse.linalg import eigsh, eigs
import gymnasium as gym
import numpy as np


def setup_env(base_args):
    env = gym.make(base_args["env_name"], render_mode="rgb_array")
    s_agg, sa_agg = get_aggregator(base_args["env_name"], bin_res=1)
    return env, s_agg, sa_agg


def setup_agent(base_args, agent_args):
    # if agent_args["policy"] == "PolicyGrad":
    #     pass
    if agent_args["policy"] == "Qlearning":
        return get_qlearning_agent(
            base_args["env_name"], agent_args["gamma"], agent_args["lr"]
        )


def reward_shaping(reward_fn):
    r_max = np.max(reward_fn)
    r_min = np.min(reward_fn)
    new_reward = reward_fn
    new_reward -= r_min
    new_reward /= r_max - r_min
    return new_reward


def collect_rollouts_from_options(env, base_args, option_args, options):
    rollouts = {"all_rollouts": None}
    for i, opt in enumerate(options):
        # rollouts[f"option{i}_rollouts"] = collect_rollouts(env, opt, base_args["env_T"], base_args["num_rollouts"], **option_args["rollout_args"])
        option_i_rollouts = collect_rollouts(
            env,
            opt,
            base_args["env_T"],
            base_args["num_rollouts"],
            **option_args["rollout_args"],
        )

        if rollouts["all_rollouts"] == None:
            rollouts["all_rollouts"] = option_i_rollouts
        else:
            rollouts["all_rollouts"].extend(option_i_rollouts)

    return rollouts


def learn_policy(env, base_args, option_args, reward_fn, transitions=None):
    option = setup_agent(base_args, option_args)

    if transitions != None:
        option.learn_offline_policy(
            transitions, option_args["offline_epochs"], reward_fn, verbose=  option_args["learning_args"]["verbose"]
        )
    _, sa_agg = get_aggregator(base_args["env_name"])

    option.learn_policy(
        env,
        base_args["env_T"],
        option_args["online_epochs"],
        reward_fn,
        **option_args["learning_args"],
    )
    return option


def collect_run_eigenoptions(base_args, option_args):
    def eig_sparse(SR, k=131):
        SR = (SR + SR.T) / 2.0

        eigenvalues, eigenvectors = eigsh(SR, sigma=None, k=k, which="LM")
        idx = np.argsort(-eigenvalues.real)
        eigenvalues = np.real_if_close(eigenvalues, tol=1e5)[idx]
        eigenvectors = np.real_if_close(eigenvectors, tol=1e5)[:, idx]
        return eigenvectors, eigenvalues

    s_agg, _ = get_aggregator(base_args["env_name"], bin_res=base_args["bin_res"])
    env = gym.make(base_args["env_name"], render_mode="rgb_array")

    options = [get_random_agent(base_args["env_name"])]
    epoch_rollouts = []

    for i in range(base_args["num_epochs"]):
        epoch_rollouts.append(
            collect_rollouts_from_options(env, base_args, option_args, options)
        )
        SR = sr_from_rollouts(epoch_rollouts[-1]["all_rollouts"], s_agg)
        p_s = p_s_from_rollouts(epoch_rollouts[-1]["all_rollouts"], s_agg)
        eigenvectors, eigenvalues = eig_sparse(SR, k=10)
        top_eig = s_agg.unflatten_s_table(eigenvectors[:, 0])
        if np.dot(top_eig.flatten(), p_s.flatten()) > 0:
            top_eig = -top_eig
        reward = SS_Reward(s_agg, top_eig)
        # learn an option
        options.append(
            learn_policy(
                env, base_args, option_args, reward, epoch_rollouts[-1]["all_rollouts"]
            )
        )

    # collect last epoch rollout
    epoch_rollouts.append(
        collect_rollouts_from_options(env, base_args, option_args, options)
    )
    return epoch_rollouts, options

def collect_run_sa_eigenoptions(base_args, option_args, node_num=0):
    def eig_sparse(SR, k=131):
        SR = (SR + SR.T) / 2.0
        eigenvalues, eigenvectors = eigsh(SR, sigma=None, k=k, which="LM")
        idx = np.argsort(-eigenvalues.real)
        eigenvalues = np.real_if_close(eigenvalues, tol=1e5)[idx]
        eigenvectors = np.real_if_close(eigenvectors, tol=1e5)[:, idx]
        return eigenvectors, eigenvalues

    s_agg, sa_agg = get_aggregator(base_args["env_name"], bin_res=base_args["bin_res"])
    env = gym.make(base_args["env_name"], render_mode="rgb_array")

    options = [get_random_agent(base_args["env_name"])]
    epoch_rollouts = []

    for i in range(base_args["num_epochs"]):
        print(f"epoch {i}")
        epoch_rollouts.append(
            collect_rollouts_from_options(env, base_args, option_args, options)
        )
        SR = sa_sr_from_rollouts(epoch_rollouts[-1]["all_rollouts"], sa_agg)
        p_sa = p_sa_from_rollouts(epoch_rollouts[-1]["all_rollouts"], sa_agg)

        eigenvectors, eigenvalues = eig_sparse(SR, k=10)
        top_eig = sa_agg.unflatten_sa_table(eigenvectors[:, 0])
        top_eig /= np.max(np.abs(top_eig))
        if np.dot(top_eig.flatten(), p_sa.flatten()) > 0:
            top_eig = -top_eig
        
        reward = SAS_Reward(sa_agg, top_eig)
        # learn an option
        options.append(
            learn_policy(
                env, base_args, option_args, reward, epoch_rollouts[-1]["all_rollouts"]
            )
        )

    # collect last epoch rollout
    epoch_rollouts.append(
        collect_rollouts_from_options(env, base_args, option_args, options)
    )
    return epoch_rollouts, options


def collect_run_codex(base_args, option_args):
    s_agg, sa_agg = get_aggregator(base_args["env_name"], bin_res=1)
    env = gym.make(base_args["env_name"], render_mode="rgb_array")

    options = [get_random_agent(base_args["env_name"])]
    epoch_rollouts = []
    for i in range(base_args["num_epochs"]):
        epoch_rollouts.append(
            collect_rollouts_from_options(env, base_args, option_args, options)
        )
        
        p_sa = p_sa_from_rollouts(epoch_rollouts[-1]["all_rollouts"], sa_agg)
        uniform_density_sa = np.ones(sa_agg.shape())
        l1_cov_reward = (
            reward_shaping(1 / (base_args["l1_eps"] * uniform_density_sa + p_sa))
        )

        reward_fn = SA_Reward(sa_agg, l1_cov_reward)
        # learn policy
        options.append(
            learn_policy(
                env,
                base_args,
                option_args,
                reward_fn,
                epoch_rollouts[-1]["all_rollouts"],
            )
        )

    epoch_rollouts.append(
        collect_rollouts_from_options(env, base_args, option_args, options)
    )
    return epoch_rollouts, options

def collect_run_maxent(base_args, option_args):
    s_agg, sa_agg = get_aggregator(base_args["env_name"], bin_res=1)
    env = gym.make(base_args["env_name"], render_mode="rgb_array")

    options = [get_random_agent(base_args["env_name"])]
    epoch_rollouts = []
    for i in range(base_args["num_epochs"]):
        epoch_rollouts.append(
            collect_rollouts_from_options(env, base_args, option_args, options)
        )
        
        p_sa = p_sa_from_rollouts(epoch_rollouts[-1]["all_rollouts"], sa_agg)
        uniform_density_sa = np.ones(sa_agg.shape())
        l1_cov_reward = (
            reward_shaping(1 / (base_args["l1_eps"] * uniform_density_sa + p_sa))
        )

        reward_fn = SA_Reward(sa_agg, l1_cov_reward)
        # learn policy
        options.append(
            learn_policy(
                env,
                base_args,
                option_args,
                reward_fn,
                epoch_rollouts[-1]["all_rollouts"],
            )
        )

    epoch_rollouts.append(
        collect_rollouts_from_options(env, base_args, option_args, options)
    )
    return epoch_rollouts, options



def collect_run_random(base_args):
    s_agg, sa_agg = get_aggregator(base_args["env_name"], bin_res=1)
    env = gym.make(base_args["env_name"], render_mode="rgb_array")

    options = [get_random_agent(env_name=base_args["env_name"])]
    epoch_rollouts = []

    for i in range(base_args["num_epochs"]):
        epoch_rollouts.append(
            collect_rollouts_from_options(env, base_args, {}, options)
        )
        options += [get_random_agent(env_name=base_args["env_name"])]

    epoch_rollouts.append(collect_rollouts_from_options(env, base_args, {}, options))
    return epoch_rollouts, options



def setup_env_exploring_starts(base_args):
    env = gym.make(base_args["env_name"], render_mode="rgb_array")

    class ESW(gym.Wrapper):
        def __init__(self, env, state_sampler, random_first_act=True):
            self.state_sampler = state_sampler
            self.random_first_action = random_first_act
            self.env = env

        def reset(self, **kwargs):
            state, info = env.reset(**kwargs)
            new_state = self.state_sampler()
            self.env.state = np.array(new_state, dtype=float)

            if self.random_first_action:
                first_action = self.env.action_space.sample()
                state, reward, terminated, truncated, info = self.env.step(first_action)
                return state, info
            return self.env.state, info

    def mountaincar_state_sampler():
        return np.random.uniform(low=[-1.2, -0.07], high=[0.6, 0.07])

    def cartpole_state_sampler():
        return np.random.uniform(
            low=[-0.5, -1.0, -0.2, -1.0], high=[0.5, 1.0, 0.2, 1.0]
        )
    def acrobot_state_sampler():
        return np.random.uniform(
            low=[-0.5, -1.0, -0.2, -1.0], high=[0.5, 1.0, 0.2, 1.0]
        )
    def pendulum_state_sampler():
        return np.random.uniform(
            low=[-0.5, -1.0, -0.2, -1.0], high=[0.5, 1.0, 0.2, 1.0]
        )



    if base_args["env_name"] == "MountainCarContinuous-v0":
        return ESW(env, mountaincar_state_sampler)
    if base_args["env_name"] == "CartPole-v1":
        return ESW(env, cartpole_state_sampler)


def compute_l1_from_run(base_args, adv_args, run):
    # s_agg, sa_agg = get_aggregator(env_name=base_args["env_name"], bin_res=1)
    # env = gym.make(base_args["env_name"], render_mode="rgb_array")

    # we allow the adverserial policy to "cheat" because we want sup(pi)

    measure_env, s_agg, sa_agg = setup_env(base_args)
    learn_env = setup_env_exploring_starts(base_args)

    l1_covs = []
    adv_policies = []


    for i, epoch in enumerate(run):

        print(f"computing epoch {i}")

        epoch_rollouts = epoch["all_rollouts"]
        p_sa = p_sa_from_rollouts(epoch_rollouts, sa_agg)

        uniform_density_sa = np.ones(sa_agg.shape())
        l1_cov_reward = 1 / (base_args["l1_eps"] * uniform_density_sa + p_sa)
        reward_fn = SA_Reward(
            sa_agg, reward_shaping(l1_cov_reward) 
        )

        adv_policy = setup_agent(base_args, adv_args)
        adv_policy.learn_policy(
            learn_env,
            base_args["env_T"],
            adv_args["online_epochs"],
            reward_fn,
            **adv_args["learning_args"],
        )
        adv_policy.learn_policy(
            measure_env,
            base_args["env_T"],
            adv_args["online_epochs"],
            reward_fn,
            **adv_args["learning_args"],
        )

        measure_reward_fn = SA_Reward(sa_agg, reward_shaping(l1_cov_reward))
        rollouts = collect_rollouts(
            measure_env,
            adv_policy,
            base_args["env_T"],
            base_args["num_rollouts"],
            measure_reward_fn,
            **adv_args["rollout_args"],
        )

        p_sa = p_sa_from_rollouts(rollouts, sa_agg)
        l1_covs.append(average_reward_from_rollouts(rollouts))
        adv_policies.append(adv_policy)

    return l1_covs, adv_policies

def compute_env_l1_cov(base_args, adv_args):
    measure_env, s_agg, sa_agg = setup_env(base_args)
    learn_env = setup_env_exploring_starts(base_args)

    adv_policy = setup_agent(base_args, adv_args)


    uniform_density_sa = np.ones(sa_agg.shape())
    p_sa =  np.ones(sa_agg.shape())/sa_agg.num_sa()
    l1_cov_reward = 1 / (base_args["l1_eps"] * uniform_density_sa + p_sa)
    reward_fn = SA_Reward(
        sa_agg, reward_shaping(l1_cov_reward) 
    )

    policy = setup_agent(base_args, adv_args)
    adv_policy.learn_policy(
        learn_env,
        base_args["env_T"],
        adv_args["online_epochs"],
        reward_fn,
        **adv_args["learning_args"],
    )
    adv_policy.learn_policy(
        measure_env,
        base_args["env_T"],
        adv_args["online_epochs"],
        reward_fn,
        **adv_args["learning_args"],
    )
    rollouts = collect_rollouts(
        measure_env,
        adv_policy,
        base_args["env_T"],
        base_args["num_rollouts"],
        reward_fn,
        **adv_args["rollout_args"],
    )

    return average_reward_from_rollouts(rollouts), adv_policy

    



def exp_test_mountaincar():
    mountaincar_args = {
        "l1_eps": 1e-4,  # regularizer epsilon for
        "bin_res": 1,
        "env_name": "MountainCarContinuous-v0",
        "env_T": 200,
        "num_rollouts": 400,
        "num_epochs": 3,
    }
    # base_args = {
    #     "gamma":0.999, # global discount factor
    #     "l1_eps":1e-4, # regularizer epsilon for
    #     "bin_res": 1,
    #     "env_name": "CartPole-v1",
    #     "env_T":200,
    #     "num_rollouts":400,
    #     "num_epochs": 5,
    #     "reward_shaping_constant": 0
    # }

    Qlearning_args = {
        "policy": "Qlearning",
        "gamma": 0.99,
        "lr": 0.01,
        "online_epochs": 100,
        "offline_epochs": 10,
        "learning_args": {
            "epsilon_start": 1,
            "epsilon_decay": 0.999,
            "decay_every": 5,
            "verbose": True,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.2,
        },
    }
    # more comprehensive qlearning args for l1 coverage
    Qlearning_args_l1 = {
        # base_args = {
        "policy": "Qlearning",
        "gamma": 0.99,
        "lr": 0.01,
        "online_epochs": 20000,
        "offline_epochs": 0,
        "learning_args": {
            "epsilon_start": 0.5,
            "epsilon_decay": 0.999,
            "decay_every": 3,
            "verbose": True,
            "print_every": 1000,
        },
        "rollout_args": {
            "epsilon": 0.2,
        },
        "print_every": 100,
    }

    
    trajectories_codex, _ = collect_run_codex(mountaincar_args, Qlearning_args)
    trajectories_eo, _ = collect_run_sa_eigenoptions(mountaincar_args, Qlearning_args)
    
    s_agg, sa_agg = get_aggregator(mountaincar_args["env_name"])
    import plotting
    for i in range(mountaincar_args["num_epochs"]):
        plotting.plot_heatmap(p_sa_from_rollouts(trajectories_codex[i]["all_rollouts"], sa_agg).reshape(12, -1), save_path=f"{i}_psa_cov.png")
        plotting.plot_heatmap(p_sa_from_rollouts(trajectories_eo[i]["all_rollouts"], sa_agg).reshape(12, -1),save_path=f"{i}_psa_eo.png")




    codex_l1_covs, _ = compute_l1_from_run(mountaincar_args, Qlearning_args_l1, trajectories_codex)
    eo_l1_covs, _ = compute_l1_from_run(mountaincar_args, Qlearning_args_l1, trajectories_eo)
    print("codex l1 covs", codex_l1_covs)
    print("EO l1 covs", eo_l1_covs)



if __name__ == "__main__":
    exp_test_mountaincar()