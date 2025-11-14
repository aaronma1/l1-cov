from tkinter import W
import plotting
from policies import average_reward_from_rollouts, get_random_agent, sr_from_rollouts_sparse, p_s_from_rollouts, p_sa_from_rollouts, collect_rollouts
from qlearningpolicy import get_qlearning_agent
from aggregation import get_aggregator, S_Reward, SA_Reward, unflatten_state
from scipy.sparse.linalg import eigsh, eigs
from plotting import gen_heatmap_epoch, plot_heatmap 
import gymnasium as gym
import numpy as np



def setup_env(base_args):
    env = gym.make(base_args["env_name"], render_mode="rgb_array")
    s_agg, sa_agg = get_aggregator(base_args["env_name"], bin_res=1)
    return env, s_agg, sa_agg


def setup_agent(base_args, agent_args):
    if agent_args["policy"] == "PolicyGrad":
        pass

    if agent_args["policy"] == "Qlearning":
        return get_qlearning_agent(base_args["env_name"], agent_args["gamma"], agent_args["lr"])

def reward_shaping(reward_fn):
    r_max = np.max(reward_fn)
    r_min = np.min(reward_fn)
    new_reward = reward_fn
    new_reward -= r_min
    new_reward /= (r_max - r_min)
    return new_reward

def l1_reward(p_sa, sa_agg, l1_eps):
    pass





def collect_rollouts_from_options(env, base_args, option_args, options):
    rollouts = {
        "all_rollouts":[]
    }
    for i,opt in enumerate(options):
        rollouts[f"option{i}_rollouts"] = collect_rollouts(env, opt, base_args["env_T"], base_args["num_rollouts"], **option_args["rollout_args"])
        rollouts["all_rollouts"].extend(rollouts[f"option{i}_rollouts"])
    return rollouts

def learn_policy(env,base_args, option_args, reward_fn, transitions = None):
    option = setup_agent(base_args, option_args)

    if transitions != None:
        for _ in range(option_args["offline_epochs"]):
            option.learn_offline_policy(transitions, reward_fn)

    option.learn_policy(env, base_args["env_T"], option_args["online_epochs"], reward_fn, **option_args["learning_args"])
    return option






def collect_run_eigenoptions(base_args, option_args):
    def eig_sparse(SR, k=131):
        SR = (SR+ SR.T)/2.0
        
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
        epoch_rollouts.append(collect_rollouts_from_options(env,base_args,option_args, options))
        SR= sr_from_rollouts_sparse(epoch_rollouts[-1]["all_rollouts"], s_agg)
        p_s =  p_s_from_rollouts(epoch_rollouts[-1]["all_rollouts"], s_agg)
        eigenvectors , eigenvalues = eig_sparse(SR, k=10)
        top_eig = s_agg.unflatten_s_table(eigenvectors[:, 0])
        if np.dot(top_eig.flatten(), p_s.flatten()) > 0:
            top_eig = -top_eig
        top_eig = reward_shaping(top_eig) + base_args["reward_shaping_constant"]
        reward = S_Reward(s_agg, top_eig)
        # learn an option
        options.append(learn_policy(env, base_args, option_args, reward, epoch_rollouts[-1]["all_rollouts"]))

    # collect last epoch rollout
    epoch_rollouts.append(collect_rollouts_from_options(env, base_args, option_args, options))
    return options, epoch_rollouts
        
def collect_run_codex(base_args, option_args):
    s_agg, sa_agg = get_aggregator(base_args["env_name"], bin_res=1)
    env = gym.make(base_args["env_name"], render_mode="rgb_array")

    options = [get_random_agent(base_args["env_name"])]
    epoch_rollouts = []
    for i in range(base_args["num_epochs"]):
        epoch_rollouts.append(collect_rollouts_from_options(env, base_args, option_args, options))
        p_sa = p_sa_from_rollouts(epoch_rollouts[-1]["all_rollouts"], sa_agg)
        plot_heatmap(p_sa.reshape(12, 11*3), save_path=f"{i}psa_codes.png")

        # define reward fn
        uniform_density_sa = np.ones(sa_agg.shape())
        l1_cov_reward = reward_shaping(1/(base_args["l1_eps"] * uniform_density_sa + p_sa))
        plot_heatmap(l1_cov_reward.reshape(12, 11*3), save_path=f"{i}l1_reward.png")
        reward_fn = SA_Reward(sa_agg, l1_cov_reward)
        # learn policy
        options.append(learn_policy(env, base_args, option_args, reward_fn, epoch_rollouts[-1]["all_rollouts"]))


    epoch_rollouts.append(collect_rollouts_from_options(env,base_args,option_args, options))
    return options, epoch_rollouts

def collect_run_random(base_args):
    s_agg, sa_agg = get_aggregator(base_args["env_name"], bin_res=1)
    env = gym.make(base_args["env_name"], render_mode="rgb_array")

    options = [get_random_agent(env_name= base_args["env_name"])]
    epoch_rollouts = []

    for i in range(base_args["num_epochs"]):
        epoch_rollouts.append(collect_rollouts_from_options(env,base_args, {}, options))
        options+=[get_random_agent(env_name= base_args["env_name"])]

    epoch_rollouts.append(collect_rollouts_from_options(env,base_args, {}, options))


    return epoch_rollouts


def compute_l1_from_run(base_args, adv_args, run):
    s_agg, sa_agg = get_aggregator(env_name=base_args["env_name"], bin_res=1)
    env = gym.make(base_args["env_name"], render_mode="rgb_array")


    l1_covs = []

    for epoch in run:
        epoch_rollouts = epoch["all_rollouts"]
        p_sa = p_sa_from_rollouts(epoch_rollouts, sa_agg)

        uniform_density_sa = np.ones(sa_agg.shape())
        l1_cov_reward = 1/(base_args["l1_eps"] * uniform_density_sa + p_sa) 
        reward_fn = SA_Reward(sa_agg, reward_shaping(l1_cov_reward))

        adv_policy= setup_agent(base_args, adv_args)

        rollouts = collect_rollouts(env, adv_policy, base_args["env_T"], base_args["num_rollouts"], reward_fn, **adv_args["rollout_args"])

        l1_covs.append(average_reward_from_rollouts(rollouts))
    return l1_covs













if __name__ == "__main__":
    mountaincar_args = {
        "l1_eps":1e-4, # regularizer epsilon for 
        "bin_res": 1,
        "env_name": "MountainCarContinuous-v0",
        "env_T":200,
        "num_rollouts":100,
        "num_epochs": 5,
        "reward_shaping_constant": 0
    }
    # base_args = {
    #     "gamma":0.999, # global discount factor
    #     "l1_eps":1e-4, # regularizer epsilon for 
    #     "bin_res": 1,
    #     "env_name": "CartPole-v1",
    #     "env_T":200,
    #     "num_rollouts":200,
    #     "num_epochs": 2,
    #     "reward_shaping_constant": 0
    # }


    Qlearning_args = {
        "policy": "Qlearning",
        "gamma":0.999,
        "lr":0.01,
        "online_epochs":3000,
        "offline_epochs":10,

        "learning_args": {
            "epsilon_start": 1.0,
            "epsilon_decay": 0.999,
            "decay_every": 1,
            "verbose": True,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.2,
        }
    }
    # more comprehensive qlearning args for l1 coverage
    Qlearning_args_l1 = {
        "policy": "Qlearning",
        "gamma":0.999,
        "lr":0.01,
        "online_epochs":5000,
        "offline_epochs":10,

        "learning_args": {
            "epsilon_start": 1.0,
            "epsilon_decay": 0.999,
            "decay_every": 3,
            "verbose": True,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.2,
        }
    }
    # options, trajectories =  collect_run_eigenoptions(mountaincar_args, Qlearning_args)
    # l1_covs = compute_l1_from_run(mountaincar_args, Qlearning_args_l1, trajectories)
    # print("l1_covs", l1_covs)


    options, trajectories =  collect_run_codex(mountaincar_args, Qlearning_args)
    l1_covs = compute_l1_from_run(mountaincar_args, Qlearning_args_l1, trajectories)
    print("l1_covs", l1_covs)