from lib.policies import (
    get_random_agent,
)
from lib.trajectories import (
    average_reward_from_rollouts,
    sr_from_rollouts,
    p_s_from_rollouts,
    p_sa_from_rollouts,
    sa_sr_from_rollouts,
    collect_rollouts,
)
from lib.qlearningpolicy import get_qlearning_agent
from lib.reinforce import get_reinforce_agent
from lib.aggregation import get_aggregator, S_Reward, SA_Reward, SAS_Reward, SS_Reward
import plotting

from scipy.sparse.linalg import eigsh
from scipy.stats import entropy
import gymnasium as gym
import numpy as np

###########################################
#           Utility Functions
###########################################

def setup_env(base_args):
    env = gym.make(base_args["env_name"], render_mode="rgb_array")
    s_agg, sa_agg = get_aggregator(base_args["env_name"], base_args["s_bins"], base_args["a_bins"])
    return env, s_agg, sa_agg

def setup_aggregator(base_args):
    return get_aggregator(base_args["env_name"], base_args["s_bins"], base_args["a_bins"])

def setup_env_exploring_starts(base_args):
    env = gym.make(base_args["env_name"], render_mode="rgb_array")
    class ESW(gym.Wrapper):
        def __init__(self, env, state_sampler):
            self.state_sampler = state_sampler
            self.env = env

        def reset(self, **kwargs):
            _, info = env.reset(**kwargs)
            new_state = self.state_sampler()
            self.env.state = np.array(new_state, dtype=float)
            return self.env.state, info

    def mountaincar_state_sampler():
        return np.random.uniform(low=[-1.2, -0.07], high=[0.6, 0.07])
    def cartpole_state_sampler():
        return np.random.uniform(low=[-0.5, -1.0, -0.2, -1.0], high=[0.5, 1.0, 0.2, 1.0])
    def acrobot_state_sampler():
        return np.random.uniform(low=[-1.0,-1.0,-1.0,-1.0, -12.566, -28.2743], high=[1.0,1.0,1.0,1.0, 12.566, 28.2743])
    def pendulum_state_sampler():
        return np.random.uniform(low=[-1.0, -1.0, -8.0], high=[1.0, 1.0,8.0])
    if base_args["env_name"] == "MountainCarContinuous-v0":
        return ESW(env, mountaincar_state_sampler)
    if base_args["env_name"] == "CartPole-v1":
        return ESW(env, cartpole_state_sampler)
    if base_args["env_name"] == "Pendulum-v1":
        return ESW(env, pendulum_state_sampler)
    if base_args["env_name"] == "Acrobot-v1":
        return ESW(env, acrobot_state_sampler)

def setup_agent(base_args, agent_args, max_rew = 0):
    if agent_args["policy"] == "Random":
        return get_random_agent(
            base_args["env_name"], base_args["a_bins"]

        )
    if agent_args["policy"] == "Qlearning":
        return get_qlearning_agent(
            base_args["env_name"], agent_args["gamma"], agent_args["lr"], a_bins = base_args["a_bins"], max_rew = max_rew
        )
    if agent_args["policy"] == "Reinforce":
        return get_reinforce_agent(
            base_args["env_name"], agent_args["gamma"], agent_args["lr"], a_bins = base_args["a_bins"]
        )


def reward_shaping(reward_fn):
    r_max = np.max(reward_fn)
    r_min = np.min(reward_fn)
    new_reward = reward_fn
    new_reward -= r_min
    if r_max != r_min:
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
    option = setup_agent(base_args, option_args, max_rew = np.max(reward_fn.reward_table))
    if transitions != None:

        option.learn_offline_policy(
            transitions, option_args["offline_epochs"], reward_fn, verbose=option_args["learning_args"]["verbose"]
        )

    option.learn_policy(
        env,
        base_args["env_T"],
        option_args["online_epochs"],
        reward_fn,
        **option_args["learning_args"],
    )
    return option

###########################################
#           Rollout Collection
###########################################

def eig_sparse(SR, k=131):
    SR = (SR + SR.T) / 2.0

    eigenvalues, eigenvectors = eigsh(SR, sigma=None, k=k, which="LM")
    idx = np.argsort(-eigenvalues.real)
    eigenvalues = np.real_if_close(eigenvalues, tol=1e5)[idx]
    eigenvectors = np.real_if_close(eigenvectors, tol=1e5)[:, idx]
    return eigenvectors, eigenvalues

def collect_run_eigenoptions(base_args, option_args):
    env, s_agg, _ = setup_env(base_args)

    options = [setup_agent(base_args, {"policy":"Random"})]
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

    env, s_agg, sa_agg = setup_env(base_args)
    options = [setup_agent(base_args, {"policy":"Random"})]
    epoch_rollouts = []

    for i in range(base_args["num_epochs"]):
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
        
        print(eigenvalues)
        plotting.plot_sa_heatmap(base_args["env_name"], top_eig, sa_agg, save_path=f"out/figs/{i}_topeig.png")
        plotting.plot_sa_heatmap(base_args["env_name"], p_sa, sa_agg, save_path=f"out/figs/{i}_psa.png")
        
        
        reward = SA_Reward(sa_agg, top_eig)
        # learn an option
        option_policy = learn_policy(
                env, base_args, option_args, reward, epoch_rollouts[-1]["all_rollouts"]
            )
        options.append(option_policy)


    # collect last epoch rollout
    epoch_rollouts.append(
        collect_rollouts_from_options(env, base_args, option_args, options)
    )
    return epoch_rollouts, options


def collect_run_codex(base_args, option_args):
    env, _, sa_agg = setup_env(base_args)
    epoch_rollouts = []
    options = [setup_agent(base_args, {"policy":"Random"})]
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
    def ent_reward(p_s):
        eps = np.sqrt(np.size(p_s))
        return 1/(p_s + eps)

    env, s_agg, _ = setup_env(base_args)
    options = [setup_agent(base_args, {"policy":"Random"})]
    epoch_rollouts = []
    for i in range(base_args["num_epochs"]):
        epoch_rollouts.append(
            collect_rollouts_from_options(env, base_args, option_args, options)
        )
        
        p_s = p_s_from_rollouts(epoch_rollouts[-1]["all_rollouts"], s_agg)
        reward_fn = S_Reward(s_agg, ent_reward(p_s))
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
    env, _, _ = setup_env(base_args)

    options = [setup_agent(base_args, {"policy":"Random"})]
    epoch_rollouts = []
    for i in range(base_args["num_epochs"]):
        epoch_rollouts.append(
            collect_rollouts_from_options(env, base_args, {"rollout_args": {}}, options)
        )
        options += [setup_agent(base_args, {"policy":"Random"})]
    epoch_rollouts.append(collect_rollouts_from_options(env, base_args, {"rollout_args": {}}, options))
    return epoch_rollouts, options

###########################################
#           Measurement Functions
###########################################

def compute_l1_from_run(base_args, adv_args, run):
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


def compute_stats_from_run(base_args, transitions):
    _, s_agg, sa_agg = setup_env(base_args)
    stats = {
        "p_sa_entropy": [],
        "p_s_entropy": [],
        "visited_s_ratio": [],
        "visited_sa_ratio": [],
    }
    for i, epoch in enumerate(transitions):
        rollouts = epoch["all_rollouts"]
        
        p_sa = p_sa_from_rollouts(rollouts, sa_agg)
        p_s = p_s_from_rollouts(rollouts, s_agg)

        stats["p_sa_entropy"].append(entropy(p_sa.flatten()))
        stats["p_s_entropy"].append(entropy(p_s.flatten()))
        stats["visited_sa_ratio"].append(np.count_nonzero(p_sa)/np.size(p_sa))
        stats["visited_s_ratio"].append(np.count_nonzero(p_s)/np.size(p_s))

    for key in stats.keys():
        stats[key] = np.array(stats[key])
    return stats


def run_exp_test(base_args, option_args, adv_args):

    trajectories_eo, _ = collect_run_sa_eigenoptions(base_args, option_args)
    trajectories_codex, _ = collect_run_codex(base_args, option_args)
    trajectories_maxent, _ = collect_run_maxent(base_args, option_args)


    _, s_agg, sa_agg = setup_env(base_args)

    for i in range(base_args["num_epochs"]):
        plotting.plot_sa_heatmap(base_args["env_name"], p_sa_from_rollouts(trajectories_codex[i]["all_rollouts"], sa_agg), sa_agg, save_path=f"out/figs/{i}_psa_cov.png")
        plotting.plot_sa_heatmap(base_args["env_name"], p_sa_from_rollouts(trajectories_eo[i]["all_rollouts"], sa_agg), sa_agg, save_path=f"out/figs/{i}_psa_eo.png")
        plotting.plot_sa_heatmap(base_args["env_name"], p_sa_from_rollouts(trajectories_maxent[i]["all_rollouts"], sa_agg), sa_agg, save_path=f"out/figs/{i}_psa_maxent.png")

    codex_l1_covs, _ = compute_l1_from_run(base_args, adv_args, trajectories_codex)
    eo_l1_covs, _ = compute_l1_from_run(base_args, adv_args, trajectories_eo)
    maxent_l1_covs, _ = compute_l1_from_run(base_args, adv_args, trajectories_maxent)
    print("codex l1 covs", codex_l1_covs)
    print("EO l1 covs", eo_l1_covs)
    print("Maxent l1 covs", maxent_l1_covs)

    stats_codex = compute_stats_from_run(base_args, trajectories_codex)
    stats_eo = compute_stats_from_run(base_args, trajectories_eo)
    stats_maxent = compute_stats_from_run(base_args, trajectories_codex)

    print("codex_stats", stats_codex)
    print("eo_stats", stats_eo)
    print("maxent_stats", stats_maxent)

    

import experiments
if __name__ == "__main__":
    args = experiments.mountaincar_qlearning_easy(epochs=5, l1_online=5000, verbose=True)    
    
    run_exp_test(*args)
