from policies import get_random_agent, sr_from_rollouts,p_s_from_rollouts, p_s_from_rollouts_1, collect_rollouts
from qlearningpolicy import get_qlearning_agent
from aggregation import get_aggregator, S_Reward, SA_Reward, unflatten_state
from plotting import plot_heatmap
import gymnasium as gym
import numpy as np


def setup_agent(base_args, agent_args):

    if agent_args["policy"] == "PolicyGrad":
        pass

    if agent_args["policy"] == "Qlearning":
        return get_qlearning_agent(base_args["env_name"], agent_args["gamma"], agent_args["lr"])





"""
The different collect_run functions should return a list of lists of trajectories
[[t1,t2,....], [t1,t2,....] ...] shape (num_epochs, num_rollouts * epoch, trajectory length)
we allow more rollouts every epoch because we have more options to evaluate
Also, we should keep track of how many environment interaction steps we have to do
"""    




def collect_run_eigenoptions(base_args, option_args, out_file):


    def eig(SR):
        SR = (SR+ SR.T)/2.0
        
        eigenvalues, eigenvectors = np.linalg.eig(SR)
        idx = np.argsort(-eigenvalues.real)
        eigenvalues = np.real_if_close(eigenvalues, tol=1e5)[idx]
        eigenvectors = np.real_if_close(eigenvectors, tol=1e5)[:, idx]
        return eigenvectors, eigenvalues

    s_agg, _ = get_aggregator(base_args["env_name"], bin_res=base_args["bin_res"])
    env = gym.make(base_args["env_name"], render_mode="rgb_array")

    options = []
    random_option = get_random_agent(base_args["env_name"])
    options.append(random_option)

    random_rollouts = collect_rollouts(env, random_option, base_args["env_T"], base_args["num_rollouts"])
    SR = sr_from_rollouts(random_rollouts,s_agg, base_args["gamma"])

    epoch_rollouts = []
    
    for i in range(base_args["num_epochs"]):

        # 1. compute SR from rollouts

        rollouts = []
        for opt in options:
            rollouts += collect_rollouts(env, opt, base_args["env_T"], base_args["num_rollouts"], epsilon=0.2)
        epoch_rollouts.append(rollouts)
        
        SR= sr_from_rollouts(rollouts, s_agg)
        p_s =  p_s_from_rollouts(rollouts, s_agg)
        eigenvectors , eigenvalues = eig(SR)

        #define reward using eigenoption, prioritize unvisited states
        top_eig = s_agg.unflatten_s_table(eigenvectors[:, 0])
        if np.dot(top_eig.flatten(), p_s.flatten()) > 0:
            top_eig = -top_eig
        reward = S_Reward(s_agg, top_eig)
        option = setup_agent(base_args, option_args)
        option.learn_policy(env, 200, 3000, reward)
        options.append(option)

    # collect last epoch rollout
    rollouts = []
    for opt in options:
        rollouts += collect_rollouts(env, opt, base_args["env_T"], base_args["num_rollouts"], epsilon=0.2)
    epoch_rollouts.append(rollouts)

    return options, epoch_rollouts

        
def collect_run_codex(base_args, option_args, adversery_args, out_file):
    # policies = []
    pass


    # for i in range():





def collect_run_random(base_args, adversery_args, out_file):

    # for i in range(epochs):
    pass



#thread safe dump pickle.
def dump_pickle(obj, save_name):
    pass



if __name__ == "__main__":
    base_args = {
        "gamma":0.999, # global discount factor
        "l1-eps":1e-4, # regularizer epsilon for 
        "bin_res": 1,
        "env_name": "MountainCarContinuous-v0",
        "env_T":200,
        "num_rollouts":100,
        "num_epochs": 15,
    }
    base_args = {
        "gamma":0.999, # global discount factor
        "l1-eps":1e-4, # regularizer epsilon for 
        "bin_res": 1,
        "env_name": "CartPole-v1",
        "env_T":200,
        "num_rollouts":100,
        "num_epochs": 15,
    }

    policy_grad_args = {
        "policy": "PolicyGrad", 
        "lr":1e-3,
        "train_steps":400,
        "train_epochs":1000,
    }

    Qlearning_args = {
        "policy": "Qlearning",
        "gamma":0.999,
        "lr":0.1,
    
        # "num-tiles": 8,
        # "num-tilings": 16, 
        # "IHT_SIZE": 4096,

        "train_steps":400,
        "train_epochs":5000,
        "eps_start":1.0,
        "eps_decay":0.999,
    }

    options, trajectories =  collect_run_eigenoptions(base_args, Qlearning_args, out_file="")
    

    







    # rod_cycle(env, state_tc)

