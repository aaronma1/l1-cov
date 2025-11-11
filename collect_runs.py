from policies import get_random_agent, sr_from_rollouts, p_s_from_rollouts, collect_rollouts
from qlearningpolicy import get_qlearning_agent
from aggregation import get_aggregator, S_Reward, SA_Reward
from plotting import plot_heatmap
import gymnasium as gym
import numpy as np


def setup_agent(base_args, agent_args):

    if agent_args["policy"] == "PolicyGrad":
        pass

    if agent_args["policy"] == "QLearning":
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
        
        eigenvalues, eigenvectors = np.linalg.eigh(SR)
        idx = np.argsort(-eigenvalues.real)
        eigenvalues = np.real_if_close(eigenvalues, tol=1e5)[idx]
        eigenvectors = np.real_if_close(eigenvectors, tol=1e5)[idx]
        return eigenvectors, eigenvalues

    s_agg, _ = get_aggregator(base_args["env_name"], bin_res=1)
    env = gym.make(base_args["env_name"], render_mode="rgb_array")

    options = []

    random_option = get_random_agent(base_args["env_name"])
    random_rollouts = collect_rollouts(env, random_option, base_args["env_T"], base_args["num_rollouts"])
    SR = sr_from_rollouts(random_rollouts,s_agg, base_args["gamma"])
    eigenvectors , _ = eig(SR)
    print(eigenvectors.shape)
    plot_heatmap(s_agg.unflatten_state_table(eigenvectors[:, 0]), save_path="eig0.png")








    for i in range(base_args["num_epochs"]):
        option = setup_agent(base_args, option_args)

        





        

    

        
     




def collect_run_codex(base_args, option_args, adversery_args, out_file):
    pass

def collect_run_random(base_args, adversery_args, out_file):
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

    policy_grad_args = {
        "policy": "PolicyGrad", 
        "lr":1e-3,
        "train_steps":400,
        "train_epochs":1000,
    }

    Qlearning_args = {
        "policy": "Qlearning",
        # "num-tiles": 8,
        # "num-tilings": 16, 
        # "IHT_SIZE": 4096,

        "train_steps":400,
        "train_epochs":5000,
        "eps_start":1.0,
        "eps_decay":0.999,
    }

    collect_run_eigenoptions(base_args, Qlearning_args, out_file="")
    





    # rod_cycle(env, state_tc)

