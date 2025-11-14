from collect_runs import collect_run_codex, collect_run_eigenoptions 
from concurrent.futures import ThreadPoolExecutor, as_completed

import pickle
import os
import threading


_pickle_lock = threading.Lock()


def dump_pickle(obj, save_dir):
    """
    Thread-safe, atomic pickle dump that overwrites the target file.
    Multiple threads can call this on the same path safely.
    """
    tmp_name = save_dir+ ".tmp"

    with _pickle_lock:
        # Write to temporary file
        with open(tmp_name, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Atomic replace; safe overwrite
        os.replace(tmp_name, save_dir)



def read_pickle(save_name):
    """
    Thread-safe pickle read.
    Safe even if other threads are simultaneously calling dump_pickle()
    on the same file.
    """
    with _pickle_lock:
        with open(save_name, "rb") as f:
            return pickle.load(f)

        






def run_experiment_eigenoptions(base_args, option_args, save_dir="out/eigenoptions/transitions.pickle", n_runs=100 ,max_workers=16):
    runs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(collect_run_eigenoptions, base_args=base_args, option_args=option_args)
            for i in range(n_runs)
        ]

        for f in as_completed(futures):
            runs.append(f.result())

    pickle.dump(runs, save_dir)




def run_experiment_codex(base_args, option_args, save_dir, num_workers=16):
    pass


def generate_plots(save_dir):
    pass


    



if __name__ == "__main__":
    mountaincar_args = {
        "gamma":0.999, # global discount factor
        "l1_eps":1e-4, # regularizer epsilon for 
        "bin_res": 1,
        "env_name": "MountainCarContinuous-v0",
        "env_T":200,
        "num_rollouts":100,
        "num_epochs": 15,
        "reward_shaping_constant": -1
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
        "lr":0.1,
        "train_steps":200,
        "train_epochs":300,

        "learning_args": {
            "epsilon_start": 0.1,
            "epsilon_decay": 0.999,
            "decay_every": 1,
            "verbose":True,
            "print_every": 1000
        }

    }