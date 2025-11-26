from collect_runs import collect_run_codex, collect_run_sa_eigenoptions, compute_l1_from_run 
from concurrent.futures import ProcessPoolExecutor, as_completed

import pickle
import os
import threading
import tempfile

import multiprocessing
import math


######################################
# Pickle Helpers
######################################

def dump_pickle(obj, save_path):
    """Atomic write — safe against parallel saves or crashes"""
    dirpath = os.path.dirname(save_path)
    fd, tmppath = tempfile.mkstemp(dir=dirpath)
    os.close(fd)
    with open(tmppath, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmppath, save_path)



def read_pickle(save_name):
    """
    Thread-safe pickle read.
    Safe even if other threads are simultaneously calling dump_pickle()
    on the same file.
    """
    with open(save_name, "rb") as f:
        obj = pickle.load(f)
        return obj


######################################
# Experiment Runners
######################################

def run_experiment_eigenoptions(base_args, option_args, save_dir, n_runs, max_workers=16):
    n_files = math.ceil(n_runs/max_workers)
    for i in range(n_files):
        filepath = os.path.join(save_dir, f"part{i}_runs.pkl")
        if not os.path.exists(filepath):
            _run_experiment_eigenoptions(base_args, option_args, filepath, max_workers)

def _run_experiment_eigenoptions(base_args, option_args, save_path, max_workers):
    runs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(collect_run_sa_eigenoptions, base_args=base_args, option_args=option_args)
            for i in range(max_workers)
        ]

        for i,f in enumerate(as_completed(futures)):
            transitions, options = f.result()
            runs.append(transitions)
            print(f"done run {i}")

    dump_pickle(runs, save_path)

def run_experiment_codex(base_args, option_args, save_dir, n_runs, max_workers):
    n_files = math.ceil(n_runs/max_workers)
    for i in range(n_files):
        filepath = os.path.join(save_dir, f"part{i}_runs.pkl")
        if not os.path.exists(filepath):
            _run_experiment_codex(base_args, option_args, filepath, max_workers)


def _run_experiment_codex(base_args, option_args, save_path, max_workers=16):
    #load checkpoint
    runs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(collect_run_codex, base_args=base_args, option_args=option_args)
            for i in range(max_workers)
        ]
        for i,f in enumerate(as_completed(futures)):
            transitions, options = f.result()
            runs.append(transitions)
            print(f"done run {i}, {len(runs)}")
    dump_pickle(runs, save_path)


######################################
# L1 coverage computation 
#######################V##############
def list_pickle_runs(directory):
    """Return a sorted list of all .pkl files in a directory."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith("_runs.pkl")
    )

def compute_l1_from_experiment(base_args, adv_args, exp_dump_path, max_workers=16):
    print(f"Computing l1 coverage for runs in {exp_dump_path}")
    pickles = list_pickle_runs(exp_dump_path)
    print(pickles)
    all_l1_covs = []
    save_path = os.path.join(exp_dump_path, "all_l1_covs.pkl")

    for fpath in pickles:
        print(fpath)
        runs = read_pickle(fpath)
        print(len(runs))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(compute_l1_from_run, base_args=base_args, adv_args=adv_args, run=runs[i])
                for i in range(len(runs))
            ]

            for _,f in enumerate(as_completed(futures)):
                l1 = f.result()
                all_l1_covs.append(l1)
    dump_pickle(all_l1_covs, save_path)





######################################
# Experiment Configurations
#######################V##############

def experiments_mountaincar():
    base_args = {
        "l1_eps":1e-4, # regularizer epsilon for 
        "bin_res": 1,
        "env_name": "MountainCarContinuous-v0",
        "env_T":200,
        "num_rollouts":400,
        "num_epochs": 15,
        "reward_shaping_constant": -1
    }

    Qlearning_args_eigenoptions = {
        "policy": "Qlearning",
        "gamma":0.999,
        "lr":0.01,
        "online_epochs":5000,
        "offline_epochs":20,


        "learning_args": {
            "epsilon_start": 0.5,
            "epsilon_decay": 0.999,
            "decay_every": 1,
            "verbose": False,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.2,
        }
    }

    Qlearning_args_adversery = {
    # base_args = {
        "policy": "Qlearning",
        "gamma":0.999,
        "lr":0.01,
        "online_epochs":10000,
        "offline_epochs":0,

        "learning_args": {
            "epsilon_start": 0.5,
            "epsilon_decay": 0.999,
            "decay_every": 3,
            "verbose": False,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.0,
        }
    }

    print("#### collecting eigenoptions rollouts ####")
    run_experiment_eigenoptions(base_args, Qlearning_args_eigenoptions, save_dir="out/mountaincar/eigenoptions", max_workers=MAX_WORKERS, n_runs=N_RUNS)
    print("#### computing codex l1 coverage ####")
    compute_l1_from_experiment(base_args, Qlearning_args_adversery, exp_dump_path="out/mountaincar/eigenoptions", max_workers=MAX_WORKERS)

    print("#### collecting codex rollouts ####")
    run_experiment_codex(base_args, Qlearning_args_eigenoptions, save_dir="out/mountaincar/codex", max_workers=MAX_WORKERS, n_runs=N_RUNS)
    print("#### computing codex l1 coverage ####")
    compute_l1_from_experiment(base_args, Qlearning_args_adversery, exp_dump_path="out/mountaincar/codex", max_workers=MAX_WORKERS,)

    # Qlearning_args_l1 = {
    # # base_args = {
    #     "policy": "Qlearning",
    #     "gamma":0.999,
    #     "lr":0.01,
    #     "online_epochs":10000,
    #     "offline_epochs":0,

    #     "learning_args": {
    #         "epsilon_start": 0.5,
    #         "epsilon_decay": 0.999,
    #         "decay_every": 3,
    #         "verbose": False,
    #         "print_every": 100,
    #     },
    #     "rollout_args": {
    #         "epsilon": 0.2,
    #     }
    # }
    # compute_l1_from_experiment(base_args, Qlearning_args_l1, "out/mountaincar/transitions_mountaincar.pickle")



def experiments_mountaincar_highbins():
    pass



    



if __name__ == "__main__":

    multiprocessing.set_start_method("spawn", force=True)
    N_RUNS=104
    MAX_WORKERS=8
    experiments_mountaincar()
