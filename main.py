from collect_runs import collect_run_codex, collect_run_sa_eigenoptions, compute_l1_from_run 
from concurrent.futures import ProcessPoolExecutor, as_completed

import pickle
import os
import threading
import tempfile

import multiprocessing



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
        return pickle.load(f)



def run_experiment_eigenoptions(base_args, adv_args, save_path, n_runs=100 ,max_workers=16, save_every=10):
    runs= []
    #load checkpoint
    if os.path.exists(save_path):
        runs = read_pickle(save_path)

    n_runs -= len(runs)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(collect_run_sa_eigenoptions, base_args=base_args, option_args=adv_args)
            for i in range(n_runs)
        ]

        for i,f in enumerate(as_completed(futures)):
            transitions, options = f.result()
            runs.append(transitions)
            print(f"done run {i}")
            if i % save_every == 0 and i != 0:
                dump_pickle(runs, save_path)

    dump_pickle(runs, save_path)



def compute_l1_from_experiment(base_args, adv_args, exp_dump_path, save_path, max_workers=4, save_every=10):
    # load checkpoint



    if not os.path.exists(exp_dump_path):
        print("Error experiments not found")
        return

    runs = read_pickle(exp_dump_path)
    l1_covs = []
    n_runs = len(runs)



    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(compute_l1_from_run, base_args=base_args, adv_args=adv_args, run=runs[i])
            for i in range(n_runs)
        ]

        for i,f in enumerate(as_completed(futures)):
            l1 = f.result()
            l1_covs.append(l1)
            print(f"done run {i}")
            if i % save_every == 0 and i != 0:
                dump_pickle(runs, save_path)
    dump_pickle(l1_covs, save_path)





def run_experiment_codex(base_args, option_args, save_path, max_workers=16, n_runs=100, save_every=10):
    #load checkpoint
    runs = []
    if os.path.exists(save_path):
        runs = read_pickle(save_path)

    n_runs -= len(runs)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(collect_run_codex, base_args=base_args, option_args=option_args)
            for i in range(n_runs)
        ]

        for i,f in enumerate(as_completed(futures)):
            transitions, options = f.result()
            runs.append(transitions)
            print(f"done run {i}")
            if i % save_every == 0 and i != 0:
                dump_pickle(runs, save_path)

    dump_pickle(runs, save_path)





def experiments_cartpole():
    pass

    

def experiments_mountaincar():
    base_args = {
        "l1_eps":1e-4, # regularizer epsilon for 
        "bin_res": 1,
        "env_name": "MountainCarContinuous-v0",
        "env_T":200,
        "num_rollouts":400,
        "num_epochs": 2,
        "reward_shaping_constant": -1
    }

    Qlearning_args_eigenoptions = {
        "policy": "Qlearning",
        "gamma":0.999,
        "lr":0.01,
        "online_epochs":1000,
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
            "epsilon": 0.2,
        }
    }

    run_experiment_eigenoptions(base_args, Qlearning_args_eigenoptions, save_path="out/mountaincar/runs_sa_eigenoptions.pickle", max_workers=MAX_WORKERS, n_runs=N_RUNS)
    compute_l1_from_experiment(base_args, Qlearning_args_adversery,exp_dump_path="out/mountaincar/runs_sa_eigenoptions.pickle", save_path="out/mountaincar/runs_sa_eigenoptions_l1.pickle", max_workers=MAX_WORKERS,)

    run_experiment_codex(base_args, Qlearning_args_eigenoptions, save_path="out/mountaincar/runs_codex.pickle", max_workers=MAX_WORKERS, n_runs=N_RUNS)
    compute_l1_from_experiment(base_args, Qlearning_args_adversery,exp_dump_path="out/mountaincar/runs_codex.pickle", save_path="out/mountaincar/runs_codex_l1.pickle", max_workers=MAX_WORKERS,)

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
    N_RUNS=5
    MAX_WORKERS=8
    experiments_mountaincar()