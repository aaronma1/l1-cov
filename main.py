from collect_runs import collect_run_codex, collect_run_eigenoptions 
from concurrent.futures import ProcessPoolExecutor, as_completed

import pickle
import os
import threading
import tempfile



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



def run_experiment_eigenoptions(base_args, option_args, save_dir, fname="transitions_eigenoptions.pickle", n_runs=100 ,max_workers=16, save_every=10):
    save_path = os.path.join(save_dir, fname)
    # load checkpoint

    runs = []
    if os.path.exists(save_path):
        runs = read_pickle(save_path)

    n_runs -= len(runs)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(collect_run_eigenoptions, base_args=base_args, option_args=option_args)
            for i in range(n_runs)
        ]

        for i,f in enumerate(as_completed(futures)):
            runs.append(f.result())

            print(f"done run {i}")

            if i % save_every == 0 and i != 0:
                dump_pickle(runs, save_path)
    dump_pickle(runs, save_path)



def compute_l1_from_exp(base_args, adv_args, save_dir):
    pass





def run_experiment_codex(base_args, option_args, save_dir, num_workers=16):
    pass


def generate_plots(save_dir):
    pass




def experiments_cartpole():
    pass

    

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

    run_experiment_eigenoptions(base_args, Qlearning_args_eigenoptions, save_dir="out/mountaincar")



def experiments_mountaincar_highbins():
    pass



    



if __name__ == "__main__":
    experiments_mountaincar()