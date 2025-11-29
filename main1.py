from collect_runs import collect_run_codex, collect_run_sa_eigenoptions, compute_l1_from_run 
from concurrent.futures import ProcessPoolExecutor, as_completed

import pickle
import os
import threading
import tempfile

import multiprocessing
import math

import argparse


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


def _run_experiment_eigenoptions(base_args, option_args, save_dir, run_id, save_fn):
    save_path_transitions = os.path.join(save_dir, f"part{run_id}_run.pkl")
    save_path_options = os.path.join(save_dir, f"part{run_id}_options.pkl")
    if os.path.exists(save_path_options) and os.path.exists(save_path_transitions):
        return
    transitions, options = collect_run_sa_eigenoptions(base_args, option_args)
    save_fn(transitions, save_path_transitions)
    save_fn(options, save_path_options)
def run_experiment_eigenoptions(base_args, option_args, save_dir, n_runs, max_workers=16):

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_experiment_eigenoptions, 
                base_args=base_args, 
                option_args=option_args, 
                save_dir= save_dir,
                run_id = i,
                save_fn=dump_pickle,
            )
            for i in range(n_runs)
        ]

        for i, f in enumerate(as_completed(futures)):
            print(f"done run {i}")
            del f


def _run_experiment_codex(base_args, option_args, save_dir, run_id, save_fn):
    save_path_transitions = os.path.join(save_dir, f"part{run_id}_run.pkl")
    save_path_options = os.path.join(save_dir, f"part{run_id}_options.pkl")
    if os.path.exists(save_path_options) and os.path.exists(save_path_transitions):
        return
    transitions, options = collect_run_codex(base_args, option_args)
    save_fn(transitions, save_path_transitions)
    save_fn(options, save_path_options)

def run_experiment_codex(base_args, option_args, save_dir, n_runs, max_workers=16):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_experiment_codex, 
                base_args=base_args, 
                option_args=option_args, 
                save_dir= save_dir,
                run_id = i,
                save_fn=dump_pickle
            )
            for i in range(n_runs)
        ]

        for i, f in enumerate(as_completed(futures)):
            print(f"done run {i}")
            del f



######################################
# L1 coverage computation 
#######################V##############
def list_pickle_runs(directory):
    """Return a sorted list of all .pkl files in a directory."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith("_run.pkl")
    )

def _compute_l1_from_experiment(base_args, adv_args, fpath, read_fn=read_pickle):
    run = read_fn(fpath)
    l1_covs, adv_policy = compute_l1_from_run(base_args, adv_args, run)
    return l1_covs



def compute_l1_from_experiment(base_args, adv_args, exp_dump_path, max_workers=16):
    pickles = list_pickle_runs(exp_dump_path)
    all_l1_covs = []
    save_path = os.path.join(exp_dump_path, "all_l1_covs.pkl")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_compute_l1_from_experiment, base_args=base_args, adv_args=adv_args, fpath=fpath)
            for fpath in pickles
        ]

        for _,f in enumerate(as_completed(futures)):
            l1 = f.result()
            all_l1_covs.append(l1)
            del f

    dump_pickle(all_l1_covs, save_path)


######################################
# Experiment Configurations
#######################V##############

def experiments_mountaincar(SAVE_DIR, MAX_WORKERS, N_RUNS, epochs=15):
    base_args = {
        "l1_eps":1e-4, # regularizer epsilon for 
        "bin_res": 1,
        "env_name": "MountainCarContinuous-v0",
        "env_T":200,
        "num_rollouts":400,
        "num_epochs": epochs,
        "reward_shaping_constant": -1
    }

    Qlearning_args_eigenoptions = {
        "policy": "Qlearning",
        "gamma":0.99,
        "lr":0.01,
        "online_epochs":1000,
        "offline_epochs":10,


        "learning_args": {
            "epsilon_start": 0.5,
            "epsilon_decay": 0.999,
            "decay_every": 1,
            "verbose": False,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.0,
        }
    }

    Qlearning_args_adversery = {
    # base_args = {
        "policy": "Qlearning",
        "gamma":0.99,
        "lr":0.01,
        "online_epochs":5000,
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
    save_dir_eigenoptions = os.path.join(SAVE_DIR, "eigenoptions/")
    os.makedirs(save_dir_eigenoptions, exist_ok=True)

    params = {
        "args_codex": (base_args, Qlearning_args_eigenoptions),
        "args_eigenoptions": (base_args, Qlearning_args_eigenoptions),
        "l1_cov_args": (base_args, Qlearning_args_adversery),
    }
    dump_pickle(params, os.path.join(SAVE_DIR, "params.pkl"))

    print("#### collecting eigenoptions rollouts ####")
    run_experiment_eigenoptions(base_args, Qlearning_args_eigenoptions, save_dir=save_dir_eigenoptions, max_workers=MAX_WORKERS, n_runs=N_RUNS)
    print("#### computing eigenoptions l1 coverage ####")
    compute_l1_from_experiment(base_args, Qlearning_args_adversery, exp_dump_path=save_dir_eigenoptions, max_workers=MAX_WORKERS)

    save_dir_codex = os.path.join(SAVE_DIR, "codex/")
    os.makedirs(save_dir_codex, exist_ok=True)

    print("#### collecting codex rollouts ####")
    run_experiment_codex(base_args, Qlearning_args_eigenoptions, save_dir=save_dir_codex, max_workers=MAX_WORKERS, n_runs=N_RUNS)
    print("#### computing codex l1 coverage ####")
    compute_l1_from_experiment(base_args, Qlearning_args_adversery, exp_dump_path=save_dir_codex, max_workers=MAX_WORKERS,)



def experiments_mountaincar_highbins():
    pass



    


def parse_args():
    parser = argparse.ArgumentParser(description="Job runner configuration")

    parser.add_argument(
        "--max_workers",
        type=int,
        required=True,
        help="Number of worker processes to spawn"
    )

    parser.add_argument(
        "--n_runs",
        type=int,
        required=True,
        help="Total number of jobs to run"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs to run (default: 1)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="out",
        help="Save path"
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    multiprocessing.set_start_method("spawn", force=True) 
    experiments_mountaincar(SAVE_DIR=args.save_path, N_RUNS=args.n_runs,MAX_WORKERS=args.max_workers, epochs=args.epochs)
