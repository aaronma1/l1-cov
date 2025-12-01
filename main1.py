from collect_runs import collect_run_codex, collect_run_sa_eigenoptions, collect_run_random, collect_run_maxent, compute_l1_from_run, compute_stats_from_run
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


def _run_experiment_eigenoptions(base_args, option_args, save_dir, run_id, save_fn=dump_pickle):
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


def _run_experiment_codex(base_args, option_args, save_dir, run_id, save_fn=dump_pickle):
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
            )
            for i in range(n_runs)
        ]
        for i, f in enumerate(as_completed(futures)):
            print(f"done run {i}")
            del f

def _run_experiment_maxent(base_args, option_args, save_dir, run_id, save_fn=dump_pickle):
    save_path_transitions = os.path.join(save_dir, f"part{run_id}_run.pkl")
    save_path_options = os.path.join(save_dir, f"part{run_id}_options.pkl")
    if os.path.exists(save_path_options) and os.path.exists(save_path_transitions):
        return
    transitions, options = collect_run_maxent(base_args, option_args)
    save_fn(transitions, save_path_transitions)
    save_fn(options, save_path_options)
def run_experiment_maxent(base_args, option_args, save_dir, n_runs, max_workers=16):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_experiment_maxent, 
                base_args=base_args, 
                option_args=option_args, 
                save_dir= save_dir,
                run_id = i,
            )
            for i in range(n_runs)
        ]
        for i, f in enumerate(as_completed(futures)):
            print(f"done run {i}")
            del f

def _run_experiment_random(base_args, save_dir, run_id, save_fn=dump_pickle):
    save_path_transitions = os.path.join(save_dir, f"part{run_id}_run.pkl")
    if os.path.exists(save_path_transitions):
        return
    transitions, _ = collect_run_random(base_args)
    save_fn(transitions, save_path_transitions)
    
def run_experiment_random(base_args, save_dir, n_runs, max_workers=16):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_experiment_random, 
                base_args=base_args, 
                save_dir= save_dir,
                run_id = i,
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


def _compute_stats_from_experiment(base_args, fpath, read_fn=read_pickle):
    runs = read_pickle(fpath)
    stats = compute_stats_from_run(base_args, runs)
    return stats
    

def compute_stats_from_experiments(base_args, exp_dump_path, max_workers=16):
    pickles = list_pickle_runs(exp_dump_path)
    save_path = os.path.join(exp_dump_path, "all_stats.pkl")
    stats = None
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_compute_stats_from_experiment, base_args=base_args, fpath=fpath)
            for fpath in pickles
        ]
        for i,f in enumerate(as_completed(futures)):
            run_stats =  f.result()

            if stats == None:
                stats = {}
                for key in run_stats.keys():
                    stats[key] = [run_stats[key]]
            else:
                for key in run_stats.keys():
                    stats[key].append(run_stats[key])
    dump_pickle(stats, save_path)


def run_experiments(base_args, option_args, adversery_args, N_RUNS, SAVE_DIR, MAX_WORKERS):
    os.makedirs(SAVE_DIR, exist_ok=True)
    params = {
        "args_codex": (base_args, option_args),
        "args_eigenoptions": (base_args, option_args),
        "l1_cov_args": (base_args, adversery_args),
    }
    dump_pickle(params, os.path.join(SAVE_DIR, "params.pkl"))

    print("#### collecting random rollouts ####")
    save_dir_random = os.path.join(SAVE_DIR, "random/")
    os.makedirs(save_dir_random, exist_ok=True)
    run_experiment_random(base_args, save_dir_random, n_runs=N_RUNS, max_workers=MAX_WORKERS)
    print("#### computing random l1 coverage ####")
    compute_l1_from_experiment(base_args, adversery_args, exp_dump_path=save_dir_random, max_workers=MAX_WORKERS)
    compute_stats_from_experiments(base_args, exp_dump_path=save_dir_random)

    print("#### collecting maxent rollouts ####")
    save_dir_maxent = os.path.join(SAVE_DIR, "maxent/")
    os.makedirs(save_dir_maxent, exist_ok=True)
    run_experiment_maxent(base_args, option_args, save_dir=save_dir_maxent, max_workers=MAX_WORKERS, n_runs=N_RUNS)
    compute_l1_from_experiment(base_args, adversery_args, exp_dump_path=save_dir_maxent, max_workers=MAX_WORKERS,)
    compute_stats_from_experiments(base_args, exp_dump_path=save_dir_maxent)


    print("#### collecting eigenoptions rollouts ####")
    save_dir_eigenoptions = os.path.join(SAVE_DIR, "eigenoptions/")
    os.makedirs(save_dir_eigenoptions, exist_ok=True)
    run_experiment_eigenoptions(base_args, option_args, save_dir=save_dir_eigenoptions, max_workers=MAX_WORKERS, n_runs=N_RUNS)
    print("#### computing eigenoptions l1 coverage ####")
    compute_l1_from_experiment(base_args, adversery_args, exp_dump_path=save_dir_eigenoptions, max_workers=MAX_WORKERS)
    compute_stats_from_experiments(base_args, exp_dump_path=save_dir_eigenoptions)


    print("#### collecting codex rollouts ####")
    save_dir_codex = os.path.join(SAVE_DIR, "codex/")
    os.makedirs(save_dir_codex, exist_ok=True)
    run_experiment_codex(base_args, option_args, save_dir=save_dir_codex, max_workers=MAX_WORKERS, n_runs=N_RUNS)
    print("#### computing codex l1 coverage ####")
    compute_l1_from_experiment(base_args, adversery_args, exp_dump_path=save_dir_codex, max_workers=MAX_WORKERS,)
    compute_stats_from_experiments(base_args, exp_dump_path=save_dir_codex)



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

    option_args = {
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

    adversery_args = {
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
    run_experiments(base_args, option_args, adversery_args, N_RUNS, SAVE_DIR, MAX_WORKERS)

def experiments_pendulum(SAVE_DIR, N_RUNS, MAX_WORKERS, epochs=15):
    base_args = {
        "l1_eps": 1e-4,  # regularizer epsilon for
        "bin_res": 1,
        "env_name": "Pendulum-v1",
        "env_T": 200,
        "num_rollouts": 400,
        "num_epochs": epochs,
    }

    option_args = {
        "policy": "Qlearning",
        "gamma": 0.99,
        "lr": 0.01,
        "online_epochs": 1000,
        "offline_epochs": 10,
        "learning_args": {
            "epsilon_start": 0.5,
            "epsilon_decay": 0.99,
            "decay_every": 1,
            "verbose": True,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.0,
        },
    }
    adversery_args = {
        # base_args = {
        "policy": "Qlearning",
        "gamma": 0.99,
        "lr": 0.01,
        "online_epochs": 20000,
        "offline_epochs": 0,
        "learning_args": {
            "epsilon_start": 0.0,
            "epsilon_decay": 0.999,
            "decay_every": 0,
            "verbose": True,
            "print_every": 1000,
        },
        "rollout_args": {
            "epsilon": 0.0,
        },
        "print_every": 100,
    }
    run_experiments(base_args, option_args, adversery_args, N_RUNS, SAVE_DIR, MAX_WORKERS)


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

    parser.add_argument(
        "--exp", 

    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    # multiprocessing.set_start_method("spawn", force=True) 
    experiments_mountaincar(SAVE_DIR=args.save_path, N_RUNS=args.n_runs,MAX_WORKERS=args.max_workers, epochs=args.epochs)
