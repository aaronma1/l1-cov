


import argparse
import os
import tempfile
import pickle



from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from collect_runs import collect_run_codex, collect_run_eigenoptions, collect_run_maxent, collect_run_random, collect_run_sa_eigenoptions, compute_l1_from_run, compute_stats_from_run


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


def _run_experiment_eigenoptions(base_args, option_args, save_dir, run_id):
    os.makedirs(save_dir, exist_ok=True)
    save_path_transitions = os.path.join(save_dir, f"part{run_id}_run.pkl")
    save_path_options = os.path.join(save_dir, f"part{run_id}_options.pkl")

    if os.path.exists(save_path_options) and os.path.exists(save_path_transitions):
        return
    try:
        transitions, options = collect_run_eigenoptions(base_args, option_args, node_num=run_id)
        dump_pickle(transitions, save_path_transitions)
        dump_pickle(options, save_path_options)
    except Exception as e:
        print("EXCEPTION:", e)
        import traceback
        traceback.print_exc()    

def _run_experiment_sa_eigenoptions(base_args, option_args, save_dir, run_id):
    os.makedirs(save_dir, exist_ok=True)
    save_path_transitions = os.path.join(save_dir, f"part{run_id}_run.pkl")
    save_path_options = os.path.join(save_dir, f"part{run_id}_options.pkl")
    if os.path.exists(save_path_options) and os.path.exists(save_path_transitions):
        return

    try:
        transitions, options = collect_run_sa_eigenoptions(base_args, option_args, node_num=run_id)
        dump_pickle(transitions, save_path_transitions)
        dump_pickle(options, save_path_options)
    except Exception as e:
        print("EXCEPTION:", e)
        import traceback
        traceback.print_exc()    


    dump_pickle(transitions, save_path_transitions)
    dump_pickle(options, save_path_options)

def _run_experiment_codex(base_args, option_args, save_dir, run_id):
    os.makedirs(save_dir, exist_ok=True)
    save_path_transitions = os.path.join(save_dir, f"part{run_id}_run.pkl")
    save_path_options = os.path.join(save_dir, f"part{run_id}_options.pkl")
    if os.path.exists(save_path_options) and os.path.exists(save_path_transitions):
        return
    try:
        transitions, options = collect_run_codex(base_args, option_args, node_num=run_id)
        dump_pickle(transitions, save_path_transitions)
        dump_pickle(options, save_path_options)
    except Exception as e:
        print("EXCEPTION:", e)
        import traceback
        traceback.print_exc()    
    dump_pickle(transitions, save_path_transitions)
    dump_pickle(options, save_path_options)
    

def _run_experiment_maxent(base_args, option_args, save_dir, run_id):
    os.makedirs(save_dir, exist_ok=True)
    save_path_transitions = os.path.join(save_dir, f"part{run_id}_run.pkl")
    save_path_options = os.path.join(save_dir, f"part{run_id}_options.pkl")
    if os.path.exists(save_path_options) and os.path.exists(save_path_transitions):
        return
    try:
        transitions, options = collect_run_maxent(base_args, option_args, node_num=run_id)
        dump_pickle(transitions, save_path_transitions)
        dump_pickle(options, save_path_options)
    except Exception as e:
        print("EXCEPTION:", e)
        import traceback
        traceback.print_exc()    

def _run_experiment_random(base_args, save_dir, run_id):
    os.makedirs(save_dir, exist_ok=True)
    save_path_transitions = os.path.join(save_dir, f"part{run_id}_run.pkl")
    if os.path.exists(save_path_transitions):
        return
    try:
        transitions, _ = collect_run_random(base_args)
        dump_pickle(transitions, save_path_transitions)
    except Exception as e:
        print("EXCEPTION:", e)
        import traceback
        traceback.print_exc()    



def collect_sample(base_args, option_args, save_dir, run_id):
    exps = [
        ("random", _run_experiment_random),
        ("eigenoptions", _run_experiment_eigenoptions),
        ("codex", _run_experiment_codex),
        ("maxent", _run_experiment_maxent),
        ("sa_eigenoptions", _run_experiment_sa_eigenoptions),
    ]


    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                exp,
                base_args,
                option_args, 
                os.path.join(save_dir, file),
                run_id
            ) 
            for file, exp in exps
        ]

        for i,f in enumerate(as_completed(futures)):
            pass
    
def _compute_l1_from_experiment(base_args, adv_args, save_dir, run_id):
    transition_file = os.path.join(save_dir, f"part{run_id}_run.pkl")
    run = read_pickle(transition_file)
    l1_covs, adv_policy = compute_l1_from_run(base_args, adv_args, run)

    dump_file = os.path.join(save_dir, f"part{run_id}_l1.pkl")
    dump_pickle(l1_covs, dump_file)


def _compute_stats_from_experiment(base_args, adv_args, save_dir, run_id):
    transition_file = os.path.join(save_dir, f"part{run_id}_run.pkl")
    run = read_pickle(transition_file)
    stats = compute_stats_from_run(base_args, adv_args, run)

    dump_file = os.path.join(save_dir, f"part{run_id}_stats.pkl")
    dump_pickle(stats, dump_file)

def collect(save_dir, prefix):

    exp_dir = os.path.join(save_dir, prefix)

    l1_pkls =  sorted(
        os.path.join(exp_dir, f)
        for f in os.listdir(exp_dir)
        if f.endswith("_l1.pkl")
    )


    stats_pkls = sorted(
        os.path.join(exp_dir, f)
        for f in os.listdir(exp_dir)
        if f.endswith("_l1.pkl")
    )

    l1_pkl = os.path.join(save_dir, f"{prefix}_l1_covs.pkl")
    all_l1_covs = []
    for fname in l1_pkls():
        l1 = read_pickle(fname)
        all_l1_covs.append(l1)
    all_l1_covs = np.array(all_l1_covs)
    dump_pickle(all_l1_covs, l1_pkl)
        

    stats_pkl = os.path.join(save_dir, f"{prefix}_stats.pkl")
    run_stats = None
    for fname in stats_pkls():
        run_stats =  read_pickle(fname)
        if stats == None:
            stats = {}
            for key in run_stats.keys():
                stats[key] = [run_stats[key]]
        else:
            for key in run_stats.keys():
                stats[key].append(run_stats[key])
        dump_pickle(run_stats, stats_pkl)


def compute(base_args, adv_args, save_dir, run_id):
    exps = [
        ("eigenoptions", _run_experiment_eigenoptions),
        ("sa_eigenoptions", _run_experiment_sa_eigenoptions),
        ("codex", _run_experiment_codex),
        ("maxent", _run_experiment_maxent),
        ("random", _run_experiment_random)
    ]
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                _compute_l1_from_experiment,
                base_args,
                adv_args, 
                os.path.join(save_dir, file),
                run_id
            ) 
            for file, exp  in exps
        ] + [
            executor.submit(
                _compute_stats_from_experiment,
                base_args,
                adv_args, 
                os.path.join(save_dir, file),
                run_id
            ) 
            for file, exp in exps
        ]

        for i,f in enumerate(as_completed(futures)):
            pass
    
        
    if run_id == 0:
        for file, _ in exps:
            collect(save_dir, file)





    



def parse_args():
    parser = argparse.ArgumentParser(description="Job runner configuration")

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to run (default: 1)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out",
        help="Save path"
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="MountainCar",
        help="MountainCar | Pendulum | CartPole"
    )
    parser.add_argument(
        "--run_id",
        type=int,
        help="run id"
    )

    return parser.parse_args()



import experiments
if __name__ == "__main__":
    
    kwargs = parse_args()
    
    if kwargs.exp_name == "MountainCar":
        base_args, option_args, adv_args = experiments.mountaincar_qlearning_easy(epochs=kwargs.epochs, l1_online=10000)
    if kwargs.exp_name == "Pendulum":
        base_args, option_args, adv_args = experiments.pendulum_default_qlearning(epochs=kwargs.epochs, l1_online=20000)
    if kwargs.exp_name == "CartPole":
        base_args, option_args, adv_args = experiments.cartpole_default(epochs=kwargs.epochs, l1_online=40000)


    save_dir = kwargs.save_dir
    run_id = kwargs.run_id


    os.makedirs(save_dir, exist_ok=True)

    if run_id == 0:
        dump_pickle({
            "base_args":base_args,
            "option_args": option_args,
            "adv_args": adv_args
            }, os.path.join(save_dir, "params.pkl"))

    collect_sample(base_args, option_args, save_dir, run_id)
    compute(base_args, adv_args, save_dir, run_id)