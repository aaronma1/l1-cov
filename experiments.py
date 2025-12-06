def mountaincar_qlearning_easy(epochs=15, l1_online=5000, verbose=False):
    base_args = {
        "l1_eps": 1e-4,  # regularizer epsilon for
        "env_name": "MountainCarContinuous-v0",
        "s_bins":[12, 11],
        "a_bins":[3],
        "env_T": 200,
        "num_rollouts": 400,
        "num_epochs": epochs,
    }

    option_args = {
        "policy": "Qlearning",
        "gamma":0.99,
        "lr":0.01,
        "online_epochs":200,
        "offline_epochs":5,
        "learning_args": {
            "epsilon_start": 0.1,
            "epsilon_decay": 0.999,
            "decay_every": 1,
            "verbose": verbose,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.0,
        }
    }
    # more comprehensive qlearning args for l1 coverage
    adv_args = {
        "policy": "Qlearning",
        "gamma": 0.99,
        "lr": 0.01,
        "online_epochs": l1_online,
        "offline_epochs": 0,
        "learning_args": {
            "epsilon_start": 0.3,
            "epsilon_decay": 0.999,
            "decay_every": 1,
            "verbose": verbose,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.0,
        },
        "print_every": 100,
    }
    return base_args, option_args, adv_args

def mountaincar_qlearning_hard(epochs=15, l1_online=5000, verbose=False):
    base_args = {
        "l1_eps": 1e-4,  # regularizer epsilon for
        "env_name": "MountainCarContinuous-v0",
        "s_bins":[18, 16],
        "a_bins":[7],
        "env_T": 200,
        "num_rollouts": 400,
        "num_epochs": epochs,
    }
    option_args = {
        "policy": "Qlearning",
        "gamma":0.99,
        "lr":0.01,
        "online_epochs":1000,
        "offline_epochs":5,
        "learning_args": {
            "epsilon_start": 0.1,
            "epsilon_decay": 0.999,
            "decay_every": 1,
            "verbose": verbose,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.1,
        }
    }
    # more comprehensive qlearning args for l1 coverage
    adv_args = {
        "policy": "Qlearning",
        "gamma": 0.99,
        "lr": 0.01,
        "online_epochs": l1_online,
        "offline_epochs": 0,
        "learning_args": {
            "epsilon_start": 0.3,
            "epsilon_decay": 0.999,
            "decay_every": 1,
            "verbose": verbose,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.0,
        },
        "print_every": 100,
    }

    return base_args, option_args, adv_args

#################################################################
# PENDULUM
#################################################################
    
def pendulum_default_qlearning(epochs=15, l1_online=5000, verbose=False):
    base_args = {
        "l1_eps": 1e-4,  # regularizer epsilon for
        "s_bins": [10, 10, 12], 
        "a_bins": [5],
        "env_name": "Pendulum-v1",
        "env_T": 200,
        "num_rollouts": 400,
        "num_epochs": epochs,
    }
        
    option_args = {
        "policy": "Qlearning",
        "gamma":0.99,
        "lr":0.01,
        "online_epochs":2000,
        "offline_epochs":5,
        "learning_args": {
            "epsilon_start": 0.1,
            "epsilon_decay": 0.999,
            "decay_every": 1,
            "verbose": verbose,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.0,
        }
    }
    # more comprehensive qlearning args for l1 coverage
    adv_args = {
        "policy": "Qlearning",
        "gamma": 0.99,
        "lr": 0.01,
        "online_epochs": l1_online,
        "offline_epochs": 10000,
        "learning_args": {
            "epsilon_start": 0.0,
            "epsilon_decay": 0.999,
            "decay_every": 1,
            "verbose": verbose,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.0,
        },
        "print_every": 100,
    }

    return base_args, option_args, adv_args

def pendulum_default(epochs=15, l1_online=5000, verbose=False):
    base_args = {
        "l1_eps": 1e-4,  # regularizer epsilon for
        "s_bins": [8,8,8], 
        "a_bins": [3],
        "env_name": "Pendulum-v1",
        "env_T": 200,
        "num_rollouts": 400,
        "num_epochs": epochs,
    }
        
    option_args = {
        "policy": "Reinforce",
        "gamma":0.99,
        "lr":0.01,
        "online_epochs":1000,
        "offline_epochs":15,
        "learning_args": {
            "update_every": 5,
            "verbose": verbose,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.1,
        }
    }
    # more comprehensive qlearning args for l1 coverage
    adv_args = {
        "policy": "Reinforce",
        "gamma": 0.99,
        "lr": 0.01,
        "online_epochs": l1_online,
        "offline_epochs": 0,
        "learning_args": {
            "update_every": 5,
            "verbose": verbose,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.0,
        },
    }

    return base_args, option_args, adv_args
    
def cartpole_default(epochs=15, l1_online=5000, verbose=False):
    base_args = {
        "l1_eps": 1e-4,  # regularizer epsilon for
        "s_bins": [8,8,10,8], 
        "a_bins": None,
        "env_name": "CartPole-v1",
        "env_T": 200,
        "num_rollouts": 300,
        "num_epochs": epochs,
    }
        
    option_args = {
        "policy": "Qlearning",
        "gamma":0.99,
        "lr":0.01,
        "online_epochs":2000,
        "offline_epochs":15,
        "learning_args": {
            "epsilon_start": 0.1,
            "epsilon_decay": 0.999,
            "decay_every": 1,
            "verbose": verbose,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.0,
        }
    }
    # more comprehensive qlearning args for l1 coverage
    adv_args = {
        "policy": "Qlearning",
        "gamma": 0.99,
        "lr": 0.01,
        "online_epochs": l1_online,
        "offline_epochs": 0,
        "learning_args": {
            "epsilon_start": 0.01,
            "epsilon_decay": 0.999,
            "decay_every": 5,
            "verbose": verbose,
            "print_every": 100,
        },
        "rollout_args": {
            "epsilon": 0.0,
        },
        "print_every": 100,
    }

    return base_args, option_args, adv_args