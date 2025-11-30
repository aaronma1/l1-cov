import numpy as np
import matplotlib.pyplot as plt
from lib.trajectories import p_s_from_rollouts, p_sa_from_rollouts, sr_from_rollouts
from lib.aggregation import get_aggregator

from scipy.stats import entropy


def plot_heatmap(mat, xlabel="x coordinate", ylabel="Velocity [-0.7, 0.7]", title=None, show_ticks=True, save_path=None, cmap="turbo"):
    """
    Plot a 2D numpy array `mat` as a heatmap using matplotlib.

    Parameters:
        mat : 2D numpy array
        xlabel, ylabel, title : optional strings for axis labels and title
        show_ticks : whether to show numeric tick labels
        save_path : if provided, saves the figure to this filepath
    """
    if not isinstance(mat, np.ndarray):
        mat = np.array(mat)
    if mat.ndim != 2:
        raise ValueError(f"Input must be a 2D array/matrix. Got ndim={mat.ndim}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        rows, cols = mat.shape
        if cols <= 20:
            ax.set_xticks(np.arange(cols))
            ax.set_xticklabels(np.arange(cols))
        else:
            ax.set_xticks([0, cols - 1])
            ax.set_xticklabels([0, cols - 1])
        if rows <= 20:
            ax.set_yticks(np.arange(rows))
            ax.set_yticklabels(np.arange(rows))
        else:
            ax.set_yticks([0, rows - 1])
            ax.set_yticklabels([0, rows - 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return None


def compute_unique_states_visited(base_args, rollouts):
    s_agg, sa_agg = get_aggregator(base_args["env_name"])
    # plot unique states visisted
    unique_states_visisted = []
    for i in range(len(rollouts)):
        epoch_rollouts = rollouts[i]["all_rollouts"]

        p_s = p_s_from_rollouts(epoch_rollouts, s_agg)

        unique_states_visisted.append(np.count_nonzero(p_s)/np.size(p_s))
    return unique_states_visisted

def compute_policy_state_entropy(base_args, rollouts):
    s_agg, sa_agg = get_aggregator(base_args["env_name"])
    # plot unique states visisted
    state_entropy = []
    for i in range(len(rollouts)):
        epoch_rollouts = rollouts[i]["all_rollouts"]

        p_s = p_s_from_rollouts(epoch_rollouts, s_agg)
        p_sa = p_sa_from_rollouts(epoch_rollouts, sa_agg)
        print(p_s.sum(), p_sa.sum())
        assert np.isclose(p_s.sum(), 1)
        assert np.isclose(p_sa.sum(), 1)
        state_entropy.append(entropy(p_s.flatten()))
    return state_entropy

        


def gen_heatmap_epoch(rollouts, base_args, save_dir="out"):
    s_agg, _ = get_aggregator(base_args["env_name"], bin_res=base_args["bin_res"])

    for i, epoch in enumerate(rollouts):

        P_S = p_s_from_rollouts(epoch, s_agg)
        SR = sr_from_rollouts(epoch, s_agg)

        plot_heatmap(P_S, save_path=f"{save_dir}/epoch{i}_mu_s")
        plot_heatmap(SR, save_path=f"{save_dir}/epoch{i}_sr")



def plot_sa_heatmap_pendulum(sa_heatmap, sa_agg, save_path=None):
    state_shape = 2*sa_agg.state_shape()
    bins_theta = 2*sa_agg.state_shape()[0] 
    bins_v = sa_agg.state_shape()[2]
    theta_samples = np.linspace(0, 2*np.pi, num=bins_theta)
    v_samples = np.linspace(-8, 8, num = state_shape[2])

    heatmap = np.zeros((bins_theta, state_shape[2], sa_agg.num_a()))


    for i, theta in enumerate(theta_samples):
        for j, v in enumerate(v_samples):

            state = [np.cos(theta), np.sin(theta), v]

            for a in range(sa_agg.num_a()):
                heatmap[i][j][a] = sa_heatmap[tuple(sa_agg.sa_to_features(state,a))]
    
    # plot_heatmap(heatmap.sum(axis=-1),save_path=save_path)
    plot_heatmap(heatmap.reshape(bins_theta, -1), xlabel="theta, [-2pi, 2pi]", ylabel="v, [-8, 8]",save_path=save_path)


import gymnasium as gym
from lib.policies import get_random_agent
from lib.trajectories import collect_rollouts, p_sa_from_rollouts
if __name__ == "__main__":
    env_name = "Pendulum-v1"
    env = gym.make(env_name)


    random_agent = get_random_agent(env_name)
    agg, sa_agg = get_aggregator(env_name)

    rollouts = collect_rollouts(env, random_agent, 200 ,200)

    psa = p_sa_from_rollouts(rollouts, sa_agg)

    plot_sa_heatmap_pendulum(psa, sa_agg, save_path="a.png")






        
        















