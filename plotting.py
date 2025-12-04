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

def plot_sa_heatmap_pendulum(sa_heatmap, sa_agg, save_path=None):
    # state_shape likely something like (cos, sin, velocity)
    S = sa_agg.state_shape()   # e.g. (n_cos, n_sin, n_v)

    # Define bins
    bins_theta = S[0] * 2   # since cos,sin discretizations wrap
    bins_v = S[2]           # velocity bins

    # Sample angles + velocities
    theta_samples = np.linspace(0, 2*np.pi, bins_theta, endpoint=False)
    v_samples = np.linspace(-8, 8, bins_v)

    # Initialize arrays
    num_a = sa_agg.num_a()
    heatmap = np.zeros((bins_theta, bins_v, num_a))
    counts  = np.zeros((bins_theta, bins_v, num_a))

    # Evaluate heatmap
    for i, theta in enumerate(theta_samples):
        c, s = np.cos(theta), np.sin(theta)
        for j, v in enumerate(v_samples):
            state = [c, s, v]

            for a in range(num_a):
                idx = tuple(sa_agg.sa_to_features(state, a))
                heatmap[i, j, a] += sa_heatmap[idx]
                counts[i, j, a] += 1

    # Avoid divide-by-zero
    mask = counts > 0
    heatmap[mask] /= counts[mask]

    # Collapse action dimension (sum/mean/max depending on your preference)
    heatmap2d = heatmap.mean(axis=-1)

    # Plot
    plot_heatmap(
        heatmap2d,
        xlabel="theta ∈ [0, 2π]",
        ylabel="velocity ∈ [-8, 8]",
        save_path=save_path
    )

def plot_sa_heatmap(env_name, sa_table, sa_agg,title="sa heatmap", save_path=None):
    print(env_name)
    if env_name == "MountainCarContinuous-v0":
        plot_heatmap(sa_table.reshape(sa_agg.shape()[0], -1).T, "x coordinate", "velocity", title, save_path=save_path)
    elif env_name == "Pendulum-v1":
        plot_sa_heatmap_pendulum(sa_table, sa_agg, save_path=save_path)
    else:
        plot_sa_heatmap_cartpole_xdot_thetadot(sa_table, sa_agg, save_path)


import numpy as np
import matplotlib.pyplot as plt

def plot_sa_heatmap_cartpole_xdot_thetadot(sa_heatmap, sa_agg, save_path=None):
    """
    Plot a 2D heatmap of state-action values over x_dot and theta_dot,
    averaging over x and theta, using sa_agg.state_to_features.

    Parameters
    ----------
    sa_heatmap : np.ndarray
        Tile-coded Q-values indexed by sa_agg.state_to_features(state, a)
    sa_agg : object
        SA aggregator / tile coder with:
        - state_shape(): returns [x_tiles, xdot_tiles, theta_tiles, thetadot_tiles]
        - num_a(): number of discrete actions
        - state_to_features(state, a): maps continuous state + action to feature indices
    save_path : str, optional
        Path to save the figure. If None, shows the plot.
    """
    # State dimensions for sampling
    bins_x, bins_xdot, bins_theta, bins_thetadot = sa_agg.state_shape()
    num_a = sa_agg.num_a()

    # Continuous sampling ranges
    x_samples = np.linspace(-2.4, 2.4, bins_x)
    theta_samples = np.linspace(-0.209, 0.209, bins_theta)  # ±12° in radians
    xdot_samples = np.linspace(-5, 5, bins_xdot)
    thetadot_samples = np.linspace(-5, 5, bins_thetadot)

    heatmap = np.zeros((bins_x,  bins_theta, num_a))
    counts = np.zeros_like(heatmap)

    # Loop over x_dot and theta_dot for the axes
    for i_xdot, xdot in enumerate(xdot_samples):
        for j_thetadot, thetadot in enumerate(thetadot_samples):
            for i_x, x in enumerate(x_samples):
                for i_theta, theta in enumerate(theta_samples):
                    state = [x, xdot, theta, thetadot]
                    for a in range(num_a):
                        idx = tuple(sa_agg.sa_to_features(state, a))
                        heatmap[i_x, i_theta, a] += sa_heatmap[idx]
                        counts[i_x, i_theta, a] += 1

    # Avoid divide-by-zero
    mask = counts > 0
    heatmap[mask] /= counts[mask]

    # Collapse action dimension (mean)
    heatmap2d = heatmap.mean(axis=-1)

    # Plot
    plt.figure(figsize=(8,6))
    plt.imshow(heatmap2d.T, origin='lower', aspect='auto',
               extent=[xdot_samples[0], xdot_samples[-1],
                       thetadot_samples[0], thetadot_samples[-1]],
               cmap='viridis')
    plt.colorbar(label='Q-value')
    plt.xlabel('Cart pos x  ')
    plt.ylabel('Pole theta')
    plt.title('SA Heatmap (averaged over x and θ, and actions)')

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()




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






        
        















