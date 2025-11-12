import numpy as np
import matplotlib.pyplot as plt
from policies import p_s_from_rollouts, p_sa_from_rollouts, sr_from_rollouts
from aggregation import get_aggregator


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




def gen_visitation_plots(rollouts, save_dir):
    # plot unique states visisted
    pass

def gen_l1_cov_plots(rollouts, base_args, adversery_args, save_dir):
    # plot l1 coverability
    l1_covs = []

    for i in range(len(rollouts)):
        epoch_rollouts = rollouts[i]
        p_sa_from_rollouts(epoch_rollouts)
        


    pass


def gen_heatmap_epoch(rollouts, base_args, save_dir="out"):
    s_agg, _ = get_aggregator(base_args["env_name"], bin_res=base_args["bin_res"])

    for i, epoch in enumerate(rollouts):

        P_S = p_s_from_rollouts(epoch, s_agg)
        SR = sr_from_rollouts(epoch, s_agg)

        plot_heatmap(P_S, save_path=f"{save_dir}/epoch{i}_mu_s")
        plot_heatmap(SR, save_path=f"{save_dir}/epoch{i}_sr")

















