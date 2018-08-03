import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as pp
import numpy
import msmexplorer as msme


def plot_spawns(inds, tica_trajs, ax=None, obs=(0, 1), color='red', base_size=100):
    if ax is None:
        ax = pp.gca()

    for traj_i, frame_i in inds:
        ax.scatter(
            tica_trajs[traj_i][frame_i, obs[0]],
            tica_trajs[traj_i][frame_i, obs[1]],
            color=color,
            s=base_size,
            marker='*',
        )
    return ax


def plot_tica_landscape(tica_trajs, ax=None, figsize=(7, 5), obs=(0, 1), cmap='magma', alpha=1):
    """
    Plots a 2D landscape of the tica trajectories
    :param tica_trajs: dictionary of tica trajectories
    :param ax: mpl.axies
    :param figsize: tuple
    :param obs: tuple
    :param cmap: str
    :return: f, ax (figure and axis)
    """
    if ax is None:
        f, ax = pp.subplots(figsize=figsize)
    else:
        f = pp.gcf()

    txx = numpy.concatenate(list(tica_trajs.values()))
    ax = msme.plot_free_energy(
        txx, ax=ax, obs=obs, alpha=alpha,
        n_levels=6,
        xlabel='tIC 1', ylabel='tIC 2',
        labelsize=14, cmap=cmap, vmin=1e-25
    )

    return f, ax


def plot_clusters(clusterer, ax=None, selection=None, obs=(0, 1), base_size=50,
                  alpha=0.5, color='yellow'):
    """
    Plots the position of clusters in a two dimensional landscape
    :param clusterer: a clusterer object from msmbuilder
    :param ax: mpl.axis
    :param selection: None or list of ints, which clusters to plot
    :param obs: tuple
    :param base_size: int
    :param alpha: float
    :param color: str
    :return: ax
    """

    if ax is None:
        ax = pp.gca()

    if selection is None:
        prune = clusterer.cluster_centers_[:, obs]
    else:
        prune = clusterer.cluster_centers_[selection][:, obs]
    scale = clusterer.counts_ + 1 / numpy.max(clusterer.counts_)  # Add one to still plot clusters with no counts

    ax.scatter(
        prune[:, 0],
        prune[:, 1],
        s=scale * base_size,
        alpha=alpha,
        color=color
    )

    return ax
