from matplotlib import pyplot as pp
import numpy


def plot_spawns(inds, tica_trajs, ax=None, figsize=(7, 5), obs=(0, 1)):
    if ax is None:
        f, ax = pp.subplots(figsize=figsize)

    txx = numpy.concatenate(list(tica_trajs.values()))
    prune = txx[:, obs]

    ax.scatter(prune[:, 0], prune[:, 1])
    for traj_i, frame_i in inds:
        ax.scatter(
            tica_trajs[traj_i][frame_i, obs[0]],
            tica_trajs[traj_i][frame_i, obs[1]],
            color='red'
        )
    return ax