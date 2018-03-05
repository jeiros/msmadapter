from ..plot_utils import plot_spawns, plot_tica_landscape, plot_clusters
import numpy
from matplotlib.axes import Axes
from . import spawns
from msmbuilder.cluster import MiniBatchKMeans

class TestPlotUtils:

    def setUp(self):
        numpy.random.seed(12)
        self.ttrajs = {
            0 : numpy.random.rand(20, 3),
            1 : numpy.random.rand(20, 3),
        }
        self.clusterer = MiniBatchKMeans(n_clusters=2)
        self.clusterer.fit(list(self.ttrajs.values()))

    def test_plot_spawns(self):
        ax = plot_spawns(
            inds=spawns,
            tica_trajs=self.ttrajs,
            ax=None
        )
        assert isinstance(ax, Axes)

    def test_plot_tica_landscape(self):
        f, ax = plot_tica_landscape(self.ttrajs)
        assert isinstance(ax, Axes)

    def test_plot_clusters(self):
        ax = plot_clusters(self.clusterer)
        assert isinstance(ax, Axes)

