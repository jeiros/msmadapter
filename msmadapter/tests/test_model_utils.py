from ..model_utils import retrieve_clusterer, retrieve_decomposer, retrieve_feat, \
    retrieve_MSM, apply_percentile_search, retrieve_scaler
from . import model
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.preprocessing import MinMaxScaler
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
import numpy

class TestModelUtils:

    def setUp(self):
        numpy.random.seed(12)
        self.msm = MarkovStateModel()
        self.msm.fit(
            [numpy.random.randint(5, size=10) for _ in range(20)]  # 20 lists of 10 random integers from 0 to 5
        )



    def test_retrieveMSM(self):
        msm = retrieve_MSM(model)
        assert isinstance(msm, MarkovStateModel)

    def test_retrieve_clusterer(self):
        clusterer = retrieve_clusterer(model)
        assert isinstance(clusterer, MiniBatchKMeans)

    def test_retrieve_feat(self):
        feat = retrieve_feat(model)
        assert isinstance(feat, DihedralFeaturizer)

    def test_retrieve_scaler(self):
        scaler = retrieve_scaler(model)
        assert isinstance(scaler, MinMaxScaler)

    def test_retrieve_decomposer(self):
        decomposer = retrieve_decomposer(model)
        assert isinstance(decomposer, tICA)

    def test_apply_percentile_search1(self):
        counts = apply_percentile_search(
            count_array=self.msm.transmat_,
            percentile=0.1,
            desired_length=10,
            search_type='clusterer',
            msm=None,
        )
        assert isinstance(counts, list)
        assert len(counts) == 10

    def test_apply_percentile_search2(self):
        counts = apply_percentile_search(
            count_array=self.msm.transmat_,
            percentile=0.1,
            desired_length=2,
            search_type='msm',
            msm=self.msm,
        )
        assert isinstance(counts, list)
        assert len(counts) == 2