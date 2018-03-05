from ..adaptive import App, Adaptive
import logging
from msmbuilder.io import NumberedRunsParser, gather_metadata
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.preprocessing import MinMaxScaler
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from sklearn.pipeline import Pipeline
import os
from ..adaptive import create_folder

logging.disable(logging.CRITICAL)


parser = NumberedRunsParser(
                    traj_fmt='run-{run}.nc',
                    top_fn='data_app/runs/structure.prmtop',
                    step_ps=200
)
meta = gather_metadata('/'.join(['data_app/runs/', '*nc']), parser)



model = Pipeline([
    ('feat', DihedralFeaturizer()),
    ('scaler', MinMaxScaler()),
    ('tICA', tICA(lag_time=1, n_components=4)),
    ('clusterer', MiniBatchKMeans(n_clusters=5)),
    ('msm', MarkovStateModel(lag_time=1, n_timescales=4))
])


spawns = [
    (0, 1),
]
epoch = 1



class TestAppBase:

    def __init__(self):
        self.app = App(
            generator_folder='data_app/generators',
            data_folder='data_app/runs',
            input_folder='data_app/inputs',
            filtered_folder='data_app/filtered_trajs',
            model_folder='data_app/model',
            build_folder='data_app/build',
            meta=meta

        )
        self.meta = meta

    def setUp(self):
        self.app.initialize_folders()
        # Create some fake Production.nc files inside folders in the input folder to test move App.move_generators_to_input
        for f in ['e1s1', 'e1s2', 'e1s3']:
            fname = os.path.join(self.app.input_folder, '{}/Production.nc'.format(f))
            create_folder(os.path.join(self.app.input_folder, f))
            with open(fname, 'w') as f:
                f.write('somebytes')


class TestAdaptiveBase:

    def __init__(self):

        self.adapt = Adaptive(
            app=TestAppBase().app,
            model=model
        )