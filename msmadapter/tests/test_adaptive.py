from . import TestAppBase
import os
import pandas as pd
from msmbuilder.io import NumberedRunsParser, gather_metadata
from shutil import rmtree
from ..pbs_utils import simulate_in_P100s

def teardown_module():
    rmtree('data_app/generators')
    rmtree('data_app/inputs')
    rmtree('data_app/filtered_trajs')
    rmtree('data_app/model')


spawns = [
    (0, 1),
]
epoch = 1
parser = NumberedRunsParser(
                    traj_fmt='run-{run}.nc',
                    top_fn='data_app/runs/structure.prmtop',
                    step_ps=200
)
meta = gather_metadata('/'.join(['data_app/runs/', '*nc']), parser)


class TestApp(TestAppBase):
    """Test the methods of the App class"""


    def test_initialize_folders(self):
        assert os.path.isdir(self.app.generator_folder)
        assert os.path.isdir(self.app.data_folder)
        assert os.path.isdir(self.app.input_folder)
        assert os.path.isdir(self.app.filtered_folder)
        assert os.path.isdir(self.app.model_folder)
        assert os.path.isdir(self.app.build_folder)


    def test_build_metadata_1(self):
        # Build metadata automatically
        output = self.app.build_metadata(meta=None)
        assert isinstance(output, pd.DataFrame)


    def test_build_metadata_2(self):
        # User defined meta
        output =  self.app.build_metadata(meta=meta)
        assert isinstance(output, pd.DataFrame)


    def test_prepare_spawns(self):
        # Use cpptraj and tleap to build a solvated and hmr prmtop
        spawns_folders = self.app.prepare_spawns(spawns, epoch)
        assert type(spawns_folders) == list
        for f in spawns_folders:
            assert type(f) == str
            assert os.path.isdir(f)

