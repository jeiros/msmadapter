from . import TestAppBase, TestAdaptiveBase, spawns, epoch
import os
import pandas as pd
from sklearn.pipeline import Pipeline

from shutil import rmtree


def teardown_module():
    rmtree('data_app/generators')
    rmtree('data_app/inputs')
    rmtree('data_app/filtered_trajs')
    rmtree('data_app/model')




class TestApp(TestAppBase):
    """Test the methods of the App class"""


    def test_initialize_folders(self):
        assert os.path.isdir(self.app.generator_folder)
        assert os.path.isdir(self.app.data_folder)
        assert os.path.isdir(self.app.input_folder)
        assert os.path.isdir(self.app.filtered_folder)
        assert os.path.isdir(self.app.model_folder)
        assert os.path.isdir(self.app.build_folder)

    def test_build_metadata(self):
        # User defined meta
        output =  self.app.build_metadata(meta=self.meta)
        assert isinstance(output, pd.DataFrame)


    def test_prepare_spawns(self):
        # Use cpptraj and tleap to build a solvated and hmr prmtop
        spawns_folders = self.app.prepare_spawns(spawns, epoch)
        assert type(spawns_folders) == list
        for f in spawns_folders:
            assert type(f) == str
            assert os.path.isdir(f)

    def test_move_trajs_to_folder(self):
        folders = self.app.input_folder + '/*'
        self.app.move_trajs_to_folder(folders)
        assert os.path.exists(os.path.join(self.app.data_folder, 'e1s1/Production.nc'))
        assert not os.path.exists(os.path.join(self.app.input_folder, 'e1s1/Production.nc'))


class TestAdaptive(TestAdaptiveBase):
    """Test the methods of the Adaptive class"""

    def test_build_model1(self):
        "If we pass no Model build one if no pkl file exists"
        model = self.adapt.build_model(None)
        assert isinstance(model, Pipeline)

    def test_build_model2(self):
        "If we pass the previously defined model just return it"
        model = self.adapt.build_model(self.adapt.model)
        assert model == self.adapt.model

    def test_fit_model(self):
        self.adapt.fit_model()

    def test_get_traj_dict(self):
        trj_dict = self.adapt.get_traj_dict()
        assert isinstance(trj_dict, dict)

    def test_get_tica_trajs(self):
        self.adapt.traj_dict = self.adapt.get_traj_dict()
        ttrajs = self.adapt.get_tica_trajs()
        print(ttrajs)
        assert isinstance(ttrajs, dict)



