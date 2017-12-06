from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.preprocessing import RobustScaler
from msmbuilder.decomposition import tICA
from mdtraj import load
from ..utils import get_ftrajs, get_sctrajs, get_ttrajs, traj_from_stateinds, \
    write_production_file, write_cpptraj_script, write_tleap_script, \
    create_folder, create_symlinks, hmr_prmtop
from .test_adaptive import meta, spawns
import numpy
import os
from shutil import rmtree
from parmed.amber import AmberParm
from glob import glob

def teardown_module():
    for f in glob('*in'):
        os.remove(f)
    os.remove('template.tleap')

class TestUtils:

    def setUp(self):
        numpy.random.seed(42)
        self.top = 'data_app/runs/structure.prmtop'
        self.traj_1 = 'data_app/runs/run-000.nc'
        self.traj_2 = 'data_app/runs/run-001.nc'
        self.feat = DihedralFeaturizer()
        self.traj_dict = {
            0 : load(self.traj_1, top=self.top),
            1 : load(self.traj_2, top=self.top)
        }
        self.scaler = RobustScaler()
        self.tica = tICA(n_components=2)
        self.ftrajs = {
            0 : numpy.random.rand(100, 50),
            1 : numpy.random.rand(100, 50),
        }

    def test_get_ftrajs(self):
        output = get_ftrajs(self.traj_dict, self.feat)
        assert len(output) == 2
        assert type(output) == dict

    def test_get_sctrajs(self):
        self.scaler.fit(list(self.ftrajs.values()))
        output = get_sctrajs(self.ftrajs, self.scaler)
        assert len(output) == 2
        assert type(output) == dict

    def test_get_ttrajs(self):
        self.tica.fit(list(self.ftrajs.values()))
        output = get_ttrajs(self.ftrajs, self.tica)
        assert len(output) == 2
        assert type(output) == dict

    def test_traj_from_stateinds(self):
        traj = traj_from_stateinds(spawns, meta)
        assert traj.n_frames == 1

    def test_write_production_file(self):
        write_production_file()
        assert os.path.exists('Production.in')
        os.remove('Production.in')

    def test_write_cpptraj_script(self):
        write_cpptraj_script(self.traj_1, self.top)
        assert os.path.exists('script.cpptraj')
        os.remove('script.cpptraj')

    def test_write_tleap_script(self):
        write_tleap_script(write=True)
        assert os.path.exists('script.tleap')
        os.remove('script.tleap')

    def test_create_folder(self):
        fname = 'foo'
        create_folder(fname)
        assert os.path.isdir(fname)
        os.removedirs(fname)

    def test_create_symlinks(self):
        create_folder('src_symlinks')
        create_folder('dst_symlinks')
        with open('src_symlinks/1.txt', 'w') as f:
            f.writelines('foo')

        create_symlinks(
            files='src_symlinks/*.txt',
            dst_folder='dst_symlinks'
        )
        assert os.path.exists('dst_symlinks/1.txt')
        rmtree('src_symlinks')
        rmtree('dst_symlinks')

    def test_hmr_prmtop(self):
        new_top = hmr_prmtop(self.top, save=False)
        assert isinstance(new_top, AmberParm)

