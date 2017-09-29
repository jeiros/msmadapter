import logging
import os
from glob import glob
from multiprocessing import Pool
from string import Template

import mdtraj
import msmbuilder
import numpy
import pandas as pd
from mdrun.Simulation import Simulation
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.decomposition import tICA, PCA
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.io import load_generic, save_generic, gather_metadata, \
    NumberedRunsParser, load_meta
from msmbuilder.io.sampling import sample_states, sample_dimension
from msmbuilder.msm import MarkovStateModel
from msmbuilder.preprocessing import RobustScaler
from parmed.amber import AmberParm
from parmed.tools import HMassRepartition
from sklearn.pipeline import Pipeline

from .pbs_settings import generate_mdrun_skeleton, simulate_in_P100s
from .traj_utils import get_ftrajs, get_sctrajs, get_ttrajs, create_folder, \
    write_cpptraj_script, write_tleap_script, create_symlinks

logger = logging.getLogger()


class App(object):

    def __init__(self, generator_folder='generators', data_folder='data',
                 input_folder='input', filtered_folder='filtered',
                 model_folder='model', ngpus=4, meta=None):
        """
        :param generator_folder:
        :param data_folder:
        :param input_folder:
        :param filtered_folder:
        :param model_folder:
        :param ngpus:
        """
        self.generator_folder = generator_folder
        self.data_folder = data_folder
        self.input_folder = input_folder
        self.filtered_folder = filtered_folder
        self.model_folder = model_folder
        self.ngpus = ngpus
        self.meta = self.build_metadata(meta)

    def __repr__(self):
        return '''App(generator_folder={}, data_folder={}, input_folder={},
                    filtered_folder={}, model_folder={}, ngpus={})'''.format(
            self.generator_folder,
            self.data_folder,
            self.input_folder,
            self.filtered_folder,
            self.model_folder,
            self.ngpus
        )

    def initialize_folders(self):
        create_folder(self.generator_folder)
        create_folder(self.data_folder)
        create_folder(self.input_folder)
        create_folder(self.filtered_folder)
        create_folder(self.model_folder)

    def build_metadata(self, meta):
        """Builds an msmbuilder metadata object"""
        if meta is None:
            parser = NumberedRunsParser(
                traj_fmt='run-{run}.nc',
                top_fn='structure.prmtop',
                step_ps=200
            )
            meta = gather_metadata('/'.join([self.data_folder, '*nc']), parser)
        else:
            if not isinstance(meta, pd.DataFrame):
                meta = load_meta(meta)
        return meta

    @property
    def finished_trajs(self):
        return len(glob('/'.join([self.data_folder, '*nc'])))

    @property
    def ongoing_trajs(self):
        return len(glob('/'.join([self.input_folder, '*nc'])))

    def prepare_spawns(self, spawns, epoch):
        sim_count = 0
        for traj_id, frame_id in spawns:
            logger.info('Building simulation {} of epoch {}'.format(sim_count, epoch))
            folder_name = 'e{:02d}s{:02d}_t{:03d}f{:04d}'.format(epoch, sim_count, traj_id, frame_id)
            destination = os.path.join(self.input_folder, folder_name)
            create_folder(destination)
            write_cpptraj_script(
                traj=self.meta.loc[traj_id]['traj_fn'],
                top=self.meta.loc[traj_id]['top_fn'],
                frame1=frame_id,
                frame2=frame_id,
                outfile=os.path.join(destination, 'seed.pdb'),
                path=os.path.join(destination, 'script.cpptraj'),
                run=True
            )
            os.symlink('../../CA2.prep', os.path.join(destination, 'CA2.prep'))
            write_tleap_script(
                pdb_file=os.path.join(destination, 'seed.pdb'),
                lig_dir=None,
                run=True,
                system_name=os.path.join(destination, 'structure'),
                path=os.path.join(destination, 'script.tleap')
            )
            sim_count += 1
            # Apply hmr to new topologies
            self.hmr_prmtop(os.path.join(destination, 'structure.prmtop'))
        epoch += 1
        return epoch

    def hmr_prmtop(self, top_fn, save=True):
        top = AmberParm(top_fn)
        hmr = HMassRepartition(top)
        hmr.execute()
        if save:
            top_out_fn = top_fn.split('.')[0]
            top_out_fn += '_hmr.prmtop'
            top.save(top_out_fn)
        return top


    def prepare_PBS_jobs(self, folders):
        folder_fnames_list = glob(folders)
        cwd = os.getcwd()
        for input_folder in folder_fnames_list:
            system_name = input_folder.split('/')[-1]
            data_folder = os.path.realpath(os.path.join(self.data_folder, system_name))
            if not os.path.exists(data_folder):
                os.mkdir(data_folder)
            os.chdir(data_folder)
            create_symlinks(files='structure.*', dst_folder=data_folder)
            skeleton = simulate_in_P100s(generate_mdrun_skeleton, system_name=system_name, destination=os.path.realpath(data_folder))
            sim = Simulation(skeleton)
            sim.writeSimulationFiles()
            os.chdir(cwd)


class Adaptive(object):
    """

    """

    def __init__(self, nmin=1, nmax=2, nepochs=20, stride=1, sleeptime=3600,
                 model=None, app=None):
        self.nmin = nmin
        self.nmax = nmax
        self.nepochs = nepochs
        self.stride = stride
        self.sleeptime = sleeptime
        if app is None:
            self.app = App()
        else:
            self.app = app
        if not isinstance(self.app, App):
            raise ValueError('self.app is not an App object')
        self.timestep = (self.app.meta['step_ps'].unique()[0] * self.stride) / 1000  # in ns
        self.model_pkl_fname = os.path.join(self.app.model_folder, 'model.pkl')
        self.model = self.build_model(model)
        self.ttrajs = None
        self.traj_dict = None
        self.current_epoch = self.app.ongoing_trajs
        self.spawns = None

    def __repr__(self):
        return '''Adaptive(nmin={}, nmax={}, nepochs={}, stride={}, sleeptime={},
                         timestep={}, model={}, app={})'''.format(
            self.nmin, self.nmax, self.nepochs, self.stride, self.sleeptime,
            self.timestep, self.model, repr(self.app))

    def run(self):
        """
        :return:
        """
        finished = False
        while not finished:
            if self.current_epoch == self.nepochs:
                finished = True
            else:
                self.app.initialize_folders()
                self.fit_model()
                self.spawns = self.find_respawn_conformations()
                self.current_epoch = self.app.prepare_spawns(self.spawns, self.current_epoch)
                logger.info('Going to sleep for {} seconds'.format(self.sleeptime))
                sleep(self.sleeptime)

    def find_respawn_conformations(self, percentile=0.5):
        """
        Find candidate frames in the trajectories to spawn new simulations from.
        We increase (or decrease) the percentile threshold of populated microstates until we have as many new candidates
        as GPUs available.

        :param percentile: float, The percentile below which to look for low populated microstates of the MSM
        :return chosen_frames: a list of tuples, each tuple being (traj_id, frame_id)
        """

        # First, identify the MarkovStateModel and Clusterer objects from the Pipeline model
        # We try to look by name first, if the user has provided a model with
        # different named steps than assumend, we look by position
        if 'msm' in self.model.named_steps.keys():
            msm = self.model.named_steps['msm']
        else:
            msm = self.model.steps[-1]
            if not getattr(msm, '__module__') == msmbuilder.msm.__name__:
                raise ValueError('The last step in the model does not belong to the msmbuilder.msm module')

        if 'clusterer' in self.model.named_steps.keys():
            clusterer = self.model.named_steps['clusterer']
        else:
            clusterer = self.model.steps[-2]
        if not getattr(clusterer, '__module__') == msmbuilder.cluster.__name__:
            raise ValueError('The penultimate step in the model does not belong to the msmbuilder.cluster module')

        # Initiate the search for candidate frames amongst the trajectories
        logger.info('Looking for low populated microstates')
        logger.info('Initial percentile threshold set to {:02f}'.format(percentile))

        low_cluster_ids = []
        iterations = 0  # to avoid getting stuck in the search
        while not len(low_cluster_ids) == self.app.ngpus:
            low_populated_msm_states = numpy.where(
                msm.populations_ < numpy.percentile(msm.populations_, percentile)
            )[0]  # numpy.where gives back a tuple with empty second element

            low_cluster_ids = []
            for state_id in low_populated_msm_states:
                # Remember that the MSM object is built with ergodic trimming, so the subspace that is found
                # does not necessarily recover all the clusters in the clusterer object.
                # Therefore, the msm.mapping_ directory stores the correspondence between the cluster labels
                # in the clusterer object and the MSM object. Keys are clusterer indices, values are MSM indices
                for c_id, msm_id in msm.mapping_.items():
                    if msm_id == state_id:
                        low_cluster_ids.append(c_id)  # Store the cluster ID in clusterer naming


            # Change percentile threshold to reach desired number of frames
            if len(low_cluster_ids) < self.app.ngpus:
                percentile += 0.5
            if len(low_cluster_ids) > self.app.ngpus:
                if (len(low_cluster_ids) - 1) == self.app.ngpus:
                    low_cluster_ids.pop()
                percentile -= 0.5
            logger.info('Percentiles is at {} and found {} frames'.format(percentile, len(low_cluster_ids)))

            iterations += 1
            # Logic to stop search
            if (percentile > 100) or (iterations > 500):
                break

        logger.info('Finished after {} iterations'.format(iterations))
        logger.info('Found {:d} frames which were below {:02f} percentile'.format(len(low_cluster_ids), percentile))

        if self.ttrajs is None:
            self.ttrajs = self.get_tica_trajs()

        # Finally, find frames in the trajectories that are nearby the selected cluster centers (low populated in the MSM)
        # Only retrieve one frame per cluster center
        return sample_states(
            trajs=self.ttrajs,
            state_centers=clusterer.cluster_centers_[low_cluster_ids]
        )

    def respawn_from_tICs(self, dims=(0, 1)):

        if self.ttrajs is None:
            self.ttrajs = self.get_tica_trajs()

        frames_per_tIC = int(self.app.ngpus / len(dims))

        assert frames_per_tIC * len(dims) == self.app.ngpus

        return [
            sample_dimension(
                self.ttrajs,
                dimension=d,
                n_frames=frames_per_tIC,
                scheme='edge'
            ) for d in dims
        ]


    def trajs_from_irrows(self, irow):
        """
        Load each trajectory in the rows of an msmbuilder.metadata object
        :param irow: iterable coming from pd.DataFrame.iterrow method
        :return i, traj: The traj id (starting at 0) and the mdtraj.Trajectory object
        """
        i, row = irow
        logger.info('Loading {}'.format(row['traj_fn']))
        traj = mdtraj.load(row['traj_fn'], top=row['top_fn'], stride=self.stride)
        return i, traj

    def get_traj_dict(self):
        """
        Load the trajectories in the disk as specified by the metadata object in parallel
        :return traj_dict: A dictionary of mdtraj.Trajectory objects
        """
        with Pool() as pool:
            traj_dict = dict(
                pool.imap_unordered(self.trajs_from_irrows, self.app.meta.iterrows())
            )
        return traj_dict

    def fit_model(self):
        """
        Fit the adaptive model onto the trajectories
        """
        logger.info('Fitting model')
        if self.traj_dict is None:
            self.traj_dict = self.get_traj_dict()
        self.model.fit(self.traj_dict.values())

    def get_tica_trajs(self):
        """
        Step through each of the steps of the adaptive model and recover the transformed trajectories after each step,
        until we reach the final tICA-transformed trajectories. We assume that the steps in the model are:
            1) A featurizer object
            2) A scaler object (optional)
            3) The tICA object
        :return ttrajs: A dict of tica-transformed trajectories, represented as np.arrays of shape (n_frames, n_components)
        """
        # Assume order of steps in model
        # Then I try to check as best as I know that it's correct
        featurizer = self.model.steps[0][1]
        scaler = self.model.steps[1][1]
        decomposer = self.model.steps[2][1]

        if getattr(featurizer, '__module__') == msmbuilder.featurizer.featurizer.__name__:
            logger.info('Featurizing trajs')
            ftrajs = get_ftrajs(self.traj_dict, featurizer)

            if getattr(scaler, '__module__') == msmbuilder.preprocessing.__name__:
                logger.info('Scaling ftrajs')
                sctrajs = get_sctrajs(ftrajs, scaler)
            elif isinstance(scaler, tICA) or isinstance(scaler, PCA):
                logger.warning('Second step in model is a decomposer and not a scaler')
                decomposer = scaler
                sctrajs = ftrajs  # We did not do any scaling of ftrajs
            else:
                logger.warning('Could not find a scaler or decomposer in your model.')

            logger.info('Getting output of tICA')
            ttrajs = get_ttrajs(sctrajs, decomposer)
        else:
            raise ValueError('The first step in the model does not belong to the msmbuilder.featurizer module')
        return ttrajs

    def build_model(self, user_defined_model=None):
        """
        Load or build a model (Pipeline from scikit-learn) to do all the transforming and fitting
        :param user_defined_model: Either a string (to load from disk) or a Pipeline object to use as model
        :return model: Return the model back
        """
        if user_defined_model is None:
            if os.path.exists(self.model_pkl_fname):
                logger.info('Loading model pkl file {}'.format(self.model_pkl_fname))
                model = load_generic(self.model_pkl_fname)
            else:
                logger.info('Building default model')
                # build a lag time of 1 ns for tICA and msm
                # if the stride is too big and we can't do that, just use 1 frame and report how much that is in ns
                lag_time = max(1, int(1 / self.timestep))
                if lag_time == 1:
                    logger.warning('Using a lag time of {:.2f} ns for the tICA and MSM'.format(self.timestep))
                model = Pipeline([
                    ('feat', DihedralFeaturizer()),
                    ('scaler', RobustScaler()),
                    ('tICA', tICA(lag_time=lag_time, kinetic_mapping=True, n_components=10)),
                    ('clusterer', MiniBatchKMeans(n_clusters=200)),
                    ('msm', MarkovStateModel(lag_time=lag_time))
                ])
        else:
            if not isinstance(user_defined_model, Pipeline):
                raise ValueError('model is not an sklearn.pipeline.Pipeline object')
            else:
                logger.info('Using user defined model')
                model = user_defined_model
        return model

    def _save_model(self):
        """
        Save a model to disk in pickle format
        """
        save_generic(self.model, self.model_pkl_fname)


if __name__ == "__main__":
    app = App(meta='meta.pandas.pickl')
    ad = Adaptive(app=app, stride=20)
    ad.fit_model()
    ad.find_respawn_conformations()
