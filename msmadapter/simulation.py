import os
import numpy
from msmbuilder.io import load_generic, save_generic, gather_metadata, NumberedRunsParser, load_meta
from msmbuilder.io.sampling import sample_states
from sklearn.pipeline import Pipeline
import msmbuilder
from msmbuilder.decomposition import tICA
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from msmbuilder.preprocessing import RobustScaler
import logging
import mdtraj
import pandas as pd
from multiprocessing import Pool
from glob import glob

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
        return 'App(generator_folder={}, data_folder={}, input_folder={}, filtered_folder={}, model_folder={}, ngpus={})'.format(
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
    def number_of_trajectories(self):
        return len(glob('/'.join([self.data_folder, '*nc'])))


class Adaptive(object):
    """

    """

    def __init__(self, nmin=1, nmax=2, nepochs=20, stride=1, sleeptime=3600, timestep=200, model=None, app=None):
        self.nmin = nmin
        self.nmax = nmax
        self.nepochs = nepochs
        self.stride = stride
        self.sleeptime = sleeptime
        self.timestep = timestep
        if app is None:
            self.app = App()
        else:
            self.app = app
        if not isinstance(self.app, App):
            raise ValueError('self.app is not an App object')
        self.model_pkl_fname = '/'.join([self.app.model_folder, 'model.pkl'])
        self.model = self.build_model(model)
        self.ttrajs = None

    def __repr__(self):
        return 'Adaptive(nmin={}, nmax={}, nepochs={}, stride={}, sleep_time={}, timestep={}, model={}, app={})'.format(
            self.nmin, self.nmax, self.nepochs, self.stride, self.sleeptime, self.timestep, self.model, repr(self.app))

    def run(self):
        """
        :return:
        """
        finished = False
        while not finished:
            if self.current_epoch == self.nepochs:
                finished = True

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

        # We now initiate the search for candidate frames amongst our trajectories
        logger.info('Looking for low populated microstates')
        logger.info('Initial percentile threshold set to {:02f}'.format(percentile))
        low_cluster_ids = []
        while not len(low_cluster_ids) == self.app.ngpus:
            low_populated_msm_states = numpy.where(
                msm.populations_ < numpy.percentile(msm.populations_, percentile)
            )[0]  # numpy.where gives back a tuple with empty second element
            low_cluster_ids = []

            for state_id in low_populated_msm_states:
                for c_id, msm_id in msm.mapping_.items():
                    if msm_id == state_id:
                        low_cluster_ids.append(c_id)

            if len(low_cluster_ids) < self.app.ngpus:
                percentile += 0.5

            if len(low_cluster_ids) > self.app.ngpus:
                if (len(low_cluster_ids) - 1) == self.app.ngpus:
                    low_cluster_ids.pop()
                percentile -= 0.5

        logger.info('Found {:d} frames which were below {:02f} percentile'.format(len(low_cluster_ids), percentile))

        if self.ttrajs is None:
            logger.warning('The tica-transformed trajs have not been calculated prior to calling find_respawn_frame')
            logger.warning('This should not happen!')
            self.get_tica_trajs()

        return sample_states(
            trajs=self.ttrajs,
            state_centers=clusterer.cluster_centers_[low_cluster_ids]
        )

    def get_traj_dict(self):
        """
        Load the trajectories in the disk as specified by the metadata object in parallel
        :return traj_dict: A dictionary of mdtraj.Trajectory objects
        """
        with Pool() as pool:
            traj_dict = dict(pool.imap_unordered(self.load_trajectories, self.app.meta.iterrows()))
        self.traj_dict = traj_dict
        return traj_dict

    def fit_model(self):
        """
        Fit the adaptive model onto the trajectories
        """
        logger.info('Fitting model')
        traj_dict = self.get_traj_dict()
        self.model.fit(traj_dict.values())

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

        if getattr(featurizer, '__module__') == msmbuilder.featurizer.__name__:
            logger.info('Featurizing trajs')
            ftrajs = get_ftrajs(self.traj_dict, featurizer)

            if getattr(scaler, '__module__') == msmbuilder.preprocessing.__name__:
                logger.info('Scaling ftrajs')
                sctrajs = get_sctrajs(ftrajs, scaler)
            elif getattr(decomposer, '__module__') == msmbuilder.decomposition.__name__:
                logger.warning('Second step in model is a decomposer and not a scaler')
                decomposer = scaler
                sctrajs = ftrajs  # We did not do any scaling of ftrajs
            else:
                logger.warning('Could not find a scaler or decomposer in your model.')

            logger.info('Getting output of tICA')
            ttrajs = get_ttrajs(sctrajs, decomposer)
        else:
            raise ValueError('The first step in the model does not belong to the msmbuilder.featurizer module')
        self.ttrajs = ttrajs
        return ttrajs

    def load_trajectories(self, irow):
        """
        Load each trajectory in the rows of an msmbuilder.metadata object
        :param irow: iterable coming from pd.DataFrame.iterrow method
        :return i, traj: The traj id (starting at 0) and the mdtraj.Trajectory object
        """
        i, row = irow
        logger.info('Loading {}'.format(row['traj_fn']))
        traj = mdtraj.load(row['traj_fn'], top=row['top_fn'], stride=self.stride)
        return i, traj

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
                model = Pipeline([
                    ('feat', DihedralFeaturizer()),
                    ('scaler', RobustScaler()),
                    ('tICA', tICA(lag_time=5, kinetic_mapping=True)),
                    ('clusterer', MiniBatchKMeans(n_clusters=200)),
                    ('msm', MarkovStateModel(lag_time=5))
                ])
        else:
            if not isinstance(user_defined_model, Pipeline):
                raise ValueError('model is not an sklearn.pipeline.Pipeline object')
            else:
                logger.info('Using user defined model')
                model = user_defined_model
        return model

    def save_model(self):
        """
        Save a model to disk in pickle format
        :return:
        """
        save_generic(self.model, self.model_pkl_fname)


def get_ftrajs(traj_dict, featurizer):
    """
    Featurize a dictionary of mdtraj.Trajectory objects
    :param traj_dict: The dictionary of trajectories
    :param featurizer: A featurizer object which must have been fit first
    :return ftrajs: A dict of featurized trajectories
    """
    ftrajs = {}
    for k, v in traj_dict.items():
        ftrajs[k] = featurizer.partial_transform(v)
    return ftrajs


def get_sctrajs(ftrajs, scaler):
    """
    Scale a dictionary of featurized trajectories
    :param traj_dict: The dictionary of featurized trajectories, represented as np.arrays of shape (n_frames, n_features)
    :param scaler: A scaler object which must have been fit first
    :return ftrajs: A dict of scaled trajectories, represented as np.arrays of shape (n_frames, n_features)
    """
    sctrajs = {}
    for k, v in ftrajs.items():
        sctrajs[k] = scaler.partial_transform(v)
    return sctrajs


def get_ttrajs(sctrajs, tica):
    """
    Reduce the dimensionality of a dictionary of scaled or featurized trajectories
    :param traj_dict: The dictionary of featurized/scaled trajectories, represented as np.arrays of shape (n_frames, n_features)
    :param tica: A tICA object which must have been fit first
    :return ttrajs: A dict of tica-transformed trajectories, represented as np.arrays of shape (n_frames, n_components)
    """
    ttrajs = {}
    for k, v in sctrajs.items():
        ttrajs[k] = tica.partial_transform(v)
    return ttrajs


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def generate_traj_from_stateinds(inds, meta):
    for state_i, state_inds in enumerate(inds):
        traj = mdtraj.join(
            mdtraj.load_frame(meta.loc[traj_i]['traj_fn'], index=frame_i, top=meta.loc[traj_i]['top_fn'])
            for traj_i, frame_i in state_inds
        )
    return traj

if __name__ == "__main__":
    app = App(meta='meta.pandas.pickl')
    ad = Adaptive(app=app, stride=20)
    ad.fit_model()
    ad.find_respawn_conformations()
