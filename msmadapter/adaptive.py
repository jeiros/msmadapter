import logging
import os
import shutil
from functools import partial
from glob import glob
from multiprocessing import Pool
from string import Template

import mdtraj
import pandas as pd
from mdrun.Simulation import Simulation
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.decomposition import tICA
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.io import load_generic, save_generic, gather_metadata, \
    NumberedRunsParser, load_meta
from msmbuilder.io.sampling import sample_states, sample_dimension
from msmbuilder.msm import MarkovStateModel
from msmbuilder.preprocessing import RobustScaler
from parmed.amber import AmberParm
from parmed.tools import HMassRepartition
from sklearn.pipeline import Pipeline

from .model_utils import retrieve_feat, retrieve_clusterer, retrieve_MSM, \
    retrieve_scaler, retrieve_decomposer, apply_percentile_search
from .pbs_utils import generate_mdrun_skeleton, simulate_in_pqigould
from .traj_utils import get_ftrajs, get_sctrajs, get_ttrajs, create_folder, \
    write_cpptraj_script, write_tleap_script, create_symlinks

logger = logging.getLogger(__name__)

class App(object):
    """
    Handles the creation of all the necessary files to set up and run the simulations
    """

    def __init__(self, generator_folder='generators', data_folder='data',
                 input_folder='input', filtered_folder='filtered',
                 model_folder='model', build_folder='build', ngpus=4, meta=None,
                 project_name='adaptive', user_HPC='je714', from_solvated=False):
        """
        :param generator_folder:
        :param data_folder:
        :param input_folder:
        :param filtered_folder:
        :param model_folder:
        :param build_folder:
        :param ngpus:
        :param meta:
        :param project_name:
        :param user_HPC:
        """
        self.generator_folder = generator_folder
        self.data_folder = data_folder
        self.input_folder = input_folder
        self.filtered_folder = filtered_folder
        self.model_folder = model_folder
        self.build_folder = build_folder
        self.ngpus = ngpus
        self.meta = self.build_metadata(meta)
        self.project_name = project_name
        self.user_HPC = user_HPC
        self.from_solvated = from_solvated

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
        create_folder(self.build_folder)

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
        """
        Prepare the prmtop and inpcrd files of the selected spawns
        :param spawns: list of tuples, (traj_id, frame_id)
        :param epoch: int, Epoch the selected spawns belong to
        """
        sim_count = 1
        basedir = os.getcwd()
        for traj_id, frame_id in spawns:
            logger.info('Building simulation {} of epoch {}'.format(sim_count, epoch))

            folder_name = 'e{}s{}_t{}f{}'.format(epoch, sim_count, traj_id, frame_id)
            destination = os.path.join(self.input_folder, folder_name)
            create_folder(destination)

            if self.from_solvated:
                # Add files from build folder to destination folder so tleap can read them
                # since we're not retrieving frame from an already solvated trajectory
                if not os.path.exists(self.build_folder):
                    raise ValueError('{} folder does not exist. Create it first.'.format(self.build_folder))
                else:
                    for fname in glob(os.path.join(self.build_folder, '*')):
                        os.symlink(
                            os.path.realpath(fname),
                            os.path.join(destination, os.path.basename(fname))
                        )
            # All files in destination, so now move into it
            os.chdir(destination)
            if self.from_solvated:
                outfile = 'seed.ncrst'
            else:
                outfile = 'seed.pdb'
            write_cpptraj_script(
                traj=os.path.relpath(os.path.join(basedir, self.meta.loc[traj_id]['traj_fn'])),
                top=os.path.relpath(self.meta.loc[traj_id]['top_abs_fn']),
                frame1=frame_id,
                frame2=frame_id,
                outfile=outfile,
                path='script.cpptraj',
                run=True
            )

            if not self.from_solvated:
                write_tleap_script(
                    pdb_file='seed.pdb',
                    run=True,
                    system_name='structure',
                    path='script.tleap'
                )
                # Apply hmr to new topologies
                self.hmr_prmtop(top_fn=os.path.join(destination, 'structure.prmtop'))
            else:
                os.symlink(
                    os.path.relpath(self.meta.loc[traj_id]['top_abs_fn']),
                    'structure.prmtop'
                )
            # Write information from provenance to file
            information = [
                'Parent trajectory:\t{}'.format(self.meta.loc[traj_id]['traj_fn']),
                'Frame number:\t{}'.format(frame_id),
                'Topology:\t{}'.format(self.meta.loc[traj_id]['top_fn']),
                ''
            ]
            provenance_fn = 'provenance.txt'
            with open(provenance_fn, 'w+') as f:
                f.write('\n'.join(information))

            # When finished, update sim_count and go back to base dir to repeat
            sim_count += 1
            os.chdir(basedir)


    def hmr_prmtop(self, top_fn, save=True):
        """
        Use parmed to apply HMR to a topology file
        :param top_fn: str, path to the prmtop file
        :param save:  bool, whether to save the hmr prmtop
        :return top: the hrm'ed prmtop file
        """
        top = AmberParm(top_fn)
        hmr = HMassRepartition(top)
        hmr.execute()
        if save:
            top_out_fn = top_fn.split('.')[0]
            top_out_fn += '_hmr.prmtop'
            top.save(top_out_fn)
        return top

    def prepare_PBS_jobs(self, folders_glob, skeleton_function):

        folder_fnames_list = glob(folders_glob)
        basedir = os.getcwd()

        for input_folder in folder_fnames_list:
            system_name = input_folder.split('/')[-1].split('_')[0]  # just eXXsYY
            data_folder = os.path.realpath(os.path.join(self.data_folder, system_name))

            if not os.path.exists(data_folder):
                os.mkdir(data_folder)
            create_symlinks(files=os.path.join(input_folder, 'structure*'),
                            dst_folder=os.path.realpath(data_folder))

            os.chdir(data_folder)
            # skeleton = skeleton_function(
            #     func=generate_mdrun_skeleton,
            #     system_name=system_name,
            #     destination=os.path.realpath(data_folder),
            #     job_directory=os.path.join('/work/je714', self.project_name, system_name)
            # )
            skeleton = skeleton_function(
                system_name=system_name,
                job_directory=os.path.join('/work/{}'.format(self.user_HPC),
                                           self.project_name, system_name),
                destination=os.path.realpath(data_folder)
            )
            sim = Simulation(skeleton)
            sim.writeSimulationFiles()

            # AMBER input files

            job_length = sim.job_length
            nsteps = int(job_length * 1e6 / 4)  # ns to steps, using 4 fs / step
            script_dir = os.path.dirname(__file__)  # Absolute path the script is in
            templates_path = 'templates'
            for input_file in glob(os.path.join(script_dir, templates_path, '*in')):
                logger.info('Copying {}'.format(input_file))
                logger.info(os.path.realpath(input_file))
                logger.info(os.path.basename(input_file))
                shutil.copyfile(
                    os.path.realpath(input_file),
                    os.path.basename(input_file)
                )

            with open('Production_cmds.in', 'r') as f:
                cmds = Template(f.read())
            cmds = cmds.substitute(
                nsteps=nsteps,
                ns=sim.job_length
            )

            with open('Production_cmds.in', 'w+') as f:
                f.write(cmds)

            os.chdir(basedir)


class Adaptive(object):
    """

    """

    def __init__(self, nmin=1, nmax=2, nepochs=20, stride=1, sleeptime=3600,
                 model=None, app=None, atoms_to_load='all'):
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
        self.atoms_to_load = atoms_to_load

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
                logger.info('Reached {} epochs. Finishing.'.format(self.current_epoch))
                finished = True
            else:
                self.app.initialize_folders()
                self.fit_model()
                #self.spawns = self.respawn_from_tICs()
                self.spawns = self.respawn_from_MSM()
                self.app.prepare_spawns(self.spawns, self.current_epoch)

                self.app.prepare_PBS_jobs(
                    folders_glob=os.path.join(self.app.input_folder, 'e{:02d}*'.format(self.current_epoch)),
                    skeleton_function=partial(
                        simulate_in_pqigould,
                        func=generate_mdrun_skeleton,
                        host='cx1-15-6-1',
                        destination=None,
                        job_directory=None,
                        system_name=None
                    )
                )
                logger.info('Going to sleep for {} seconds'.format(self.sleeptime))
                # sleep(self.sleeptime)
                self.current_epoch += 1
                finished = True

    def respawn_from_MSM(self, percentile=0.5):
        """
        Find candidate frames in the trajectories to spawn new simulations from.
        We look for frames in the trajectories that are nearby regions with low population in the MSM equilibrium

        :param percentile: float, The percentile below which to look for low populated microstates of the MSM
        :return: a list of tuples, each tuple being (traj_id, frame_id)
        """

        msm = retrieve_MSM(self.model)
        clusterer = retrieve_clusterer(self.model)

        low_counts_ids = apply_percentile_search(
            count_array=msm.populations_,
            percentile=percentile,
            desired_length=self.app.ngpus,
            search_type='msm',
            msm=msm
        )

        if self.ttrajs is None:
            self.ttrajs = self.get_tica_trajs()

        # Finally, find frames in the trajectories that are nearby the selected cluster centers (low populated in the MSM)
        # Only retrieve one frame per cluster center
        return sample_states(
            trajs=self.ttrajs,
            state_centers=clusterer.cluster_centers_[low_counts_ids]
        )

    def respawn_from_tICs(self, dims=(0, 1)):
        """
        Find candidate frames in the trajectories to spawn new simulations from.
        Look for frames in the trajectories that are nearby the edge regions of the tIC converted space

        :param dims: tICs to sample from
        :return chosen_frames: a list of tuples, each tuple being (traj_id, frame_id)
        """

        if self.ttrajs is None:
            self.ttrajs = self.get_tica_trajs()

        frames_per_tIC = max(1, int(self.app.ngpus / len(dims)))

        chosen_frames = []
        for d in dims:
            sampled_pairs = sample_dimension(
                self.ttrajs,
                dimension=d,
                n_frames=frames_per_tIC,
                scheme='edge'
            )
            for pair in sampled_pairs:
                chosen_frames.append(pair)

        return chosen_frames

    def respawn_from_clusterer(self, percentile=0.5):
        """
        Find candidate frames in the trajectories to spawn new simulations from.
        Look for frames in the trajectories that are nearby the cluster centers that have low counts

        :param percentile: float, The percentile below which to look for low populated microstates of the MSM
        :return: a list of tuples, each tuple being (traj_id, frame_id)
        """

        clusterer = retrieve_clusterer(self.model)

        low_counts_ids = apply_percentile_search(
            count_array=clusterer.counts_,
            percentile=percentile,
            desired_length=self.app.ngpus,
            search_type='clusterer'
        )

        if self.ttrajs is None:
            self.ttrajs = self.get_tica_trajs()

        return sample_states(
            trajs=self.ttrajs,
            state_centers=clusterer.cluster_centers_[low_counts_ids]
        )

    def trajs_from_irrows(self, irow):
        """
        Load each trajectory in the rows of an msmbuilder.metadata object
        :param irow: iterable coming from pd.DataFrame.iterrow method
        :return i, traj: The traj id (starting at 0) and the mdtraj.Trajectory object
        """
        i, row = irow
        logger.info('Loading {}'.format(row['traj_fn']))
        atom_ids = mdtraj.load_topology(row['top_fn']).select(self.atoms_to_load)
        logger.debug('Will load {} atoms'.format(len(atom_ids)))
        traj = mdtraj.load(row['traj_fn'], top=row['top_fn'], stride=self.stride, atom_indices=atom_ids)
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
        featurizer = retrieve_feat(self.model)
        scaler = retrieve_scaler(self.model)
        decomposer = retrieve_decomposer(self.model)


        logger.info('Featurizing trajs')
        ftrajs = get_ftrajs(self.traj_dict, featurizer)

        logger.info('Scaling ftrajs')
        sctrajs = get_sctrajs(ftrajs, scaler)

        logger.info('Getting output of tICA')
        ttrajs = get_ttrajs(sctrajs, decomposer)

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
                    ('msm', MarkovStateModel(lag_time=lag_time, ergodic_cutoff='off', reversible_type=None))
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
    ad.respawn_from_MSM()
