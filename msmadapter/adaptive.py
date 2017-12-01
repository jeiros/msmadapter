import logging
import os
import shutil
from functools import partial
from glob import glob
from multiprocessing import Pool
from string import Template
import subprocess
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

from sklearn.pipeline import Pipeline
import numpy
from .model_utils import retrieve_feat, retrieve_clusterer, retrieve_MSM, \
    retrieve_scaler, retrieve_decomposer, apply_percentile_search
from .pbs_utils import generate_mdrun_skeleton, simulate_in_pqigould
from .utils import get_ftrajs, get_sctrajs, get_ttrajs, create_folder, \
    write_cpptraj_script, write_tleap_script, create_symlinks, hmr_prmtop

logger = logging.getLogger(__name__)


class App(object):
    """
    Handles the creation of all the necessary files to set up and run the
    simulations
    """

    def __init__(self, generator_folder='generators', data_folder='data',
                 input_folder='input', filtered_folder='filtered',
                 model_folder='model', build_folder='build', ngpus=4,
                 meta=None, project_name='adaptive', user='je714',
                 from_solvated=False):
        """
        Parameters
        ----------
        :generator_folder: str
        :data_folder: str
        :input_folder: str
        :filtered_folder: str
        :model_folder: str
        :build_folder: str
        :ngpus: int
        :meta: str
        :project_name: str
        :user: str
        :from_solvated: bool
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
        self.user = user
        self.from_solvated = from_solvated

    def __repr__(self):
        doc = """
App: {total} GPUs, {in_use} in use
"""
        return doc.format(
            total=self.ngpus,
            in_use=self.gpus_in_use
        )

    @property
    def finished_trajs(self):
        "Count how many trajs are inside the data_folder"
        return len(glob('/'.join([self.data_folder, '*nc'])))

    @property
    def completed_trajs(self):
        "Count how many trajs there are inside the input_folder"
        counter = 0
        folders_in_input_folder = glob('{}/*'.format(self.input_folder))
        for folder in folders_in_input_folder:
            if os.path.isdir(folder):
                counter += 1
        return counter

    @property
    def gpus_in_use(self):
        """
        This command shows how many GPUS are being used:
        $ nvidia-smi --query-compute-apps=pid --format=csv
        pid
        27044
        27278

        So we can pipe it into wc -l to count the lines and then deduct
        one from it to know how many GPUs are in use
        """
        bash_cmd = 'nvidia-smi \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv | wc -l'
        output = subprocess.check_output(['bash', '-c', bash_cmd])
        return int(output) - 1

    @property
    def available_gpus(self):
        return self.ngpus - self.gpus_in_use

    def initialize_folders(self):
        """
        Create folders for adaptive simulation if they do not exist
        under the current path
        """
        logger.info('Initializing folders')
        create_folder(self.generator_folder)
        create_folder(self.data_folder)
        create_folder(self.input_folder)
        create_folder(self.filtered_folder)
        create_folder(self.model_folder)
        create_folder(self.build_folder)

    def build_metadata(self, meta):
        """
        Builds an msmbuilder metadata object
        """
        if meta is None:
            try:
                parser = NumberedRunsParser(
                    traj_fmt='run-{run}.nc',
                    top_fn='structure.prmtop',
                    step_ps=200
                )
                meta = gather_metadata('/'.join([self.data_folder, '*nc']), parser)
            except:
                logger.warning("Could not automatically build metadata")
                return None
        else:
            if not isinstance(meta, pd.DataFrame):
                meta = load_meta(meta)
        return meta

    def prepare_spawns(self, spawns, epoch):
        """
        Prepare the prmtop and inpcrd files of the selected list of spawns.
        :param spawns: list of tuples, (traj_id, frame_id)
        :param epoch: int, Epoch the selected spawns belong to
        """
        sim_count = 1
        basedir = os.getcwd()
        for traj_id, frame_id in spawns:
            logger.info('Building simulation {} of epoch {}'.format(sim_count, epoch))

            folder_name = 'e{:02d}s{:02d}_{}f{:04d}'.format(epoch, sim_count, traj_id, frame_id)
            destination = os.path.join(self.input_folder, folder_name)
            create_folder(destination)

            if not self.from_solvated:
                # Add files from build folder to destination folder so tleap
                # can read them since we're not retrieving frame from an
                # already solvated trajectory
                create_symlinks(
                    files=glob(os.path.join(self.build_folder, '*')),
                    dst_folder=os.path.realpath(destination)
                )

            # All files in destination, so now move into it
            os.chdir(destination)

            # Structure
            if self.from_solvated:
                outfile = 'seed.ncrst'
            else:
                outfile = 'seed.pdb'
            write_cpptraj_script(
                traj=os.path.relpath(
                    os.path.join(
                        basedir,
                        self.meta.loc[traj_id]['traj_fn']
                    )
                ),
                top=os.path.relpath(
                    self.meta.loc[traj_id]['top_abs_fn']
                ),
                frame1=frame_id,
                frame2=frame_id,
                outfile=outfile,
                path='script.cpptraj',
                run=True
            )

            # Topology
            if not self.from_solvated:
                write_tleap_script(
                    pdb_file='seed.pdb',
                    run=True,
                    system_name='structure',
                    path='script.tleap'
                )
                # Apply hmr to new topologies
                hmr_prmtop(top_fn='structure.prmtop')
            else:
                os.symlink(
                    os.path.relpath(
                        self.meta.loc[traj_id]['top_abs_fn']
                    ),
                    'structure.prmtop'
                )

            # AMBER input files
            job_length = sim.job_length
            nsteps = int(job_length * 1e6 / 4)  # ns to steps, using 4 fs / step
            script_dir = os.path.dirname(__file__)  # Absolute path the script is in
            templates_path = 'templates'
            for input_file in glob(os.path.join(script_dir, templates_path, '*in')):
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


    def prepare_PBS_jobs(self, folders_glob, skeleton_function):
        """
        Uses the mdrun package to automate the generation of PBS scripts for the launch of each
        simulation to be run on GPUs in the HPC facility at ICL.
        Each simulation can be split in several shorter 'jobs'.
        Paramters
        ----------
        :folders_glob: str, a glob expression of the folders containing the necessary files for AMBER simulation
            (a prmtop and a restart files)
        :skeleton_function: callable, The function to specify the settings of the HPC job

        Returns
        -------
        None
        """

        folder_fnames_list = glob(folders_glob)
        basedir = os.getcwd()

        for input_folder in folder_fnames_list:
            # get eXXsYY from input/eXXsYY
            system_name = input_folder.split('/')[-1].split('_')[0]
            # create data/eXXsYY if it does not exist already
            data_folder = os.path.realpath(
                os.path.join(
                    self.data_folder,
                    system_name
                )
            )
            create_folder(data_folder)
            # Symlink the files inside the input folder to the data folder
            create_symlinks(files=os.path.join(input_folder, 'structure*'),
                            dst_folder=os.path.realpath(data_folder))
            create_symlinks(files=os.path.join(input_folder, '*.in'),
                            dst_folder=os.path.realpath(data_folder))
            # Move inside the data folder
            os.chdir(data_folder)
            skeleton = skeleton_function(
                system_name=system_name,
                job_directory=os.path.join('/work/{}'.format(self.user),
                                           self.project_name, system_name),
                destination=os.path.realpath(data_folder)
            )
            sim = Simulation(skeleton)
            sim.writeSimulationFiles()

            os.chdir(basedir)

    def run_local_GPU(self, folders_glob):
        bash_cmd = "export CUDA_VISIBLE_DEVICES=0\n"
        if len(glob(folders_glob)) > (self.ngpus - self.gpus_in_use):
            raise ValueError("Cannot run jobs of {} folders as only {} GPUs are available".format(len(glob(folders_glob)), self.ngpus - self.gpus_in_use))

        for folder in glob(folders_glob):
            bash_cmd += 'cd {}\n'.format(folder)
            bash_cmd += """nohup pmemd.cuda_SPFP -O -i Production.in \
            -c seed.ncrst -p structure.prmtop -r Production.rst \
            -x Production.nc &
            ((CUDA_VISIBLE_DEVICES++))
            cd ..
            """

        with open('run.sh', 'w') as f:
            f.write(bash_cmd)

        output = subprocess.check_output(['bash', './run.sh'])
        return output





class Adaptive(object):

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
        self.current_epoch = 1
        self.spawns = None
        self.atoms_to_load = atoms_to_load

    def __repr__(self):

        doc = """Adaptive Search
nmin : {nmin}
nmin : {nmin},
nmax : {nmax},
nepochs : {nepochs},
stride : {stride},
sleeptime : {sleeptime},
timestep : {timestep},
model : {model},
atoms_to_load : {atoms_to_load},
app : {app},
mode : {mode}
"""
        return doc.format(
            nmin=self.nmin,
            nmin=self.nmin,
            nmax=self.nmax,
            nepochs=self.nepochs,
            stride=self.stride,
            sleeptime=self.sleeptime,
            timestep=self.timestep,
            model=self.model,
            atoms_to_load=self.atoms_to_load,
            app=self.app,
            mode=self.mode,
        )

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

    def respawn_from_MSM(self, percentile=0.5, search_type='populations'):
        """
        Find candidate frames in the trajectories to spawn new simulations from.

        i) We can look for frames in the trajectories that are nearby regions with low population in the MSM equilibrium
        ii) We can also look for microstates that have low counts of transitions out of them

        Parameters
        ----------
        percentile: float, The percentile below which to look for low populated microstates of the MSM
        search_type: str, either 'populations' or 'counts'

        Returns
        -------
        selected_states: a list of tuples, each tuple being (traj_id, frame_id)
        """

        msm = retrieve_MSM(self.model)
        clusterer = retrieve_clusterer(self.model)

        if search_type not in ['populations', 'counts']:
            raise ValueError("search_type is not 'populations' or 'counts'")

        if search_type == 'counts':
            # Counts amount of transitions out of each microstate of the MSM
            count_matrix = numpy.sum(msm.countsmat_, axis=1)
        else:
            # The equilibrium population (stationary eigenvector) of transmat_
            count_matrix = msm.populations_

        low_counts_ids = apply_percentile_search(
            count_array=count_matrix,
            percentile=percentile,
            desired_length=self.app.ngpus,
            search_type='msm',
            msm=msm
        )

        if self.ttrajs is None:
            self.ttrajs = self.get_tica_trajs()

        # Find frames in the trajectories that are nearby the selected cluster centers
        # Only retrieve one frame per cluster center
        selected_states = sample_states(
            trajs=self.ttrajs,
            state_centers=clusterer.cluster_centers_[low_counts_ids]
        )
        return selected_states

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
                logger.info('Building default model based on dihedrals')
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
