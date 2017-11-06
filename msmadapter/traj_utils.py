import mdtraj
from subprocess import call, PIPE
import os
from string import Template
import shutil
from glob import glob


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


def traj_from_stateinds(inds, meta):
    """
    Generate a 'fake' trajectory from the traj_ids and frame_ids inside the inds list
    In principle this are the chosen frame to spawn new simulations from
    :param inds: list of tuples, each being (traj_id, frame_id)
    :param meta: an msmbuilder.metadata object
    :return traj: an mdtraj.Trajectory object
    """
    traj = mdtraj.join(
        mdtraj.load_frame(meta.loc[traj_i]['traj_fn'],
                          top=meta.loc[traj_i]['top_fn'],
                          index=frame_i) for traj_i, frame_i in inds
    )
    return traj


def write_cpptraj_script(traj, top, frame1=1, frame2=1, outfile=None,
                         write=True, path='script.cpptraj', run=False):
    """
    Create a cpptraj script to load specific range of frames from a trajectory and write them out to a file

    :param traj: str, Location in disk of trajectories to load
    :param top: str, Location in disk of the topology file
    :param frame1: int, The first frame to load
    :param frame2: int, The last frame to load
    :param outfile: str, Name (with file format extension) of the output trajectory
    :param write: bool, Whether to write the script to a file in disk
    :param path: str, Where to save the script in disk.
    :param run: bool, Whether to run the script after writing it to disk
    :return cmds: str, the string representing the cpptraj script
    """
    if run and not write:
        raise ValueError('Cannot call the script without writing it to disk')
    if outfile is None:
        outfile = 'pdbs/' + traj.split('.')[0] + '.pdb'

    script_dir = os.path.dirname(__file__)  # Absolute path the script is in
    relative_path = 'templates/template.cpptraj'
    shutil.copyfile(
        os.path.join(script_dir, relative_path), './template.cpptraj')
    with open('./template.cpptraj', 'r') as f:
        cmds = Template(f.read())
    cmds = cmds.substitute(
        top=top,
        traj=traj,
        frame1=frame1,
        frame2=frame2,
        outfile=outfile
    )
    if write:
        with open(path, 'w') as f:
            f.write(cmds)
        if run:
            call(['cpptraj', '-i', path], stdout=PIPE)

    return cmds


def write_tleap_script(pdb_file='seed.pdb', box_dimensions='25 25 35', counterions='220',
                       system_name='seed_wat', write=True, path='script.tleap',
                       run=False):
    """
    Write and optionally run a tleap script to build an inpcrd and prmtop file from a pdb file
    :param pdb_file: str, path to the pdb file
    :param box_dimensions: str, should be readable by tleaps solvatebox function
    :param counterions: str or int, number of Na+ and Cl- ions to add (of each one)
    :param system_name: str, name of the built system
    :param write: bool, whether to write the script to a file or not
    :param path: str, filename of the tleap script
    :param run: bool, whether to run the script with tleap or not (should be in the path)
    :return cmds: list of str
    """
    script_dir = os.path.dirname(__file__)  # Absolute path the script is in
    relative_path = 'templates/template.tleap'
    shutil.copyfile(
        os.path.join(script_dir, relative_path), './template.tleap')
    with open('./template.tleap', 'r') as f:
        cmds = Template(f.read())
    cmds = cmds.substitute(
        pdb_file=pdb_file,
        box_dimensions=box_dimensions,
        counterions=counterions,
        system_name=system_name
    )
    if write:
        with open(path, 'w') as f:
            f.write(cmds)
        if run:
            call(['tleap', '-f', path], stdout=PIPE)
    return cmds


def create_symlinks(files, dst_folder):
    """
    Create symlinks inside dst_folder for all the files passed as a glob expression
    :param files: str, glob expression matching the files
    :param dst_folder: str, the destination folder of the symlinks
    """
    fns_list = glob(files)
    for fn in fns_list:
        fname_trimmed = fn.split('/')[-1]
        dst_fname = os.path.join(dst_folder, fname_trimmed)
        os.symlink(os.path.realpath(fn), dst_fname)
