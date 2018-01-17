from mdrun.Simulation import Simulation
import os
from glob import glob


def pbs_settings(walltime='72:0:0', nnodes=1, ncpus=1, ngpus=1, mem='2gb',
                 host='', queue='gpgpu', gpu_type='P100'):
    return dict(
        walltime=walltime,
        nnodes=nnodes,
        ncpus=ncpus,
        ngpus=ngpus,
        mem=mem,
        host=host,
        queue=queue,
        gpu_type=gpu_type
    )


def simulation_details(system_name="protein", inpcrd_file="structure.inpcrd",
                       topology_file="structure.prmtop", start_rst="Heated_eq.rst",
                       input_file="Production_cmds.in", start_time=0,
                       final_time=500, job_length=50, job_directory="/work/je714/protein",
                       cuda_version="8.0.44", binary_location="pmemd.cuda_SPFP",
                       pre_simulation_cmd=None, pre_simulation_type="gpu"):
    if pre_simulation_cmd is None:
        pre_simulation_cmd = [
            "{} -O -i min1.in -o premin.out -c $inpcrd -p $prmtop -r premin.rst -ref $inpcrd".format(binary_location),
            "{} -O -i min2.in -o sandermin1.out -c premin.rst -p $prmtop -r sandermin1.rst".format(binary_location),
            "{} -O -i 02_Heat.in -o 02_Heat.out -c sandermin1.rst -p $prmtop -r 02_Heat.rst -ref sandermin1.rst -x 02_Heat.nc".format(binary_location),
            "{} -O -i 03_Heat2.in -o 03_Heat2.out -c 02_Heat.rst -p $prmtop -r Heated_eq.rst -ref 02_Heat.rst -x 03_Heat2.nc".format(binary_location)
        ]
    return dict(
        system_name=system_name,
        inpcrd_file=inpcrd_file,
        topology_file=topology_file,
        start_rst=start_rst,
        input_file=input_file,
        start_time=start_time,
        final_time=final_time,
        job_length=job_length,
        job_directory=job_directory,
        cuda_version=cuda_version,
        binary_location=binary_location,
        pre_simulation_cmd=pre_simulation_cmd,
        pre_simulation_type=pre_simulation_type
    )


def local_machine_details(user="je714", hostname="ch-knuth.ch.ic.ac.uk",
                          destination="/Users/je714/protein"):
    return dict(
        user=user,
        hostname=hostname,
        destination=destination
    )


def master_node_details(user_m="username", hostname_m="master_node-hostname",
                        job_directory_m="/home/username/protein1"):
    return dict(
        user_m=user_m,
        hostname_m=hostname_m,
        job_directory_m=job_directory_m,
    )


def generate_mdrun_skeleton(scheduler='pbs', HPC_job='True', pbs_settings_kwargs=None,
                            simulation_details_kwargs=None, local_machine_kwargs=None,
                            master_node_kwargs=None):
    if pbs_settings_kwargs is None:
        pbs_settings_kwargs = {}
    if simulation_details_kwargs is None:
        simulation_details_kwargs = {}
    if local_machine_kwargs is None:
        local_machine_kwargs = {}
    if master_node_kwargs is None:
        master_node_kwargs = {}
    return dict(
        scheduler=scheduler,
        HPC_job=HPC_job,
        pbs_settings=pbs_settings(**pbs_settings_kwargs),
        simulation_details=simulation_details(**simulation_details_kwargs),
        local_machine=local_machine_details(**local_machine_kwargs),
        master_node=master_node_details(**master_node_kwargs)
    )


def simulate_in_P100s(func, job_directory, system_name, destination):
    pbs = pbs_settings(walltime='48:0:0', queue=None)
    sim = simulation_details(
        system_name=system_name, inpcrd_file="structure.inpcrd",
        topology_file="structure_hmr.prmtop", start_rst="Heated_eq.rst",
        input_file="Production.in", start_time=0,
        final_time=500, job_length=25, job_directory=job_directory,
        cuda_version="9.0", binary_location="/home/igould/amber16/bin/pmemd.cuda_SPFP",
        pre_simulation_cmd=None, pre_simulation_type="gpu"
    )
    local = local_machine_details(
        destination=destination
    )
    return func(pbs_settings_kwargs=pbs, simulation_details_kwargs=sim, local_machine_kwargs=local)


def simulate_in_pqigould(func, host, job_directory, system_name, destination):
    pbs = pbs_settings(
        walltime='48:0:0', nnodes=1, ncpus=1, ngpus=1, mem='2gb',
        host=host, queue='pqigould', gpu_type=''
    )
    sim = simulation_details(
        system_name=system_name, inpcrd_file="structure.inpcrd",
        topology_file="structure_hmr.prmtop", start_rst="Heated_eq.rst",
        input_file="Production.in", start_time=0,
        final_time=500, job_length=25, job_directory=job_directory,
        cuda_version="9.0", binary_location="/home/igould/amber16/bin/pmemd.cuda_SPFP",
        pre_simulation_cmd=None, pre_simulation_type="gpu"
    )
    local = local_machine_details(
        destination=destination
    )
    return func(pbs_settings_kwargs=pbs, simulation_details_kwargs=sim, local_machine_kwargs=local)


def remove_pbs_files():
    pbs_fnames = glob('*.pbs')
    for fname in pbs_fnames:
        os.remove(fname)

if __name__ == '__main__':
    skel = generate_mdrun_skeleton()
    print(skel)
    sim = Simulation(skel)
    sim.writeSimulationFiles()
    remove_pbs_files()
