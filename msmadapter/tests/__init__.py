from ..adaptive import App
import logging
from msmbuilder.io import NumberedRunsParser, gather_metadata
import os
from ..adaptive import create_folder

logging.disable(logging.CRITICAL)


parser = NumberedRunsParser(
                    traj_fmt='run-{run}.nc',
                    top_fn='data_app/runs/structure.prmtop',
                    step_ps=200
)
meta = gather_metadata('/'.join(['data_app/runs/', '*nc']), parser)

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