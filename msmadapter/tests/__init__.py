from ..adaptive import App
import logging
logging.disable(logging.CRITICAL)
from shutil import rmtree


class TestAppBase:

    def __init__(self):
        self.app = App(
            generator_folder='data_app/generators',
            data_folder='data_app/runs',
            input_folder='data_app/inputs',
            filtered_folder='data_app/filtered_trajs',
            model_folder='data_app/model',
            build_folder='data_app/build'

        )

    def setUp(self):
        self.app.initialize_folders()
