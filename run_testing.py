from typing import Optional
import click
import pathlib

from dneg_ml_toolkit.src.utils.logger import Logger, LogLevel
from dneg_ml_toolkit.src.globals import Globals
from dneg_ml_toolkit import run_experiment_utils
from dneg_ml_toolkit.src.register_components import register_toolkit_components

from src.register_components import register_project_components
from src.test import run_testing

current_folder = pathlib.Path(__file__).parent.resolve()


@click.command()
@click.option('--experiment', help='Name of the experiment to run testing on.', required=True)
@click.option('--device', default="cpu",
              help="Device selection: 'cpu' will run the testing on cpu; '1' will run on the given number of gpus;"
                   "'[1]' will run on the gpu with the given index; '[0,1]' "
                   "will run on multiple gpus with the given indices",
              required=True)
@click.option('--checkpoint', help="Name of the checkpoint in the experiment's Checkpoint folder to load."
                                   "If not specified, will load the latest checkpoint.")
def run(experiment: str, device: str, checkpoint: Optional[str] = None) -> None:
    # 1. Get the experiment folder
    experiments_root_folder = current_folder.joinpath(Globals().EXPERIMENTS_FOLDER)
    experiment_folder = experiments_root_folder.joinpath(experiment)
    assert experiment_folder.is_dir(), "Cannot run testing for {}, experiment folder does " \
                                       "not exist at: {}. Run run_experiment make-experiment " \
                                       "or manually create the experiment folder and configuration " \
                                       "file".format(experiment, experiment_folder)

    # 2. Configure the system log
    Logger().configure(log_name="{}_{}".format(experiment, Globals().SYSTEM_LOG_NAME),
                       min_console_log_level=LogLevel.INFO,
                       min_file_log_level=LogLevel.DEBUG,
                       save_path=str(experiment_folder))

    # 3. Register the core toolkit components, then the components for the template project
    register_toolkit_components()
    register_project_components()

    # Load the testing config for the experiment
    test_experiment = "{}{}".format(experiment, Globals().TEST_CONFIG_SUFFIX)
    # 4. The test config must be built after all the components have been registered,
    # so that it has access to the registered components
    config = run_experiment_utils.build_experiment_config(experiment_folder=experiment_folder,
                                                          experiment=test_experiment,
                                                          device=device)

    run_testing(config, resume_checkpoint=checkpoint)


if __name__ == "__main__":
    run()
