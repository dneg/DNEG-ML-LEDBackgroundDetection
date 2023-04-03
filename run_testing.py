from typing import Optional, Union
import click
import pathlib

from dneg_ml_toolkit.src.globals import Globals
from dneg_ml_toolkit import run_experiment_utils
from dneg_ml_toolkit.src.register_components import register_toolkit_components

from src.register_components import register_project_components
from src.test import run_testing

current_folder = pathlib.Path(__file__).parent.resolve()


@click.command()
@click.option('--experiment', help='Name of the experiment to run testing on.', required=True)
@click.option('--run', help='Number of the experiment run to test.', required=True)
@click.option('--device', default="cpu",
              help="Device selection: 'cpu' will run the testing on cpu; '1' will run on the given number of gpus;"
                   "'[1]' will run on the gpu with the given index; '[0,1]' "
                   "will run on multiple gpus with the given indices",
              required=True)
@click.option('--checkpoint', help="Name of the checkpoint in the experiment's Checkpoint folder to load."
                                   "If not specified, will load the latest checkpoint.")
def run_test(experiment: str, run: Union[str, int], device: str, checkpoint: Optional[str] = None) -> None:
    # 1. Register the core toolkit components, then the components for the template project
    register_toolkit_components()
    register_project_components()

    # 2. The test config must be built after all the components have been registered,
    # so that it has access to the registered components
    config = run_experiment_utils.build_experiment_config(project_root_folder=current_folder,
                                                          experiment=experiment, run=run, device=device,
                                                          config_file_suffix=Globals().TEST_CONFIG_SUFFIX)

    run_testing(config, resume_checkpoint=checkpoint)


if __name__ == "__main__":
    run_test()
