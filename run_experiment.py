from typing import Optional
import pathlib

import click
from dneg_ml_toolkit import run_experiment_utils
from dneg_ml_toolkit.src.utils.logger import Logger, LogLevel
from dneg_ml_toolkit.src.globals import Globals
from dneg_ml_toolkit.src.register_components import register_toolkit_components
from src.register_components import register_project_components
from src.train import train as run_train

current_folder = pathlib.Path(__file__).parent.resolve()


@click.group()
def cli():
    """
    click group for the commands that are called from commandline (i.e. train, make-experiment)

    Returns:
        None
    """

    pass  # pylint: disable=unnecessary-pass


@cli.command()
@click.option('--experiment', help='Name of the experiment to run.', required=True)
@click.option('--device', default="cpu",
              help="Device selection: 'cpu' will run the experiment on cpu; '1' will run on the given number of gpus;"
                   "'[1]' will run on the gpu with the given index; '[0,1]' "
                   "will run on multiple gpus with the given indices",
              required=True)
@click.option('--resume', '-r', is_flag=True, help="Attempt to resume the the training from the latest checkpoint. "
                                                   "If no checkpoint is found, will start from the beginning.")
@click.option('--resume_from_checkpoint', help="Path to a specific checkpoint to resume the training from. "
                                               "Cannot be use if --resume is enabled.")
def train(experiment: str, device: str, resume: bool, resume_from_checkpoint: Optional[str] = None) -> None:
    """
    Run a training on the specified experiment

    """
    # 1. Get the experiment folder
    experiments_root_folder = current_folder.joinpath(Globals().EXPERIMENTS_FOLDER)
    experiment_folder = experiments_root_folder.joinpath(experiment)
    assert experiment_folder.is_dir(), "Cannot run experiment {}, experiment folder does " \
                                       "not exist at: {}. Run run_experiment make-experiment " \
                                       "or manually create the experiment folder and configuration " \
                                       "file".format(experiment, experiment_folder)

    # 2. Configure the system log
    Logger().configure(log_name="{}_{}".format(experiment, Globals().SYSTEM_LOG_NAME),
                       min_console_log_level=LogLevel.INFO, # TODO Make log level configurable
                       min_file_log_level=LogLevel.DEBUG,
                       save_path=str(experiment_folder))

    if resume and resume_from_checkpoint is not None:
        raise ValueError("Cannot use --resume_from_checkpoint when --resume is enabled. Use --resume to resume from"
                         "the latest checkpoint, or --resume_from_checkpoint with a valid checkpoint path to resume"
                         "from a specific checkpoint.")

    # 3. Register the core toolkit components, then the components for the current project
    register_toolkit_components()
    register_project_components()

    # 4. Build the experiment config by parsing the json configuration in the experiment's folder.
    # The experiment config must be built after all the components have been registered,
    # so that it has access to the registered components
    config = run_experiment_utils.build_experiment_config(experiment_folder=experiment_folder, experiment=experiment,
                                                          device=device)

    # 5. Call the trainer to run the training
    run_train(training_config=config, resume=resume, resume_checkpoint=resume_from_checkpoint)


@cli.command()
@click.option('--name', help='Name of the experiment to create.', required=True)
@click.option('--template', help="Template for experiment to create. This is the name of a json template "
                                 "in the project's config_templates folder"
                                 " (without the .json extension).",
              type=click.Choice(run_experiment_utils.get_json_templates(current_folder)), required=True)
def make_experiment(name: str, template: str) -> None:
    """
    Create a new experiment in the experiments folder, using the selected json template
    """

    run_experiment_utils.create_experiment_from_template(current_folder=current_folder, experiment_name=name,
                                                         template=template)


if __name__ == "__main__":
    cli()
