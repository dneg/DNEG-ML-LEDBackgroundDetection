from typing import Optional, Union
import pathlib

import click
from dneg_ml_toolkit import run_experiment_utils
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
@click.option('--run', help='Number of the experiment run to train.', required=True)
@click.option('--device', default="cpu",
              help="Device selection: 'cpu' will run the experiment on cpu; '1' will run on the given number of gpus;"
                   "'[1]' will run on the gpu with the given index; '[0,1]' "
                   "will run on multiple gpus with the given indices",
              required=True)
@click.option('--resume', '-r', is_flag=True, help="Attempt to resume the the training from the latest checkpoint. "
                                                   "If no checkpoint is found, will start from the beginning.")
@click.option('--resume_from_checkpoint', help="Path to a specific checkpoint to resume the training from. "
                                               "Cannot be use if --resume is enabled.")
@click.option('--load', '-l', help="Path to a specific checkpoint to initialize a new training run"
                                   "(not resuming an old run)")
def train(experiment: str,
          run: Union[str, int],
          device: str,
          resume: bool,
          resume_from_checkpoint: Optional[str] = None,
          load: Optional[str] = None) -> None:
    """
    Run a training on the specified experiment

    """

    # 1. Check the resume args
    num_checkpoint_options = sum(bool(x) for x in [resume, resume_from_checkpoint, load])
    if num_checkpoint_options > 1:
        raise ValueError("Cannot use more than one of --resume_from_checkpoint, --resume, or --load")

    # 2. Register the core toolkit components, then the components for the current project
    register_toolkit_components()
    register_project_components()

    # 3. The experiment config must be built after all the components have been registered,
    # so that it has access to the registered components
    config = run_experiment_utils.build_experiment_config(project_root_folder=current_folder,
                                                          experiment=experiment, run=run, device=device)

    # 4. Call the trainer to run the training
    run_train(training_config=config, resume=resume, resume_checkpoint=resume_from_checkpoint, load_checkpoint=load)


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

@cli.command()
@click.option('--name', help='Name of the experiment to create a new run for.', required=True)
def make_run(name: str) -> None:
    """
    Create a new run for an existing experiment in the experiments folder
    """

    run_experiment_utils.create_new_experiment_run(project_root_folder=current_folder, experiment_name=name)


if __name__ == "__main__":
    cli()
