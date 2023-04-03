from typing import Optional
import click
import pathlib
from dneg_ml_toolkit.src.utils.logger import Logger, LogLevel
from dneg_ml_toolkit.src.Component.component_store import ComponentStore

from dneg_ml_toolkit.src.register_components import register_toolkit_components
from src.register_components import register_project_components


@click.command()
@click.option("--name", help="Name of the Component to create.", required=True)
@click.option("--target_folder", help="Folder relative to the project's src folder to create the Component in.",
              required=True)
@click.option("--parent_component", help="Name of the Component to inherit from. E.g. if creating a new Network,"
                                         " inherit from Base_Network. "
                                         "If not provided, will inherit from ML Toolkit's base Component.")
@click.option('--is_base_component', '-b', is_flag=True, help="Set to true if the Component is intended to be an "
                                                              "abstract base Component with shared functionality "
                                                              "that other components will inherit from.")
def make_component(name: str, target_folder: str, parent_component: Optional[str], is_base_component: bool):
    """
    Development helper tool to create the scaffolding for a Component, including the Component Package, config class,
    and component class.
    """

    # 1. Configure the system log
    Logger().configure(log_name="make_component_log",
                       min_console_log_level=LogLevel.INFO,
                       min_file_log_level=LogLevel.DEBUG)

    # 2. Register the core toolkit components, then the components for the current project
    # This is necessary so the tool can find the parent Component when creating the new Component class
    register_toolkit_components()
    register_project_components()

    current_folder = pathlib.Path(__file__).parent.resolve()
    project_src_folder = current_folder.joinpath("src")

    # 3. Make the new Component
    ComponentStore().make_component(component_name=name, target_folder=target_folder,
                                    project_root=str(project_src_folder),
                                    parent_component=parent_component,
                                    is_base_component=is_base_component)


if __name__ == '__main__':
    make_component()
