import os
import time

from dneg_ml_toolkit.src.register_components import register_components
from dneg_ml_toolkit.src.utils.logger import Logger


def register_project_components() -> None:
    """
    Register all the Components in the project to make them available to DNEG ML Toolkit's
    configuration system.

    Call this along with the Toolkit's register_components when initializing the system

    Returns:
        None
    """

    Logger().Log("--------------------Registering DNEG ML Template Components--------------------")
    start_time = time.time()
    package_dir = os.path.dirname(os.path.abspath(__file__))
    register_components(package_dir, namespace_root="src")
    time_taken = time.time() - start_time
    Logger().Log("Toolkit Components registered in: {}".format(Logger().format_time(time_taken)))
    Logger().Log("--------------------DNEG ML Template Components registered--------------------")
