import sys

import typer
from loguru import logger

from new_modeling_toolkit.core.utils.util import DirStructure
from new_modeling_toolkit.recap import recap_model


def run_recap2_wrapper(
    data_folder: str = typer.Option(
        "data-benchmark",
        help="Name of data folder, which is assumed to be in the same folder as `new_modeling_toolkit` folder.",
    ),
    log_level: str = typer.Option(
        "DEBUG",
        help="Any Python logging level: [DEBUG, INFO, WARNING, ERROR, CRITICAL]. "
        "Choosing DEBUG will also enable Pyomo `tee=True` and `symbolic_solver_labels` options.",
    ),
):
    # Remove default loguru logger to stderr
    logger.remove()
    # Set stdout logging level
    logger.add(sys.__stdout__, level=log_level)

    # Get directory structure
    dir_str = DirStructure(data_folder=data_folder)
    dir_str.make_recap_dir()

    # Instantiate RecapModel instance
    # RecapModel --> RecapCase --> MonteCarloDraw
    recap_model_instance = recap_model.RecapModel(dir_str, use_recap2_wrapper=True)

    # Call RecapModel.run_model()
    recap_model_instance.run_recap2_wrappers()


if __name__ == "__main__":
    typer.run(run_recap2_wrapper)
