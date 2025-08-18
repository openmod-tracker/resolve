import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger

from new_modeling_toolkit.core.utils.gurobi_utils import GurobiCredentials
from new_modeling_toolkit.core.utils.util import DirStructure
from new_modeling_toolkit.recap.recap_case import RecapCase

_DEFAULT_GUROBI_LICENSE_LOCATION = str(Path(__file__).parents[2].joinpath("gurobi.lic"))

def run_recap_model(
    cases_to_run: Optional[str] = typer.Argument(
        None,
        help="Name of a RECAP case (under ./data/settings/recap). If `None`, will run all cases listed in ./data/settings/recap/cases_to_run.csv",
    ),
    data_folder: str = typer.Option(
        "data-test",
        help="Name of data folder, which is assumed to be in the same folder as `new_modeling_toolkit` folder.",
    ),
    log_level: str = typer.Option(
        "DEBUG",
        help="Any Python logging level: [DEBUG, INFO, WARNING, ERROR, CRITICAL]. "
        "Choosing DEBUG will also enable Pyomo `tee=True` and `symbolic_solver_labels` options.",
    ),
):
    # Remove default loguru logger to stderr + set stdout logging level
    logger.remove()
    logger.add(sys.__stdout__, level=log_level)

    # Get directory structure + setup RECAP folder structure
    dir_str = DirStructure(data_folder=data_folder)
    dir_str.make_recap_dir()

    # Get Gurobi credentials
    try:
        gurobi_credentials = GurobiCredentials.from_license_file(_DEFAULT_GUROBI_LICENSE_LOCATION)
    except FileNotFoundError as error:
        raise ValueError(
            f"Gurobi license file was not found at expected location `{_DEFAULT_GUROBI_LICENSE_LOCATION}`. "
            f"Put your license file at the default location: `{_DEFAULT_GUROBI_LICENSE_LOCATION}`"
        ) from error

    # Get cases to run
    if cases_to_run is None:
        cases_to_run = pd.read_csv(dir_str.recap_settings_dir / "cases_to_run.csv").values.flatten()
    else:
        cases_to_run = [cases_to_run]

    # Run cases in order
    start = time.time()
    failed_cases = []  # Initialize list to catch failed cases
    for case_name in cases_to_run:
        # Create copy of dir_str for case
        case_dir_str = dir_str.copy()
        # Update directory structure / output directory
        case_dir_str.make_recap_dir(case_name)
        # Initialize RECAP case from directory
        recap_case = RecapCase.from_dir(
            case_name=case_name, dir_str=case_dir_str, gurobi_credentials=gurobi_credentials
        )

        # Add case logger to logger
        i = logger.add(recap_case.dir_str.recap_output_dir / "recap.log", level="DEBUG")
        t = logger.add(recap_case.dir_str.recap_output_dir / "timing_log.log", level="SUCCESS", format="{message}")

        # Execute RECAP case
        try:
            logger.info(f"Executing case {case_name}")
            # Setup Monte Carlo draws for case
            recap_case.setup_monte_carlo_draws()
            # Run case (calculate reliability / perfect capacity shortfall / ELCCs)
            recap_case.run_case()
            # Save case results
            recap_case.report_results()
            # Save timing results
            recap_case.print_timing_df()

        # Catch failed case / error
        except Exception as e:
            logger.warning(f"Case {recap_case.case_name} could not be completed, error below:")
            # Print error
            logger.error(traceback.format_exc())
            # Record name of failed case
            failed_cases.append(case_name)
            # Save case results
            recap_case.report_results()

        # Remove case logger from logger
        logger.remove(i)
        logger.remove(t)

    # Report final status of attempted cases
    if len(failed_cases) > 0:
        logger.warning(f"All cases attempted. The following cases failed: {failed_cases}")
    else:
        end = time.time()
        logger.info(f"All cases completed!: {end - start}")

if __name__ == "__main__":
    typer.run(run_recap_model)
