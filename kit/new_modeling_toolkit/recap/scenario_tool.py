import os
import pathlib
import sys
from typing import Optional

import numpy as np
import pandas as pd
import xlwings as xw

# Set traceback limit to 0 so that error message is more readable in Excel popup window
sys.tracebacklimit = 0

"""
Save model-specific case settings for RECAP.
"""


def save_RECAP_case_settings(sheet_name: str, model: str = "recap", data_folder: Optional[os.PathLike] = None):
    """Save RECAP case settings from Scenario Tool named ranges."""
    # Open workbook and define system worksheet
    wb = xw.Book.caller()
    sheet = wb.sheets[sheet_name]

    # Define current working directory and path to settings directory
    if data_folder is None:
        data_folder = pathlib.Path(wb.fullname).parent / "data"
    data_folder = pathlib.Path(data_folder)

    settings_dir = data_folder / "settings" / model / sheet.range("settings_name").value
    settings_dir.mkdir(parents=True, exist_ok=True)

    # Save case settings
    settings_files_dict = {
        "case_settings.csv": "case_settings",
        "scenarios.csv": "settings_scenarios",
    }
    for filename, setting in settings_files_dict.items():
        df = sheet.range(setting).options(pd.DataFrame).value
        df = df.loc[df.index.dropna()]
        df.to_csv(settings_dir / filename)

    # Create ELCC surfaces directory
    ELCC_surfaces_dir = settings_dir / "ELCC_surfaces"
    ELCC_surfaces_dir.mkdir(parents=True, exist_ok=True)
    save_custom_ELCC_surface(sheet, ELCC_surfaces_dir)

    # Save marginal resources ELCC surface
    # Read in marginal resources + step size
    marginal_ELCC_step_size = sheet.range("marginal_ELCC_step_size").value
    marginal_resources = (
        sheet.range("marginal_resources")
        .options(pd.DataFrame)
        .value.reset_index()
        .dropna(subset=["value"])
        .values.flatten()
    )
    # Create marginal ELCC points matrix
    marginal_ELCC_points_matrix = pd.DataFrame(
        columns=marginal_resources,
        data=np.diag([marginal_ELCC_step_size for resource in marginal_resources]),
    )
    marginal_ELCC_points_matrix["incremental"] = True  # Marginal ELCCs always incremental
    # Save ELCC surface to ELCC surfaces directory
    marginal_ELCC_points_matrix.to_csv(ELCC_surfaces_dir / "marginal_ELCC.csv", index=False)

    # Save incremental last-in resources ELCC surface
    # Read in incremental last-in resources
    incremental_resources = (
        sheet.range("incremental_last_in_resources")
        .options(pd.DataFrame)
        .value.reset_index()
        .dropna(subset=["value"])
        .values.flatten()
    )
    # Create incremental last-in ELCC points matrix
    incremental_ELCC_points_matrix = pd.DataFrame(
        columns=incremental_resources,
        data=np.diag([np.nan for resource in incremental_resources]),
    )
    incremental_ELCC_points_matrix["incremental"] = False  # ELCC point values are absolute, not incremental
    # Save ELCC surface to ELCC surfaces directory
    incremental_ELCC_points_matrix.to_csv(ELCC_surfaces_dir / "incremental_last_in_ELCC.csv", index=False)

    # Save decremental last-in resources ELCC surface
    # Read in decremental last-in resources
    decremental_resources = (
        sheet.range("decremental_last_in_resources")
        .options(pd.DataFrame)
        .value.reset_index()
        .dropna(subset=["value"])
        .values.flatten()
    )
    # Create decremental last-in ELCC points matrix
    decremental_ELCC_points_matrix = pd.DataFrame(
        columns=decremental_resources, data=np.full((len(decremental_resources), len(decremental_resources)), np.nan)
    )
    # Set diagonal elements to 0
    np.fill_diagonal(decremental_ELCC_points_matrix.values, 0)
    decremental_ELCC_points_matrix["incremental"] = False  # ELCC point values are absolute, not incremental
    # Save ELCC surface to ELCC surfaces directory
    decremental_ELCC_points_matrix.to_csv(ELCC_surfaces_dir / "decremental_last_in_ELCC.csv", index=False)


def save_custom_ELCC_surface(sheet, ELCC_surfaces_dir):
    """Save ELCC surface data to ELCC surfaces folder for current RECAP case"""

    # Get ELCC surface data
    ELCC_surface_incremental_flag = sheet.range("ELCC_surface_incremental_flag").value
    ELCC_points_matrix = (
        sheet.range("ELCC_points_matrix")
        .options(pd.DataFrame)
        .value.dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
    )

    # Convert incremental flag to boolean
    if type(ELCC_surface_incremental_flag) != bool:
        ELCC_surface_incremental_flag = ELCC_surface_incremental_flag.lower() in ("yes", "true", "t", "1")
    # Add "incremental flag" column to ELCC_points_matrix
    ELCC_points_matrix["incremental"] = ELCC_surface_incremental_flag

    # Write out ELCC surface to ELCC surfaces directory
    ELCC_points_matrix.to_csv(ELCC_surfaces_dir / "custom_ELCC_surface.csv", index=False)


def load_RECAP_case_settings(sheet_name: str, model: str = "recap", data_folder: Optional[os.PathLike] = None):
    """Load RECAP case settings and populate Scenario Tool named ranges."""
    # Open workbook and define system worksheet
    wb = xw.Book.caller()
    sheet = wb.sheets[sheet_name]

    # Define current working directory and path to settings directory
    if data_folder is None:
        data_folder = pathlib.Path(wb.fullname).parent / "data"
    data_folder = pathlib.Path(data_folder)

    settings_dir = data_folder / "settings" / model / sheet.range("recap_case_to_load").value

    # Save case settings
    settings_files_dict = {
        "case_settings.csv": "case_settings",
        "scenarios.csv": "settings_scenarios",
    }
    for filename, setting in settings_files_dict.items():
        df = pd.read_csv(settings_dir / filename, header=None)
        sheet.range(setting).clear_contents()
        sheet.range(setting).value = df.values

    # Create ELCC surfaces directory
    ELCC_surfaces_dir = settings_dir / "ELCC_surfaces"
    df_ELCC_surface = pd.read_csv(ELCC_surfaces_dir / "custom_ELCC_surface.csv")
    sheet.range("ELCC_points_matrix").clear_contents()
    if not df_ELCC_surface.empty:
        sheet.range("ELCC_surface_incremental_flag").value = str(df_ELCC_surface["incremental"].values[0])
        sheet.range("ELCC_points_matrix").value = df_ELCC_surface.drop(columns=["incremental"])

    # Populate marginal resources ELCC table
    df_marginal_ELCC = pd.read_csv(ELCC_surfaces_dir / "marginal_ELCC.csv")
    sheet.range("marginal_resources").clear_contents()
    if not df_marginal_ELCC.empty:
        marginal_resource = np.insert(df_marginal_ELCC.columns[:-1].values, 0, "value")
        sheet.range("marginal_resources").value = marginal_resource.reshape(-1, 1)
        sheet.range("marginal_ELCC_step_size").value = df_marginal_ELCC.iloc[0, 0]
    else:
        sheet.range("marginal_resources").value = ["value"]

    # Populate incremental last-in resources ELCC surface
    df_increment_ELCC = pd.read_csv(ELCC_surfaces_dir / "incremental_last_in_ELCC.csv")
    sheet.range("incremental_last_in_resources").clear_contents()
    if not df_increment_ELCC.empty:
        incremental_resource = np.insert(df_increment_ELCC.columns[:-1].values, 0, "value")
        sheet.range("incremental_last_in_resources").value = incremental_resource.reshape(-1, 1)
    else:
        sheet.range("incremental_last_in_resources").value = ["value"]

    # Populate decremental last-in resources ELCC surface
    df_decrement_ELCC = pd.read_csv(ELCC_surfaces_dir / "decremental_last_in_ELCC.csv")
    sheet.range("decremental_last_in_resources").clear_contents()
    if not df_decrement_ELCC.empty:
        decremental_resource = np.insert(df_decrement_ELCC.columns[:-1].values, 0, "value")
        sheet.range("decremental_last_in_resources").value = decremental_resource.reshape(-1, 1)
    else:
        sheet.range("decremental_last_in_resources").value = ["value"]


if __name__ == "__main__":
    # Create mock caller
    curr_dir = pathlib.Path(__file__).parent
    xw.Book(curr_dir / ".." / ".." / "Recap-Resolve Scenario Tool.xlsm").set_mock_caller()
    wb = xw.Book.caller()
    # Call functions
    save_RECAP_case_settings(sheet_name="RECAP Case Settings")
    # load_RECAP_case_settings(sheet_name="RECAP Case Settings")
