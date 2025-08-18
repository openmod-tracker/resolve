import pathlib

import pandas as pd
import xlwings as xw

# TODO 11-21-2022 JLC currently excluding loads in pathways. If we add them back, need to list Load, DeviceToLoad,and EnergyDemandSubsectorToLoad
components = [
    ("CandidateFuel", "CandidateFuel"),
    ("SRS_Key_Drivers", "StockRolloverSubsector"),
    ("SRS_Device_Inputs", "Device"),
    ("Sector", "Sector"),
    ("Pollutant", "Pollutant"),
    ("Emissions_Only_Inputs", "NonEnergySubsector"),
    ("ENOS_Energy_Demand_Inputs", "EnergyDemandSubsector"),
    ("FuelTypes", "FinalFuel"),
]

linkages = [
    ("CandidateFuel", "CandidateFuelToFinalFuel"),
    ("DeviceToFinalFuel", "DeviceToFinalFuel"),
    ("SRS_Device_Inputs", "StockRolloverSubsectorToDevice"),
    ("SRS_Key_Drivers", "StockRolloverSubsectorToSector"),
    ("ENOS_Energy_Demand_Inputs", "EnergyDemandSubsectorToFinalFuel"),
    ("ENOS_Energy_Demand_Inputs", "EnergyDemandSubsectorToSector"),
    ("Emissions_Only_Inputs", "NonEnergySubsectorToPollutant"),
    ("Emissions_Only_Inputs", "NonEnergySubsectorToSector"),
    ("EmissionsFactors", "CandidateFuelToPollutant"),
]

three_way_linkages = [
    ("SectorCandidateFuelBlending", "SectorCandidateFuelBlending"),
    ("ENOS_Fuel_Switching", "EnergyDemandSubsectorFuelSwitching"),
]


def get_system(sheet_name: str):
    """
    Gets unique component, linkage, and three way linkage instances from scenario tool workbook and saves them in csv files
    in the system data folder. When run after save_attributes_files in scenario_tool.py, it overrides the csv files with
    the entire superset of instances that exists in the named ranges in the workbook.

    """
    # Open workbook and define system worksheet
    wb = xw.Book.caller()
    sheet = wb.sheets[sheet_name]

    # Define current working directory from calling workbook and path to interim data directory
    curr_dir = pathlib.Path(wb.fullname).parent
    interim_dir = curr_dir / "data" / "interim"

    # Get system name / directory
    system_dir = interim_dir / "systems" / sheet.range("system_name").value
    system_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataframe
    df_components = pd.DataFrame()
    df_linkages = pd.DataFrame()
    df_three_way_linkages = pd.DataFrame()

    # Append unique instances of each component type to the dataframe
    for component_sheet, component in components:
        df = xw.sheets[component_sheet].range(component).options(pd.DataFrame, header=0, index=0).value
        df = df.drop_duplicates(0)
        df["component"] = [component] * len(df)
        df_components = pd.concat([df_components, df])

    for linkage_sheet, linkage in linkages:
        df = xw.sheets[linkage_sheet].range(linkage).options(pd.DataFrame, header=0, index=0).value
        df = df.drop_duplicates()
        df["linkage"] = [linkage] * len(df)
        df_linkages = pd.concat([df_linkages, df])

    for three_way_linkage_sheet, three_way_linkage in three_way_linkages:
        df = xw.sheets[three_way_linkage_sheet].range(three_way_linkage).options(pd.DataFrame, header=0, index=0).value
        df = df.drop_duplicates()
        df["linkage"] = [three_way_linkage] * len(df)
        df_three_way_linkages = pd.concat([df_three_way_linkages, df])

    ###  Remove Null Values and reformat ###
    df_components = df_components.dropna(axis=0)
    df_components = df_components[["component", 0]]
    df_components = df_components.rename(columns={0: "instance"})

    df_linkages = df_linkages.dropna(axis=0)
    df_linkages = df_linkages[["linkage", 0, 1]]
    df_linkages = df_linkages.rename(columns={0: "component_from", 1: "component_to"})
    # Remove rows with '0' components
    df_linkages = df_linkages.loc[~(df_linkages["component_from"] == 0)]

    df_three_way_linkages = df_three_way_linkages.dropna(axis=0)
    df_three_way_linkages = df_three_way_linkages[["linkage", 0, 1, 2]]
    df_three_way_linkages = df_three_way_linkages.rename(columns={0: "component_1", 1: "component_2", 2: "component_3"})
    # Remove rows with '0' components
    df_three_way_linkages = df_three_way_linkages.loc[~(df_three_way_linkages["component_1"] == 0)]

    # Save to components.csv and linkages.csv
    df_components.to_csv(system_dir / "components.csv", index=False)
    df_linkages.to_csv(system_dir / "linkages.csv", index=False)
    df_three_way_linkages.to_csv(system_dir / "three_way_linkages.csv", index=False)


def main():
    # Create mock caller
    print("Saving PATHWAYS system...")
    curr_dir = pathlib.Path(__file__).parent
    xw.Book(curr_dir / ".." / ".." / "Pathways Scenario Tool.xlsb").set_mock_caller()
    wb = xw.Book.caller()

    get_system(sheet_name="System")


if __name__ == "__main__":
    main()
