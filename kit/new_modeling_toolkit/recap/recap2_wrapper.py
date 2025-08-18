import copy
import shutil

import numpy as np
import pandas as pd
from loguru import logger

from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.recap.recap_case import RecapCase
from new_modeling_toolkit.system.electric.resource_group import ResourceGroupCategory


class Recap2Wrapper(RecapCase):
    #### MANAGERIAL FUNCTIONS ####
    # Overwrite some methods from parent class

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create RECAP 2.0 directory structure
        self.dir_str.make_recap2_dir(case_name=self.case_name)

        # Initialize gross load
        self._gross_load = None

        # Set minimum resource nameplate capacity to 1e-5 to avoid scaling/numerical issues throughout RECAP code
        for resource in self.system.resources.keys():
            resource_obj = self.system.resources[resource]
            if resource_obj.capacity_planned.slice_by_year(self.model_year) < 1e-5:
                resource_obj.capacity_planned.data.loc[pd.Timestamp(year=self.model_year, month=1, day=1)] = 1e-5
            self.system.resources[resource] = resource_obj

        # Need to keep track of cases_to_run in the base case to create "cases_to_run.csv" in RECAP 2.0 input folder
        self.cases_to_run = []

    def run_wrapper(self):
        """Runs full RECAP case"""

        if self.case_settings.calculate_reliability or self.case_settings.calculate_perfect_capacity_shortfall:
            # skip_prm_calculations, calculate_capacity_shortfall_only
            self.create_recap2_input_files()

        if self.case_settings.calculate_portfolio_ELCC:
            raise ValueError("calculate_perfect_capacity_shortfall does not work in RECAP 2.0 wrapper")

        if self.case_settings.calculate_marginal_ELCC:
            logger.info(f"Calculating marginal ELCCs of resources in case {self.case_name}")
            marginal_ELCC_points_matrix = pd.read_csv(
                self.dir_str.recap_settings_dir / self.case_name / "ELCC_surfaces" / "marginal_ELCC.csv"
            )
            self.create_recap2_ELCC_points(marginal_ELCC_points_matrix, ELCC_surface_name="marginal_ELCC")

        if self.case_settings.calculate_incremental_last_in_ELCC:
            logger.info(f"Calculating last-in incremental ELCCs of resources in case {self.case_name}")
            incremental_last_in_ELCC_points_matrix = pd.read_csv(
                self.dir_str.recap_settings_dir / self.case_name / "ELCC_surfaces" / "incremental_last_in_ELCC.csv"
            )
            self.create_recap2_ELCC_points(
                incremental_last_in_ELCC_points_matrix, ELCC_surface_name="incremental_last_in_ELCC"
            )

        if self.case_settings.calculate_decremental_last_in_ELCC:
            logger.info(f"Calculating last-in decremental ELCCs of resources in case {self.case_name}")
            decremental_last_in_ELCC_points_matrix = pd.read_csv(
                self.dir_str.recap_settings_dir / self.case_name / "ELCC_surfaces" / "decremental_last_in_ELCC.csv"
            )
            self.create_recap2_ELCC_points(
                decremental_last_in_ELCC_points_matrix, ELCC_surface_name="decremental_last_in_ELCC"
            )

        if self.case_settings.calculate_ELCC_surface:
            logger.info(f"Calculating custom ELCC surface for case {self.case_name}")
            ELCC_points_matrix = pd.read_csv(
                self.dir_str.recap_settings_dir / self.case_name / "ELCC_surfaces" / "custom_ELCC_surface.csv"
            )
            self.create_recap2_ELCC_points(
                ELCC_points_matrix,
                ELCC_surface_name="custom_ELCC_surface",
            )

    def create_copy(self, case_name):
        kwargs = {
            "system": copy.deepcopy(self.system),
            "dir_str": copy.deepcopy(self.dir_str),
            "case_name": case_name,
            "case_settings": self.case_settings,
            "gurobi_credentials": self.gurobi_credentials,
            "monte_carlo_draws": copy.deepcopy(self.monte_carlo_draws),
            "skip_monte_carlo_draw_setup": True,
        }

        recap2_wrapper = Recap2Wrapper(**kwargs)

        return recap2_wrapper

    def rescale_portfolio(self, portfolio_vector: pd.Series, incremental: bool = False):
        # Avoid scaling nameplate capacities to 0; must be a small number > 0 (avoids scaling issues)
        portfolio_vector = portfolio_vector.dropna().clip(lower=1e-5)

        for resource in portfolio_vector.index:
            logger.info(f"Rescaling {resource} capacity")
            assert resource in self.system.resources.keys(), KeyError(
                f"Resource {resource} not in 'MC_draw.system.resources'"
            )
            resource_obj = self.system.resources[resource]
            resource_obj.rescale_resource_capacity(
                self.model_year, portfolio_vector.loc[resource], incremental=incremental
            )
            self.system.resources[resource] = resource_obj  # Save back to self.system

    def create_recap2_ELCC_points(self, ELCC_points_matrix: pd.DataFrame, ELCC_surface_name: str):
        """
        Iteratively creates RECAP 2.0 input folders for portfolio vectors defined in rows of ELCC_points_matrix

        Args:
            ELCC_points_matrix: DataFrame with resource name columns and rows of portfolio nameplate vectors
        """

        # Create base case
        self.create_recap2_input_files()

        # Get incremental flags for surface points and remove incremental flag column
        incremental = ELCC_points_matrix["incremental"]
        ELCC_points_matrix.drop(columns=["incremental"], inplace=True)

        # Loop through ELCC surface points and calculate portfolio capacity shortfall
        for k, portfolio_vector in ELCC_points_matrix.iterrows():
            assert ELCC_surface_name in [
                "marginal_ELCC",
                "incremental_last_in_ELCC",
                "decremental_last_in_ELCC",
                "custom_ELCC_surface",
            ]

            # Logic to determine case name in RECAP 2.0
            if ELCC_surface_name == "marginal_ELCC":
                # Get marginal resource name
                marginal_resource_name = portfolio_vector.index[portfolio_vector > 0][0]
                marginal_step_size = int(portfolio_vector.loc[portfolio_vector > 0][0])
                case_name = f"{self.case_name}_marginal_{marginal_step_size}MW_{marginal_resource_name}"
            elif ELCC_surface_name == "incremental_last_in_ELCC":
                # Get incremental last-in resource name
                incremental_name = portfolio_vector.index[portfolio_vector.isna()][0]
                case_name = f"{self.case_name}_incremental_LI_{incremental_name}"
            elif ELCC_surface_name == "decremental_last_in_ELCC":
                # Get decremental last-in resource name
                decremental_name = portfolio_vector.index[portfolio_vector == 0][0]
                case_name = f"{self.case_name}_decremental_LI_{decremental_name}"
            elif ELCC_surface_name == "custom_ELCC_surface":
                # Make case name = point # on surface and write portfolio vector to input folder
                case_name = f"{self.case_name}_ELCC_surface_point_{k}"

            # Create copy of base case for ELCC calculation
            ELCC_case = self.create_copy(case_name=case_name)
            # Re-scale resource profiles and resource capacity in ELCC case
            ELCC_case.rescale_portfolio(portfolio_vector, incremental=incremental.loc[k])
            # Calculate ELCC case perfect capacity shortfall
            ELCC_case.create_recap2_input_files()
            # Save portfolio vector and incremental flag
            portfolio_vector["incremental flag"] = int(incremental.loc[k])
            portfolio_vector.to_csv(ELCC_case.dir_str.recap2_input_dir / "portfolio_vector.csv")
            # Add case name to cases_to_run
            self.cases_to_run.append(ELCC_case.case_name)

    #### MASTER FUNCTION FOR RECAP 2.0 INPUT CONVERSION ####

    def create_recap2_input_files(self):
        """Writes all RECAP 2.0 input files"""

        # Write out input files
        self.write_case_settings()
        self.write_load_profile_settings()
        self.write_generator_module()
        self.write_hydro()
        self.write_vg_group_settings()
        self.write_vg_profile_settings()
        self.write_storage()
        self.write_dynamic_storage()
        self.write_demand_response()
        self.write_flexible_load()
        self.write_imports()
        self.write_outage_distributions()

        # Add case name to cases_to_run
        self.cases_to_run.append(self.case_name)

    #### HELPER FUNCTIONS ####

    def map_inputs(self, obj, df_map):
        df_obj_values = pd.DataFrame(columns=df_map.index)  # Store values for object
        df_map = df_map.fillna(False)  # Replace NaN with empty string for logic below
        for column in df_map.index:
            # Go through each row of map file
            attr, helper_function, default_value = df_map.loc[column]
            if attr:
                value = getattr(obj, attr)
                if type(value) in ts.Timeseries.__subclasses__():
                    value = value.slice_by_year(self.model_year)
            elif helper_function:
                value = getattr(self, helper_function)(obj)
            elif default_value is not None:
                if default_value == "blank":
                    default_value = ""
                value = default_value
            else:
                raise ValueError(
                    f"No attribute, helper function, or default value specified for column {column} in mapping file"
                )
            df_obj_values.loc[0, column] = value
        return df_obj_values

    def get_target_prm(self, case_settings):
        is_calculating_elcc = np.array(
            [
                case_settings.calculate_perfect_capacity_shortfall,
                case_settings.calculate_portfolio_ELCC,
                case_settings.calculate_marginal_ELCC,
                case_settings.calculate_incremental_last_in_ELCC,
                case_settings.calculate_decremental_last_in_ELCC,
                case_settings.calculate_ELCC_surface,
            ]
        ).any()

        if is_calculating_elcc:
            target_prm = "calculate_capacity_short_only"
        elif case_settings.calculate_reliability:
            target_prm = "skip_prm_calculations"
        else:
            logger.warning(
                "No run setting selected for RECAP 3.0 case; setting RECAP 2.0 target PRM to 'skip_prm_calculations.'"
            )
            target_prm = "skip_prm_calculations"

        # calculate_reliability and is_calculating_elcc may not both be True --> ERROR
        # TODO: if both are specified, it's probably OK to just use "calculate_capacity_shortfall" since RECAP 2.0
        #  will automatically calculate the "untuned" system results (which you get from calculate_reliability)
        if case_settings.calculate_reliability and is_calculating_elcc:
            raise ValueError(
                "Calculate_reliability and perfect capacity shortfall calculations are both True."
                "You must choose one to use RECAP 2.0 wrapper."
            )

        return target_prm

    def get_target_metric(self, case_settings):
        return self.case_settings.target_metric.value.lower()

    @property
    def gross_load(self) -> pd.Series:
        """Scales load profiles to the desired model year and calculate the aggregate zonal gross load"""
        if self._gross_load is None:
            for load in self.system.loads.values():
                if self.case_settings.zone_to_analyze in load.zones:
                    load.forecast_load((self.model_year, self.model_year))
            self._gross_load = self.system.zones[self.case_settings.zone_to_analyze].get_aggregated_load_profile(
                self.model_year
            )
        return self._gross_load

    def get_operating_reserves_up_mw(self, case_settings):
        # Actually ignore case settings, get max of sum of upward operating reserves
        up_reserves = [
            reserve
            for reserve in self.system.reserves.values()
            if reserve.direction == "up" and self.case_settings.zone_to_analyze in reserve.zones
        ]
        if len(up_reserves) == 0:
            reserves = pd.Series(index=self.gross_load.index, data=[0.0] * len(self.gross_load))
        else:
            for reserve in up_reserves:
                if not reserve.requirement and self.case_settings.zone_to_analyze in reserve.zones.keys():
                    requirement_fraction_of_gross_load = reserve.zones[
                        self.case_settings.zone_to_analyze
                    ].requirement_fraction_of_gross_load.data[0]
                    requirement = self.gross_load.copy() * requirement_fraction_of_gross_load
                    reserve.requirement = ts.NumericTimeseries(
                        name="requirement",
                        default_freq="H",
                        up_method="interpolate",
                        down_method="first",
                        data=requirement,
                    )
            reserves = pd.DataFrame(pd.concat([reserve.requirement.data for reserve in up_reserves], axis=1)).sum(
                axis=1
            )
        max_reserves = reserves.max()
        return max_reserves

    def get_load_scale_type(self, load_component):
        # Get load scale type for load component
        if load_component.scale_by_energy and not load_component.scale_by_capacity:
            load_scale_type = "energy"
        elif not load_component.scale_by_energy and load_component.scale_by_capacity:
            load_scale_type = "peak"
        elif load_component.scale_by_energy and load_component.scale_by_capacity:
            load_scale_type = "energy_and_peak"
        else:
            raise ValueError(
                f"Load scale type not specified for load component {load_component.name}."
                "Please specify 'scale_by_energy' or 'scale_by_capacity'"
            )
        return load_scale_type

    def get_zone(self, obj):  # resource, load
        # Access "_instance_to" name of LoadToZone or ResourceToZone linkage
        inst = list(obj.zones.keys())[0]
        zone = obj.zones[inst].instance_to.name
        return zone

    def get_resource_group(self, resource):
        # Access "_instance_to" name of ResourceToResourceGroup linkage
        inst = list(resource.resource_groups.keys())[0]
        resource_group = resource.resource_groups[inst].instance_to.name
        return resource_group

    def get_dispatch_window(self, dr_resource):
        if not dr_resource.max_monthly_calls:
            return "annual"
        else:
            return "monthly"

    #### RECAP 2.0 INPUT CONVERSION FUNCTIONS ####

    def write_case_settings(self):
        # Read in mapping file
        df_map = pd.read_csv(
            self.dir_str.code_recap_dir / "recap2_wrapper_inputs_mapping" / "calculation_settings_mapping.csv",
            index_col=[0],
        )
        # Convert inputs
        df_values = self.map_inputs(self.case_settings, df_map)

        # Note that each field has a row rather than a column in case_settings
        # transpose and rename to put into case_settings format
        df_case_settings = df_values.T.reset_index()
        df_case_settings.columns = ["setting", "value"]
        df_case_settings.to_csv(self.dir_str.recap2_input_dir / "calculation_settings.csv", index=False)

    def write_load_profile_settings(self):
        # Read in load_profile_settings input mapping file
        df_map = pd.read_csv(
            self.dir_str.code_recap_dir / "recap2_wrapper_inputs_mapping" / "load_profile_settings_inputs_mapping.csv",
            index_col=[0],
        )
        # Create vg_group_settings input file
        df_load_profile_settings = pd.DataFrame(columns=df_map.index)
        # Get all load components for study zone
        load_components = [
            load_component
            for load_component in self.system.loads.values()
            if self.get_zone(load_component) == self.case_settings.zone_to_analyze
        ]
        # Loop through each load component
        for load_component in load_components:
            df_values = self.map_inputs(load_component, df_map)
            df_load_profile_settings = pd.concat([df_load_profile_settings, df_values])
        # Write out load_profile_settings.csv input file
        df_load_profile_settings.to_csv(self.dir_str.recap2_input_dir / "load_profile_settings.csv", index=False)

        # Extra: check to ensure that load_component_shapes.csv exists in common_inputs folder,
        # and that all load components are represented by a column of load_component_shapes.csv
        # If does not exist, attempt to write load profiles to common_inputs folder
        load_component_shapes_filepath = self.dir_str.recap2_common_inputs_dir / "load_component_shapes.csv"
        if load_component_shapes_filepath.exists():
            df_load_component_shapes = pd.read_csv(load_component_shapes_filepath, index_col=[0])
        else:
            df_load_component_shapes = pd.DataFrame([])
        change = False  # Whether to re-write load_component_shapes.csv file
        for load_component in load_components:
            if load_component.name not in df_load_component_shapes.columns:
                logger.warning(
                    f"Missing load profile for load component '{load_component.name}' in 'load_component_shapes.csv'"
                    f"in RECAP 2.0 common_inputs folder. Attempting to write profile for load component..."
                )
                profile = load_component.profile.data
                df_load_component_shapes[load_component.name] = profile
                df_load_component_shapes.index.name = "date"
                change = True
        if change:
            df_load_component_shapes.to_csv(load_component_shapes_filepath)

    def write_generator_module(self):
        # Read in generator_module input mapping file
        df_map = pd.read_csv(
            self.dir_str.code_recap_dir / "recap2_wrapper_inputs_mapping" / "generator_module_inputs_mapping.csv",
            index_col=[0],
        )
        # Create vg_group_settings input file
        df_generator_module = pd.DataFrame(columns=df_map.index)
        # Get thermal or firm resource groups
        firm_resource_groups = [
            group
            for group in self.system.resource_groups.values()
            if group.category in [ResourceGroupCategory.THERMAL, ResourceGroupCategory.FIRM]
        ]
        # Loop through each firm resource
        for firm_group in firm_resource_groups:
            for resource in firm_group.resource_dict.values():
                df_values = self.map_inputs(resource, df_map)
                df_generator_module = pd.concat([df_generator_module, df_values])
        # Write out vg_group_settings.csv input file
        df_generator_module.to_csv(self.dir_str.recap2_input_dir / "generator_module.csv", index=False)

    def write_hydro(self):
        # Read in hydro input mapping file (not used for hydro - just used to store extra column names)
        df_map = pd.read_csv(
            self.dir_str.code_recap_dir / "recap2_wrapper_inputs_mapping" / "hydro_inputs_mapping.csv",
            index_col=[0],
        )
        # Create hydro input file
        df_hydro = pd.DataFrame(columns=df_map.index)
        # Get hydro resource groups
        hydro_resource_groups = [
            group for group in self.system.resource_groups.values() if group.category == ResourceGroupCategory.HYDRO
        ]
        if hydro_resource_groups:
            logger.warning(
                "Warning: RECAP 2.0 wrapper makes rigid assumptions about hydro operations. Hydro budgets and dispatch "
                "windows are assumed to be monthly. RECAP 2.0 wrapper infers monthly values for Pmin and Pmax from "
                "'power_output_min' and 'power_output_max'. RECAP 2.0 wrapper ignores daily and"
                "annual energy budgets, and time-varying Pmin / Pmax profiles."
            )
        # Loop through each firm resource
        for hydro_group in hydro_resource_groups:
            for resource in hydro_group.resource_dict.values():
                # Check that resource has monthly energy budget
                assert hasattr(
                    resource, "energy_budget_monthly"
                ), "Hydro monthly budgets required for hydro resources in RECAP 2.0 wrapper."
                df_values = pd.DataFrame(columns=df_map.index)
                # Get hydro monthly budget profile
                energy_budget_monthly = resource.energy_budget_monthly.data
                # Infer hydro years and months for resource
                hydro_years = energy_budget_monthly.index.year
                hydro_months = energy_budget_monthly.index.month
                # Get hydro Pmin and Pmax profiles
                hydro_pmin = resource.power_output_min.data * resource.capacity_planned.slice_by_year(self.model_year)
                hydro_pmax = resource.power_output_max.data * resource.capacity_planned.slice_by_year(self.model_year)
                # Get monthly hydro pmin and pmax
                hydro_pmin = hydro_pmin.groupby([hydro_pmin.index.year, hydro_pmin.index.month])
                hydro_pmax = hydro_pmax.groupby([hydro_pmax.index.year, hydro_pmax.index.month])
                # Adjust pmin and pmax so that monthly budgets are feasible
                hours_per_month = {group: len(hydro_pmax.get_group(group)) for group in hydro_pmax.groups}
                hydro_pmin = hydro_pmin.min()
                hydro_pmax = hydro_pmax.max()
                for year, month in hours_per_month.keys():
                    timestamp = pd.Timestamp(year=year, month=month, day=1)
                    budget = energy_budget_monthly.loc[timestamp]
                    if hydro_pmin.loc[(year, month)] > budget / hours_per_month[(year, month)]:
                        # logger.debug(f"Pmin ({year}, {month}): {hydro_pmin.loc[(year, month)]} -> {budget/hours_per_month[(year, month)]}")
                        hydro_pmin.loc[(year, month)] = budget / hours_per_month[(year, month)]
                    elif hydro_pmax.loc[(year, month)] < budget / hours_per_month[(year, month)]:
                        # logger.debug(f"Pmax ({year}, {month}): {hydro_pmax.loc[(year, month)]} -> {budget/hours_per_month[(year, month)]}")
                        hydro_pmax.loc[(year, month)] = budget / hours_per_month[(year, month)]
                # Get data with monthly frequency
                df_values["name"] = [resource.name for i in hydro_years]
                df_values["zone"] = [self.get_zone(resource) for i in hydro_years]
                df_values["hydro_year"] = hydro_years
                df_values["month"] = hydro_months
                df_values["pmin_mw"] = hydro_pmin.values
                df_values["pmax_mw"] = hydro_pmax.values
                df_values["monthly_budget_mwh"] = energy_budget_monthly.values
                # Concatenate with other resource data
                df_hydro = pd.concat([df_hydro, df_values])
        df_hydro.to_csv(self.dir_str.recap2_input_dir / "hydro.csv", index=False)

    def write_vg_group_settings(self):
        # Read in vg_group_settings input mapping file
        df_map = pd.read_csv(
            self.dir_str.code_recap_dir / "recap2_wrapper_inputs_mapping" / "vg_group_settings_inputs_mapping.csv",
            index_col=[0],
        )
        # Create vg_group_settings input file
        df_vg_group_settings = pd.DataFrame(columns=df_map.index)
        # Get variable resource groups
        variable_resource_groups = [
            group for group in self.system.resource_groups.values() if group.category == ResourceGroupCategory.VARIABLE
        ]
        # Loop through each variable resource group
        for variable_group in variable_resource_groups:
            df_values = self.map_inputs(variable_group, df_map)
            df_vg_group_settings = pd.concat([df_vg_group_settings, df_values])
        # Write out vg_group_settings.csv input file
        df_vg_group_settings.to_csv(self.dir_str.recap2_input_dir / "vg_group_settings.csv", index=False)

    def write_vg_profile_settings(self):
        # Read in vg_group_settings input mapping file
        df_map = pd.read_csv(
            self.dir_str.code_recap_dir / "recap2_wrapper_inputs_mapping" / "vg_profile_settings_inputs_mapping.csv",
            index_col=[0],
        )
        # Create vg_group_settings input file
        df_vg_profile_settings = pd.DataFrame(columns=df_map.index)
        # Get variable resource groups
        variable_resource_groups = [
            group for group in self.system.resource_groups.values() if group.category == ResourceGroupCategory.VARIABLE
        ]
        # Loop through each variable resource
        for variable_group in variable_resource_groups:
            for resource in variable_group.resource_dict.values():
                df_values = self.map_inputs(resource, df_map)
                df_vg_profile_settings = pd.concat([df_vg_profile_settings, df_values])
                # Extra: check to ensure that profiles for all variable resources exist in common_inputs folder
                # If profile does not exist, attempt to write profile for resource to common_inputs folder
                if not (self.dir_str.recap2_common_inputs_dir / f"{resource.name}.csv").exists():
                    logger.warning(
                        f"Missing profile for resource '{resource.name}' in RECAP 2.0 common_inputs folder."
                        "Attempting to write profile for resource to common_inputs folder..."
                    )
                    profile = resource.power_output_max.data.reset_index()
                    profile.columns = ["date", "mw"]
                    profile.to_csv(self.dir_str.recap2_common_inputs_dir / f"{resource.name}.csv", index=False)
        # Write out vg_profile_settings.csv input file
        df_vg_profile_settings.to_csv(self.dir_str.recap2_input_dir / "vg_profile_settings.csv", index=False)

    def write_storage(self):
        # Read in storage input mapping file
        df_map = pd.read_csv(
            self.dir_str.code_recap_dir / "recap2_wrapper_inputs_mapping" / "storage_inputs_mapping.csv", index_col=[0]
        )
        # Create storage input file
        df_storage = pd.DataFrame(columns=df_map.index)
        # Get storage resource groups
        storage_resource_groups = [
            group for group in self.system.resource_groups.values() if group.category == ResourceGroupCategory.STORAGE
        ]
        # Loop through each storage resource
        for storage_group in storage_resource_groups:
            for resource in storage_group.resource_dict.values():
                df_values = self.map_inputs(resource, df_map)
                df_storage = pd.concat([df_storage, df_values])
        # Write out storage.csv input file
        df_storage.to_csv(self.dir_str.recap2_input_dir / "storage.csv", index=False)

    def write_dynamic_storage(self):
        # Read in dynamic storage input mapping file
        df_map = pd.read_csv(
            self.dir_str.code_recap_dir / "recap2_wrapper_inputs_mapping" / "dynamic_storage_inputs_mapping.csv",
            index_col=[0],
        )
        # Create storage input file
        df_dynamic_storage = pd.DataFrame(columns=df_map.index)
        # Get storage resource groups
        hybrid_storage_resource_groups = [
            group
            for group in self.system.resource_groups.values()
            if group.category == ResourceGroupCategory.HYBRID_STORAGE
        ]
        # Loop through each hybrid storage resource
        for hybrid_storage_group in hybrid_storage_resource_groups:
            for resource in hybrid_storage_group.resource_dict.values():
                df_values = self.map_inputs(resource, df_map)
                df_dynamic_storage = pd.concat([df_dynamic_storage, df_values])
        # Write out storage.csv input file
        df_dynamic_storage.to_csv(self.dir_str.recap2_input_dir / "dynamic_storage.csv", index=False)

    def write_demand_response(self):
        # NOTE: SHAPED DR NOT CURRENTLY SUPPORTED (only dispatched DR)
        df_map = pd.read_csv(
            self.dir_str.code_recap_dir / "recap2_wrapper_inputs_mapping" / "demand_response_inputs_mapping.csv",
            index_col=[0],
        )
        # Create demand_response input file
        df_demand_response = pd.DataFrame(columns=df_map.index)
        # Get demand response resource groups
        dr_resource_groups = [
            group
            for group in self.system.resource_groups.values()
            if group.category == ResourceGroupCategory.DEMAND_RESPONSE
        ]
        # Loop through each demand response resource
        for dr_group in dr_resource_groups:
            for resource in dr_group.resource_dict.values():
                df_values = self.map_inputs(resource, df_map)
                df_demand_response = pd.concat([df_demand_response, df_values])
        # Write out demand_response.csv input file
        df_demand_response.to_csv(self.dir_str.recap2_input_dir / "demand_response.csv", index=False)

        # Extra: create empty "shaped_dr_component_shapes.csv" in common_inputs folder
        shaped_dr_component_shapes_filepath = self.dir_str.recap2_common_inputs_dir / "shaped_dr_component_shapes.csv"
        if not shaped_dr_component_shapes_filepath.exists():
            df_shaped_dr_component_shapes = pd.DataFrame(
                index=pd.date_range(start="2010-01-01", end="2011-01-01", freq="H")
            )
            df_shaped_dr_component_shapes.index.name = "date"
            df_shaped_dr_component_shapes.to_csv(shaped_dr_component_shapes_filepath)

    def write_flexible_load(self):
        df_map = pd.read_csv(
            self.dir_str.code_recap_dir / "recap2_wrapper_inputs_mapping" / "flexible_load_inputs_mapping.csv",
            index_col=[0],
        )
        # Create flexible_load input file
        df_flex_load = pd.DataFrame(columns=df_map.index)
        # Get flex load resource groups
        flex_load_resource_groups = [
            group
            for group in self.system.resource_groups.values()
            if group.category == ResourceGroupCategory.FLEXIBLE_LOAD
        ]
        # Loop through each demand response resource
        for flex_load_group in flex_load_resource_groups:
            for resource in flex_load_group.resource_dict.values():
                df_values = self.map_inputs(resource, df_map)
                df_flex_load = pd.concat([df_flex_load, df_values])
        # Write out flexible_load.csv input file
        df_flex_load.to_csv(self.dir_str.recap2_input_dir / "flexible_load.csv", index=False)

    def write_imports(self):
        # THIS CODE CURRENTLY OUTPUTS A BLANK CSV, but is built to support the translation of tx limits
        # if they are built into recap 3.0
        # CURRENTLY NO IMPORTS LIMITS BUILT INTO RECAP 3.0
        # todo: test code when there are tx paths
        df_map = pd.read_csv(
            self.dir_str.code_recap_dir / "recap2_wrapper_inputs_mapping" / "imports_inputs_mapping.csv", index_col=[0]
        )
        # Create imports input file
        df_imports = pd.DataFrame(columns=df_map.index)
        # Loop through each tx path
        for tx_path in self.system.tx_paths:
            df_values = self.map_inputs(tx_path, df_map)
            df_imports = pd.concat([df_imports, df_values])
        # Write out imports.csv input file
        df_imports.to_csv(self.dir_str.recap2_input_dir / "imports.csv", index=False)

    def write_outage_distributions(self):
        # Stand-alone function; copies default outage_distributions.csv input file into inputs directory
        shutil.copy(
            self.dir_str.code_recap_dir
            / "recap2_wrapper_inputs_mapping"
            / "outage_distributions_input_files"
            / "outage_distributions.csv",
            self.dir_str.recap2_input_dir / "outage_distributions.csv",
        )
        logger.warning(
            "RECAP 2.0 wrapper does not use NMT OutageDistribution components or linkages."
            "All outages assumed to be full outages (no partial derates)."
        )

    #### RECAP 2.0 CODE EXECUTION FUNCTION ####

    def execute_recap2_code(self):
        # Write out cases_to_run
        df_cases_to_run = pd.DataFrame(data=self.cases_to_run).drop_duplicates()
        df_cases_to_run.to_csv(self.dir_str.recap2_code_dir / "cases_to_run.csv", header=False, index=False)

        # Execute RECAP 2.0 code (NOT WORKING)
        # Environment-related issues encountered with commented-out code below
        # For now manually execute RECAP 2.0 code by either
        # 1. Update cases_to_run.csv in recap 2.0 folder and execute
        # runbatch_parallel.py with e3recap environment OR
        # 2. Use bash script run_recap2_wrapper.bat

        # 1. recommended

        # original code w issue (CG code)
        # os.chdir(self.dir_str.recap2_code_dir)
        # process = subprocess.Popen(f"conda run -n e3recap python runbatch.py".split())  # nosec

        # attempt 2 code not working - attempt #2 (SK and KW code)
        # current_path = Path(subprocess.run("cd", shell=True, capture_output=True).stdout.decode().replace("\r\n", ""))
        # current_path = Path("\\".join(current_path.parts[1:]))
        # process = subprocess.run("conda activate e3recap && python runbatch.py", shell=True, capture_output=True)
        # output, error = process.communicate()
        # https://unix.stackexchange.com/questions/622383/subprocess-activate-conda-environment-from-python-script/690330#690330
