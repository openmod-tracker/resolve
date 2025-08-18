import contextlib
import copy
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from new_modeling_toolkit.core import stream
from new_modeling_toolkit.core.temporal.new_temporal import NewTemporalSettings
from new_modeling_toolkit.core.utils.core_utils import cantor_pairing_function
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.core.utils.util import DirStructure
from new_modeling_toolkit.recap.dispatch_model import DispatchModel
from new_modeling_toolkit.recap.recap_case_settings import RecapCaseSettings
from new_modeling_toolkit.recap.recap_case_settings import ResourceGrouping
from new_modeling_toolkit.system import System
from new_modeling_toolkit.system.electric.resource_group import ResourceGroupCategory
from new_modeling_toolkit.system.electric.resources import FlexLoadResource
from new_modeling_toolkit.system.electric.resources import HydroResource
from new_modeling_toolkit.system.electric.resources import ShedDrResource

# Define constants
_PERIOD_START_BUFFER = pd.Timedelta(hours=-168)
_PERIOD_END_BUFFER = pd.Timedelta(hours=168)
_PERIOD_MIN_SEPARATION_WINDOW = pd.Timedelta(hours=168 * 2)
_PERIOD_CONSECUTIVE_DELTA = pd.Timedelta(hours=24)


def _identify_positive_net_load_windows(net_load: pd.Series) -> pd.DatetimeIndex:
    """Identifies timestamps of positive net load periods for use in optimized dispatch.

    The method adds leading and lagging buffer periods and combines periods within a certain distance of one another.

    Args:
        net_load: the series of net load with a datetime index

    Returns:
        a datetime index including timestamps with positive net load and buffer periods
    """
    # Get list of timesteps, dates, and subproblem span
    timesteps = net_load.index
    dates = np.unique(timesteps.date)
    # Get positive net load dates
    pos_timesteps = timesteps[net_load.values > 0]
    pos_dates = np.unique(pos_timesteps.date)

    # Loop through pos_dates to get in dates
    dates_to_include = pd.DatetimeIndex([])  # Initialize
    # Add +/- 1 week buffer
    for date in pos_dates:
        start, end = max(date + _PERIOD_START_BUFFER, dates.min()), min(date + _PERIOD_END_BUFFER, dates.max())
        window_dates = pd.date_range(start, end, freq="D")
        dates_to_include = dates_to_include.union(window_dates)

    # Consolidate windows < 2 weeks apart
    dates_to_include = dates_to_include.sort_values()
    i = 0
    while i < len(dates_to_include) - 1:
        start, end = dates_to_include[i], dates_to_include[i + 1]
        time_delta = end - start
        if (time_delta > _PERIOD_CONSECUTIVE_DELTA) and (
            time_delta < _PERIOD_MIN_SEPARATION_WINDOW
        ):  # non-consecutive dates < 2 weeks apart
            window_dates = pd.date_range(start, end, freq="D")
            dates_to_include = dates_to_include.union(window_dates)
        i += 1

    # Use modular arithmetic to consolidate first and last periods if < 2 weeks apart in "loop"
    if len(dates_to_include) > 0:
        start, end = dates_to_include[-1], dates_to_include[0]
        time_delta = end - start
        span = dates.max() - dates.min()
        time_delta = pd.Timedelta(seconds=np.mod(time_delta.total_seconds(), span.total_seconds()))
        if time_delta < _PERIOD_MIN_SEPARATION_WINDOW:
            window_dates = pd.date_range(start, dates.max(), freq="D")
            dates_to_include = dates_to_include.union(window_dates)
            window_dates = pd.date_range(dates.min(), end, freq="D")
            dates_to_include = dates_to_include.union(window_dates)

    return dates_to_include


class MonteCarloDraw:
    subclasses_dispatch_order: Optional[List[str]] = None

    def __init__(
        self,
        dir_str: DirStructure,
        case_settings: RecapCaseSettings,
        system: System,
        random_seed: int,
    ):
        self.dir_str = dir_str
        self.case_settings = case_settings
        self.system = system
        self.random_seed = random_seed

        # Set name
        self.name = f"MC_draw_{self.random_seed}"

        # Get model year from case settings
        self.model_year = self.case_settings.analysis_year

        # Get day draw directory from directory structure
        self.day_draw_dir = self.dir_str.day_draw_dir

        # Initialize gross load and reserves private attributes
        self._gross_load = None
        self._reserves = None

        # Initialize upsample_called and simulate_outages_called attributes
        self.upsample_called = False
        self.simulate_outages_called = False

    @property
    def gross_load(self) -> pd.Series:
        """Returns aggregated gross load for study zone"""
        if self._gross_load is None:
            self._gross_load = self.system.zones[self.case_settings.zone_to_analyze].get_aggregated_load_profile(
                self.model_year
            )
        return self._gross_load

    @property
    def reserves(self):
        """Returns aggregated upward reserves for study zone"""
        if self._reserves is None:
            self._reserves = self.system.zones[self.case_settings.zone_to_analyze].get_aggregated_up_reserves(
                self.model_year
            )
        return self._reserves

    @timer
    def create_day_draw_map_for_resource_group(self):
        """
        Create day draw map for weather e variable resource groups.
        """
        logger.info("Creating day draw map for weather dependent resource groups")
        if self.case_settings.draw_settings != "random":
            day_draw_filepath = (
                self.dir_str.recap_settings_dir
                / self.case_settings.draw_settings
                / "day_draw_map"
                / f"day_draw_map_{self.random_seed}.csv"
            )
            logger.info(f"Reading in existing MC draw day draw map from {day_draw_filepath}")
            assert (
                day_draw_filepath.exists()
            ), f"cannot find existing day draw map {str(day_draw_filepath)}, please double check it exists or change the draw setting to random"
            # Load the day draw map from another scenario and convert it to the required dictionary format
            df_day_draw = pd.read_csv(day_draw_filepath, infer_datetime_format=True, index_col=[0])
            df_day_draw.index = pd.to_datetime(df_day_draw.index)
            for column in df_day_draw.columns:
                df_day_draw[column] = pd.to_datetime(df_day_draw[column])
            df_day_draw.rename(columns=lambda x: x.replace("_date", ""), inplace=True)
            for name, group in self.system.resource_groups.items():
                if (
                    group.category == ResourceGroupCategory.VARIABLE
                    or group.category == ResourceGroupCategory.HYBRID_VARIABLE
                ) and group.resource_dict:
                    group.day_draw_map = df_day_draw[name]
        else:
            logger.info("Generating random day draw map")
            day_draws_by_group = pd.DataFrame()
            for name, group in self.system.resource_groups.items():
                if (
                    group.category == ResourceGroupCategory.VARIABLE
                    or group.category == ResourceGroupCategory.HYBRID_VARIABLE
                ) and group.resource_dict:
                    # create random day draw map for each group
                    group.draw_days_by_group(
                        self.gross_load, self.model_year, self.case_settings.day_window_variable_draws, self.random_seed
                    )
                    day_draws_by_group = pd.concat([day_draws_by_group, group.day_draw_map], axis=1)
                else:
                    logger.warning(f"SKIPPING RESOURCE GROUP {name} - check that resources are in group")
            # Save the generated day draw map
            day_draws_by_group.add_suffix("_date").to_csv(
                self.day_draw_dir / ("day_draw_map_{}.csv".format(self.random_seed))
            )

    def net_load(self, heuristic_net_load_subclasses: list = []):
        """
        Calculates net load time series for system (load - thermal - renewables)
        """
        # TODO: make this LESS dynamic and simpler - net load should really just be load - thermal - renewables
        # Where is this even used? Within DispatchModel and compression... leave for now, but can eventually remove?

        # Define list of resource sub-classes whose heuristic dispatch will be included in net load calculation
        # (i.e. heuristic dispatch from sub-classes will be subtracted from net load)
        heuristic_net_load_subclasses = copy.deepcopy(heuristic_net_load_subclasses)
        heuristic_net_load_subclasses += ResourceGrouping.NO_ELRS.resource_subclasses
        heuristic_net_load_subclasses = list(set(heuristic_net_load_subclasses))  # Get rid of duplicates

        # Calculate net load (subtracting heuristic dispatch from resources included in subclasses)
        net_load = self.gross_load.copy()
        for subclass in heuristic_net_load_subclasses:
            resources = list(getattr(self.system, subclass).values())
            for resource in resources:
                net_load -= resource.heuristic_provide_power_mw

        return net_load

    @timer
    def upsample(self, deterministic_upsampling=False, probabilistic_upsampling=True):
        """
        Upsample data for load, reserves, resources in System
        By default, only upsamples resources with randomized data (probabilistic upsampling)
        If specified, will upsample resources with non-randomized data (deterministic upsampling)

        Args:
            deterministic_upsampling: bool - whether to upsample resources with non-randomized data
            probabilistic_upsampling: bool - whether to upsample resources with randomized data

        """
        logger.debug("Upsampling data for system")

        if self.upsample_called:
            logger.warning(f"MonteCarloDraw.upsample() already called for {self.name}")

        # Upsample resources with non-randomized data (by default, this is handled in RecapCase.setup_monte_carlo_draws)
        if deterministic_upsampling:
            for subclass in ResourceGrouping.DETERMINISTIC_UPSAMPLING.resource_subclasses:
                resources = getattr(self.system, subclass).values()
                for resource in resources:
                    resource.upsample(load_calendar=self.gross_load.index.copy())

        # Upsample resources with randomized data with Monte Carlo draw random seed
        if probabilistic_upsampling:
            # Create day draw maps for renewable resources
            self.create_day_draw_map_for_resource_group()
            for subclass in ResourceGrouping.PROBABILISTIC_UPSAMPLING.resource_subclasses:
                resources = getattr(self.system, subclass).values()
                for resource in resources:
                    resource.upsample(load_calendar=self.gross_load.index.copy(), random_seed=self.random_seed)

        self.upsample_called = True

    @timer
    def simulate_outages(self):
        """
        Simulate outages for all resources in System
        """
        logger.debug("Simulating outages for resources in system")
        if not self.upsample_called:
            logger.warning("MonteCarloDraw.upsample() not called before MonteCarloDraw.simulate_outages()")
        if self.simulate_outages_called:
            logger.warning(f"MonteCarloDraw.simulate_outages() already called for {self.name}")
        for resource in self.system.resources.values():
            resource.simulate_outages(
                model_year=self.model_year, random_seed=cantor_pairing_function(resource.random_seed, self.random_seed)
            )

        self.simulate_outages_called = True

    def rescale(self, portfolio_vector, incremental=False):
        """
        Re-scale resource capacities / attributes to capacities in portfolio_vector
        Args:
            portfolio_vector: pd.Series with resource (incremental) nameplate capacities
            incremental: boolean indicator for whether portfolio vector is incremental or absolute
        """
        for resource_name in portfolio_vector.index:
            resource = self.system.resources[resource_name]
            resource.rescale(
                model_year=self.model_year, capacity=portfolio_vector.loc[resource_name], incremental=incremental
            )

    def split(self, max_num_years: int) -> dict[int, "MonteCarloDraw"]:
        """
        When splitting each MonteCarloDraw into sub-problems, do so via a new MonteCarloDraw.split() method, which
        returns multiple copies of the MonteCarloDraw instance whose System objects contain only N-year long slices of
        the original System’s load/reserves/upsampled resource profiles
        """
        # Define groups for all time series indices
        hourly_index = self.gross_load.index
        hourly_index_grouped = hourly_index.groupby((hourly_index.year - min(hourly_index.year)) // max_num_years)

        daily_index = self.gross_load.index.to_period(freq="D").to_timestamp().unique()
        daily_index_grouped = daily_index.groupby((daily_index.year - min(daily_index.year)) // max_num_years)

        monthly_index = self.gross_load.index.to_period(freq="M").to_timestamp().unique()
        monthly_index_grouped = monthly_index.groupby((monthly_index.year - min(monthly_index.year)) // max_num_years)

        annual_index = self.gross_load.index.to_period(freq="Y").to_timestamp().unique()
        annual_index_grouped = annual_index.groupby((annual_index.year - min(annual_index.year)) // max_num_years)

        # Now, loop through each group, getting the set of indices (hourly, daily, monthly, annual) corresponding
        # to the group, and create a copy of system within which we will simply slice out the timestamps for group
        # and make new MonteCarloDraw instance for sliced system

        # Loop through groups and make new MC draws
        split_monte_carlo_draws = {}
        for group in hourly_index_grouped.keys():
            # Get subset of hourly/daily/monthly/annual indices
            hourly_inds = hourly_index_grouped[group]
            daily_inds = daily_index_grouped[group]
            monthly_inds = monthly_index_grouped[group]
            annual_inds = annual_index_grouped[group]

            # Create deep copy of System
            system = copy.deepcopy(self.system)

            ### Slice out timestamps of relevant timeseries within System copy ###

            # Loads
            for load in system.loads.values():
                load.profile.data = load.profile.data.loc[hourly_inds]
                for model_year_profile in load.model_year_profiles.values():
                    model_year_profile.data = model_year_profile.data.loc[hourly_inds]

            # Reserves
            for reserve in system.reserves.values():
                if reserve.requirement:
                    reserve.requirement.data = reserve.requirement.data.loc[hourly_inds]

            # Resources
            for resource in system.resources.values():
                # Find all timeseries attributes that we will split (weather year-indexed attributes + outages)
                timeseries_attrs = [
                    attr
                    for attr in resource.timeseries_attrs
                    if (
                        resource.model_fields[attr].json_schema_extra is not None
                        and "weather_year" in resource.model_fields[attr].json_schema_extra.keys()
                        and resource.model_fields[attr].json_schema_extra["weather_year"]
                    )
                ]

                # Loop through attributes and slice
                for attr in timeseries_attrs:
                    # Get default frequency of attribute
                    default_freq = resource.model_fields[attr].json_schema_extra["default_freq"]
                    # Get corresponding indices for frequency
                    if default_freq == "H":
                        inds = hourly_inds
                    elif default_freq == "D":
                        inds = daily_inds
                    elif default_freq in ["M", "MS"]:
                        inds = monthly_inds
                    elif default_freq in ["Y", "YS"]:
                        inds = annual_inds
                    else:
                        raise AttributeError
                    # Get / slice / set attribute
                    temp = getattr(resource, attr)
                    if temp is not None:
                        temp.data = temp.data.loc[inds]
                        setattr(resource, attr, temp)

                # If outages or heuristic dispatch attributes exist, split them as well
                for attr in ["outages", "heuristic_provide_power_mw", "heuristic_storage_SOC_mwh"]:
                    if hasattr(resource, attr):
                        temp = getattr(resource, attr)
                        if temp is not None:
                            temp = temp.loc[hourly_inds]
                            setattr(resource, attr, temp)

                # Re-set the cache
                resource.clear_calculated_properties()

            # Make new MonteCarloDraw
            mc_draw_name = f"{self.name}_{group}"
            new_mc_draw = MonteCarloDraw(
                dir_str=self.dir_str,
                case_settings=self.case_settings,
                system=system,
                random_seed=self.random_seed,
            )

            new_mc_draw.name = mc_draw_name

            # Define hidden attributes for new MC draw to avoid re-scaling load after splitting profile
            # (Results in incorrect scaled load if scaling only a subset of original load profile)
            new_mc_draw._gross_load = self.gross_load.loc[hourly_inds].copy()
            new_mc_draw._reserves = self.reserves.loc[hourly_inds].copy()

            split_monte_carlo_draws[mc_draw_name] = new_mc_draw

        return split_monte_carlo_draws

    @timer
    def compress(self, perfect_capacity: float, heuristic_net_load_subclasses: list = []):
        # Calculate net load (optionally subtracting heuristic dispatch from resources included in subclasses)
        net_load = self.net_load(heuristic_net_load_subclasses) + self.reserves - perfect_capacity
        net_load = net_load.round(4)
        # Should we be including reserves here?

        # Determine dispatch windows
        in_dates = _identify_positive_net_load_windows(net_load)

        # Label windows
        df_in = pd.DataFrame(index=pd.DatetimeIndex(np.unique(net_load.index.date)))
        df_in.loc[in_dates, "include"] = 1
        df_in = df_in.fillna(0)
        df_in["window_label"] = (df_in.shift(1) > df_in).cumsum() + 1

        # Join period after last window with first window
        df_in.loc[df_in["window_label"] == df_in["window_label"].max(), "window_label"] = df_in["window_label"].min()

        # Re-sample at hourly frequency
        df_in.loc[df_in.index[-1] + pd.Timedelta(hours=24)] = df_in.loc[df_in.index[-1]]
        df_in = df_in.resample("H").ffill()[:-1]

        # Re-order df_in by dispatch window label
        first_window = df_in["window_label"].min()
        last_window = df_in["window_label"].max()
        loop_inds = (df_in["window_label"] == first_window) & (
            df_in.index > max(df_in.loc[df_in["window_label"] == last_window].index)
        )
        df_in = pd.concat([df_in[loop_inds], df_in[~loop_inds]])

        # Adjust monthly / annual budgets: loop through hydro, flex load, and shed DR resources and skip if budgets are all none
        # In RECAP, only hydro, flex load, and shed DR resources have a heuristic dispatch that obeys energy budgets
        for resource in self.system.resources.values():
            if resource.has_energy_budget:
                if type(resource) not in [FlexLoadResource, ShedDrResource, HydroResource]:
                    raise TypeError(
                        "Energy budgets can only be used on hydro, flex load, and shed DR resources in RECAP. Remove the energy budget from "
                        + resource.name
                    )
                else:
                    resource.adjust_budgets_for_optimization(
                        model_year=self.model_year, timestamps_included_in_optimization_flags=df_in.loc[:, "include"]
                    )

        # Determine initial storage SOC for each dispatch window
        for storage_resource in self.system.storage_resources.values():
            storage_resource.set_initial_SOC_for_optimization(
                timestamps_included_in_optimization_flags=df_in.loc[:, "include"],
                window_labels=df_in.loc[:, "window_label"],
            )

        for hybrid_storage_resource in self.system.hybrid_storage_resources.values():
            hybrid_storage_resource.set_initial_SOC_for_optimization(
                timestamps_included_in_optimization_flags=df_in.loc[:, "include"],
                window_labels=df_in.loc[:, "window_label"],
            )

        # adjust shed dr and flex load call limits
        for shed_dr_resource in self.system.shed_dr_resources.values():
            shed_dr_resource.adjust_remaining_calls_for_optimization(timestamps_to_include=df_in.loc[:, "include"])

        for flex_load_resource in self.system.flex_load_resources.values():
            flex_load_resource.adjust_remaining_calls_for_optimization(timestamps_to_include=df_in.loc[:, "include"])

        df_in = df_in.loc[df_in["include"] == 1]
        df_in = df_in.reset_index().set_index(["window_label", "index"])
        df_in["weight"] = None

        # Save df_in as temporal settings
        self.temporal_settings = NewTemporalSettings(
            name="recap",
            dispatch_window_edge_effects="fixed_initial_soc",
            dispatch_windows_df=df_in,
            modeled_years=[self.model_year],
        )

    @timer
    def heuristic_dispatch(self, perfect_capacity: float, mode: ResourceGrouping = ResourceGrouping.DEFAULT):
        self.subclasses_dispatch_order = mode.resource_subclasses

        # Get resource dispatch order
        resources_dispatch_order = []
        for subclass in self.subclasses_dispatch_order:
            resources = list(getattr(self.system, subclass).values())
            # Sort storage resources by duration (longest to shortest)
            if subclass == "storage_resources" or subclass == "hybrid_storage_resources":
                resources = sorted(
                    resources, key=lambda r: r.duration.data.loc[f"01-01-{self.model_year}"], reverse=True
                )
            resources_dispatch_order.extend(resources)

        # Initialize net load with load + reserves - perfect_capacity
        logger.debug("Heuristic dispatch...")
        net_load = self.gross_load + self.reserves - perfect_capacity  # Should we be adding reserves?
        for resource in resources_dispatch_order:
            logger.debug(f"Dispatching {resource.name}...")
            # Dispatch resource with respect to net load and subtract resource dispatch from net load
            net_load = resource.dispatch(net_load=net_load, model_year=self.model_year)

        # Calculate unserved energy and reserve
        self.unserved_energy_and_reserve = net_load.clip(lower=0)
        # Save perfect capacity
        self.perfect_capacity = perfect_capacity

    @timer
    def optimized_dispatch(self, perfect_capacity: float, calculate_duals_flag: bool = False):
        if len(self.temporal_settings.dispatch_windows_df) > 0:
            # Create and solve dispatch model
            logger.debug("Optimized dispatch...")
            dispatch_model = DispatchModel(
                self, perfect_capacity=perfect_capacity, calculate_duals_flag=calculate_duals_flag
            )

            with contextlib.redirect_stdout(stream):
                dispatch_model.solve(
                    output_dir=self.dir_str.recap_output_dir, calculate_duals_flag=calculate_duals_flag
                )

            # Save optimization dispatch results to model / resources
            dispatch_model.calculate_resource_dispatch_results()

            # Get unserved energy and reserve from dispatch model
            self.unserved_energy_and_reserve = dispatch_model.unserved_energy_and_reserve

            # Get unserved energy and reserve duals from dispatch model
            if hasattr(dispatch_model, "unserved_energy_and_reserve_duals"):
                self.unserved_energy_and_reserve_duals = dispatch_model.unserved_energy_and_reserve_duals

            # Save perfect capacity
            self.perfect_capacity = perfect_capacity

        else:
            logger.info(f"Skipping optimized dispatch for {self.name} because no positive net load periods")

    @timer
    def calculate_dispatch_results(self):
        # Get unserved energy and reserve (either from heuristic dispatch or optimized dispatch)
        unserved_energy_and_reserve = self.unserved_energy_and_reserve.copy()

        # Save to dataframe
        dispatch_df = pd.DataFrame(index=unserved_energy_and_reserve.index)
        dispatch_df["unserved_energy_and_reserve"] = unserved_energy_and_reserve

        if hasattr(self, "unserved_energy_and_reserve_duals"):
            unserved_energy_and_reserve_duals = self.unserved_energy_and_reserve_duals.copy()
            dispatch_df["unserved_energy_and_reserve_duals"] = unserved_energy_and_reserve_duals

        # Save gross load, reserves to dataframe
        dispatch_df["gross_load"] = self.gross_load.copy()
        dispatch_df["reserves"] = self.reserves.copy()

        # Save perfect capacity
        dispatch_df["perfect_capacity"] = self.perfect_capacity

        provide_power_dict = {}
        provide_reserves_dict = {}
        # Get resource dispatch (heuristic or combination of heuristic and optimized)
        for resource in self.system.resources.values():
            # Get heuristic provide power if available
            if hasattr(resource, "heuristic_provide_power_mw"):
                provide_power_mw = resource.heuristic_provide_power_mw.copy()
                provide_power_mw.name = resource.name
                # Combine heuristic with optimized provide power if available
                if hasattr(resource, "optimized_provide_power_mw"):
                    provide_power_mw.loc[
                        resource.optimized_provide_power_mw.index
                    ] = resource.optimized_provide_power_mw.squeeze()
                # Save to dataframe
                provide_power_dict[resource.name] = provide_power_mw

            if hasattr(resource, "optimized_provide_reserves"):
                provide_reserves = resource.optimized_provide_reserves.copy().squeeze()
                provide_reserves.name = f"{resource.name} reserves"
                provide_power_dict[f"{resource.name} reserves"] = provide_reserves

        resource_df = pd.concat(provide_power_dict.values(), axis=1)
        dispatch_df = pd.concat([dispatch_df, resource_df], axis=1)

        return dispatch_df
