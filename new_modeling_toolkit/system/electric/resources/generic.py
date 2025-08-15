from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from numpy.random import binomial
from pydantic import Field

from new_modeling_toolkit.core import linkage
from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.system.asset import Asset
from new_modeling_toolkit.system.electric.resource_group import ResourceGroup
from new_modeling_toolkit.system.outage_distribution import OutageDistribution


# TODO 2023-04-23: Need to collapse vintages for operations, otherwise significant increase in operational variables...
# Potentially only create separate (nested/tranched) vintages of the "build" attributes (instead of fully separate vintage instances)
# A slightly slower way would be to build all the vintage instances, and then have a "combine_vintages" method that collapses operational variables (that would wrap the `construct_operational_block`)?

# Also still need to figure out how to organize all the vintages...currently they'll just all be different instances in the system (e.g., System.resources["a.2020"]
# Having a nested dict would make it more obvious how to combine resource vintages (instead of just doing it based on some naming pattern)

# Why not ConstraintList? Because fundamentally ConstraintList is not indexed

# Could separating vintages happen on a per-resource level? How does Plexos do this? Ask Jimmy/Chen?


class GenericResource(Asset):
    ############
    # Linkages #
    ############
    reserves: dict[str, linkage.ResourceToReserve] = Field(
        default_factory=dict,
        description="[RESOLVE,RECAP]. String Input. This input links the resource to a specific reserve type that it can contribute to. For example, Storage can provide both regulation up and down, but might not provide non-spin reserves.",
    )
    resource_groups: Optional[dict[str, linkage.ResourceToResourceGroup]] = Field(
        default_factory=dict,
        description="[RECAP only]. String Input. This gives each resource a group for RECAP. Depending on the Resource class, the upsampling method could change. For example, with Solar and Wind, these will be upsampled using a day draw methodology.",
    )

    outage_distributions: Optional[dict[str, linkage.ResourceToOutageDistribution]] = Field(
        default_factory=dict,
        description="[RECAP only]. String Input. This input links resources to a specific OutageDistribution component. When a random or planned outage occurs, the outage distribution dictates what the possible outage state are for each resource. For example, if a unit is either on or offline, then it's outage distribution is 0,1.",
    )
    #################################
    # Build & Retirement Attributes #
    #################################

    ###################
    # Cost Attributes #
    ###################

    # TODO 2023-04-30: Think about whether variable costs should vary by modeled year or weather year, or both. Currently assumes by modeled year
    variable_cost_power_output: ts.NumericTimeseries = Field(
        default_factory=ts.NumericTimeseries.zero,
        description="[RESOLVE only]. $/MWh. float. Variable O&M cost per MWh charged.",
        default_freq="H",
        up_method="ffill",
        down_method="mean",
        alias="variable_cost_provide_power",
    )

    variable_cost_power_input: ts.NumericTimeseries = Field(
        default_factory=ts.NumericTimeseries.zero,
        description="[RESOLVE only]. $/MWh. float. Variable O&M cost per MWh generated.",
        default_freq="H",
        up_method="ffill",
        down_method="mean",
        alias="variable_cost_increase_load",
    )

    ##########################
    # Operational Attributes #
    ##########################
    power_output_min: ts.FractionalTimeseries = Field(
        default_factory=ts.FractionalTimeseries.zero,
        description="[RESOLVE,RECAP]. MW or %. Fixed shape of resource's potential power output (e.g., solar or wind shape or flat shape"
        "for firm resources or storage resources). Used in conjunction with "
        ":py:attr:`new_modeling_toolkit.common.resource.Resource.curtailable`.",
        default_freq="H",
        up_method="interpolate",
        down_method="mean",
        weather_year=True,
        alias="provide_power_min_profile",
    )
    power_output_max: ts.FractionalTimeseries = Field(
        default_factory=ts.FractionalTimeseries.one,
        description="[RESOLVE,RECAP]. MW or %. Fixed shape of resource's potential power output (e.g., solar or wind shape or flat shape"
        "for firm resources or storage resources). Used in conjunction with "
        ":py:attr:`new_modeling_toolkit.common.resource.Resource.curtailable`.",
        default_freq="H",
        up_method="interpolate",
        down_method="mean",
        weather_year=True,
        alias="provide_power_potential_profile",
    )
    power_input_min: ts.FractionalTimeseries = Field(
        default_factory=ts.FractionalTimeseries.zero,
        description="[RESOLVE,RECAP]. MW or %. Fixed shape of resource's potential power draw (e.g. flat shape for storage resources)."
        " Used in conjunction with "
        ":py:attr:`new_modeling_toolkit.common.resource.Resource.curtailable`.",
        default_freq="H",
        up_method="interpolate",
        down_method="mean",
        weather_year=True,
    )
    power_input_max: ts.FractionalTimeseries = Field(
        default_factory=ts.FractionalTimeseries.zero,
        description="[RESOLVE,RECAP]. MW or %. Fixed shape of resource's potential power draw (e.g. flat shape for storage resources)."
        " Used in conjunction with "
        ":py:attr:`new_modeling_toolkit.common.resource.Resource.curtailable`.",
        default_freq="H",
        up_method="interpolate",
        down_method="mean",
        weather_year=True,
        alias="increase_load_potential_profile",
    )

    outage_profile: Optional[ts.FractionalTimeseries] = Field(
        default_factory=ts.FractionalTimeseries.one,
        description="[RESOLVE,RECAP]. MW or %. Fixed profile of simulated outages, where a value of 1.0 represents availability of full nameplate "
        "capacity and a value less than 1.0 represents a partial outage.",
        default_freq="H",
        up_method="interpolate",
        down_method="mean",
        weather_year=True,
    )

    unit_size: ts.NumericTimeseries = Field(
        default_factory=ts.Timeseries.zero,
        description="[RECAP, RESOLVE] Size of each unit that can be independently committed, in MW.",
        default_freq="YS",
        up_method="ffill",
        down_method="annual",
        alias="unit_size_mw",
    )

    @property
    def num_units(self) -> Dict[int, pd.Series]:
        if getattr(self, "_num_units", None) is None:
            self._num_units = {}
            for model_year in self.capacity_planned.data.index.year:
                if model_year in self.unit_size.data.index.year:
                    self._num_units[model_year] = (
                        self.capacity_planned.data.at[f"01-01-{model_year}"]
                        / self.unit_size.data.at[f"01-01-{model_year}"]
                    )
                else:
                    self._num_units[model_year] = 1.0
        return self._num_units

    energy_budget_daily: Optional[ts.FractionalTimeseries] = Field(
        None,
        description="Daily fraction of energy capacity allowed for daily dispatch [dimensionless].",
        default_freq="D",
        up_method="ffill",
        down_method="mean",
        weather_year=True,
    )

    energy_budget_monthly: Optional[ts.FractionalTimeseries] = Field(
        None,
        description="Monthly fraction of energy capacity allowed for monthly dispatch [dimensionless]",
        default_freq="MS",
        up_method="ffill",
        down_method="mean",
        weather_year=True,
    )

    energy_budget_annual: Optional[ts.FractionalTimeseries] = Field(
        None,
        description="Annual fraction of energy capacity allowed for annual dispatch [dimensionless].",
        default_freq="YS",
        up_method="ffill",
        down_method="mean",
        weather_year=True,
    )

    ###########
    # Methods #
    ###########

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def outage_distribution(self) -> Optional[OutageDistribution]:
        if self.outage_distributions is None or len(self.outage_distributions) == 0:
            outage_distribution = None
        else:
            outage_distribution_name = list(self.outage_distributions.keys())[0]
            outage_distribution = self.outage_distributions[outage_distribution_name].instance_to

        return outage_distribution

    @property
    def resource_group(self) -> ResourceGroup:
        if self.resource_groups is None or len(self.resource_groups) == 0:
            resource_group = None
        else:
            resource_group_name = list(self.resource_groups.keys())[0]
            resource_group = self.resource_groups[resource_group_name].instance_to

        return resource_group

    @property
    def has_energy_budget(self) -> bool:
        return (
            (self.energy_budget_daily is not None)
            or (self.energy_budget_monthly is not None)
            or (self.energy_budget_annual is not None)
        )

    @property
    def scaled_imax_profile(self) -> Dict[int, pd.Series]:
        if getattr(self, "_scaled_imax_profile", None) is None:
            self._scaled_imax_profile = {
                model_year: (
                    self.capacity_planned.data.at[f"01-01-{model_year}"]
                    * self.power_input_max.data
                    * self.outage_profile.data
                )
                for model_year in self.capacity_planned.data.index.year
            }
        return self._scaled_imax_profile

    @property
    def scaled_imin_profile(self) -> Dict[int, pd.Series]:
        if getattr(self, "_scaled_imin_profile", None) is None:
            self._scaled_imin_profile = {
                model_year: (
                    self.capacity_planned.data.at[f"01-01-{model_year}"]
                    * self.power_input_min.data
                    * self.outage_profile.data
                )
                for model_year in self.capacity_planned.data.index.year
            }
        return self._scaled_imin_profile

    @property
    def scaled_pmax_profile(self) -> Dict[int, pd.Series]:
        if getattr(self, "_scaled_pmax_profile", None) is None:
            self._scaled_pmax_profile = {
                model_year: (
                    self.capacity_planned.data.at[f"01-01-{model_year}"]
                    * self.power_output_max.data
                    * self.outage_profile.data
                )
                for model_year in self.capacity_planned.data.index.year
            }

        return self._scaled_pmax_profile

    @property
    def scaled_pmin_profile(self):
        if getattr(self, "_scaled_pmin_profile", None) is None:
            self._scaled_pmin_profile = {
                model_year: (
                    self.capacity_planned.data.at[f"01-01-{model_year}"]
                    * self.power_output_min.data
                    * self.outage_profile.data
                )
                for model_year in self.capacity_planned.data.index.year
            }
        return self._scaled_pmin_profile

    @property
    def scaled_daily_energy_budget(self):
        if self.energy_budget_daily is None:
            self._scaled_daily_energy_budget = None
        elif getattr(self, "_scaled_daily_energy_budget", None) is None:
            self._scaled_daily_energy_budget = {
                model_year: self.capacity_planned.data.at[f"01-01-{model_year}"] * self.energy_budget_daily.data * 24
                for model_year in self.capacity_planned.data.index.year
            }
        return self._scaled_daily_energy_budget

    @property
    def scaled_monthly_energy_budget(self):
        if self.energy_budget_monthly is None:
            self._scaled_monthly_energy_budget = None
        elif getattr(self, "_scaled_monthly_energy_budget", None) is None:
            self._scaled_monthly_energy_budget = {
                model_year: (
                    self.capacity_planned.data.at[f"01-01-{model_year}"]
                    * self.energy_budget_monthly.data
                    * 24
                    * self.energy_budget_monthly.data.index.days_in_month
                )
                for model_year in self.capacity_planned.data.index.year
            }
        return self._scaled_monthly_energy_budget

    @property
    def scaled_annual_energy_budget(self):
        if self.energy_budget_annual is None:
            self._scaled_annual_energy_budget = None
        elif getattr(self, "_scaled_annual_energy_budget", None) is None:
            self._scaled_annual_energy_budget = {
                model_year: (
                    self.capacity_planned.data.at[f"01-01-{model_year}"]
                    * self.energy_budget_annual.data
                    * 24
                    * self.energy_budget_annual.days_in_year
                )
                for model_year in self.capacity_planned.data.index.year
            }
        return self._scaled_annual_energy_budget

    def clear_calculated_properties(self):
        """
        Clear the property cache so scaled profiles are recalculated after rescaling
        """
        self._scaled_pmax_profile = None
        self._scaled_pmin_profile = None
        self._scaled_imax_profile = None
        self._scaled_imin_profile = None
        self._num_units = None
        self._scaled_daily_energy_budget = None
        self._scaled_monthly_energy_budget = None
        self._scaled_annual_energy_budget = None

    def upsample(self, load_calendar: pd.DatetimeIndex, random_seed: int = None):
        # Upsample weather year-indexed attributes to match length of full load profile
        years_tuple = (min(load_calendar.year), max(load_calendar.year))
        self.resample_ts_attributes(
            years_tuple,
            years_tuple,
            resample_weather_year_attributes=True,
            resample_non_weather_year_attributes=False,
        )

    def rescale(self, model_year: int, capacity: float, incremental: bool = False):
        # Scale resource by incremental/absolute capacity

        # Define scaling factor; scale by 1 + capacity ratio if incremental; else scale by capacity ratio
        scaling_factor = int(incremental) + capacity / self.capacity_planned.slice_by_year(model_year)

        # Re-scale planned installed capacity
        self.capacity_planned.data.at[pd.Timestamp(year=model_year, month=1, day=1)] *= scaling_factor

        # clear the cache
        self.clear_calculated_properties()

    def simulate_outages(self, model_year: int, random_seed: int = None):
        """
        Simulate stochastic outages (define "outages" attribute) for resource
        Args:
            random_seed: random seed used in stochastic outage simulation
        """

        # Check whether necessary attributes / linkages are defined
        bool_has_mttr = self.mean_time_to_repair is not None
        bool_has_for = self.stochastic_outage_rate is not None and (self.stochastic_outage_rate.data != 0).any()
        bool_has_outage_distribution = len(self.outage_distributions.keys()) > 0
        inputs_given_bool_arr = [bool_has_mttr, bool_has_for, bool_has_outage_distribution]

        # Initialize outages dataframe
        outages = pd.Series(index=self.power_output_max.data.index, data=1)

        # If any inputs are missing
        if not all(inputs_given_bool_arr):
            pass
        # If all inputs are given
        else:
            # If no random seed provided, use resource's random seed
            if not random_seed:
                random_seed = self.random_seed
            # Set random seed
            np.random.seed(random_seed)

            # Define outage simulation length, FOR profile, MTTR, and number of units
            T = len(outages)
            FOR = self.stochastic_outage_rate.data.values
            MTTR = self.mean_time_to_repair  # Hours
            n = int(np.round(self.num_units[model_year], 0))  # Number of units (nearest whole number)

            # Get random derates from resource outage distribution
            derate_fracs = self.outage_distribution.get_random_derate_fraction_arr(size=n * (T + 1), seed=random_seed)

            # Calculate MTTF
            MTTF = MTTR * (1 - FOR) / FOR
            n0 = binomial(n, 1 - FOR[0])  # Get number of available units in first period
            k, l = 0, n - n0  # Index to track total number of unit recoveries/failures up to time t
            c0 = 1 - sum(derate_fracs[k:l]) / n  # Get fraction of resource's available capacity in first period
            n_arr = []  # List to track n_t
            c_arr = []  # List to track fraction of resource's available capacity

            for t in range(T):
                n_arr.append(n0)
                c_arr.append(c0)
                # Simulate number of failures
                f = binomial(n0, min(1 / MTTF[t], 1))
                # Simulate number of recoveries
                r = binomial(n - n0, min(1 / MTTR, 1))
                # Update n0
                n0 = n0 + r - f
                # Update k, total number of repairs
                k += r
                # Update l, total number of failures
                l += f
                # Update c0
                c0 = 1 - sum(derate_fracs[k:l]) / n

            outages.loc[:] = c_arr

        # Save attribute to resource object
        self.outage_profile = ts.FractionalTimeseries(name="outage_profile", data=outages)

    def adjust_budgets_for_optimization(self, model_year: int, timestamps_included_in_optimization_flags: pd.Series):
        """Adjusts monthly and annual budgets for RECAP optimization based on heuristic dispatch.

        The total heuristic dispatch power output during timestamps that will not be included in the optimization is
        subtracted from the budgets for that month and/or year.

        Args:
            model_year: int
            timestamps_included_in_optimization_flags: A series with a datetime index with values of 1.0 for timestamps that will be
              included in optimization, and 0.0 for timestamps that will not.

        """
        for budget_attribute, freq in [
            (self.scaled_monthly_energy_budget, "M"),
            (self.scaled_annual_energy_budget, "Y"),
        ]:
            if budget_attribute is not None and (budget_attribute[model_year] < np.inf).any():
                heuristic_provide_power_mw_in_budget_periods = self._calculate_heuristic_provide_power_in_window(
                    heuristic_provide_power=self.heuristic_provide_power_mw,
                    timestamps_included_in_optimization_flags=timestamps_included_in_optimization_flags,
                    freq=freq,
                    target_index=budget_attribute[model_year].index,
                )
                # Calculate remaining budget
                remaining_budget = budget_attribute[model_year] - heuristic_provide_power_mw_in_budget_periods
                budget_attribute[model_year] = remaining_budget

    @timer
    def dispatch(self, net_load: pd.Series, model_year: int) -> pd.Series:
        """Performs heuristic dispatch for the resource against a timeseries of net load data for a given model year.

        The method also sets the resource instance's `heuristic_provide_power_mw` attribute for access in other methods.

        Each resource subclass can and should define its own dispatch heuristic, if necessary. The default is to
        dispatch at its Pmax, after accounting for outage simulations, for each timestamp.

        Args:
            net_load: the net load to dispatch the resource against.
            model_year: which model year to use for scaling the output profile of the resource

        Returns:
            The new net load, which is the input net load minus the dispatch profile of the resource.
        """
        self.heuristic_provide_power_mw = self.scaled_pmax_profile[model_year].copy()
        return net_load - self.heuristic_provide_power_mw.values

    def _calculate_heuristic_provide_power_in_window(
        self,
        heuristic_provide_power: pd.Series,
        timestamps_included_in_optimization_flags: pd.Series,
        freq: str,
        target_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Calculates the total heuristic provide power in the set of timestamps that will not be included in the
        optimization model for RECAP.

        This method is primarily used to adjust attributes like hydro budgets before optimization. The total amount of
        power provided in timestamps outside the optimization windows should be subtracted from the energy budgets
        that are passed to the optimization model to ensure that the budgets are respected.

        Note that a value of 1.0 in `timestamps_included_in_optimization_flags` means that that timestamp will *not*
        be included in the returned total, because it means that timestamp will be included in the optimization window.

        Args:
            heuristic_provide_power: the heuristic dispatch timeseries
            timestamps_included_in_optimization_flags: a timeseries indicating which timestamps are included in
              optimization.
            freq: the frequency at which to aggregate the heuristic provide power outside of the optimization windows
            target_index: the desired output index

        Returns:
            The total heuristic provide power in all timestamps per period that are not included in optimization
        """
        # Get heuristic dispatch outside of windows
        heuristic_provide_power_mw_out = heuristic_provide_power.multiply(
            (1 - timestamps_included_in_optimization_flags), axis=0
        )
        heuristic_provide_power_mw_in_window = heuristic_provide_power_mw_out.groupby(
            [heuristic_provide_power_mw_out.index.to_period(freq).to_timestamp()]
        ).sum()

        heuristic_provide_power_mw_in_budget_periods = heuristic_provide_power_mw_in_window.reindex(
            target_index
        ).fillna(0.0)

        return heuristic_provide_power_mw_in_budget_periods

    def construct_investment_block(self, model: pyo.ConcreteModel):
        super().construct_investment_block(model)

    def construct_operational_block(self, model: pyo.ConcreteModel):
        super().construct_operational_block(model)
        block = model.blocks[self.name]
        block.RESERVES = pyo.Set(initialize=list(self.reserves.keys()))
        block.power_output = pyo.Var(
            model.MODELED_YEARS,
            model.DISPATCH_WINDOWS_AND_TIMESTAMPS,
            within=pyo.NonNegativeReals,
        )
        block.power_input = pyo.Var(
            model.MODELED_YEARS,
            model.DISPATCH_WINDOWS_AND_TIMESTAMPS,
            within=pyo.NonNegativeReals,
        )
        block.provide_reserve = pyo.Var(
            model.MODELED_YEARS,
            model.DISPATCH_WINDOWS_AND_TIMESTAMPS,
            within=pyo.NonNegativeReals,
        )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def power_output_max_constraint(block: pyo.Block, modeled_year, dispatch_window, timestamp):
            return (
                block.power_output[modeled_year, dispatch_window, timestamp]
                + block.provide_reserve[modeled_year, dispatch_window, timestamp]
                <= self.scaled_pmax_profile[modeled_year].at[timestamp]
            )

        if not (self.power_output_min.data == 0).all():

            @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
            def power_output_min_constraint(block, modeled_year, dispatch_window, timestamp):
                return (
                    block.power_output[modeled_year, dispatch_window, timestamp]
                    >= self.scaled_pmin_profile[modeled_year].at[timestamp]
                )

        if (self.power_input_max.data == 0).all():
            # fix all timestamps at zero without looping through
            block.power_input.fix(0)
        else:

            @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
            def power_input_max_constraint(block, modeled_year, dispatch_window, timestamp):
                return (
                    block.power_input[modeled_year, dispatch_window, timestamp]
                    <= self.scaled_imax_profile[modeled_year].at[timestamp]
                )

            @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
            def mileage_constraint(block, modeled_year, dispatch_window, timestamp):
                return block.power_output[modeled_year, dispatch_window, timestamp] + block.power_input[
                    modeled_year, dispatch_window, timestamp
                ] <= max(
                    self.scaled_pmax_profile[modeled_year].at[timestamp],
                    self.scaled_imax_profile[modeled_year].at[timestamp],
                )

        # Hard-code tolerance for budget/other checks (for numerical issues that create infeasible dispatch problems)
        tol = 1  # MWh

        if not getattr(self, "energy_budget_annual", None) is None:

            @block.Constraint(model.MODELED_YEARS, model.WEATHER_YEARS)
            def annual_energy_budget_constraint(block, modeled_year, weather_year):
                if self.energy_budget_annual is None or np.isinf(self.energy_budget_annual.data.at[weather_year]):
                    constraint = pyo.Constraint.Skip
                else:
                    constraint = (
                        sum(
                            block.power_output[modeled_year, dispatch_window, timestamp]
                            for dispatch_window, timestamp in model.WEATHER_YEAR_TO_TIMESTAMPS_MAPPING[weather_year]
                        )
                        <= self.scaled_annual_energy_budget[modeled_year].at[weather_year] + tol
                    )

                return constraint

        if not getattr(self, "energy_budget_monthly", None) is None:

            @block.Constraint(model.MODELED_YEARS, model.MONTHS)
            def monthly_energy_budget_constraint(block, modeled_year, month):
                if self.energy_budget_monthly is None or np.isinf(self.energy_budget_monthly.data.at[month]):
                    constraint = pyo.Constraint.Skip
                else:
                    constraint = (
                        sum(
                            block.power_output[modeled_year, dispatch_window, timestamp]
                            for dispatch_window, timestamp in model.MONTH_TO_TIMESTAMPS_MAPPING[month]
                        )
                        <= self.scaled_monthly_energy_budget[modeled_year].at[month] + tol
                    )

                return constraint

        if not getattr(self, "energy_budget_daily", None) is None:

            @block.Constraint(model.MODELED_YEARS, model.DAYS)
            def daily_energy_budget_constraint(block, modeled_year, day):
                if self.energy_budget_daily is None or np.isinf(self.energy_budget_daily.data.at[day]):
                    constraint = pyo.Constraint.Skip
                else:
                    constraint = (
                        sum(
                            block.power_output[modeled_year, dispatch_window, timestamp]
                            for dispatch_window, timestamp in model.DAY_TO_TIMESTAMPS_MAPPING[day]
                        )
                        <= self.scaled_daily_energy_budget[modeled_year].at[day] + tol
                    )

                return constraint
