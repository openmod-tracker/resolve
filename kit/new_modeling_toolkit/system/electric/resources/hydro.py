import numpy as np
import pandas as pd
from loguru import logger
from pydantic import model_validator
from scipy import optimize

from new_modeling_toolkit.core.utils.core_utils import cantor_pairing_function
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.system.electric.resources.generic import GenericResource

class HydroResource(GenericResource):
    """Resource with storage capacity.

    Adds state-of-charge tracking.

    It feels like BatteryResource could be composed of a GenericResource + Asset (that represents the storage costs),
    but need to think about this more before implementing.
    """

    ############
    # Linkages #
    ############

    #################################
    # Build & Retirement Attributes #
    #################################

    ##########################
    # Operational Attributes #
    ##########################

    ###########
    # Methods #
    ###########

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def validate_budget_exists(self) -> "Hydro":
        if (
            self.energy_budget_daily is None
            and self.energy_budget_monthly is None
            and self.energy_budget_annual is None
        ):
            raise ValueError(f"Hydro resource {self.name} must have at least one energy budget defined.")
        return self

    def upsample(self, load_calendar: pd.DatetimeIndex, random_seed: int = None):
        """
        Upsample hydro profiles (including Pmin, Pmax, and energy budget profiles).
        """

        # define a set of timeseries attribute to upsample
        timeseries_attrs = [
            "power_output_max",
            "power_output_min",
            "energy_budget_daily",
            "energy_budget_monthly",
            "energy_budget_annual",
        ]

        input_year_min = self.power_output_max.data.index.year.min()
        input_year_max = self.power_output_max.data.index.year.max()
        # hydro_year: year of data input
        hydro_years = list(self.power_output_max.data.copy().index.year.unique())

        # Set random seed
        np.random.seed(cantor_pairing_function(random_seed, int(self.random_seed)))

        # Construct a dictionary of {load year: hydro_year}
        repeat_year_dict = dict(
            zip(
                load_calendar.year.unique(),
                np.random.choice(hydro_years, size=(max(load_calendar.year) - min(load_calendar.year) + 1)),
            )
        )
        self.resample_ts_attributes([input_year_min, input_year_max], [input_year_min, input_year_max], True, False)
        for attr in timeseries_attrs:
            temp = getattr(self, attr)
            # skip timeseries input that have only one data point (upsample using 'resample_ts_attributes')
            if temp is None or len(temp.data) == 1:
                continue
            # repeat weather year data using repeat_year_dict to upsample to length of full load calendar
            else:
                logger.debug(f"Repeating {temp.name} profile")
                # repeat the timeseries input following the load year <-> hydro year pair in the dictionary
                temp.repeat_ts(repeat_year_dict)
                # return the repeated timeseries back to Hydro resource
                setattr(self, attr, temp)

        super().upsample(load_calendar=load_calendar)

    @timer
    def dispatch(self, net_load, model_year):
        # Skip dispatch if insufficient capacity
        if self.capacity_planned.data.at[f"01-01-{model_year}"] < 0.1:
            self.heuristic_provide_power_mw = 0.0 * self.scaled_pmax_profile[model_year].copy()
            return net_load

        # Note: heuristic dispatch will only use ONE set of energy budgets (daily, monthly, or annual)
        if self.energy_budget_daily is not None and (self.energy_budget_daily.data < np.inf).any().any():
            budget = self.scaled_daily_energy_budget[model_year]
            freq = "D"
        elif self.energy_budget_monthly is not None and (self.energy_budget_monthly.data < np.inf).any().any():
            budget = self.scaled_monthly_energy_budget[model_year]
            freq = "M"
        elif self.energy_budget_annual is not None and (self.energy_budget_annual.data < np.inf).any().any():
            budget = self.scaled_annual_energy_budget[model_year]
            freq = "Y"

        # Initialize dispatch with pmin profile
        provide_power_mw = self.scaled_pmin_profile[model_year].copy()

        # Adjust net load, and budgets (subtract pmin)
        remaining_net_load = net_load - provide_power_mw
        remaining_budget = (
            budget - provide_power_mw.groupby([provide_power_mw.index.to_period(freq).to_timestamp()]).sum()
        )
        remaining_pmax = self.scaled_pmax_profile[model_year].copy() - provide_power_mw

        # Get dispatch periods and critical periods
        dispatch_periods = remaining_net_load.index.to_period(freq).to_timestamp()  # Not unique list yet
        remaining_net_load_grouped = remaining_net_load.groupby(dispatch_periods)
        critical_periods = dispatch_periods.unique()[remaining_net_load_grouped.max() > 0]

        # Iterate through critical budget periods and simulate dispatch
        for period in critical_periods:
            # Get index for period
            period_index = remaining_net_load_grouped.groups[period]

            # Initialize hourly remaining net load over period (after subtraction of pmin)
            remaining_net_load_period = remaining_net_load_grouped.get_group(period)

            # Initialize hourly provide power over period (currently set to pmin)
            provide_power_mw_period = provide_power_mw.loc[period_index]

            # Initialize remaining pmax (after subtraction of pmin)
            remaining_pmax_period = remaining_pmax.loc[period_index]

            # Initialize remaining budget (after subtraction of pmin)
            remaining_budget_period = remaining_budget.loc[period]

            # Hard-code tolerance for budget/other checks
            tol = 1  # MWh

            # Assert non-negative remaining budget after subtracting pmin
            assert remaining_budget_period > -tol

            # If remaining budget close to 0 (i.e. sum of pmin over period ~= budget), keep provide power at pmin
            if abs(remaining_budget_period) < tol:
                # Keep provide power at pmin
                provide_power_mw.loc[period_index] = provide_power_mw_period
                continue  # Continue to next period

            # Else if sum of remaining pmax over period <= remaining budget, provide power at pmax
            elif remaining_pmax_period.sum() <= remaining_budget_period + tol:
                # Add remaining pmax (pmax - pmin) to provide power (pmin) to get pmax (pmax = pmin + pmax - pmin)
                provide_power_mw_period += remaining_pmax_period
                provide_power_mw.loc[period_index] = provide_power_mw_period
                continue  # Continue to next period

            # Else if sum of remaining pmax over period > remaining budget, need to "pseudo-optimize" dispatch
            elif remaining_pmax_period.sum() > remaining_budget_period + tol:
                pass  # Execute dispatch code below

            else:
                raise ValueError

            # Search for threshold T such that we discharge hydro at max( min(net load - T, pmax), 0)
            # and use full remaining budget
            # I.e. we discharge when net load > T up to pmax, and do not discharge below this threshold
            # This function is decreasing in T, hence we can bound its root with a and b and search for it with opti

            # a should be a lower bound for the threshold, and hence the remaining budget after dispatching above the
            # threshold a should be negative
            # b should be an upper bound for the threshold, and hence the remaining budget after dispatching above the
            # threshold should be positive
            a = remaining_net_load_period.min() - remaining_pmax_period.max() - tol
            b = remaining_net_load_period.max() + remaining_pmax_period.max() + tol

            # a should be a lower bound for the threshold, and hence the remaining budget after dispatching above the
            # threshold a should be negative
            # b should be an upper bound for the threshold, and hence the remaining budget after dispatching above the
            # threshold should be positive

            def dispatch_above_threshold(T, net_load, pmax):
                # Provide power when net_load > T up to pmax...
                provide_power_mw = np.minimum(net_load - T, pmax).clip(lower=0)
                return provide_power_mw

            def remaining_budget_after_dispatch(T, net_load, pmax, budget):
                provide_power_mw = dispatch_above_threshold(T, net_load, pmax)
                return budget - provide_power_mw.sum()

            T = optimize.bisect(
                remaining_budget_after_dispatch,
                a,
                b,
                args=(remaining_net_load_period, remaining_pmax_period, remaining_budget_period),
                xtol=0.001,
            )
            provide_power_mw_period += dispatch_above_threshold(T, remaining_net_load_period, remaining_pmax_period)

            # Save final provide_power_mw
            provide_power_mw.loc[period_index] = provide_power_mw_period

        self.heuristic_provide_power_mw = provide_power_mw

        return net_load - self.heuristic_provide_power_mw.values
