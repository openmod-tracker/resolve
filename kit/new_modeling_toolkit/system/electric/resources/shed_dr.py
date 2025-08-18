from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from loguru import logger
from pydantic import Field
from typing_extensions import Annotated

from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.system.electric.resources.unit_commitment import UnitCommitmentMethod
from new_modeling_toolkit.system.electric.resources.unit_commitment import UnitCommitmentResource


class ShedDrResource(UnitCommitmentResource):
    max_call_duration: Optional[Annotated[int, Field(gt=0)]] = Field(
        None, description="[RECAP, RESOLVE] Maximum duration of a single shed demand response event call [hrs]."
    )

    max_annual_calls: Optional[ts.NumericTimeseries] = Field(
        None,
        description="[RECAP, RESOLVE] Annual number of allowable calls for a shed DR resource.",
        default_freq="YS",
        up_method="ffill",
        down_method="annual",
        weather_year=True,
    )

    max_monthly_calls: Optional[ts.NumericTimeseries] = Field(
        None,
        description="[RECAP, RESOLVE] Monthly number of allowable calls for a shed DR resource.",
        default_freq="MS",
        up_method="ffill",
        down_method="monthly",
        weather_year=True,
    )

    max_daily_calls: Optional[ts.NumericTimeseries] = Field(
        default_factory=ts.NumericTimeseries.one,
        description="[RECAP, RESOLVE] Daily number of allowable calls for a shed DR resource.",
        default_freq="D",
        up_method="ffill",
        down_method="daily",
        weather_year=True,
    )

    unit_commitment: UnitCommitmentMethod = Field(
        UnitCommitmentMethod.INTEGER,
        description="[RECAP, RESOLVE] linear or integer. See unit_commitment attribute description for further detail.",
    )

    def rescale(self, model_year: int, capacity: float, incremental: bool = False):
        super().rescale(model_year, capacity, incremental)
        self.unit_size.data.at[pd.Timestamp(year=model_year, month=1, day=1)] = self.capacity_planned.data.at[
            pd.Timestamp(year=model_year, month=1, day=1)
        ]

    def upsample(self, load_calendar: pd.DatetimeIndex, random_seed: int = None):
        # define a set of timeseries attribute to upsample
        timeseries_attrs = [
            "power_output_max",
            "power_output_min",
            "power_input_max",
            "energy_budget_daily",
            "energy_budget_monthly",
            "energy_budget_annual",
            "max_annual_calls",
            "max_monthly_calls",
            "max_daily_calls",
        ]

        input_year = list(self.power_output_max.data.index.year.unique())[0]
        repeat_year_dict = {model_year: input_year for model_year in load_calendar.year.unique()}

        self.resample_ts_attributes([input_year, input_year], [input_year, input_year], True, False)
        for attr in timeseries_attrs:
            temp = getattr(self, attr, None)
            if temp is not None and isinstance(temp, ts.Timeseries):
                temp.repeat_ts(repeat_year_dict)
                setattr(self, attr, temp)

        # Upsample weather year-indexed attributes to match length of full load profile
        super().upsample(load_calendar)

    def _return_freq_windows_n_calls(self, windows: pd.Series, freq: str, calls: Union[int, float]):
        if calls is None:
            return None
        elif freq == "D":
            freq_id = windows.index.date
        elif freq == "M":
            freq_id = windows.index.month
        elif freq == "Y":
            freq_id = windows.index.year

        freq_windows = windows.groupby(freq_id).nlargest(int(calls)).reset_index(level=0, drop=True)
        return freq_windows

    def _return_overlapping_id(self, window_series: pd.Series):
        """
        Identify overlapping hours when the windows are expanded to the max call duration
        """
        window_series = window_series.sort_index()
        try:
            overlap = abs(pd.Series(window_series.index).diff().dt.total_seconds()) < self.max_call_duration * 60 * 60
            return window_series[overlap.values], overlap.sum()
        except:
            return pd.Series(), 0

    def _return_idx_max_window(
        self, windows: pd.Series, freq: str, calls: Optional[Union[int, float]], min_calls: Optional[Union[int, float]]
    ):
        """
        Return index of last hour of max avoided unserved energy by frequency
        """
        if calls is None:
            return windows
        else:
            freq_windows = self._return_freq_windows_n_calls(windows, freq, calls)
            overlapping_id, overlap_sum = self._return_overlapping_id(freq_windows)
            freq_windows = freq_windows.drop(overlapping_id.index)
            n = 1
            # if ids are overlapping, add more calls, then drop the overlapping until 0 overlapping and minimum call number is met
            while len(freq_windows) < min_calls:
                freq_windows_extended = self._return_freq_windows_n_calls(windows, freq, min_calls + n)
                overlapping_id, overlap_sum = self._return_overlapping_id(freq_windows_extended)
                freq_windows = freq_windows_extended.drop(overlapping_id.index)
                n += 1

            return freq_windows

    def _calculate_min_calls(self, windows):
        if self.max_annual_calls is not None:
            years = len(set(windows.index.to_period("Y").unique())) * self.max_annual_calls
        else:
            years = np.nan
        if self.max_monthly_calls is not None:
            months = len(set(windows.index.to_period("M").unique())) * self.max_monthly_calls
        else:
            months = np.nan
        if self.max_daily_calls is not None:
            days = len(set(windows.index.to_period("D").unique())) * self.max_daily_calls
        else:
            days = np.nan
        return min(years, months, days)

    def _identify_heuristic_dispatch_calls(
        self, call_window_potentials: pd.Series, freq: str, overlap_window_length: int
    ) -> pd.Series:
        """
        Identify demand response call start with greatest potential given call limits
        """
        if call_window_potentials.empty:
            return call_window_potentials

        if not (call_window_potentials > 0).any():
            selected_call_window_potentials = call_window_potentials.loc[call_window_potentials > 0]
            return selected_call_window_potentials

        assert freq in ["D", "M", "Y"]

        # Get max calls time series
        max_calls = None
        if freq == "D" and self.max_daily_calls is not None:
            max_calls = self.max_daily_calls
        elif freq == "M" and self.max_monthly_calls is not None:
            max_calls = self.max_monthly_calls
        elif freq == "Y" and self.max_annual_calls is not None:
            max_calls = self.max_annual_calls

        # If max_calls is None --> no call limits specified for given frequency --> return unfiltered call windows
        if max_calls is None:
            return call_window_potentials

        # Define min_down_time for use in heuristics
        if self.min_down_time is not None:
            min_down_time = self.min_down_time
        else:
            min_down_time = 0

        # Define function to get top n calls from a set of grouped calls
        def get_top_n_calls(call_group, n):
            """ "
            Greedy algorithm for choosing top n non-overlapping calls in a group of calls
            """

            # Initialize list of call starts
            call_starts = []

            # Make copy of call_group to modify in call selection process
            remaining_call_group = call_group.copy()

            # Get top n calls
            while len(call_starts) < n and len(remaining_call_group) > 0 and (remaining_call_group > 0).any():
                # Select call start with largest potential from remaining call starts in call_group
                top_call_start = remaining_call_group.idxmax()

                # Add to list of call starts
                call_starts.extend([top_call_start])

                # Remove overlapping call starts so they will not get selected
                overlap_start = top_call_start - pd.DateOffset(hours=overlap_window_length - 1 + min_down_time)
                overlap_end = top_call_start + pd.DateOffset(hours=overlap_window_length - 1 + min_down_time)
                overlap_period = remaining_call_group.loc[
                    (remaining_call_group.index >= overlap_start) & (remaining_call_group.index <= overlap_end)
                ].index
                if overlap_period is not None:
                    remaining_call_group = remaining_call_group.drop(labels=overlap_period)

                # If this doesn't work, switch back to while loop of (nlargest --> remove overlap)

            # Create datetime index from list of call starts
            call_start_index = pd.DatetimeIndex(call_starts)

            # Return call group potentials for selected top n call starts
            return call_group.loc[call_start_index].copy()

        # Group calls by frequency (day, month, or year)
        call_groups = call_window_potentials.groupby(
            call_window_potentials.index.to_period(freq=freq).to_timestamp(), group_keys=False
        )

        # Get top n calls in every group
        # Here, call_group.name is day/month/year (label of group)
        selected_call_window_potentials_by_group = call_groups.apply(
            lambda call_group: get_top_n_calls(call_group, n=max_calls.data.loc[call_group.name])
        )

        selected_call_window_potentials = selected_call_window_potentials_by_group.copy()

        return selected_call_window_potentials

    def _expand_call_index(
        self, call_start_index: pd.DatetimeIndex, beginning_hours_offset: int = 0, end_hours_offset: int = 0
    ):
        """
        Expand call index to include hours after the call start based on the max_call_duration
        """
        expanded_call_index_list = []
        for call_start in call_start_index:
            expanded_call_index = pd.date_range(
                start=call_start - pd.DateOffset(hours=beginning_hours_offset),
                end=call_start + pd.DateOffset(hours=end_hours_offset),
                freq="H",
            )
            expanded_call_index_list.extend(expanded_call_index)
        expanded_call_index = pd.DatetimeIndex(expanded_call_index_list).drop_duplicates()
        # drop any index outside of weather year
        expanded_call_index = expanded_call_index[
            expanded_call_index.year.isin(call_start_index.year.unique().tolist())
        ]
        return expanded_call_index

    @timer
    def dispatch(self, net_load, model_year):
        """
        Heuristic dispatch of shed dr resources

        Identify periods with highest potential avoided unserved energy from DR calls
        Choose top N in each day/month/year
        """

        # Skip dispatch if insufficient capacity
        if self.capacity_planned.data.at[f"01-01-{model_year}"] < 0.1:
            self.heuristic_provide_power_mw = 0.0 * self.scaled_pmax_profile[model_year].copy()
            self.call_starts = pd.Series()
            return net_load

        # Initialize provide_power_mw
        provide_power_mw = pd.Series(index=net_load.index, data=0)
        pmax = self.scaled_pmax_profile[model_year]

        # Define unserved energy
        potential_avoided_unserved_energy = net_load.clip(lower=0, upper=pmax)

        # Get avoided unserved energy of DR calls starting in each hour
        call_window_potentials = (
            potential_avoided_unserved_energy.rolling(self.max_call_duration, min_periods=1)
            .sum()
            .shift(-self.max_call_duration + 1)
        )

        # Get top calls for each day / month / year
        top_daily_calls = self._identify_heuristic_dispatch_calls(
            call_window_potentials, freq="D", overlap_window_length=self.max_call_duration
        )
        top_monthly_calls = self._identify_heuristic_dispatch_calls(
            top_daily_calls, freq="M", overlap_window_length=self.max_call_duration
        )
        top_annual_calls = self._identify_heuristic_dispatch_calls(
            top_monthly_calls, freq="Y", overlap_window_length=self.max_call_duration
        )

        # Get index of call starts
        call_start_index = top_annual_calls.index

        # Save call starts series (for use in adjusting boundary conditions)
        self.call_starts = pd.Series(index=call_start_index, data=1)

        # add dr calls to provide power
        expanded_call_index = self._expand_call_index(call_start_index, end_hours_offset=self.max_call_duration - 1)
        provide_power_mw.loc[expanded_call_index] = pmax.loc[expanded_call_index]

        # Adjust net load with provide power
        self.heuristic_provide_power_mw = provide_power_mw

        return net_load - self.heuristic_provide_power_mw.values

    def adjust_remaining_calls_for_optimization(self, timestamps_to_include: pd.Series):
        """
        If DR calls are used outside of dispatch periods, subtract the number of calls from the
        call limits so that optimized dispatch only uses the remaining calls
        """
        # Get call starts outside of dispatch optimization windows
        calls_not_included = self.call_starts.loc[self.call_starts.index.difference(timestamps_to_include.index)]
        # Adjust remaining calls if necessary
        if calls_not_included.empty:
            pass
        else:
            for call_limit_attr, freq in [
                ("max_daily_calls", "D"),
                ("max_monthly_calls", "M"),
                ("max_annual_calls", "Y"),
            ]:
                call_limits = getattr(self, call_limit_attr)
                if call_limits is not None:
                    # Get excluded calls in each period
                    calls_not_included_by_period = calls_not_included.groupby(
                        calls_not_included.index.to_period(freq=freq).to_timestamp()
                    ).sum()
                    # Adjust remaining calls
                    call_limits.data.loc[calls_not_included_by_period.index] -= calls_not_included_by_period

    @timer
    def construct_operational_block(self, model: pyo.ConcreteModel):
        block = model.blocks[self.name]
        logger.info(f"Construction operational block for {self.name}.")
        super().construct_operational_block(model)

        if self.max_annual_calls is not None:

            @block.Constraint(model.MODELED_YEARS, model.WEATHER_YEARS)
            def annual_dr_call_limit_constraint(block, model_year, weather_year):
                """
                Max # of call per year.
                """
                constraint = (
                    sum(
                        block.start_units[model_year, dispatch_window, timestamp]
                        for dispatch_window, timestamp in model.WEATHER_YEAR_TO_TIMESTAMPS_MAPPING[weather_year]
                    )
                    <= self.max_annual_calls.data.at[f"01-01-{weather_year.year}"] * self.num_units[model_year]
                )

                return constraint

            @block.Constraint(model.MODELED_YEARS, model.WEATHER_YEARS)
            def annual_dr_call_limit_power_output_constraint(block, model_year, weather_year):
                """
                Max # of call per year.
                """
                constraint = (
                    sum(
                        block.start_units_power_output[model_year, dispatch_window, timestamp]
                        for dispatch_window, timestamp in model.WEATHER_YEAR_TO_TIMESTAMPS_MAPPING[weather_year]
                    )
                    <= self.max_annual_calls.data.at[f"01-01-{weather_year.year}"] * self.num_units[model_year]
                )

                return constraint

        if self.max_monthly_calls is not None:

            @block.Constraint(model.MODELED_YEARS, model.MONTHS)
            def monthly_dr_call_limit_constraint(block, model_year, month):
                """
                Max # of call per month.
                """
                constraint = (
                    sum(
                        block.start_units[model_year, dispatch_period, timestamp]
                        for dispatch_period, timestamp in model.MONTH_TO_TIMESTAMPS_MAPPING[month]
                    )
                    <= self.max_monthly_calls.data.at[month] * self.num_units[model_year]
                )
                return constraint

            @block.Constraint(model.MODELED_YEARS, model.MONTHS)
            def monthly_dr_call_limit_power_output_constraint(block, model_year, month):
                """
                Max # of call per month.
                """
                constraint = (
                    sum(
                        block.start_units_power_output[model_year, dispatch_period, timestamp]
                        for dispatch_period, timestamp in model.MONTH_TO_TIMESTAMPS_MAPPING[month]
                    )
                    <= self.max_monthly_calls.data.at[month] * self.num_units[model_year]
                )
                return constraint

        if self.max_daily_calls is not None:

            @block.Constraint(model.MODELED_YEARS, model.DAYS)
            def daily_dr_call_limit_constraint(block, model_year, day):
                """
                Max # of call per day.
                """
                constraint = (
                    sum(
                        block.start_units[model_year, dispatch_period, timestamp]
                        for dispatch_period, timestamp in model.DAY_TO_TIMESTAMPS_MAPPING[day]
                    )
                    <= self.max_daily_calls.data.at[day] * self.num_units[model_year]
                )

                return constraint

            @block.Constraint(model.MODELED_YEARS, model.DAYS)
            def daily_dr_call_limit_power_output_constraint(block, model_year, day):
                """
                Max # of call per day.
                """
                constraint = (
                    sum(
                        block.start_units_power_output[model_year, dispatch_period, timestamp]
                        for dispatch_period, timestamp in model.DAY_TO_TIMESTAMPS_MAPPING[day]
                    )
                    <= self.max_daily_calls.data.at[day] * self.num_units[model_year]
                )

                return constraint

        if self.max_call_duration is not None:

            @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
            def max_dr_call_constraint(block, model_year, dispatch_window, timestamp):
                """
                Constrain maximum up time (power output) for unit commitment resources.
                I.e., length of one DR call in terms of hours.
                """
                hour_offsets = range(0, self.max_call_duration)
                return block.committed_units_power_output[model_year, dispatch_window, timestamp] <= sum(
                    block.start_units_power_output[
                        model_year, model.DISPATCH_WINDOWS_AND_TIMESTAMPS.prevw((dispatch_window, timestamp), step=t)
                    ]
                    for t in hour_offsets
                )
