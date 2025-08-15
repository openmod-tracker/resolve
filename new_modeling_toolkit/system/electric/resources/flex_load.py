import copy
import enum
from typing import Optional

import pandas as pd
import pyomo.environ as pyo
from loguru import logger
from pydantic import Field
from pydantic import root_validator
from typing_extensions import Annotated

from new_modeling_toolkit import get_units
from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.system.electric.resources.shed_dr import ShedDrResource
from new_modeling_toolkit.system.electric.resources.storage import StorageResource


@enum.unique
class FlexLoadShiftDirection(enum.Enum):
    PRE_CONSUMPTION = "pre_consumption"
    DEFERRED_CONSUMPTION = "deferred_consumption"
    EITHER = "either"


class FlexLoadResource(ShedDrResource, StorageResource):
    ###########################
    # Flexible Load Attribute #
    ###########################

    adjacency: Annotated[int, Field(gt=0)] = Field(
        ...,
        description="[RECAP, RESOLVE] Number of adjacent hours to constrain energy shifting. Adjacency constraints ensure that if load is shifted down in one hour, an equivalent amount of load is shifted up at most X hours away, and vice versa. [hours]",
        units=get_units("adjacency"),
    )

    shift_direction: FlexLoadShiftDirection = Field(
        ...,
        description="[RECAP, RESOLVE] If pre_consumption, flexible load resources always need to increase load first before providing power."
        "An example of this is pre-cooling."
        "If deferred_consumption, flexible load resources always will provide power first before increasing load."
        "An example of this is deferring to use appliances. If either, the resource can do provide power or increase load first depending on what is optimal.",
    )

    duration: Optional[ts.NumericTimeseries] = Field(
        None,
        description="[RECAP, RESOLVE] Operational time of the battery at a specified power level before it runs out of energy [hours]. Note: not all flex load resources will require a duration. Only EV resources. For all others, the duration will default to the length of the adjacency window.",
        default_freq="YS",
        up_method="ffill",
        down_method="annual",
        alias="storage_duration",
    )

    energy_budget_daily: ts.FractionalTimeseries = Field(
        None,
        description="[RECAP, RESOLVE] Daily fraction of energy capacity allowed for daily dispatch [dimensionless].",
        default_freq="D",
        up_method="ffill",
        down_method="mean",
        weather_year=True,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.warning(f"Setting {self.name} resource unit size equal to planned capacity.")
        self.unit_size.data = self.capacity_planned.data

    @root_validator(skip_on_failure=True)
    def update_duration_default(cls, values):
        """
        Set default of duration equal to adjacency if not provided
        """
        adjacency = values.get("adjacency")
        duration = values.get("duration")

        if duration is None and adjacency is not None:
            new_duration = ts.NumericTimeseries(
                name="storage_capacity_planned",
                data=copy.deepcopy(values["capacity_planned"].data),
            )
            new_duration.data[:] = adjacency
            new_duration.name = "duration"
            values["duration"] = new_duration
            values["_scaled_SOC_max_profile"] = None
        return values

    def _identify_initial_SOC(self, dispatch_window_df: pd.DataFrame, modeled_year: int):
        """
        Set the initial storage SOC (needed for storage constraints) based on shift direction inputs
        """
        dispatch_window_df["initial_SOC"] = self.power_input_max.data.loc[dispatch_window_df["index"].values].values
        initial_soc = dispatch_window_df.groupby(dispatch_window_df.index.get_level_values(0)).first()["initial_SOC"]
        self.initial_storage_SOC = pd.Series(initial_soc)
        if self.shift_direction == FlexLoadShiftDirection.PRE_CONSUMPTION:
            self.initial_storage_SOC.loc[:] = 0
        elif self.shift_direction == FlexLoadShiftDirection.DEFERRED_CONSUMPTION:
            self.initial_storage_SOC.loc[:] = self.storage_capacity_planned.slice_by_year(modeled_year)
        else:
            self.initial_storage_SOC.loc[:] = self.storage_capacity_planned.slice_by_year(modeled_year) / 2

    def dispatch(self, net_load: pd.Series, model_year: int) -> pd.Series:
        """
        Heuristic dispatch of shed dr resources

        Identify periods with highest potential avoided unserved energy from flex load calls
        Choose top N in each day/month/year based on input call limits.

        Remove second call if two calls are overlapping.

        Call start is defined as when shift down begins
        Adjacency window start is when shift up or shift down begins

        Loop through each call start, provide power in positive net load hours and increase load in negative net load hours.
        Adjust total dispatched load so that provide power == increase load at the end of the adjacency window.
        Also ensure the total provide power does not exceed the daily energy budget defined for the resource.
        """

        # Skip dispatch if insufficient capacity
        if self.capacity_planned.data.at[f"01-01-{model_year}"] < 0.1:
            self.heuristic_provide_power_mw = 0.0 * self.scaled_pmax_profile[model_year].copy()
            self.call_starts = pd.Series()
            return net_load

        # Initialize provide_power_mw
        provide_power_mw = pd.Series(index=net_load.index, data=0)
        pmax = self.scaled_pmax_profile[model_year]
        increase_load_mw = pd.Series(index=net_load.index, data=0)
        imax = self.scaled_imax_profile[model_year]

        # Define call duration, adjacency
        call_duration = self.max_call_duration
        adjacency = self.adjacency

        # Define "adjacency shift", used to determine how to align shift up potential with shift down potential
        if self.shift_direction == FlexLoadShiftDirection.PRE_CONSUMPTION:
            window_length = adjacency + call_duration
            adjacency_shift = call_duration
        elif self.shift_direction == FlexLoadShiftDirection.DEFERRED_CONSUMPTION:
            window_length = adjacency + call_duration
            adjacency_shift = adjacency + call_duration
        elif self.shift_direction == FlexLoadShiftDirection.EITHER:
            window_length = 2 * adjacency + call_duration
            adjacency_shift = adjacency + call_duration

        # Define flex up and flex down potential
        potential_flex_down = net_load.clip(lower=0, upper=pmax)  # Potential to avoid unserved energy in this period
        potential_flex_up = net_load.clip(lower=-imax, upper=0)  # Potential to shift energy to this period

        # Get avoided unserved energy of flex load calls starting in each hour
        call_window_potential_flex_down = (
            potential_flex_down.rolling(call_duration, min_periods=1).sum().shift(-call_duration + 1)
        )
        # Get shiftable energy of flex load calls starting in each hour
        call_window_potential_flex_up = (
            potential_flex_up.rolling(window_length, min_periods=1).sum().shift(-adjacency_shift + 1)
        )
        # Adjust flex up potential during overlap with flex down period
        call_window_potential_flex_up -= (
            potential_flex_up.rolling(call_duration, min_periods=1).sum().shift(-call_duration + 1)
        )
        # Energy budgets
        daily_energy_budgets = self.scaled_daily_energy_budget[model_year].resample("H").asfreq().ffill()
        # Call window potentials is the minimum of (negative) flex up and flex down potential and daily budget
        call_window_potentials = (
            pd.concat(
                [
                    -call_window_potential_flex_up,
                    call_window_potential_flex_down,
                    daily_energy_budgets,
                ],
                axis=1,
            )
            .dropna(axis=0)
            .min(axis=1)
        )

        # Define overlap window length
        overlap_window_length = self.max_call_duration + self.adjacency

        # Get top calls for each day / month / year
        top_daily_calls = self._identify_heuristic_dispatch_calls(
            call_window_potentials, freq="D", overlap_window_length=overlap_window_length
        )
        top_monthly_calls = self._identify_heuristic_dispatch_calls(
            top_daily_calls, freq="M", overlap_window_length=overlap_window_length
        )
        top_annual_calls = self._identify_heuristic_dispatch_calls(
            top_monthly_calls, freq="Y", overlap_window_length=overlap_window_length
        )

        # Get index of call starts
        call_start_index = top_annual_calls.index.sort_values()

        # Remove overlapping call starts
        if not call_start_index.empty:
            call_start = call_start_index[0]
            non_overlapping_call_start_index = [call_start]
            for next_call_start in call_start_index[1:]:
                if next_call_start > call_start + pd.Timedelta(hours=overlap_window_length):
                    non_overlapping_call_start_index.append(next_call_start)
                    call_start = next_call_start
            call_start_index = pd.DatetimeIndex(non_overlapping_call_start_index)

        # Save call starts series (for use in adjusting boundary conditions)
        self.call_starts = pd.Series(index=call_start_index, data=1)

        # Get dispatch for each call start
        for call_start in call_start_index:
            # Expand flex down call
            expanded_call_index = self._expand_call_index(
                pd.DatetimeIndex([call_start]), end_hours_offset=call_duration - 1
            )
            # Expand flex up call
            expanded_adjacency_index = self._expand_call_index(
                pd.DatetimeIndex([call_start]),
                beginning_hours_offset=window_length - adjacency_shift,
                end_hours_offset=adjacency_shift - 1,
            )
            expanded_adjacency_index = expanded_adjacency_index.difference(
                expanded_call_index
            )  # Remove flex down periods
            # Remove negative net load hours from call index (align with call potential definition)
            expanded_call_index = expanded_call_index[net_load.loc[expanded_call_index] > 0]
            # Remove positive net load hours from adjacency index (align with call potential definition)
            expanded_adjacency_index = expanded_adjacency_index[net_load.loc[expanded_adjacency_index] < 0]

            # Get flex down / flex up profile
            provide_power_mw_call = pmax.loc[expanded_call_index]
            increase_load_mw_call = imax.loc[expanded_adjacency_index]
            discharging_efficiency_call = self.discharging_efficiency.data.loc[expanded_call_index]
            charging_efficiency_call = self.charging_efficiency.data.loc[expanded_adjacency_index]
            provide_power_mw_call_plus_losses = (provide_power_mw_call / discharging_efficiency_call).sum()
            increase_load_mw_call_less_losses = (increase_load_mw_call * charging_efficiency_call).sum()
            # Adjust flex down / flex up profile to preserve energy balance
            if provide_power_mw_call_plus_losses < increase_load_mw_call_less_losses:
                increase_load_mw_call *= provide_power_mw_call_plus_losses / increase_load_mw_call_less_losses
                increase_load_mw_call_less_losses = (increase_load_mw_call * charging_efficiency_call).sum()
            elif provide_power_mw_call_plus_losses > increase_load_mw_call_less_losses:
                provide_power_mw_call *= increase_load_mw_call_less_losses / provide_power_mw_call_plus_losses
                provide_power_mw_call_plus_losses = (provide_power_mw_call / discharging_efficiency_call).sum()
            # Apply daily energy budget to dispatch (scale down if necessary)
            daily_energy_budget = daily_energy_budgets.loc[call_start]
            if (
                provide_power_mw_call.sum() > daily_energy_budget
            ):  # energy provided after discharging efficiency losses must be less than or equal to energy budget
                provide_power_mw_call *= daily_energy_budget / provide_power_mw_call.sum()
                provide_power_mw_call_plus_losses = (provide_power_mw_call / discharging_efficiency_call).sum()
                increase_load_mw_call *= provide_power_mw_call_plus_losses / increase_load_mw_call_less_losses

            provide_power_mw.loc[provide_power_mw_call.index] = provide_power_mw_call
            increase_load_mw.loc[increase_load_mw_call.index] = increase_load_mw_call

        # Adjust net load with provide power
        self.heuristic_provide_power_mw = provide_power_mw - increase_load_mw

        return net_load - self.heuristic_provide_power_mw.values

    #####################################################################################################
    @timer
    def construct_operational_block(self, model: pyo.ConcreteModel):
        self._identify_initial_SOC(
            model.temporal_settings.dispatch_windows_df.index.to_frame(), model.temporal_settings.modeled_years[0]
        )

        block = model.blocks[self.name]
        super().construct_operational_block(model)

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def adjacency_constraint(block, model_year, dispatch_window, timestamp):
            """
            Define adjacency window for a shift dr event. The SOC of the resource must return to the initial state
            within x hours before or after the DR (power output only) event.
            I.e., length of one DR call in terms of hours.
            """
            if getattr(self, "max_call_duration") is not None:
                hour_offsets = range(0, int(self.max_call_duration) + 2 * self.adjacency + 1)
            else:
                hour_offsets = range(0, 2 * self.adjacency + 1)
            return block.committed_units[model_year, dispatch_window, timestamp] <= sum(
                block.start_units[
                    model_year, model.DISPATCH_WINDOWS_AND_TIMESTAMPS.prevw((dispatch_window, timestamp), step=t)
                ]
                for t in hour_offsets
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def energy_balance_committed_units_constraint(block, model_year, dispatch_window, timestamp):
            """
            Coupled with energy_balance_uncommitted_units_constraint: Forces hours where units are uncommited to return to the initial state of charge
            When committed_units is 0: state_of_charge <= initial_charge
            """
            initial_charge = self.initial_storage_SOC[dispatch_window]
            return (
                block.state_of_charge[model_year, dispatch_window, timestamp] - initial_charge
                <= self.scaled_SOC_max_profile[model_year].at[timestamp]
                * block.committed_units[model_year, dispatch_window, timestamp]
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def energy_balance_uncommitted_units_constraint(block, model_year, dispatch_window, timestamp):
            """
            Coupled with energy_balance_committed_units_constarint: Forces hours where units are uncommited to return to the initial state of charge
            When committed_units is 0 (aka uncommitted): state_of_charge >= initial_charge
            """
            initial_charge = self.initial_storage_SOC[dispatch_window]
            return (
                block.state_of_charge[model_year, dispatch_window, timestamp]
                >= initial_charge * (block.committed_units[model_year, dispatch_window, timestamp] - 1) * -1
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def start_and_shutdown_unit_exclusivity_constraint(block, model_year, dispatch_window, timestamp):
            """
            Prevents a unit from starting and shutting down in the same hour. Necessary to enforce the energy balance constraints
            """
            return (
                block.start_units[model_year, dispatch_window, timestamp]
                + block.shutdown_units[model_year, dispatch_window, timestamp]
                <= 1
            )

        if self.shift_direction is not FlexLoadShiftDirection.EITHER:

            @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS)
            def shift_direction_constraint(block, model_year, dispatch_window):
                """
                Constraint to direct the shift direction of flexible load resources
                If pre_consumption = True, flexible load resources always need to increase load first before providing power.
                An example of this is pre-cooling.
                If deferred_consumption = True, flexible load resources always will provide power first before increasing load.
                An example of this is deferring to use appliances.

                model: model
                year: model_year in timestep format
                tp_index: timepoint index of the variable in a tuple format. Example: RECAP (hour,), RESOLVE (year, rep_period,hour,)
                variable_name: variable name to apply the constraint. Example: RECAP: "Storage_SOC_MWh", RESOLVE "SOC_Intra_Period"
                """
                if self.shift_direction == FlexLoadShiftDirection.PRE_CONSUMPTION:
                    return (
                        block.state_of_charge[
                            model_year, dispatch_window, model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].first()
                        ]
                        == 0
                    )

                elif self.shift_direction == FlexLoadShiftDirection.DEFERRED_CONSUMPTION:
                    full_charge = self.storage_capacity_planned.slice_by_year(model_year)
                    return (
                        block.state_of_charge[
                            model_year, dispatch_window, model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].first()
                        ]
                        == full_charge
                    )
                else:
                    raise NotImplementedError(
                        f"Flex load shift direction constraint for shift direction `{self.shift_direction}` is not "
                        f"implemented."
                    )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS)
        def first_hour_state_of_charge_tracking(block, modeled_year, dispatch_window):
            """
            Handle the edge case where the first timestamp has a fixed initial SOC
            """
            timestamp = model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].first()
            prev_timestamp = model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].prevw(timestamp)
            constraint = block.state_of_charge[modeled_year, dispatch_window, timestamp] == (
                block.state_of_charge[modeled_year, dispatch_window, prev_timestamp]
                + block.power_input[modeled_year, dispatch_window, prev_timestamp]
                * self.charging_efficiency.data.at[timestamp]
                - block.power_output[modeled_year, dispatch_window, prev_timestamp]
                / self.discharging_efficiency.data.at[timestamp]
            )

            return constraint
