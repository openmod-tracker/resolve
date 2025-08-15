import enum
from typing import Dict

import pandas as pd
import pyomo.environ as pyo
from pydantic import Field
from typing_extensions import Annotated

from new_modeling_toolkit import get_units
from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.core.temporal.new_temporal import DispatchWindowEdgeEffects
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.system.electric.resources.generic import GenericResource


@enum.unique
class StorageDurationConstraint(enum.Enum):
    FIXED = "fixed"
    MINIMUM = "minimum"


class StorageResource(GenericResource):
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
    duration: ts.NumericTimeseries = Field(
        default_factory=ts.NumericTimeseries.zero,
        description="[RESOLVE, RECAP]. MWh. operational time of the battery at a specified power level before it runs out of energy [hours]",
        default_freq="YS",
        up_method="ffill",
        down_method="annual",
        alias="storage_duration",
    )

    power_input_max: ts.FractionalTimeseries = Field(
        default_factory=ts.FractionalTimeseries.one,
        description="[RESOLVE,RECAP]. MW or %. Fixed shape of resource's potential power draw (e.g. flat shape for storage resources)."
        " Used in conjunction with "
        ":py:attr:`new_modeling_toolkit.common.resource.Resource.curtailable`.",
        default_freq="H",
        up_method="interpolate",
        down_method="mean",
        weather_year=True,
        alias="increase_load_potential_profile",
    )

    duration_constraint: StorageDurationConstraint = StorageDurationConstraint.FIXED

    ###################
    # Cost Attributes #
    ###################
    storage_cost_fixed_om: ts.NumericTimeseries = Field(
        default_factory=ts.NumericTimeseries.zero,
        description="[RESOLVE only]. $/kWh-yr. For the planned portion of the resource's storage capacity, "
        "the ongoing fixed O&M cost",
        default_freq="YS",
        up_method="interpolate",
        down_method="annual",
        alias="new_storage_capacity_fixed_om_by_vintage",
    )
    storage_cost_investment: ts.NumericTimeseries = Field(
        default_factory=ts.NumericTimeseries.zero,
        description="[RESOLVE only]. $/kWh-yr. For new storage capacity, the annualized fixed cost of investment. "
        "This is an annualized version of an overnight cost that could include financing costs ($/kWh-year).",
        default_freq="YS",
        up_method="interpolate",
        down_method="annual",
        alias="new_storage_annual_fixed_cost_dollars_per_kwh_yr_by_vintage",
    )

    ##########################
    # Operational Attributes #
    ##########################
    charging_efficiency: ts.FractionalTimeseries = Field(
        default_factory=ts.FractionalTimeseries.one,
        description="[RESOLVE, RECAP]. % of Charging MW. Efficiency losses associated with charging (increasing load), typically expressed as a % of nameplate unless charging power specified "
        'to increase storage "state of charge".',
        units=get_units("charging_efficiency"),
        default_freq="H",
        up_method="ffill",
        down_method="mean",
        weather_year=True,
    )
    discharging_efficiency: ts.FractionalTimeseries = Field(
        default_factory=ts.FractionalTimeseries.one,
        description="[RESOLVE, RECAP]. % of Discharging MW, Efficiency losses associated with discharging (providing power), typically expressed as a % of nameplate unless charging power specified, "
        'taking energy out of storage "state of charge".',
        units=get_units("discharging_efficiency"),
        default_freq="H",
        up_method="ffill",
        down_method="mean",
        weather_year=True,
    )
    parasitic_loss: Annotated[float, Field(ge=0, le=1)] = Field(
        0, description="[Storage] Hourly state of charge losses.", units=get_units("parasitic_loss")
    )

    ###########
    # Methods #
    ###########

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_storage_SOC = None

    @property
    def storage_capacity_planned(self) -> ts.NumericTimeseries:
        """
        This property is for storage only now.
        """
        if self.duration is None:
            return ts.NumericTimeseries(
                name="storage_capacity_planned",
                data=self.capacity_planned.data,
            )
        else:
            return ts.NumericTimeseries(
                name="storage_capacity_planned",
                data=self.capacity_planned.data * self.duration.data,
            )

    @property
    def scaled_SOC_max_profile(self) -> Dict[int, pd.Series]:
        """
        This property is for storage only now.
        """
        if getattr(self, "_scaled_SOC_max_profile", None) is None:
            self._scaled_SOC_max_profile = {
                model_year: pd.Series(
                    index=self.scaled_pmax_profile[model_year].index,
                    data=self.storage_capacity_planned.data.at[f"01-01-{model_year}"],
                )
                for model_year in self.storage_capacity_planned.data.index.year
            }
        return self._scaled_SOC_max_profile

    @timer
    def dispatch(self, net_load: pd.Series, model_year: int) -> pd.Series:
        """Dispatches a storage resource against a net load timeseries for a given model year.

        The storage resource is dispatched with a greedy algorithm. When net load is negative, the resource charges
        as much as possible until it is full. When net load is positive, it is discharged as much as possible until
        it is empty.

        Current implementation assumes there is no SOC_min or P_min.

        Args:
            net_load: the net load to dispatch against
            model_year: which model year to use for nameplate capacity and duration

        Returns:
            the load net of the storage dispatch
        """
        # Skip dispatch if insufficient capacity
        if self.capacity_planned.data.at[f"01-01-{model_year}"] < 0.1:
            self.heuristic_provide_power_mw = 0.0 * self.scaled_pmax_profile[model_year].copy()
            self.heuristic_storage_SOC_mwh = 0.0 * self.heuristic_provide_power_mw.copy()
            return net_load

        # Get storage resource attributes
        SOC_max = self.storage_capacity_planned.data.at[f"01-01-{model_year}"]
        charging_efficiency_numpy = self.charging_efficiency.data.to_numpy()
        discharging_efficiency_numpy = self.discharging_efficiency.data.to_numpy()

        # Initialize net_provide_power_mw, SOC_mwh
        net_provide_power_mw = net_load.copy().clip(
            lower=-self.scaled_imax_profile[model_year], upper=self.scaled_pmax_profile[model_year]
        )
        SOC_mwh = pd.Series(index=net_load.copy().index, data=0.0)

        # Identify last positive net load hour
        pos_net_load_periods = net_provide_power_mw.loc[net_provide_power_mw > 0].index
        if pos_net_load_periods.empty:
            self.heuristic_provide_power_mw = 0.0 * self.scaled_pmax_profile[model_year].copy()
            self.heuristic_storage_SOC_mwh = 0.0 * self.heuristic_provide_power_mw.copy()
            return net_load
        final_pos_net_load_period = pos_net_load_periods[-1]

        # Shift start of period to last positive net load hour
        index = net_provide_power_mw.index.copy()
        net_provide_power_mw = pd.concat(
            [
                net_provide_power_mw.loc[index > final_pos_net_load_period],
                net_provide_power_mw.loc[index <= final_pos_net_load_period],
            ]
        )
        SOC_mwh = pd.concat(
            [
                SOC_mwh.loc[index > final_pos_net_load_period],
                SOC_mwh.loc[index <= final_pos_net_load_period],
            ]
        )

        timestamps = net_provide_power_mw.index.to_numpy()
        net_provide_power_mw_numpy = net_provide_power_mw.copy().to_numpy()
        SOC_mwh_numpy = SOC_mwh.copy().to_numpy()

        # Initialize SOC
        SOC = 0.0
        SOC_mwh_numpy[0] = SOC
        for i, ts in enumerate(timestamps):
            curr_net_provide_power = net_provide_power_mw_numpy[i]
            charging_efficiency = charging_efficiency_numpy[i]
            discharging_efficiency = discharging_efficiency_numpy[i]
            # Calculate net provide power and SOC
            if curr_net_provide_power <= 0:
                curr_net_provide_power = -min((SOC_max - SOC) / charging_efficiency, -curr_net_provide_power)
                SOC -= charging_efficiency * curr_net_provide_power  # Net provide power negative
            else:
                curr_net_provide_power = min(SOC * discharging_efficiency, curr_net_provide_power)
                SOC -= (1 / discharging_efficiency) * curr_net_provide_power  # Net provide power positive
            # Update net provide power and SOC
            net_provide_power_mw_numpy[i] = curr_net_provide_power
            if i < len(timestamps) - 1:
                SOC_mwh_numpy[i + 1] = SOC

        # Save data and sort index to return to correct ordering of timestamps
        new_index = pd.DatetimeIndex(timestamps)
        self.heuristic_provide_power_mw = pd.Series(index=new_index, data=net_provide_power_mw_numpy).sort_index()
        self.heuristic_storage_SOC_mwh = pd.Series(index=new_index, data=SOC_mwh_numpy).sort_index()

        return net_load - self.heuristic_provide_power_mw.values

    def set_initial_SOC_for_optimization(
        self, timestamps_included_in_optimization_flags: pd.Series, window_labels: pd.Series
    ):
        """Calculates what the initial SOC should be for each dispatch window in the optimization problem for RECAP.

        Args:
            df_in:

        Returns:

        """
        # Get initial SOC for each dispatch window
        # (Can just label by dispatch window, no need to specify timestamp)
        windows = window_labels.unique()
        self.initial_storage_SOC = pd.Series(index=windows)
        for window in windows:
            timestamps_included_in_window = timestamps_included_in_optimization_flags.loc[(window_labels == window)]
            timestamps_included_in_window = (
                timestamps_included_in_window.shift(1).fillna(timestamps_included_in_window.iloc[-1])
                - timestamps_included_in_window
            )
            if (timestamps_included_in_window == 0).all():
                first_timepoint = timestamps_included_in_window.index[0]
            else:
                first_timepoint = timestamps_included_in_window.loc[timestamps_included_in_window < 0].index[0]
            self.initial_storage_SOC.loc[window] = self.heuristic_storage_SOC_mwh.loc[first_timepoint]

    # fmt: off
    def construct_investment_block(self, model: pyo.ConcreteModel):
        super().construct_investment_block(model)
        block = model.blocks[self.name]
        block.storage_capacity_selected = pyo.Var(within=pyo.NonNegativeReals)
        block.storage_capacity_retired = pyo.Var(model.MODELED_YEARS, within=pyo.NonNegativeReals)
        block.storage_capacity_operational = pyo.Expression(
            model.MODELED_YEARS,
            rule=lambda block, modeled_year:
                self.storage_capacity_planned.data[modeled_year] +
                block.storage_capacity_selected -
                block.storage_capacity_retired[modeled_year]
        )

        @block.Constraint(model.MODELED_YEARS)
        def storage_duration_build(block, modeled_year):
            rhs = block.capacity_operational[modeled_year] * self.duration
            lhs = block.storage_capacity_operational[modeled_year]
            if self.duration_constraint == StorageDurationConstraint.FIXED:
                return rhs == lhs
            elif self.duration_constraint == StorageDurationConstraint.MINIMUM:
                return rhs <= lhs
            else:
                raise NotImplementedError(f"Duration specification ({self.duration_constraint}) for {self.name} has not been implemented yet.")

        # Append costs to already-existing `block.investment_costs` expression inherited from parent class
        # TODO: this doesn't work
        for modeled_year in block.investment_costs:
            block.investment_costs[modeled_year] += (
                block.storage_capacity_selected * self.storage_cost_investment.data[modeled_year] * 10**3
            ) + (block.storage_capacity_operational * self.storage_cost_fixed_om.data[modeled_year] * 10**3)

    @timer
    def construct_operational_block(self, model: pyo.ConcreteModel):
        super().construct_operational_block(model)
        block = model.blocks[self.name]
        block.state_of_charge = pyo.Var(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=pyo.NonNegativeReals)

        if self.duration is not None:
            @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
            def state_of_charge_max_constraint(block, modeled_year: int, dispatch_window: int, timestamp: pd.Timestamp):

                return (
                    block.state_of_charge[modeled_year, dispatch_window, timestamp]
                    <= self.scaled_SOC_max_profile[modeled_year].at[timestamp]
                )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def state_of_charge_tracking(block, modeled_year: int, dispatch_window: int, timestamp: pd.Timestamp):
            charging_efficiency = self.charging_efficiency.data.at[timestamp]
            discharging_efficiency = self.discharging_efficiency.data.at[timestamp]

            # Handle the edge case where the first timestamp has a fixed initial SOC
            if timestamp == model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].first() and model.dispatch_window_edge_effects == DispatchWindowEdgeEffects.FIXED_INITIAL_SOC:
                constraint = (
                        block.state_of_charge[modeled_year, dispatch_window, timestamp] == self.initial_storage_SOC.at[dispatch_window]
                )
            else:
                prev_timestamp = model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].prevw(timestamp)
                constraint = (
                        block.state_of_charge[modeled_year, dispatch_window, timestamp] == (
                            block.state_of_charge[modeled_year, dispatch_window, prev_timestamp]
                            + block.power_input[
                                modeled_year, dispatch_window, prev_timestamp] * charging_efficiency
                            - block.power_output[
                                modeled_year, dispatch_window, prev_timestamp] / discharging_efficiency)
                        )

            return constraint

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def state_of_charge_operating_reserve_up_max(block, modeled_year: int, dispatch_window: int, timestamp: pd.Timestamp):
            """Upward reserves are limited by storage state-of-charge "headroom"."""
            return (
                # charging efficiency not applied to power_input
                # because storage could provide full amount of MW charging as reserve by just stopping its charging
                # Ex: Battery w 100 MWh SOC (discharge_eff of 0.8), charging at 200 MW may provide 280 MW reserve
                # bc it could stop charging and then begin discharging
                block.provide_reserve[modeled_year, dispatch_window, timestamp] <=
                block.state_of_charge[modeled_year, dispatch_window, timestamp] * self.discharging_efficiency.data.at[timestamp]
                + block.power_input[modeled_year, dispatch_window, timestamp]
                - block.power_output[modeled_year, dispatch_window, timestamp]
            )

        # fmt: on
