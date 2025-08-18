import enum
from typing import Optional

import pyomo.environ as pyo
from pydantic import Field
from typing_extensions import Annotated

from new_modeling_toolkit import get_units
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.system.electric.resources.generic import GenericResource


@enum.unique
class UnitCommitmentMethod(enum.Enum):
    INTEGER = "integer"
    LINEAR = "linear"

    @property
    def var_type(self):
        if self.value == "integer":
            return pyo.NonNegativeIntegers
        elif self.value == "linear":
            return pyo.NonNegativeReals


class UnitCommitmentResource(GenericResource):
    unit_commitment: UnitCommitmentMethod = Field(
        UnitCommitmentMethod.LINEAR,
        description="[RECAP, RESOLVE] To strictly the number of shift events, set to integer. Otherwise, the default is ‘linear’ and does not fully limit the number of events. Linear is the default, because this is a much simpler optimization problem and you will see minimal increase in run time. This is an acceptable assumption for many projects if you are modeling a large area which could have several Demand Response programs in different locations, each with their own call limits. When modeling smaller geographic areas, or if the project needs strict call limits set the unit_commitment attribute to ‘integer’. Note that this will increase the run time. How much will vary depending on the model complexity, but a good rule of thumb is to assume 1.5x the run time if the variables were linear.",
    )
    min_down_time: Optional[Annotated[int, Field(gt=0)]] = Field(
        None,
        description="[RECAP, RESOLVE] Minimum downtime between commitments (hours).",
        units=get_units("min_down_time"),
    )
    min_up_time: Optional[Annotated[int, Field(gt=0)]] = Field(
        None, description="[RECAP, RESOLVE] Minimum uptime during a commitment (hours).", units=get_units("min_up_time")
    )
    min_stable_level: Optional[Annotated[float, Field(gt=0)]] = Field(
        None, description="[RESOLVE] Minimum stable level when committed", units=get_units("min_stable_level")
    )
    start_cost: Optional[Annotated[float, Field(ge=0)]] = Field(
        None,
        description="[RESOLVE] Cost for each unit startup ($/unit start). "
        "If using linearized UC, this cost will be linearized as well.",
        units=get_units("start_cost"),
    )
    shutdown_cost: Optional[Annotated[float, Field(ge=0)]] = Field(
        None,
        description="[RESOLVE] Cost for each unit shutdown ($/unit shutdown). "
        "If using linearized UC, this cost will be linearized as well.",
        units=get_units("shutdown_cost"),
    )

    start_fuel_use: Optional[Annotated[float, Field(ge=0)]] = Field(
        None, description="[RESOLVE] Amount of fuel used per unit start", units=get_units("start_fuel_use")
    )

    @timer
    def construct_operational_block(self, model: pyo.ConcreteModel):
        super().construct_operational_block(model)
        block = model.blocks[self.name]

        # tracking variables for power input and/or power output
        block.start_units = pyo.Var(
            model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=pyo.NonNegativeReals
        )
        block.shutdown_units = pyo.Var(
            model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=pyo.NonNegativeReals
        )

        block.committed_units = pyo.Var(
            model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=pyo.NonNegativeReals
        )

        block.committed_units_power_input = pyo.Var(
            model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=self.unit_commitment.var_type
        )

        # tracking variables for power output only
        block.start_units_power_output = pyo.Var(
            model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=pyo.NonNegativeReals
        )
        block.shutdown_units_power_output = pyo.Var(
            model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=pyo.NonNegativeReals
        )
        block.committed_units_power_output = pyo.Var(
            model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=self.unit_commitment.var_type
        )

        block.operational_units_in_timepoint = pyo.Expression(
            model.MODELED_YEARS,
            model.DISPATCH_WINDOWS_AND_TIMESTAMPS,
            rule=lambda block, year, dispatch_window, timestamp: self.num_units[year],
        )

        block.committed_capacity_mw_power_output = pyo.Expression(
            model.MODELED_YEARS,
            model.DISPATCH_WINDOWS_AND_TIMESTAMPS,
            rule=lambda block, year, dispatch_window, timestamp: block.committed_units_power_output[
                year, dispatch_window, timestamp
            ]
            * self.unit_size.data.at[f"01-01-{year}"],
        )

        block.committed_capacity_mw_power_input = pyo.Expression(
            model.MODELED_YEARS,
            model.DISPATCH_WINDOWS_AND_TIMESTAMPS,
            rule=lambda block, year, dispatch_window, timestamp: block.committed_units_power_input[
                year, dispatch_window, timestamp
            ]
            * self.unit_size.data.at[f"01-01-{year}"],
        )

        # Derate hourly operational (or committed) capacity by `increase_load_potential_profile`
        block.plant_increase_load_capacity_in_timepoint_mw = pyo.Expression(
            model.MODELED_YEARS,
            model.DISPATCH_WINDOWS_AND_TIMESTAMPS,
            rule=lambda block, year, dispatch_window, timestamp: block.committed_capacity_mw_power_input[
                year, dispatch_window, timestamp
            ]
            * self.power_input_max.data.at[timestamp],
        )

        # Derate hourly operational (or committed) capacity by `provide_power_potential_profile`.
        block.plant_provide_power_capacity_in_timepoint_mw = pyo.Expression(
            model.MODELED_YEARS,
            model.DISPATCH_WINDOWS_AND_TIMESTAMPS,
            rule=lambda block, year, dispatch_window, timestamp: block.committed_capacity_mw_power_output[
                year, dispatch_window, timestamp
            ]
            * self.power_output_max.data.at[timestamp],
        )

        # allow user to set a lower bound on hourly generation
        # note that there is a separate pmin constraint for unit commitment resources
        block.plant_power_min_capacity_in_timepoint_mw = pyo.Expression(
            model.MODELED_YEARS,
            model.DISPATCH_WINDOWS_AND_TIMESTAMPS,
            rule=lambda block, year, dispatch_window, timestamp: block.committed_capacity_mw_power_output[
                year, dispatch_window, timestamp
            ]
            * self.power_output_min.data.at[timestamp],
        )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS)
        def zero_committed_units_in_first_timepoint(block, model_year, dispatch_window):
            return (
                block.committed_units[
                    model_year, dispatch_window, model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].first()
                ]
                == 0
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS)
        def zero_committed_units_in_last_timepoint(block, model_year, dispatch_window):
            return (
                block.committed_units[
                    model_year, dispatch_window, model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].last()
                ]
                == 0
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS)
        def zero_start_units_in_first_timepoint(block, model_year, dispatch_window):
            return (
                block.start_units[
                    model_year, dispatch_window, model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].first()
                ]
                == 0
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def max_provide_power_domain_constraint(block, model_year, dispatch_window, timestamp):
            """
            provide power cannot exceed max provide power (restricts domain of provide power)
            """
            return (
                block.power_output[model_year, dispatch_window, timestamp]
                <= block.plant_provide_power_capacity_in_timepoint_mw[model_year, dispatch_window, timestamp]
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def min_provide_power_domain_constraint(block, model_year, dispatch_window, timestamp):
            """
            provide power cannot be less than min provide power (restricts domain of provide power)
            """
            return (
                block.power_output[model_year, dispatch_window, timestamp]
                >= block.plant_power_min_capacity_in_timepoint_mw[model_year, dispatch_window, timestamp]
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def max_increase_load_domain_constraint(block, model_year, dispatch_window, timestamp):
            """
            increase load cannot exceed max increase load (restricts domain of increase load)
            """
            return (
                block.power_input[model_year, dispatch_window, timestamp]
                <= block.plant_increase_load_capacity_in_timepoint_mw[model_year, dispatch_window, timestamp]
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def committed_units_ub_constraint(block, model_year, dispatch_window, timestamp):
            """
            Calculate the maximum number of units that can be committed based on operational capacity in model year.
            """
            return (
                block.committed_units[model_year, dispatch_window, timestamp]
                <= block.operational_units_in_timepoint[model_year, dispatch_window, timestamp]
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def start_units_ub_constraint(block, model_year, dispatch_window, timestamp):
            """Calculate the maximum number of units that can be started based on operational capacity in model year."""
            return (
                block.start_units[model_year, dispatch_window, timestamp]
                <= block.operational_units_in_timepoint[model_year, dispatch_window, timestamp]
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def shutdown_units_ub_constraint(block, model_year, dispatch_window, timestamp):
            """Calculate the maximum number of units that can be shutdown based on operational capacity in model year."""
            return (
                block.shutdown_units[model_year, dispatch_window, timestamp]
                <= block.operational_units_in_timepoint[model_year, dispatch_window, timestamp]
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def committed_unit_constraint(block, model_year, dispatch_window, timestamp):
            return (
                block.committed_units[model_year, dispatch_window, timestamp]
                == block.committed_units_power_input[model_year, dispatch_window, timestamp]
                + block.committed_units_power_output[model_year, dispatch_window, timestamp]
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def commitment_tracking_constraint(block, model_year, dispatch_window, timestamp):
            """Track unit commitment status."""
            next_timestamp = model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].nextw(timestamp)
            return (
                block.committed_units[model_year, dispatch_window, next_timestamp]
                == block.committed_units[model_year, dispatch_window, timestamp]
                + block.start_units[model_year, dispatch_window, next_timestamp]
                - block.shutdown_units[model_year, dispatch_window, next_timestamp]
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def commitment_power_output_tracking_constraint(block, model_year, dispatch_window, timestamp):
            """Track unit commitment status of power output only."""
            next_timestamp = model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].nextw(timestamp)
            return (
                block.committed_units_power_output[model_year, dispatch_window, next_timestamp]
                == block.committed_units_power_output[model_year, dispatch_window, timestamp]
                + block.start_units_power_output[model_year, dispatch_window, next_timestamp]
                - block.shutdown_units_power_output[model_year, dispatch_window, next_timestamp]
            )

        if self.min_stable_level is not None:

            @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
            def unit_commitment_pmin_constraint(block, model_year, dispatch_window, timestamp):
                return (
                    block.power_output[model_year, dispatch_window, timestamp]
                    >= self.min_stable_level
                    * block.committed_capacity_mw_power_output[model_year, dispatch_window, timestamp]
                )

        if self.min_up_time is not None:

            @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
            def min_uptime_constraint(block, model_year, dispatch_window, timestamp):
                """Constrain minimum up time for unit commitment resources."""
                hour_offsets = range(1, self.min_up_time + 1)
                return block.committed_units[model_year, dispatch_window, timestamp] >= sum(
                    block.start_units[
                        model_year,
                        dispatch_window,
                        model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].prevw(timestamp, t),
                    ]
                    for t in hour_offsets
                )

        if self.min_down_time is not None:

            @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
            def min_downtime_constraint(block, model_year, dispatch_window, timestamp):
                """Constrain minimum down time for unit commitment resources.
                Treat hourly datasets differently from rep_periods with variable timesteps:
                For variable timesteps, may require longer up time if min_up_time falls between timesteps
                """
                hour_offsets = range(1, self.min_down_time + 1)
                return block.operational_units_in_timepoint[
                    model_year, dispatch_window, timestamp
                ] - block.committed_units[model_year, dispatch_window, timestamp] >= sum(
                    block.shutdown_units[
                        model_year,
                        dispatch_window,
                        model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].prevw(timestamp, t),
                    ]
                    for t in hour_offsets
                )
