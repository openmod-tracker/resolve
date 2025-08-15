from typing import Optional

import pandas as pd
import pyomo.environ as pyo
from loguru import logger
from pydantic import Field

import new_modeling_toolkit.core.temporal.timeseries as ts
from new_modeling_toolkit import get_units
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.system.electric.resources.generic import GenericResource


class VariableResource(GenericResource):
    ############
    # Linkages #
    ############

    #################################
    # Build & Retirement Attributes #
    #################################

    ###################
    # Cost Attributes #
    ###################
    curtailment_cost: Optional[ts.NumericTimeseries] = Field(
        default=None,
        description="[RESOLVE Only]. $/MWh. float. Cost of curtailment - the exogeneously assumed cost"
        "at which different contract zones would be willing to curtail their"
        "variable renewable generation",
        default_freq="H",
        up_method="ffill",
        down_method="mean",
        units=get_units("curtailment_cost"),
    )

    ##########################
    # Operational Attributes #
    ##########################
    curtailable: bool = Field(
        True,
        description="[RESOLVE only]. TRUE/FALSE. boolean.  Whether resource's power output can be curtailed relative to "
        ":py:attr:`new_modeling_toolkit.common.resource.Resource.potential_provide_power_profile`.",
    )

    ###########
    # Methods #
    ###########

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.curtailable:
            logger.info(f"Setting power_output_min to power_output_max for non-curtailable resource {self.name}.")
            self.power_output_min.data = self.power_output_max.data

    # upsample renewable resource
    def upsample(self, load_calendar: pd.DatetimeIndex, random_seed: int = None):
        """
        Upsample profiles based on resource group.day_draw_map
        Upsampled renewables profiles over length of full weather/load records.
        """
        group_obj = self.resource_group
        df_day_draw = group_obj.day_draw_map.squeeze()
        # resample day_draw_map to be hourly
        df_day_draw_hourly = df_day_draw.reindex(load_calendar).fillna(method="ffill")
        df_day_draw_hourly += pd.to_timedelta(df_day_draw_hourly.index.hour, unit="H")

        def _remap_with_day_draw_map(
            attribute_name: str,
            day_draw_map: pd.Series,
            load_calendar: pd.DatetimeIndex,
            default_val: Optional[float] = None,
        ):
            """
            Remap df to day_draw_map
            """
            df = getattr(self, attribute_name).data
            df = df.reindex(day_draw_map.values)
            df.index = load_calendar
            df.name = "value"
            if default_val is not None:
                df = df.fillna(default_val)
            elif df.isnull().sum():
                raise ValueError(
                    f"Attribute `{attribute_name}` for `{self.__class__.__name__}` `{self.name}` has null values "
                    f"introduced after day-draw mapping was applied."
                )

            return df

        for att, default_val in [("power_output_max", None), ("power_output_min", 0), ("outage_profile", 1)]:
            getattr(self, att).data = _remap_with_day_draw_map(
                attribute_name=att,
                day_draw_map=df_day_draw_hourly,
                load_calendar=load_calendar,
                default_val=default_val,
            )

        super().upsample(load_calendar)

    def construct_investment_block(self, model: pyo.ConcreteModel):
        super().construct_investment_block(model)

    @timer
    def construct_operational_block(self, model: pyo.ConcreteModel):
        super().construct_operational_block(model)
