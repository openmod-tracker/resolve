from typing import Literal
from typing import Optional

from loguru import logger
from pydantic import Field

from new_modeling_toolkit import get_units
from new_modeling_toolkit.core import component
from new_modeling_toolkit.core import dir_str
from new_modeling_toolkit.core import linkage
from new_modeling_toolkit.core.temporal import timeseries as ts


class Reserve(component.Component):

    direction: Literal["up", "down"] = "up"
    exclusive: bool = True
    load_following_percentage: Optional[float] = None

    requirement: Optional[ts.NumericTimeseries] = Field(
        default=None,
        default_freq="H",
        up_method="interpolate",
        down_method="first",
        units=get_units("requirement"),
        weather_year=True,
    )
    _dynamic_requirement: ts.NumericTimeseries
    category: Optional[str] = None

    ######################
    # Mapping Attributes #
    ######################
    plants: Optional[dict[str, linkage.Linkage]] = dict()
    tx_paths: Optional[dict[str, linkage.Linkage]] = dict()
    zones: Optional[dict[str, linkage.ReserveToZone]] = dict()

    #######################################
    # Unserved Reserve Penalty #
    #######################################
    penalty_unserved_reserve: float = Field(
        10000,
        description="[RESOLVE Only]. $/MW. float. Modeled penalty for unserved operating reserves.",
        units=get_units("penalty_unserved_reserve"),
    )  # $10,000 / MW

    def revalidate(self):
        """Warn that operating reserve % of gross load will override any ``requirement`` timeseries."""
        if self.zones:
            for zone, l in self.zones.items():
                if l.incremental_requirement_hourly_scalar and self.requirement:
                    logger.warning(
                        f"For {self.name}: Operating reserve requirement as percentage of {zone} gross load (from {l.__class__.__name__} linkage) will override `requirement` timeseries attribute."
                    )


if __name__ == "__main__":
    r = Reserve(name="load following")
    # print(r)

    r3 = Reserve.from_dir(dir_str.data_dir / "interim" / "reserves")
    print(r3)
