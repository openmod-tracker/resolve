from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import get_args
from typing import Union

import pandas as pd
import pint
import pydantic
from pydantic import ConfigDict
from pydantic.fields import FieldInfo


def df_encoder(df, date_format="iso"):
    """Convert pandas dataframe to JSON.

    Need to do json.loads because df.to_json() returns a string in JSON format.
    """
    # if "pint" in df.dtypes:
    #     df = df.pint.dequantify()
    # except AttributeError:
    #     # If the dataframe doesn't have pint units

    # Dropping repeated values (if we ever reload a JSON file, will need to `ffill`)
    df = df[(df.shift(1) != df)]
    return json.loads(df.to_json(date_format="iso", date_unit="s"))


class CustomModel(pydantic.BaseModel):
    """Standard pydantic BaseModel configuration."""

    name: Union[str, tuple]
    # TODO[pydantic]: The following keys were removed: `copy_on_model_validation`, `underscore_attrs_are_private`, `json_encoders`.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        populate_by_name=True,
        loc_by_alias=True,
        json_encoders={
            pd.DataFrame: lambda df: df_encoder(df),
            pd.Series: lambda df: df_encoder(df),
            pint.Unit: lambda unit: str(unit),
            # pint.Quantity: lambda quantity: str(quantity),
        },
    )

    @classmethod
    def get_field_type(cls, *, field_info: FieldInfo) -> tuple:
        """Return a tuple of a field's type(s)."""
        if nested_types := get_args(field_info.annotation):
            return nested_types
        else:
            return tuple([field_info.annotation])

    @classmethod
    def get_subclasses(cls):
        """Get all subclasses recursively."""
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __rich_repr__(self):
        """WORKAROUND for Rich Repr Protocol.

        [Rich Repr Protocol](https://rich.readthedocs.io/en/latest/pretty.html#rich-repr-protocol)
        doesn't seem to work right now due to highly recursive nature of NMT's `pydantic` model and/or some unknown
        interaction with `loguru` trying to also pretty print the error.
        """
        yield None


class FieldCategory(enum.Enum):
    BUILD = "Build Parameters"
    OPERATIONS = "Operational Parameters"
    RELIABILITY = "Reliability Parameters"
    PYOMO_VARS = "Decision Variables"


import pint

units = pint.UnitRegistry()
units.define("dollar = [currency] = \u0024")
units.define("W_year = watt / year")
units.define("unitless = []")
units.define("MMBtu = 1e6 * Btu")
units.define("month = [time]")


@pint.register_unit_format("e3")
def format_unit(unit, registry, **kwargs):
    string = ""
    if unit == units.dollar / units.kW_year or unit == units.dollar / units.kilowatt / units.year:
        return "$/kW⋅year"
    elif unit == units.dollar / units.MWh or unit == units.dollar / units.megawatt / units.hour:
        return "$/MWh"
    elif unit == units.dollar / units.kWh or unit == units.dollar / units.kilowatt / units.hour:
        return "$/kWh"

    for u, power in unit.items():
        match u:
            case units.W_year:
                u = "W⋅year"
            case units.dollar:
                u = "$"
            case units.hour:
                u = "h"
            case units.year:
                u = "year"
            case units.megawatt_hour:
                u = "MWh"
            case units.kilowatt_hour:
                u = "kWh"
            case units.megawatt:
                u = "MW"
            case units.kilowatt:
                u = "kW"
            case _:
                u = f"{u}"

        match power:
            case -1:
                string += f"/{u}"
            case _:
                string += f"⋅{u}"
    return string.lstrip("⋅")


@enum.unique
class ModelType(enum.Enum):
    PATHWAYS = "Pathways"
    RECAP = "Recap"
    RESOLVE = "Resolve"
    TEMPLATE = "Template"


@dataclass
class Metadata:
    category: None | FieldCategory = None
    units: pint.Unit = units.unitless
    excel_short_title: str = ""
    tools: None | set[ModelType] = None
    warning_bounds: None | tuple[float, float] = None
