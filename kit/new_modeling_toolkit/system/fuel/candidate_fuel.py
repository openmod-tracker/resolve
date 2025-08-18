from typing import Optional

import pandas as pd
from pydantic import condecimal
from pydantic import Field
from pydantic import model_validator

from new_modeling_toolkit import get_units
from new_modeling_toolkit.core import component
from new_modeling_toolkit.core import linkage
from new_modeling_toolkit.core import three_way_linkage
from new_modeling_toolkit.core.temporal import timeseries as ts

# todo: add validator to confirm there is at least one pollutant for candidate fuel. Only for pathways unless this is true for resolve too


class CandidateFuel(component.Component):
    """A candidate fuel is one type of fuel that can be used to meet a final fuel demand.

    Gasoline is a *final fuel*; E85 ethanol and fossil gasoline are *candidate fuels*.

    Every candidate fuel has three ways in which it can be made, which can be turned on and off via parameters
    as applicable: 1) production from fossil extraction, 2) conversion from a biomass resource, and 3) conversion from
    an electrolytic fuel production tech.

    Methods:
        from_csv: instantiate fuel objects from a csv input file

    TODO: check that either biomass production cost or commodity cost is specified for a given candidate fuel

    """

    ######################
    # Mapping Attributes #
    ######################
    biomass_resources: dict[str, linkage.Linkage] = {}
    electrolyzers: dict[str, linkage.Linkage] = {}
    fuel_storages: dict[str, linkage.Linkage] = {}
    fuel_transportations: dict[str, linkage.Linkage] = {}
    fuel_zones: dict[str, linkage.Linkage] = {}
    emission_types: dict[str, linkage.Linkage] = {}
    fuel_conversion_plants: dict[str, linkage.Linkage] = {}
    final_fuels: dict[str, linkage.Linkage] = {}
    resources: dict[str, linkage.Linkage] = {}
    policies: dict[str, linkage.Linkage] = {}
    pollutants: dict[str, linkage.Linkage] = {}
    sector_candidate_fuel_blending: dict[tuple[str, str], three_way_linkage.ThreeWayLinkage] = {}

    ######################
    # Attributes #
    ######################
    fuel_is_commodity_bool: bool = Field(
        True,
        description='Set to `False` if this fuel is an electrolytic fuel; otherwise, it will be considered a "commodity" fuel with a fixed price stream.',
    )

    # used to make sure demand for electricity is not included in fuels optimization
    fuel_is_electricity: bool = False

    apply_electrofuel_SOC: bool = Field(False, description="Track hourly electrolytical fuel storage.")

    electrofuel_parasitic_loss: Optional[condecimal(ge=0, le=1)] = Field(
        None,
        description="[For candidate fuels that are electrofuels] Hourly state of charge losses,"
        "if SOC constraints are being applied.",
        units=get_units("electrofuel_parasitic_loss"),
    )

    electrofuel_storage_limit_mmbtu: Optional[ts.NumericTimeseries] = Field(
        None,
        default_freq="YS",
        up_method="interpolate",
        down_method="annual",
        description="[For candidate fuels that are electrofuels] Storage reservoir size (mmbtu),"
        "if SOC constraints are being applied",
        units=get_units("electrofuel_storage_limit_mmbtu"),
    )

    # TODO: Rename to something else
    fuel_price_per_mmbtu: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="H", up_method="ffill", down_method="mean", units=get_units("fuel_price_per_mmbtu")
    )

    monthly_price_multiplier: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="M", up_method=None, down_method=None
    )
    annual_price: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="YS", up_method="interpolate", down_method="annual", units=get_units("annual_price")
    )

    @property
    def final_fuel_name(self):
        return list(self.final_fuels.keys())[0]

    @property
    def pollutant_list(self):
        return self._return_linkage_list("pollutants")

    @property
    def fuel_production_plants(self):
        # TODO (2/29/24)- these attributes are never None, revisit or update
        if self.electrolyzers is None and self.fuel_conversion_plants is None:
            electrofuel_plants = None
        else:
            electrofuel_plants = (self.electrolyzers or dict()) | (self.fuel_conversion_plants or dict())

        return electrofuel_plants

    @model_validator(mode="after")
    def validate_or_calculate_hourly_fuel_prices(self):
        """Hourly price stream or combination of monthly price shape + annual price shape must be passed.

        # TODO 2022-03-31: This should be rewritten/made more robust. The current implementation should work
                           in most cases but takes a brute-force approach.

        Steps to calculate the monthly price shape from the monthly shape and annual price:
            #. Interpolate & extrapolate annual price to 2000-2100 (this is currently hard-coded)
            #. Resample to monthly
            #. Map monthly price shape to all months in the 2000-2100 time horizon
            #. Multiply annual price by monthly_price_multiplier
        """
        if self.fuel_is_commodity_bool == 0:
            assert (self.fuel_price_per_mmbtu is None) and (self.annual_price is None), (
                "If fuel is not a commodity (i.e., connected to fuel production components), fuel prices should not be "
                "defined"
            )
            return self
        if self.fuel_price_per_mmbtu is not None:
            assert not any([self.monthly_price_multiplier, self.annual_price]), (
                f"For {self.name}, if `fuel_price_per_mmbtu` is provided, `monthly_price_multiplier` and "
                f"`annual_price` cannot be passed."
            )
        elif all([self.monthly_price_multiplier, self.annual_price]):
            assert self.fuel_price_per_mmbtu is None, (
                f"For {self.name}, if `monthly_price_multiplier` and `annual_price` are provided, "
                f"`fuel_price_per_mmbtu` cannot be passed"
            )
            # TODO 2022-03-31: Can just warn that one will be ignored

            # Calculate hourly price shape from two other attributes (first to interpolate annual prices, aligned with
            #   field settings, then to monthly ffill)
            df = self.annual_price.data.resample("YS").interpolate().resample("H", closed="right").ffill()

            # Multiply by monthly_price_multiplier
            temp = self.monthly_price_multiplier.data.copy(deep=True)
            temp.index = temp.index.month
            multipliers = pd.Series(df.index.month.map(temp), index=df.index)
            df = df * multipliers

            self.fuel_price_per_mmbtu = ts.NumericTimeseries(data=df, name="fuel_price_per_mmbtu")

        else:
            raise ValueError(
                f"For {self.name}, fuel price can be entered via `fuel_price_per_mmbtu` or by providing both `monthly_price_multiplier` and `annual_price`"
            )

        return self

    production_limit_mmbtu: Optional[ts.NumericTimeseries] = Field(
        None,
        default_freq="YS",
        up_method="interpolate",
        down_method="annual",
        units=get_units("production_limit_mmbtu"),
    )

    opt_candidate_fuel_production_for_final_fuel_demands: Optional[ts.NumericTimeseries] = Field(
        None,
        description="Optimized production for final fuel demands.",
        default_freq="YS",
        up_method=None,
        down_method="annual",
    )

    opt_candidate_fuel_production_from_biomass_mt: Optional[ts.NumericTimeseries] = Field(
        None,
        description="Optimized candidate fuel production from biomass in metric tons.",
        default_freq="YS",
        up_method=None,
        down_method="annual",
    )

    opt_candidate_fuel_commodity_production_mmbtu: Optional[ts.NumericTimeseries] = Field(
        None,
        description="Optimized candidate fuel production from commodity pathway in MMBTU.",
        default_freq="YS",
        up_method=None,
        down_method="annual",
    )

    def candidate_fuel_blend(self, sector: str) -> pd.Series:
        """

        Args:
            sector: str of Sector name

        Returns: pd.Series of blend_override data for the input sector name saved on the SectorCandidateFuelBlending ThreeWayLinkage

        """
        return self.sector_candidate_fuel_blending[(sector, self.final_fuel_name)].blend_override.data

    def calc_energy_demand(self, sector: str, final_fuel_energy_demand: pd.Series) -> pd.Series:
        """
        CandidateFuel energy demand = FinalFuel energy demand * candidate fuel blending %

        Args:
            sector: str of Sector name, used for filtering input data
            final_fuel_energy_demand: pd.Series of energy demand for the FinalFuel

        Returns: pd.Series of energy demand for the CandidateFuel

        """
        ed = final_fuel_energy_demand * self.candidate_fuel_blend(sector).loc[final_fuel_energy_demand.index]

        # temporarily save the candidate fuel energy demand results on the FinalFuelToCandidateFuel linkage, used for aggregating results in PW on the FinalFuel component
        # this will be overwritten since many sectors/subsectors can have the same FinalFuelToCandidateFuel linkage
        fuel_linkage = self.final_fuels[self.final_fuel_name]
        getattr(fuel_linkage, "out_energy_demand").data = ed
        return ed

    def calc_fuel_cost(self, candidate_fuel_energy_demand: pd.Series):
        """
        CandidateFuel fuel cost = CandidateFuel energy demand * candidate fuel fuel_price per mmbtu

        Args:
            candidate_fuel_energy_demand: pd.Series of calculated CandidateFuel energy demand in mmbtu

        Returns: pd.Series of CandidateFuel fuel cost in $

        """
        fuel_linkage = self.final_fuels[self.final_fuel_name]
        # temporarily save the candidate fuel cost results on the FinalFuelToCandidateFuel linkage, used for aggregating results in PW on the FinalFuel component
        # this will be overwritten since many sectors/subsectors can have the same FinalFuelToCandidateFuel linkage
        fuel_cost = (
            candidate_fuel_energy_demand * self.fuel_price_per_mmbtu.data.loc[candidate_fuel_energy_demand.index]
        )
        getattr(fuel_linkage, "out_fuel_cost").data = fuel_cost
        return fuel_cost

    @model_validator(mode="after")
    def check_fuel_price(self):
        """Check that fuel price is specified if commodity_bool is True."""
        if (self.fuel_is_commodity_bool == 1) and (self.fuel_price_per_mmbtu is None):
            raise ValueError(
                "Error in fuel {}: fuel_price_per_mmbtu must be specified if fuel_is_commodity_bool is set to 1.".format(
                    self.name
                )
            )
        else:
            return self
