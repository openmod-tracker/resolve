import pandas as pd
from typing_extensions import Literal

from new_modeling_toolkit.core import component
from new_modeling_toolkit.core import dir_str
from new_modeling_toolkit.core import linkage


# add to kit
class Pollutant(component.Component):
    """
    ADD DESCRIPTION
    """

    ######################
    # Mapping Attributes #
    ######################

    non_energy_subsectors: dict[str, linkage.Linkage] = {}
    negative_emissions_technologies: dict[str, linkage.Linkage] = {}
    resources: dict[str, linkage.ResourceToPollutant] = {}
    policies: dict[str, linkage.Linkage] = {}
    tx_paths: dict[str, linkage.TransmissionPathToPollutant] = {}
    candidate_fuels: dict[str, linkage.CandidateFuelToPollutant] = {}
    ccs_plants: dict[str, linkage.Linkage] = {}

    ######################
    # Attributes #
    ######################

    GWP: float

    def _calc_emissions(
        self,
        emissions_type: Literal["net", "gross", "upstream"],
        linkage: linkage.CandidateFuelToPollutant,
        energy_demand: pd.Series,
    ) -> [pd.Series, pd.Series]:
        """
        Calculate net, gross, or upstream emissions for the pollutant type and also in CO2e
        Args:
            emissions_type: str. 'net','gross', or 'upstream' indicating which emission type
            linkage: Linkage object for CandidateFuelToPollutant
            energy_demand: pd.Series of energy demand for the linked CandidateFuel

        Returns: tuple(pd.Series of emissions, pd.Series of emissions in CO2e)

        """
        emissions_trajectory_override = getattr(linkage, f"{emissions_type}_emissions_trajectory_override")
        if emissions_trajectory_override:
            # emissions calculated in pathways_system.export_results()
            emissions = energy_demand * 0
            emissions_CO2e = energy_demand * 0
        else:
            emission_factor = getattr(linkage, f"{emissions_type}_emission_factor").data
            emissions = emission_factor.mul(energy_demand, axis=0)
            emissions_CO2e = emissions * self.GWP
        # save to linkage for later access
        getattr(linkage, f"out_{emissions_type}_emissions").data = emissions
        getattr(linkage, f"out_{emissions_type}_emissions_CO2e").data = emissions_CO2e
        return emissions, emissions_CO2e

    def calc_emissions(self, linkage: linkage.CandidateFuelToPollutant, energy_demand: pd.Series) -> None:
        """
        Call functions to calculate net, gross, and upstream emissions for pollutant type.

        Args:
            linkage: Linkage object for CandidateFuelToPollutant
            energy_demand: pd.Series of energy demand for the linked CandidateFuel

        Returns: Emissions results are saved on the linkage directly for later access/aggregation

        """
        self.calc_net_emissions(linkage, energy_demand)
        self.calc_gross_emissions(linkage, energy_demand)
        self.calc_upstream_emissions(linkage, energy_demand)

    def calc_net_emissions(
        self, linkage: linkage.CandidateFuelToPollutant, energy_demand: pd.Series
    ) -> [pd.Series, pd.Series]:
        """
        Calculate net emissions

        Args:
            linkage: Linkage object for CandidateFuelToPollutant
            energy_demand: pd.Series of energy demand for the linked CandidateFuel

        Returns: tuple(pd.Series of emissions, pd.Series of emissions in CO2e)

        """
        return self._calc_emissions("net", linkage, energy_demand)

    def calc_gross_emissions(
        self, linkage: linkage.CandidateFuelToPollutant, energy_demand: pd.Series
    ) -> [pd.Series, pd.Series]:
        """
        Calculate gross emissions

        Args:
            linkage: Linkage object for CandidateFuelToPollutant
            energy_demand: pd.Series of energy demand for the linked CandidateFuel

        Returns: tuple(pd.Series of emissions, pd.Series of emissions in CO2e)

        """
        return self._calc_emissions("gross", linkage, energy_demand)

    def calc_upstream_emissions(
        self, linkage: linkage.CandidateFuelToPollutant, energy_demand: pd.Series
    ) -> [pd.Series, pd.Series]:
        """
        Calculate upstream emissions

        Args:
            linkage: Linkage object for CandidateFuelToPollutant
            energy_demand: pd.Series of energy demand for the linked CandidateFuel

        Returns: tuple(pd.Series of emissions, pd.Series of emissions in CO2e)

        """
        return self._calc_emissions("upstream", linkage, energy_demand)


if __name__ == "__main__":
    # path to data folder
    data_path = dir_str.data_dir / "interim" / "pollutants" / "CH4 (CO2e)"
    pollutants = Pollutant.from_dir(data_path)
    print("pollutant testing complete!")
