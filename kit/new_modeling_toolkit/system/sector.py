from typing import Dict
from typing import Optional

from new_modeling_toolkit.core import component
from new_modeling_toolkit.core.linkage import Linkage
from new_modeling_toolkit.core.three_way_linkage import ThreeWayLinkage


class Sector(component.Component):
    """This class defines a Sector object and its methods."""

    ######################
    # Mapping Attributes #
    ######################
    building_shell_subsectors: Optional[Dict[str, Linkage]] = None
    stock_rollover_subsectors: dict[str, Linkage] = {}
    energy_demand_subsectors: dict[str, Linkage] = {}
    non_energy_subsectors: dict[str, Linkage] = {}
    sector_candidate_fuel_blending: Optional[dict[tuple[str, str], ThreeWayLinkage]] = None
    negative_emissions_technologies: dict[str, Linkage] = {}
