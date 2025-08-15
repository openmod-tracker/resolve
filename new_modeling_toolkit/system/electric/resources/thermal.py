from pydantic import Field

from new_modeling_toolkit.core.linkage import Linkage
from new_modeling_toolkit.system.electric.resources.generic import GenericResource
from new_modeling_toolkit.system.electric.resources.unit_commitment import UnitCommitmentResource


class ThermalResource(GenericResource):
    """Fuel-burning resource."""

    ############
    # Linkages #
    ############
    candidate_fuels: dict[str, Linkage] = Field(
        default_factory=dict,
        description="[RESOLVE only]. String Input. This input links a specified `candidate_fuels` to this `ThermalResource` . (e.g. Natural_Gas to gas_CCGT).",
    )


class ThermalUnitCommitmentResource(UnitCommitmentResource, ThermalResource):
    pass
