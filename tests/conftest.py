import copy
from typing import Optional

import pandas as pd
import pyomo.environ as pyo
import pytest

import new_modeling_toolkit.core.temporal.timeseries as ts
from new_modeling_toolkit.core.linkage import Linkage
from new_modeling_toolkit.core.linkage import ResourceToResourceGroup
from new_modeling_toolkit.core.linkage import ResourceToZone
from new_modeling_toolkit.core.utils.pyomo_utils import convert_pyomo_object_to_dataframe
from new_modeling_toolkit.core.utils.util import DirStructure
from new_modeling_toolkit.recap.dispatch_model import DispatchModel
from new_modeling_toolkit.recap.monte_carlo_draw import MonteCarloDraw
from new_modeling_toolkit.recap.recap_case import RecapCase
from new_modeling_toolkit.recap.recap_case_settings import RecapCaseSettings
from new_modeling_toolkit.system import System
from new_modeling_toolkit.system.electric.resource_group import ResourceGroup
from new_modeling_toolkit.system.electric.resources import GenericResource
from new_modeling_toolkit.system.electric.resources import HybridVariableResource
from new_modeling_toolkit.system.electric.resources import ThermalResource
from new_modeling_toolkit.system.electric.resources import VariableResource
from new_modeling_toolkit.system.electric.zone import Zone

collect_ignore = ["resolve/test_run_opt.py"]

#########################
### Helpful functions ###
#########################

_TEST_CASE_NAME = "RECAP_test"


def return_non_zero_df(dispatch_model, resource, component):
    df = convert_pyomo_object_to_dataframe(getattr(dispatch_model.blocks[resource], component))
    df.index = df.index.get_level_values(-1)
    df = df[df.values != 0]
    return df


def print_outputs(dispatch_model):
    for component in pyo.Var, pyo.Expression, pyo.Constraint, pyo.Param:
        components_to_print = list(dispatch_model.component_objects(component, active=True))
        for cp in components_to_print:
            df = convert_pyomo_object_to_dataframe(cp)
            df.to_csv(f"{cp.name}.csv")


@pytest.fixture(scope="module")
def dir_structure():
    dir_str = DirStructure(data_folder="data-test")
    dir_str.make_recap_dir()

    return dir_str


@pytest.fixture(scope="module")
def recap_df_in(dir_structure):
    return pd.read_csv(
        dir_structure.data_raw_dir / "df_in.csv", index_col=[0, 1], infer_datetime_format=True, parse_dates=True
    )


@pytest.fixture(scope="module")
def recap_case_generator(dir_structure):
    class RecapCaseGenerator:
        def __init__(self):
            dir_structure.make_recap_dir("RECAP_test")
            self.recap_case = RecapCase.from_dir(
                case_name=_TEST_CASE_NAME,
                dir_str=dir_structure,
                gurobi_credentials=None,
            )
            self.recap_case.setup_monte_carlo_draws()

        def get(self) -> RecapCase:
            return copy.deepcopy(self.recap_case)

    return RecapCaseGenerator()


@pytest.fixture(scope="module")
def recap_case_settings(dir_structure):
    return RecapCaseSettings.from_csv(dir_structure.recap_settings_dir / _TEST_CASE_NAME / "case_settings.csv")

@pytest.fixture(scope="module")
def system_generator(recap_case_generator):
    class SystemGenerator:
        def __init__(self):
            self.system = recap_case_generator.get().system

        def get(self) -> System:
            return copy.deepcopy(self.system)

    return SystemGenerator()


@pytest.fixture(scope="module")
def monte_carlo_draw_generator(recap_case_generator):
    class MonteCarloDrawGenerator:
        def __init__(self):
            recap_case = recap_case_generator.get()
            self.monte_carlo_draw = recap_case.monte_carlo_draws["MC_draw_0"]

        def get(self) -> MonteCarloDraw:
            return copy.deepcopy(self.monte_carlo_draw)

    return MonteCarloDrawGenerator()


@pytest.fixture(scope="module")
def dispatch_model_generator(monte_carlo_draw_generator):
    class DispatchModelGenerator:
        def __init__(self):
            mc_draw = monte_carlo_draw_generator.get()
            mc_draw.heuristic_dispatch(perfect_capacity=0)
            mc_draw.compress(
                perfect_capacity=0,
                heuristic_net_load_subclasses=[
                    "thermal_resources",
                    "variable_resources",
                    "hydro_resources",
                    "storage_resources",
                    "flex_load_resources",
                    "shed_dr_resources",
                ],
            )
            self.dispatch_model = DispatchModel(monte_carlo_draw=mc_draw, perfect_capacity=0)

        def get(self):
            return copy.deepcopy(self.dispatch_model)

    return DispatchModelGenerator()


@pytest.fixture(scope="module")
def thermal_resource(dir_structure):
    _, resource = ThermalResource.from_csv(
        filename=dir_structure.data_interim_dir / "resources" / "thermal" / "CA_Thermal_1.csv", return_type=tuple
    )

    return resource


@pytest.fixture(scope="module")
def thermal_resource_group(dir_structure):
    _, group = ResourceGroup.from_csv(
        filename=dir_structure.data_interim_dir / "resource_groups" / "Thermal.csv", return_type=tuple
    )

    return group


@pytest.fixture(scope="module")
def solar_resource_1(dir_structure):
    _, resource = VariableResource.from_csv(
        filename=dir_structure.data_interim_dir / "resources" / "variable" / "Arizona_Solar.csv", return_type=tuple
    )
    return resource


@pytest.fixture(scope="module")
def solar_resource_2(dir_structure):
    _, resource = VariableResource.from_csv(
        filename=dir_structure.data_interim_dir / "resources" / "variable" / "DER_Solar.csv", return_type=tuple
    )
    return resource


@pytest.fixture(scope="module")
def hybrid_solar_resource_1(dir_structure):
    _, resource = HybridVariableResource.from_csv(
        filename=dir_structure.data_interim_dir / "resources" / "hybrid_variable" / "CA_Solar_Hybrid.csv",
        return_type=tuple,
    )
    resource.capacity_planned = ts.NumericTimeseries(
        name="capacity_planned",
        data=pd.Series(
            index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
            data=[100.0, 100.0],
            name="value",
        ),
    )

    return resource


@pytest.fixture(scope="module")
def variable_resource_group(dir_structure):
    _, group = ResourceGroup.from_csv(
        filename=dir_structure.data_interim_dir / "resource_groups" / "Solar.csv", return_type=tuple
    )

    return group


@pytest.fixture(scope="module")
def hybrid_storage_resource_group(dir_structure):
    _, group = ResourceGroup.from_csv(
        filename=dir_structure.data_interim_dir / "resource_groups" / "HybridBatteryStorage.csv", return_type=tuple
    )

    return group


@pytest.fixture(scope="module")
def caiso_zone(dir_structure):
    _, zone = Zone.from_csv(filename=dir_structure.data_interim_dir / "zones" / "CAISO.csv", return_type=tuple)

    return zone


@pytest.fixture(scope="module")
def thermal_resource_to_zone_linkage(thermal_resource, caiso_zone):
    return ResourceToZone(name=("CA_Thermal_1", "CAISO"), instance_from=thermal_resource, instance_to=caiso_zone)


@pytest.fixture(scope="module")
def thermal_resource_to_resource_group_linkage(thermal_resource, thermal_resource_group):
    return ResourceToResourceGroup(
        name=("CA_Thermal_1", "Thermal"), instance_from=thermal_resource, instance_to=thermal_resource_group
    )


@pytest.fixture(scope="module")
def simple_system_generator(
    dir_structure,
    thermal_resource,
    thermal_resource_group,
    thermal_resource_to_zone_linkage,
    thermal_resource_to_resource_group_linkage,
    caiso_zone,
):
    class SimpleSystemGenerator:
        def __init__(self):
            system = System(
                name=_TEST_CASE_NAME,
                dir_str=dir_structure,
                scenarios=[],
            )

            system.generic_assets = {}
            system.generic_resources = {}
            system.flex_load_resources = {}
            system.hydro_resources = {}
            system.shed_dr_resources = {}
            system.storage_resources = {}
            system.thermal_resources = {}
            system.thermal_uc_resources = {}
            system.tx_paths = {}
            system.variable_resources = {}
            system.elcc_surfaces = {}
            system.outage_distributions = {}
            system.reserves = {}
            system.linkages = {}
            system.three_way_linkages = {}

            system.thermal_resources = {"CA_Thermal_1": thermal_resource}
            system.resource_groups = {"Thermal": thermal_resource_group}
            system.linkages = {
                "ResourceToResourceGroup": [thermal_resource_to_resource_group_linkage],
                "ResourceToZone": [thermal_resource_to_zone_linkage],
            }
            Linkage.announce_linkage_to_instances()

            self.system = system

        def get(
            self,
            component_dict_name: Optional[str] = None,
            resource: Optional[GenericResource] = None,
            resource_group: Optional[ResourceGroup] = None,
        ) -> System:
            system = copy.deepcopy(self.system)
            if resource is not None:
                assert all(arg is not None for arg in [component_dict_name, resource])
                if component_dict_name == "thermal_resources":
                    assert type(resource) == ThermalResource
                    system.thermal_resources[resource.name] = resource
                    resource_group = system.resource_groups["Thermal"]
                else:
                    assert hasattr(system, component_dict_name)
                    assert (
                        resource_group is not None
                    ), "A resource group must be specified if the resource is not a thermal resource"
                    setattr(system, component_dict_name, {resource.name: resource})
                    system.resource_groups[resource_group.name] = resource_group

                system.linkages["ResourceToResourceGroup"].append(
                    ResourceToResourceGroup(
                        name=(resource.name, resource_group.name),
                        instance_from=resource,
                        instance_to=resource_group,
                    )
                )
                system.linkages["ResourceToZone"].append(
                    ResourceToZone(
                        name=(resource.name, "CAISO"),
                        instance_from=resource,
                        instance_to=caiso_zone,
                    )
                )
                Linkage.announce_linkage_to_instances()

            return system

    return SimpleSystemGenerator()


@pytest.fixture(scope="module")
def single_resource_dispatch_model_generator(
    recap_case_generator,
    thermal_resource,
    thermal_resource_group,
    thermal_resource_to_zone_linkage,
    thermal_resource_to_resource_group_linkage,
    caiso_zone,
):
    class SingleResourceDispatchModelGenerator:
        def __init__(self):
            recap_case = recap_case_generator.get()

            recap_case.system.generic_assets = {}
            recap_case.system.generic_resources = {}
            recap_case.system.flex_load_resources = {}
            recap_case.system.hydro_resources = {}
            recap_case.system.shed_dr_resources = {}
            recap_case.system.storage_resources = {}
            recap_case.system.hybrid_storage_resources = {}
            recap_case.system.thermal_resources = {}
            recap_case.system.thermal_uc_resources = {}
            recap_case.system.tx_paths = {}
            recap_case.system.variable_resources = {}
            recap_case.system.hybrid_variable_resources = {}
            recap_case.system.elcc_surfaces = {}
            recap_case.system.outage_distributions = {}
            recap_case.system.reserves = {}
            recap_case.system.linkages = {}
            recap_case.system.three_way_linkages = {}

            recap_case.system.thermal_resources = {"CA_Thermal_1": thermal_resource}
            recap_case.system.resource_groups = {"Thermal": thermal_resource_group}
            recap_case.system.linkages = {
                "ResourceToResourceGroup": [thermal_resource_to_resource_group_linkage],
                "ResourceToZone": [thermal_resource_to_zone_linkage],
            }
            Linkage.announce_linkage_to_instances()

            self.recap_case = recap_case

        def _copy_and_update_case(
            self, component_dict_name: str, resource: GenericResource, resource_group: Optional[ResourceGroup]
        ) -> RecapCase:
            system = copy.deepcopy(self.recap_case.system)
            if component_dict_name == "thermal_resources":
                assert type(resource) == ThermalResource
                system.thermal_resources[resource.name] = resource
                resource_group = system.resource_groups["Thermal"]
            else:
                assert hasattr(system, component_dict_name)
                setattr(system, component_dict_name, {resource.name: resource})
                system.resource_groups[resource_group.name] = resource_group

            system.linkages["ResourceToResourceGroup"].append(
                ResourceToResourceGroup(
                    name=(resource.name, resource_group.name),
                    instance_from=resource,
                    instance_to=resource_group,
                )
            )
            system.linkages["ResourceToZone"].append(
                ResourceToZone(
                    name=(resource.name, "CAISO"),
                    instance_from=resource,
                    instance_to=caiso_zone,
                )
            )
            Linkage.announce_linkage_to_instances()

            recap_case = RecapCase(
                system=system,
                dir_str=self.recap_case.dir_str,
                case_name=self.recap_case.case_name,
                case_settings=self.recap_case.case_settings,
                gurobi_credentials=None,
                monte_carlo_draws=None,
            )

            return recap_case

        def get(
            self,
            component_dict_name: str,
            resource: GenericResource,
            resource_group: Optional[ResourceGroup],
            perfect_capacity: float,
        ):
            resource.resample_ts_attributes([2030, 2030], [2010, 2010])
            recap_case = self._copy_and_update_case(
                component_dict_name=component_dict_name, resource=resource, resource_group=resource_group
            )
            recap_case.setup_monte_carlo_draws()
            recap_case.monte_carlo_draws["MC_draw_0"].heuristic_dispatch(perfect_capacity=perfect_capacity)
            recap_case.monte_carlo_draws["MC_draw_0"].compress(
                perfect_capacity=perfect_capacity,
                heuristic_net_load_subclasses=[
                    "thermal_resources",
                    "variable_resources",
                    "hydro_resources",
                    "storage_resources",
                    "flex_load_resources",
                    "shed_dr_resources",
                ],
            )
            recap_case.monte_carlo_draws["MC_draw_0"].subclasses_dispatch_order = [
                "thermal_resources",
                "variable_resources",
                "generic_resources",
                "hydro_resources",
                "hybrid_storage_resources",
                "storage_resources",
                "flex_load_resources",
                "shed_dr_resources",
            ]
            dispatch_model = DispatchModel(
                monte_carlo_draw=recap_case.monte_carlo_draws["MC_draw_0"], perfect_capacity=perfect_capacity
            )

            return dispatch_model

    return SingleResourceDispatchModelGenerator()


@pytest.fixture(scope="module")
def mini_fuels_system(dir_structure):
    """Creates an instance of the system that is associated with the class to be tested.

    This method is not intended to be called directly. Instead, use the `make_resource_group_copy()` fixture,
    which is a factory for generating a clean copy of the resource group.
    """
    group = System(
        name="TEST_fuels",
        dir_str=dir_structure,
        year_start=2020,
        year_end=2050,
        scenarios=["Test Scenario"],
        model_name="pathways",
    )

    return group


@pytest.fixture(scope="module")
def pathways_series_index():
    index = pd.DatetimeIndex(
        [
            "2031-01-01",
            "2032-01-01",
            "2033-01-01",
            "2034-01-01",
            "2035-01-01",
        ]
    )
    return index
