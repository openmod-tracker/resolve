import copy
from pathlib import Path

import pandas as pd
import pytest

from new_modeling_toolkit.core.linkage import HybridStorageResourceToHybridVariableResource
from new_modeling_toolkit.core.linkage import Linkage
from new_modeling_toolkit.core.linkage import ResourceToResourceGroup
from new_modeling_toolkit.core.linkage import ResourceToZone
from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.recap.dispatch_model import DispatchModel
from new_modeling_toolkit.recap.recap_case import RecapCase
from new_modeling_toolkit.system.electric.resources import HybridStorageResource
from new_modeling_toolkit.system.electric.resources import HybridVariableResource
from tests.system.electric.resources import test_generic
from tests.system.electric.resources import test_variable


class TestHybridVariableResource(test_variable.TestVariableResource):
    _RESOURCE_CLASS = HybridVariableResource
    _RESOURCE_PATH = Path("resources/hybrid_variable/CA_Solar_Hybrid.csv")
    _RESOURCE_GROUP_PATH = Path("resource_groups/Solar.csv")
    _SYSTEM_COMPONENT_DICT_NAME = "hybrid_variable_resources"

    _RESOURCE_INIT_KWARGS = dict()

    def test_dispatch(self, make_resource_copy):
        resource = make_resource_copy()

        # Run dispatch against net load
        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[50.0, -20.0, 0, 150.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

        pd.testing.assert_series_equal(net_load, updated_net_load)

        assert not hasattr(resource, "heuristic_provide_power_MW")

    def test_dispatch_second_model_year(self, make_resource_copy):
        resource = make_resource_copy()

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[50.0, -20.0, 0, 150.0],
        )
        updated_net_load = resource.dispatch(net_load=net_load, model_year=2030)

        pd.testing.assert_series_equal(net_load, updated_net_load)

        assert not hasattr(resource, "heuristic_provide_power_MW")

    def test_dispatch_all_positive(self, make_resource_copy):
        """Test the `dispatch()` function when all net load hours are positive."""
        resource = make_resource_copy()

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[50.0, 100.0, 10.0, 300.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

        pd.testing.assert_series_equal(net_load, updated_net_load)

        assert not hasattr(resource, "heuristic_provide_power_MW")

    def test_dispatch_all_negative(self, make_resource_copy):
        """Test the dispatch() function when all net load hours are negative."""
        resource = make_resource_copy()

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-50.0, -100.0, -10.0, -300.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

        pd.testing.assert_series_equal(net_load, updated_net_load)

        assert not hasattr(resource, "heuristic_provide_power_MW")

    def test_dispatch_all_zero(self, make_resource_copy):
        """Test the `dispatch()` function when all net load hours are zero."""
        resource = make_resource_copy()

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[0.0, 0.0, 0.0, 0.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

        pd.testing.assert_series_equal(net_load, updated_net_load)

        assert not hasattr(resource, "heuristic_provide_power_MW")

    def test_variable_bounds(self, make_dispatch_model_copy):
        """Copied from TestGenericResource"""

        dispatch_model = make_dispatch_model_copy()
        block = dispatch_model.blocks["Example_Resource"]

        for index in block.power_output:
            assert block.power_output[index].lb == 0
            assert block.power_output[index].ub is None

        for index in block.power_input:
            assert block.power_input[index].lb == 0
            assert block.power_input[index].ub is None

        for index in block.provide_reserve:
            assert block.provide_reserve[index].lb == 0
            assert block.provide_reserve[index].ub is None

    def test_power_output_max_constraint(self, make_dispatch_model_copy, first_index):
        """Copied from TestGenericResource"""

        dispatch_model = make_dispatch_model_copy()

        block = dispatch_model.blocks["Example_Resource"]
        model_year, dispatch_window_id, timestamp = first_index
        block.power_output[model_year, dispatch_window_id, timestamp].fix(160.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(120.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(30.0)
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].body() == 190
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].upper() == 200
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].expr()

        block.power_output[model_year, dispatch_window_id, timestamp].fix(600.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(50.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(30.0)
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].body() == 630
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].upper() == 200
        assert not block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].expr()

        block.power_output[model_year, dispatch_window_id, timestamp].fix(0.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(0.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(0.0)
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].body() == 0
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].upper() == 200
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].expr()

    def test_power_output_min_constraint(self, make_dispatch_model_copy, first_index):
        """Copied from TestGenericResource"""

        dispatch_model = make_dispatch_model_copy()
        block = dispatch_model.blocks["Example_Resource"]
        model_year, dispatch_window_id, timestamp = first_index

        block.power_output[model_year, dispatch_window_id, timestamp].fix(160.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(120.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(30.0)
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].body() == 160
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].lower() == 100
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].expr()

        block.power_output[model_year, dispatch_window_id, timestamp].fix(50.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(120.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(0.0)
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].body() == 50
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].lower() == 100
        assert not block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].expr()

        block.power_output[model_year, dispatch_window_id, timestamp].fix(0.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(0.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(0.0)
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].body() == 0
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].lower() == 100
        assert not block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].expr()

    def test_power_input_max_constraint(self, make_dispatch_model_copy, first_index):
        """Copied from TestGenericResource"""

        dispatch_model = make_dispatch_model_copy()
        block = dispatch_model.blocks["Example_Resource"]
        model_year, dispatch_window_id, timestamp = first_index

        block.power_output[model_year, dispatch_window_id, timestamp].fix(120.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(160.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(30.0)
        assert block.power_input_max_constraint[model_year, dispatch_window_id, timestamp].body() == 160
        assert block.power_input_max_constraint[model_year, dispatch_window_id, timestamp].upper() == 200
        assert block.power_input_max_constraint[model_year, dispatch_window_id, timestamp].expr()

        block.power_output[model_year, dispatch_window_id, timestamp].fix(120.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(250.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(30.0)
        assert block.power_input_max_constraint[model_year, dispatch_window_id, timestamp].body() == 250
        assert block.power_input_max_constraint[model_year, dispatch_window_id, timestamp].upper() == 200
        assert not block.power_input_max_constraint[model_year, dispatch_window_id, timestamp].expr()

        block.power_output[model_year, dispatch_window_id, timestamp].fix(0.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(0.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(0.0)
        assert block.power_input_max_constraint[model_year, dispatch_window_id, timestamp].body() == 0
        assert block.power_input_max_constraint[model_year, dispatch_window_id, timestamp].upper() == 200
        assert block.power_input_max_constraint[model_year, dispatch_window_id, timestamp].expr()


class TestHybridStorageResource(test_generic.TestGenericResource):
    _RESOURCE_CLASS = HybridStorageResource
    _RESOURCE_PATH = Path("resources/hybrid_storage/Hybrid_Battery_Storage.csv")
    _RESOURCE_GROUP_PATH = Path("resource_groups/HybridBatteryStorage.csv")
    _SYSTEM_COMPONENT_DICT_NAME = "hybrid_storage_resources"

    _RESOURCE_INIT_KWARGS = dict(
        duration=ts.NumericTimeseries(
            name="storage_duration",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[4.0, 4.0],
                name="value",
            ),
        ),
        charging_efficiency=ts.FractionalTimeseries(
            name="charging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[0.85, 0.90],
                name="value",
            ),
        ),
        discharging_efficiency=ts.FractionalTimeseries(
            name="discharging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[0.75, 0.80],
                name="value",
            ),
        ),
        # power_input_max and power_input_min inherited from generic resource tests
        # are not representative of a true storage resource. However, methods related to power_input_max
        # and power_output_min were not re-written in the Storage class, so attributes were kept as-is
    )

    @pytest.fixture(scope="class")
    # Fixture passed to make_resource_copy which is then used by tests directly or
    # to generate the hybrid_storage_resource fixtures passed to dispatch tests
    def resources_and_linkage(self, make_resource_copy, hybrid_solar_resource_1):
        resource = make_resource_copy()
        linkage_1 = HybridStorageResourceToHybridVariableResource(
            name=("Example_Resource", "CA_Solar_Hybrid"),
            instance_from=resource,
            instance_to=hybrid_solar_resource_1,
            interconnection_limit_mw=ts.NumericTimeseries(
                name="interconnection_limit_mw",
                data=pd.Series(
                    index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                    data=[20.0, 40.0],
                    name="value",
                ),
            ),
            grid_charging_allowed=True,
        )
        return resource, hybrid_solar_resource_1, linkage_1

    # Similar to fixtures in conftest.py and test_generic.py
    # But those are only for a single resource (plus thermal)
    # This allows for a hybrid storage and hybrid variable resource to be set up
    @pytest.fixture(scope="class")
    def hybrid_resource_dispatch_model_generator(
        self,
        recap_case_generator,
        thermal_resource,
        thermal_resource_group,
        thermal_resource_to_zone_linkage,
        thermal_resource_to_resource_group_linkage,
        caiso_zone,
        resources_and_linkage,
        hybrid_storage_resource_group,
        variable_resource_group,
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

            def _copy_and_update_case(self, grid_charging_allowed: bool = True) -> RecapCase:
                resource, hybrid_solar_resource_1, hybrid_linkage = resources_and_linkage
                hybrid_linkage.grid_charging_allowed = grid_charging_allowed
                system = copy.deepcopy(self.recap_case.system)
                system.hybrid_storage_resources[resource.name] = resource
                system.hybrid_variable_resources[hybrid_solar_resource_1.name] = hybrid_solar_resource_1
                system.resource_groups[hybrid_storage_resource_group.name] = hybrid_storage_resource_group
                system.resource_groups[variable_resource_group.name] = variable_resource_group

                system.linkages["ResourceToZone"].append(
                    ResourceToZone(
                        name=(resource.name, "CAISO"),
                        instance_from=resource,
                        instance_to=caiso_zone,
                    )
                )
                system.linkages["ResourceToZone"].append(
                    ResourceToZone(
                        name=(hybrid_solar_resource_1.name, "CAISO"),
                        instance_from=hybrid_solar_resource_1,
                        instance_to=caiso_zone,
                    )
                )
                system.linkages["HybridStorageResourceToHybridVariableResource"] = hybrid_linkage
                system.linkages["ResourceToResourceGroup"].append(
                    ResourceToResourceGroup(
                        name=(resource.name, "HybridBatteryStorage"),
                        instance_from=resource,
                        instance_to=hybrid_storage_resource_group,
                    )
                )
                system.linkages["ResourceToResourceGroup"].append(
                    ResourceToResourceGroup(
                        name=(hybrid_solar_resource_1.name, "Solar"),
                        instance_from=hybrid_solar_resource_1,
                        instance_to=variable_resource_group,
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

            def get(self, perfect_capacity: float, grid_charging_allowed: bool = True):
                recap_case = self._copy_and_update_case(grid_charging_allowed=grid_charging_allowed)
                recap_case.setup_monte_carlo_draws()
                recap_case.monte_carlo_draws["MC_draw_0"].heuristic_dispatch(perfect_capacity=perfect_capacity)
                recap_case.monte_carlo_draws["MC_draw_0"].compress(
                    perfect_capacity=perfect_capacity,
                    heuristic_net_load_subclasses=[
                        "thermal_resources",
                        "variable_resources",
                        "hydro_resources",
                        "hybrid_storage_resources",
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

    @pytest.fixture(scope="class")
    def make_dispatch_model_copy(self, hybrid_resource_dispatch_model_generator):
        """Factory fixture for generating a copy of a dispatch model containing one thermal resource and the hybrid storage and hybrid variable resources
        being tested.
        """
        dispatch_model = hybrid_resource_dispatch_model_generator.get(perfect_capacity=0)

        def _dispatch_model_factory():
            return copy.deepcopy(dispatch_model)

        return _dispatch_model_factory

    @pytest.fixture(scope="class")
    def make_dispatch_model_copy_no_grid_charging(self, hybrid_resource_dispatch_model_generator):
        """Factory fixture for generating a copy of a dispatch model containing one thermal resource and the hybrid storage and hybrid variable resources
        being tested.
        """
        dispatch_model = hybrid_resource_dispatch_model_generator.get(perfect_capacity=0, grid_charging_allowed=False)

        def _dispatch_model_factory():
            return copy.deepcopy(dispatch_model)

        return _dispatch_model_factory

    @pytest.fixture(scope="class")
    def hybrid_storage_resource_no_paired_var_6_timestamps(self, make_dispatch_model_copy):
        storage_dispatch_model = make_dispatch_model_copy()
        resource = storage_dispatch_model.system.resources["Example_Resource"]

        resource.duration = ts.NumericTimeseries(
            name="duration", data=pd.Series(index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]), data=4.0)
        )
        resource.capacity_planned = ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]),
                data=[200.0, 400.0],
            ),
        )

        resource.paired_variable_resource.capacity_planned = ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]),
                data=0,
            ),
        )

        resource.paired_variable_resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                        "2010-01-01 04:00",
                        "2010-01-01 05:00",
                        "2010-01-01 06:00",
                    ]
                ),
                data=1,
            ),
        )

        resource.hybrid_linkage.grid_charging_allowed = True

        resource.hybrid_linkage.interconnection_limit_mw = ts.NumericTimeseries(
            name="interconnection_limit",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]),
                data=800,
            ),
        )
        resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                        "2010-01-01 04:00",
                        "2010-01-01 05:00",
                        "2010-01-01 06:00",
                    ]
                ),
                data=1.0,
            ),
        )
        resource.power_input_max = ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                        "2010-01-01 04:00",
                        "2010-01-01 05:00",
                        "2010-01-01 06:00",
                    ]
                ),
                data=1.0,
            ),
        )
        resource.outage_profile = ts.FractionalTimeseries(
            name="outage_profile",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                        "2010-01-01 04:00",
                        "2010-01-01 05:00",
                        "2010-01-01 06:00",
                    ]
                ),
                data=1.0,
            ),
        )

        resource.clear_calculated_properties()
        resource.paired_variable_resource.clear_calculated_properties()

        def _make_resource_copy():
            return copy.deepcopy(resource)

        return _make_resource_copy

    @pytest.fixture(scope="class")
    def hybrid_storage_resource_4_timestamps(self, make_dispatch_model_copy):
        storage_dispatch_model = make_dispatch_model_copy()
        resource = storage_dispatch_model.system.resources["Example_Resource"]

        resource.duration = ts.NumericTimeseries(
            name="duration", data=pd.Series(index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]), data=4.0)
        )
        resource.capacity_planned = ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]),
                data=[200.0, 400.0],
            ),
        )

        resource.paired_variable_resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                    ]
                ),
                data=1,
            ),
        )

        resource.hybrid_linkage.grid_charging_allowed = True

        resource.hybrid_linkage.interconnection_limit_mw = ts.NumericTimeseries(
            name="interconnection_limit",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]),
                data=800,
            ),
        )
        resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                    ]
                ),
                data=1.0,
            ),
        )
        resource.power_input_max = ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                    ]
                ),
                data=1.0,
            ),
        )
        resource.outage_profile = ts.FractionalTimeseries(
            name="outage_profile",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                    ]
                ),
                data=1.0,
            ),
        )

        resource.clear_calculated_properties()
        resource.paired_variable_resource.clear_calculated_properties()

        def _make_resource_copy():
            return copy.deepcopy(resource)

        return _make_resource_copy

    @pytest.fixture(scope="class")
    def hybrid_storage_resource_no_paired_var_4_timestamps(self, make_dispatch_model_copy):
        storage_dispatch_model = make_dispatch_model_copy()
        resource = storage_dispatch_model.system.resources["Example_Resource"]

        resource.duration = ts.NumericTimeseries(
            name="duration", data=pd.Series(index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]), data=4.0)
        )
        resource.capacity_planned = ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]),
                data=[200.0, 400.0],
            ),
        )

        resource.paired_variable_resource.capacity_planned = ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]),
                data=0,
            ),
        )

        resource.paired_variable_resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                    ]
                ),
                data=1,
            ),
        )

        resource.hybrid_linkage.grid_charging_allowed = True

        resource.hybrid_linkage.interconnection_limit_mw = ts.NumericTimeseries(
            name="interconnection_limit",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]),
                data=800,
            ),
        )
        resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                    ]
                ),
                data=1.0,
            ),
        )
        resource.power_input_max = ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                    ]
                ),
                data=1.0,
            ),
        )
        resource.outage_profile = ts.FractionalTimeseries(
            name="outage_profile",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                    ]
                ),
                data=1.0,
            ),
        )

        resource.clear_calculated_properties()
        resource.paired_variable_resource.clear_calculated_properties()

        def _make_resource_copy():
            return copy.deepcopy(resource)

        return _make_resource_copy

    @pytest.fixture(scope="class")
    def hybrid_storage_resource_zero_capacity_4_timestamps(self, make_dispatch_model_copy):
        storage_dispatch_model = make_dispatch_model_copy()
        resource = storage_dispatch_model.system.resources["Example_Resource"]

        resource.capacity_planned = ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2030-01-01"]),
                data=0.0,
            ),
        )

        resource.outage_profile = ts.FractionalTimeseries(
            name="outage_profile",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=1.0,
            ),
        )

        resource.paired_variable_resource.capacity_planned = ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2030-01-01"]),
                data=400.0,
            ),
        )

        resource.paired_variable_resource.power_output_max = ts.FractionalTimeseries(
            name="scaled_pmax_paired_variable",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[1.0, 0.5, 0, 1.0],
            ),
        )

        resource.hybrid_linkage.grid_charging_allowed = True

        resource.hybrid_linkage.interconnection_limit_mw = ts.NumericTimeseries(
            name="interconnection_limit",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]),
                data=300.0,
            ),
        )

        resource.clear_calculated_properties()
        resource.paired_variable_resource.clear_calculated_properties()

        def _make_resource_copy():
            return copy.deepcopy(resource)

        return _make_resource_copy

    @pytest.fixture(scope="class")
    def first_index(self, make_dispatch_model_copy):
        """Fixture that returns an example index for the dispatch model, for use in testing."""
        model = make_dispatch_model_copy()
        modeled_year = 2030
        dispatch_window, timestamp = model.DISPATCH_WINDOWS_AND_TIMESTAMPS.first()

        return modeled_year, dispatch_window, timestamp

    # Hybrid unique constraint

    def test_interconnection_limit(self, first_index, make_dispatch_model_copy):
        modeled_year, dispatch_window, timestamp = first_index
        storage_dispatch_model = make_dispatch_model_copy()
        storage_resource = storage_dispatch_model.system.resources["Example_Resource"]
        resource_block = storage_dispatch_model.blocks[storage_resource.name]
        paired_variable_block = storage_dispatch_model.blocks[storage_resource.paired_variable_resource.name]

        resource_block.power_output[modeled_year, dispatch_window, timestamp].fix(10)
        resource_block.power_input[modeled_year, dispatch_window, timestamp].fix(10)
        resource_block.provide_reserve[modeled_year, dispatch_window, timestamp].fix(10)
        paired_variable_block.power_output[modeled_year, dispatch_window, timestamp].fix(10)
        paired_variable_block.provide_reserve[modeled_year, dispatch_window, timestamp].fix(10)

        assert resource_block.hybrid_resource_interconnection_limit_constraint[
            modeled_year, dispatch_window, timestamp
        ].expr()

        paired_variable_block.power_output[modeled_year, dispatch_window, timestamp].fix(770)
        assert not resource_block.hybrid_resource_interconnection_limit_constraint[
            modeled_year, dispatch_window, timestamp
        ].expr()

    def test_hybrid_charging_constraint_grid_charging_allowed(self, first_index, make_dispatch_model_copy):
        modeled_year, dispatch_window, timestamp = first_index
        storage_dispatch_model = make_dispatch_model_copy()
        storage_resource = storage_dispatch_model.system.resources["Example_Resource"]
        resource_block = storage_dispatch_model.blocks[storage_resource.name]
        paired_variable_block = storage_dispatch_model.blocks[storage_resource.paired_variable_resource.name]

        # Test 1
        # Interconnection limit not violated
        resource_block.power_input[modeled_year, dispatch_window, timestamp].fix(10)
        paired_variable_block.power_output[modeled_year, dispatch_window, timestamp].fix(8)

        assert resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].upper() == 40.0
        assert resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].body() == 10 - 8
        assert resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].expr()

        # Interconnection limit violated
        resource_block.power_input[modeled_year, dispatch_window, timestamp].fix(61)
        paired_variable_block.power_output[modeled_year, dispatch_window, timestamp].fix(20)
        assert resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].upper() == 40.0
        assert resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].body() == 61 - 20
        assert not resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].expr()

    def test_hybrid_charging_constraint_grid_charging_not_allowed(
        self, first_index, make_dispatch_model_copy_no_grid_charging
    ):
        modeled_year, dispatch_window, timestamp = first_index
        storage_dispatch_model = make_dispatch_model_copy_no_grid_charging()
        storage_resource = storage_dispatch_model.system.resources["Example_Resource"]
        resource_block = storage_dispatch_model.blocks[storage_resource.name]
        paired_variable_block = storage_dispatch_model.blocks[storage_resource.paired_variable_resource.name]

        # Test 1
        # Interconnection limit not violated
        resource_block.power_input[modeled_year, dispatch_window, timestamp].fix(8)
        paired_variable_block.power_output[modeled_year, dispatch_window, timestamp].fix(10)

        assert resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].upper() == 0.0
        assert resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].body() == 8 - 10
        assert resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].expr()

        # Interconnection limit violated
        resource_block.power_input[modeled_year, dispatch_window, timestamp].fix(10)
        paired_variable_block.power_output[modeled_year, dispatch_window, timestamp].fix(5)
        assert resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].upper() == 0.0
        assert resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].body() == 10 - 5
        assert not resource_block.hybrid_charging_constraint[modeled_year, dispatch_window, timestamp].expr()

    def test_variable_bounds(self, resource_block):
        block = resource_block

        for index in block.power_output:
            assert block.power_output[index].lb == 0
            assert block.power_output[index].ub is None

        for index in block.power_input:
            assert block.power_input[index].lb == 0
            assert block.power_input[index].ub is None

        for index in block.provide_reserve:
            assert block.provide_reserve[index].lb == 0
            assert block.provide_reserve[index].ub is None

    def test_rescale(self, make_resource_copy):
        resource = make_resource_copy()

        resource.rescale(model_year=2030, capacity=100, incremental=False)
        pd.testing.assert_series_equal(
            resource.capacity_planned.data,
            pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[200.0, 100.0],
                name="value",
            ),
        )
        pd.testing.assert_series_equal(
            resource.storage_capacity_planned.data,
            pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[800.0, 400.0],
                name="value",
            ),
        )

    def test_rescale_incremental(self, make_resource_copy):
        resource: HybridStorageResource = make_resource_copy()
        resource.rescale(model_year=2020, capacity=50, incremental=True)
        pd.testing.assert_series_equal(
            resource.capacity_planned.data,
            pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[250.0, 400.0],
                name="value",
            ),
        )
        pd.testing.assert_series_equal(
            resource.storage_capacity_planned.data,
            pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[1000.0, 1600.0],
                name="value",
            ),
        )

    def test_scaled_SOC_max_profile(self, make_resource_copy):
        resource = make_resource_copy()

        expected_SOC_max_profile = {
            2020: pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00:00", "2010-01-01 01:00:00", "2010-01-01 02:00:00", "2010-01-01 03:00:00"],
                    name="timestamp",
                ),
                data=[800.0, 800.0, 800.0, 800.0],
            ),
            2030: pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00:00", "2010-01-01 01:00:00", "2010-01-01 02:00:00", "2010-01-01 03:00:00"],
                    name="timestamp",
                ),
                data=[1600.0, 1600.0, 1600.0, 1600.0],
            ),
        }

        assert resource.scaled_SOC_max_profile.keys() == expected_SOC_max_profile.keys()

        for model_year in resource.scaled_SOC_max_profile.keys():
            pd.testing.assert_series_equal(
                resource.scaled_SOC_max_profile[model_year], expected_SOC_max_profile[model_year]
            )

    def test_dispatch(self, hybrid_storage_resource_4_timestamps):
        hybrid_storage_resource = hybrid_storage_resource_4_timestamps()
        hybrid_variable_resource = hybrid_storage_resource.paired_variable_resource

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, 50.0, 600.0],
        )

        updated_net_load = hybrid_storage_resource.dispatch(net_load=net_load, model_year=2030)

        # Note: the `heuristic_provide_power_mw` for the hybrid variable resource only captures power that is sent
        #  to the grid out of the interconnection, not power that is sent to the battery for charging.
        pd.testing.assert_series_equal(
            hybrid_variable_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 50.0, 100.0],
            ),
        )

        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-300.0, -200.0, 0.0, 400.0],
            ),
        )

        # Note: the SOC includes power from the variable resource, which is not captured in its
        # `heuristic_provide_power_mw`, and the storage resource's `heuristic_provide_power_mw`, so it appears that the
        # SOC is higher than it should be based only on the storage provide power.
        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[
                    0,
                    400.0 * 0.85,
                    (400 + 300.0) * 0.85,
                    (400 + 300 + 50) * 0.85,
                ],
            ),
        )

        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-200.0, 0.0, 0.0, 100.0],
            ),
        )

    def test_dispatch_all_negative(self, hybrid_storage_resource_4_timestamps):
        hybrid_storage_resource = hybrid_storage_resource_4_timestamps()
        hybrid_variable_resource = hybrid_storage_resource.paired_variable_resource

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, -50.0, -600.0],
        )

        updated_net_load = hybrid_storage_resource.dispatch(net_load=net_load, model_year=2030)

        # Note: the `heuristic_provide_power_mw` for the hybrid variable resource only captures power that is sent
        #  to the grid out of the interconnection, not power that is sent to the battery for charging.
        pd.testing.assert_series_equal(
            hybrid_variable_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 0.0],
            ),
        )

        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-300.0, -200.0, -50.0, -300.0],
            ),
        )

        # Note: the SOC includes power from the variable resource, which is not captured in its
        # `heuristic_provide_power_mw`, and the storage resource's `heuristic_provide_power_mw`, so it appears that the
        # SOC is higher than it should be based only on the storage provide power.
        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[
                    0,
                    400.0 * 0.85,
                    (400 + 300.0) * 0.85,
                    (400 + 300 + 150) * 0.85,
                ],
            ),
        )

        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-200.0, 0.0, 0.0, -300.0],
            ),
        )

    def test_dispatch_all_positive(self, hybrid_storage_resource_4_timestamps):
        hybrid_storage_resource = hybrid_storage_resource_4_timestamps()
        hybrid_variable_resource = hybrid_storage_resource.paired_variable_resource

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[500.0, 100.0, 50.0, 600.0],
        )

        updated_net_load = hybrid_storage_resource.dispatch(net_load=net_load, model_year=2030)

        # Note: the `heuristic_provide_power_mw` for the hybrid variable resource only captures power that is sent
        #  to the grid out of the interconnection, not power that is sent to the battery for charging.
        pd.testing.assert_series_equal(
            hybrid_variable_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[100.0, 100.0, 50.0, 100.0],
            ),
        )

        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 50.0 * 0.85 * 0.75],
            ),
        )

        # Note: the SOC includes power from the variable resource, which is not captured in its
        # `heuristic_provide_power_mw`, and the storage resource's `heuristic_provide_power_mw`, so it appears that the
        # SOC is higher than it should be based only on the storage provide power.
        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 50.0 * 0.85],
            ),
        )

        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[400.0, 0.0, 0.0, 600.0 - 100.0 - 50 * 0.85 * 0.75],
            ),
        )

    def test_dispatch_all_zero(self, hybrid_storage_resource_4_timestamps):
        hybrid_storage_resource = hybrid_storage_resource_4_timestamps()
        hybrid_variable_resource = hybrid_storage_resource.paired_variable_resource

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[0.0, 0.0, 0.0, 0.0],
        )

        updated_net_load = hybrid_storage_resource.dispatch(net_load=net_load, model_year=2030)

        # Note: the `heuristic_provide_power_mw` for the hybrid variable resource only captures power that is sent
        #  to the grid out of the interconnection, not power that is sent to the battery for charging.
        pd.testing.assert_series_equal(
            hybrid_variable_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 0.0],
            ),
        )

        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 0.0],
            ),
        )

        # Note: the SOC includes power from the variable resource, which is not captured in its
        # `heuristic_provide_power_mw`, and the storage resource's `heuristic_provide_power_mw`, so it appears that the
        # SOC is higher than it should be based only on the storage provide power.
        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 100.0 * 0.85, 200.0 * 0.85, 300.0 * 0.85],
            ),
        )

        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 0.0],
            ),
        )

    def test_dispatch_time_varying_charging_efficiency(self, hybrid_storage_resource_4_timestamps):
        hybrid_storage_resource = hybrid_storage_resource_4_timestamps()
        hybrid_variable_resource = hybrid_storage_resource.paired_variable_resource

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, 50.0, 600.0],
        )

        hybrid_storage_resource.charging_efficiency = ts.FractionalTimeseries(
            name="charging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.8, 0.85, 0.9, 0.95],
                name="value",
            ),
        )
        hybrid_storage_resource.discharging_efficiency = ts.FractionalTimeseries(
            name="discharging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.75, 0.75, 0.85, 0.85],
                name="value",
            ),
        )

        updated_net_load = hybrid_storage_resource.dispatch(net_load=net_load, model_year=2030)

        # Note: the `heuristic_provide_power_mw` for the hybrid variable resource only captures power that is sent
        #  to the grid out of the interconnection, not power that is sent to the battery for charging.
        pd.testing.assert_series_equal(
            hybrid_variable_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 50.0, 100.0],
            ),
        )

        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-300.0, -200.0, 0.0, 400.0],
            ),
        )

        # Note: the SOC includes power from the variable resource, which is not captured in its
        # `heuristic_provide_power_mw`, and the storage resource's `heuristic_provide_power_mw`, so it appears that the
        # SOC is higher than it should be based only on the storage provide power.
        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[
                    0,
                    400.0 * 0.8,
                    (400 * 0.8) + (300.0 * 0.85),
                    (400 * 0.8) + (300.0 * 0.85) + (50 * 0.9),
                ],
            ),
        )

        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-200.0, 0.0, 0.0, 100.0],
            ),
        )

    def test_dispatch_time_varying_discharging_efficiency(self, hybrid_storage_resource_4_timestamps):
        hybrid_storage_resource = hybrid_storage_resource_4_timestamps()
        hybrid_variable_resource = hybrid_storage_resource.paired_variable_resource

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[500.0, 100.0, 50.0, 600.0],
        )

        hybrid_storage_resource.charging_efficiency = ts.FractionalTimeseries(
            name="charging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.8, 0.85, 0.9, 0.95],
                name="value",
            ),
        )
        hybrid_storage_resource.discharging_efficiency = ts.FractionalTimeseries(
            name="discharging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.75, 0.75, 0.85, 0.85],
                name="value",
            ),
        )

        updated_net_load = hybrid_storage_resource.dispatch(net_load=net_load, model_year=2030)

        # Note: the `heuristic_provide_power_mw` for the hybrid variable resource only captures power that is sent
        #  to the grid out of the interconnection, not power that is sent to the battery for charging.
        pd.testing.assert_series_equal(
            hybrid_variable_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[100.0, 100.0, 50.0, 100.0],
            ),
        )

        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 50.0 * 0.90 * 0.85],
            ),
        )

        # Note: the SOC includes power from the variable resource, which is not captured in its
        # `heuristic_provide_power_mw`, and the storage resource's `heuristic_provide_power_mw`, so it appears that the
        # SOC is higher than it should be based only on the storage provide power.
        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 50.0 * 0.90],
            ),
        )

        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[400.0, 0.0, 0.0, 600.0 - 100.0 - 50 * 0.90 * 0.85],
            ),
        )

    def test_dispatch_second_model_year(self, hybrid_storage_resource_4_timestamps):
        hybrid_storage_resource = hybrid_storage_resource_4_timestamps()
        hybrid_variable_resource = hybrid_storage_resource.paired_variable_resource

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, 50.0, 600.0],
        )

        updated_net_load = hybrid_storage_resource.dispatch(net_load=net_load, model_year=2020)

        # Note: the `heuristic_provide_power_mw` for the hybrid variable resource only captures power that is sent
        #  to the grid out of the interconnection, not power that is sent to the battery for charging.
        pd.testing.assert_series_equal(
            hybrid_variable_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 50.0, 100.0],
            ),
        )

        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-100.0, -100.0, 0.0, 200.0],
            ),
        )

        # Note: the SOC includes power from the variable resource, which is not captured in its
        # `heuristic_provide_power_mw`, and the storage resource's `heuristic_provide_power_mw`, so it appears that the
        # SOC is higher than it should be based only on the storage provide power.
        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[
                    0,
                    200.0 * 0.85,
                    (200 + 200) * 0.85,
                    (200 + 200 + 50) * 0.85,
                ],
            ),
        )

        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-400.0, -100.0, 0.0, 300.0],
            ),
        )

    def test_dispatch_interconnection_limit(self, hybrid_storage_resource_4_timestamps):
        hybrid_storage_resource = hybrid_storage_resource_4_timestamps()
        hybrid_variable_resource = hybrid_storage_resource.paired_variable_resource
        hybrid_storage_resource.hybrid_linkage.interconnection_limit_mw = ts.NumericTimeseries(
            name="interconnection_limit_mw",
            data=pd.Series(index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"]), data=[200.0, 200.0]),
        )

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, 50.0, 600.0],
        )

        updated_net_load = hybrid_storage_resource.dispatch(net_load=net_load, model_year=2030)

        # Note: the `heuristic_provide_power_mw` for the hybrid variable resource only captures power that is sent
        #  to the grid out of the interconnection, not power that is sent to the battery for charging.
        pd.testing.assert_series_equal(
            hybrid_variable_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 50.0, 100.0],
            ),
        )

        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-200.0, -200.0, 0.0, 100.0],
            ),
        )

        # Note: the SOC includes power from the variable resource, which is not captured in its
        # `heuristic_provide_power_mw`, and the storage resource's `heuristic_provide_power_mw`, so it appears that the
        # SOC is higher than it should be based only on the storage provide power.
        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[
                    0,
                    300.0 * 0.85,
                    (300 + 300) * 0.85,
                    (300 + 300 + 50) * 0.85,
                ],
            ),
        )

        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-300.0, 0.0, 0.0, 400.0],
            ),
        )

    def test_dispatch_no_grid_charging(self, hybrid_storage_resource_4_timestamps):
        hybrid_storage_resource = hybrid_storage_resource_4_timestamps()
        hybrid_variable_resource = hybrid_storage_resource.paired_variable_resource
        hybrid_storage_resource.hybrid_linkage.grid_charging_allowed = False

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, 50.0, 600.0],
        )

        updated_net_load = hybrid_storage_resource.dispatch(net_load=net_load, model_year=2030)

        # Note: the `heuristic_provide_power_mw` for the hybrid variable resource only captures power that is sent
        #  to the grid out of the interconnection, not power that is sent to the battery for charging.
        pd.testing.assert_series_equal(
            hybrid_variable_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 50.0, 100.0],
            ),
        )

        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, (250 * 0.85 * 0.75)],
            ),
        )

        # Note: the SOC includes power from the variable resource, which is not captured in its
        # `heuristic_provide_power_mw`, and the storage resource's `heuristic_provide_power_mw`, so it appears that the
        # SOC is higher than it should be based only on the storage provide power.
        pd.testing.assert_series_equal(
            hybrid_storage_resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[
                    0,
                    100.0 * 0.85,
                    (100 + 100) * 0.85,
                    (100 + 100 + 50) * 0.85,
                ],
            ),
        )

        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-500.0, -200.0, 0.0, 600 - 100 - (250 * 0.85 * 0.75)],
            ),
        )

    def test_dispatch_no_variable(self, hybrid_storage_resource_no_paired_var_4_timestamps):
        hybrid_storage_resource = hybrid_storage_resource_no_paired_var_4_timestamps()
        resource = copy.deepcopy(hybrid_storage_resource)

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, 50.0, 600.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2030)

        ## Test Proper ##
        # battery capacity, SOC, and preventing LOL are respected (0 variable output)
        # hr 1: charge from grid s.t. battery capacity limit (-400)
        # hr 2: charge but do not cause an LOL event (-200)
        # hr 3: discharge s.t. SOC limit
        #    ((MW_in_hr_1 + MW_in_hr_2) * 0.85) * 0.75 = (300 + 200) * 0.85 * 0.75 = 382.5
        # hr 4: SOC depleted
        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-400.0, -200.0, 50.0, 0.75 * (600 * 0.85 - 50 / 0.75)],
            ),
        )
        pd.testing.assert_series_equal(
            resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0, 400 * 0.85, 600 * 0.85, 600 * 0.85 - 50.0 / 0.75],
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-500 + 400, -200 + 200, 50 - 50, 600 - 332.5],
            ),
        )

    def test_dispatch_no_storage(self, hybrid_storage_resource_zero_capacity_4_timestamps):
        hybrid_storage_resource = hybrid_storage_resource_zero_capacity_4_timestamps()
        resource = copy.deepcopy(hybrid_storage_resource)

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, 50.0, 600.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2030)

        # Test Set 1
        # battery capacity, SOC, and preventing LOL are respected (0 variable output)
        # hr 1-4: renewables go directly to grid s.t. transmission limit (300 MW)
        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 0.0],
            ),
        )

        pd.testing.assert_series_equal(
            resource.paired_variable_resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[300.0, 0.5 * 400, 0.0, 300.0],
            ),
        )

        pd.testing.assert_series_equal(
            resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 0.0],
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-500.0 - 300.0, -200.0 - 200.0, 50.0 - 0.0, 600.0 - 300.0],
            ),
        )

    def test_dispatch_pmax_imax(self, hybrid_storage_resource_4_timestamps):
        resource = hybrid_storage_resource_4_timestamps()
        resource.power_input_max = ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.1, 0.5, 1.0, 1.0],
            ),
        )
        resource.clear_calculated_properties()

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, 100.0, 600.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2030)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, -100.0, 0.0, (40 + 200) * 0.85 * 0.75],
            ),
        )
        pd.testing.assert_series_equal(
            resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 40 * 0.85, (40 + 200) * 0.85, (40 + 200) * 0.85],
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-560.0, -100.0, 0.0, 600.0 - 100.0 - (40 + 200) * 0.85 * 0.75],
            ),
        )

    def test_dispatch_no_variable_second_model_year(self, hybrid_storage_resource_no_paired_var_4_timestamps):
        resource = hybrid_storage_resource_no_paired_var_4_timestamps()

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, 50.0, 600.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-200.0, -200.0, 50.0, 200.0],
            ),
        )
        pd.testing.assert_series_equal(
            resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 170.0, 340.0, 273.33333333],
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-300.0, 0.0, 0.0, 400.0],
            ),
        )

    def test_dispatch_no_variable_all_negative(self, hybrid_storage_resource_no_paired_var_6_timestamps):
        resource = hybrid_storage_resource_no_paired_var_6_timestamps()

        net_load = pd.Series(
            index=pd.DatetimeIndex(
                [
                    "2010-01-01 00:00",
                    "2010-01-01 01:00",
                    "2010-01-01 02:00",
                    "2010-01-01 03:00",
                    "2010-01-01 04:00",
                    "2010-01-01 05:00",
                    "2010-01-01 06:00",
                ]
            ),
            data=[-500.0, -200.0, -50.0, -150.0, -300.0, -600.0, -500.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                        "2010-01-01 04:00",
                        "2010-01-01 05:00",
                        "2010-01-01 06:00",
                    ]
                ),
                data=[-200.0, -200.0, -50.0, -150.0, -200.0, -141.176471, 0.0],
            ),
        )
        pd.testing.assert_series_equal(
            resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                        "2010-01-01 04:00",
                        "2010-01-01 05:00",
                        "2010-01-01 06:00",
                    ]
                ),
                data=[0.0, 170.0, 340.0, 382.5, 510.0, 680.0, 800.0],
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2010-01-01 00:00",
                        "2010-01-01 01:00",
                        "2010-01-01 02:00",
                        "2010-01-01 03:00",
                        "2010-01-01 04:00",
                        "2010-01-01 05:00",
                        "2010-01-01 06:00",
                    ]
                ),
                data=[-300.0, 0.0, 0.0, 0.0, -100.0, -458.823529, -500.0],
            ),
        )

    def test_dispatch_no_variable_all_positive(self, hybrid_storage_resource_no_paired_var_4_timestamps):
        resource = hybrid_storage_resource_no_paired_var_4_timestamps()

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[500.0, 200.0, 50.0, 150.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2030)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 0.0],
            ),
        )
        pd.testing.assert_series_equal(
            resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 0.0],
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[500.0, 200.0, 50.0, 150.0],
            ),
        )

    def test_dispatch_no_variable_all_zero(self, hybrid_storage_resource_no_paired_var_4_timestamps):
        resource = hybrid_storage_resource_no_paired_var_4_timestamps()

        resource.paired_variable_resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=0,
            ),
        )

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[0.0, 0.0, 0.0, 0.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2030)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 0.0],
            ),
        )
        pd.testing.assert_series_equal(
            resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 0.0],
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 0.0, 0.0, 0.0],
            ),
        )

    def test_power_output_max_constraint(self, make_dispatch_model_copy, first_index):
        dispatch_model = make_dispatch_model_copy()

        block = dispatch_model.blocks["Example_Resource"]
        model_year, dispatch_window_id, timestamp = first_index

        block.power_output[model_year, dispatch_window_id, timestamp].fix(160.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(120.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(30.0)
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].body() == 190
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].upper() == 200
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].expr()

        block.power_output[model_year, dispatch_window_id, timestamp].fix(600.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(50.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(30.0)
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].body() == 630
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].upper() == 200
        assert not block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].expr()

        block.power_output[model_year, dispatch_window_id, timestamp].fix(0.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(0.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(0.0)
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].body() == 0
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].upper() == 200
        assert block.power_output_max_constraint[model_year, dispatch_window_id, timestamp].expr()

    def test_set_initial_SOC_for_optimization(self, make_resource_copy):
        resource = make_resource_copy()

        df_in = pd.DataFrame(
            index=pd.DatetimeIndex(
                [
                    "2010-01-12 00:00",
                    "2010-01-12 01:00",
                    "2010-01-12 02:00",
                    "2010-01-12 03:00",
                    "2010-06-08 00:00",
                    "2010-06-08 01:00",
                    "2010-06-08 02:00",
                    "2010-06-08 03:00",
                ]
            ),
            data={
                "window_label": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                "include": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            },
        )

        resource.heuristic_storage_SOC_mwh = pd.Series(
            index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H"), data=0.0
        )
        resource.heuristic_storage_SOC_mwh.loc["2010-01-12 01:00"] = 30.0
        resource.heuristic_storage_SOC_mwh.loc["2010-06-08 00:00"] = 50.0

        resource.set_initial_SOC_for_optimization(
            timestamps_included_in_optimization_flags=df_in.loc[:, "include"],
            window_labels=df_in.loc[:, "window_label"],
        )
        pd.testing.assert_series_equal(pd.Series(index=[1.0, 2.0], data=[30.0, 50.0]), resource.initial_storage_SOC)

    def test_state_of_charge_max_constraint(self, resource_block, first_index):
        modeled_year, dispatch_window, timestamp = first_index

        # scaled max SOC is 1600
        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp].fix(10)
        assert resource_block.state_of_charge_max_constraint[modeled_year, dispatch_window, timestamp].expr()

        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp].fix(1610)
        assert not resource_block.state_of_charge_max_constraint[modeled_year, dispatch_window, timestamp].expr()

    def test_state_of_charge_tracking(self, make_dispatch_model_copy, first_index):
        storage_dispatch_model = make_dispatch_model_copy()
        storage_resource = storage_dispatch_model.system.resources["Example_Resource"]
        resource_block = storage_dispatch_model.blocks[storage_resource.name]

        # test first time step
        modeled_year, dispatch_window, timestamp = first_index
        initial_soc = storage_resource.initial_storage_SOC.loc[dispatch_window]
        # initial state_of_charge must be initial_soc
        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp].fix(initial_soc - 10)
        assert not resource_block.state_of_charge_tracking[modeled_year, dispatch_window, timestamp].expr()
        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp].fix(initial_soc)
        assert resource_block.state_of_charge_tracking[modeled_year, dispatch_window, timestamp].expr()

        # conservation of SOC
        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp].fix(initial_soc)
        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=1)].fix(
            initial_soc + 5
        )
        resource_block.power_input[modeled_year, dispatch_window, timestamp].fix(2)
        resource_block.power_output[modeled_year, dispatch_window, timestamp].fix(3)
        assert not resource_block.state_of_charge_tracking[
            modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=1)
        ].expr()

        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=1)].fix(
            initial_soc - 2.3
        )
        assert resource_block.state_of_charge_tracking[
            modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=1)
        ].expr()

        # When constraint is satisfied - the difference in the constraint is zero
        assert (
            resource_block.state_of_charge_tracking[
                modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=1)
            ].body()
            == 0
        )

    def test_state_of_charge_operating_reserve_up_max(self, make_dispatch_model_copy, first_index):
        storage_dispatch_model = make_dispatch_model_copy()
        storage_resource = storage_dispatch_model.system.resources["Example_Resource"]
        resource_block = storage_dispatch_model.blocks[storage_resource.name]

        modeled_year, dispatch_window, timestamp = first_index
        storage_dispatch_model = make_dispatch_model_copy()
        storage_resource = storage_dispatch_model.system.resources["Example_Resource"]

        resource_block.power_input[modeled_year, dispatch_window, timestamp].fix(0)
        resource_block.power_output[modeled_year, dispatch_window, timestamp].fix(0)

        # Test 1a: Basic : Should not be able to provide more reserve than SOC * discharge_efficiency
        resource_block.provide_reserve[modeled_year, dispatch_window, timestamp].fix(1200)
        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp].fix(1300)
        assert not resource_block.state_of_charge_operating_reserve_up_max[
            modeled_year, dispatch_window, timestamp
        ].expr()

        # Test 1b Basic : Reserve / discharge_eff - SOC <= 0 (LHS should return 225)
        assert (
            resource_block.state_of_charge_operating_reserve_up_max[modeled_year, dispatch_window, timestamp].body()
            == 225
        )

        # Test 2a: Reserve / discharging_eff <= SOC + power_input - power_output / discharging_efficiency
        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp].fix(1800)
        resource_block.power_output[modeled_year, dispatch_window, timestamp].fix(200)
        resource_block.provide_reserve[modeled_year, dispatch_window, timestamp].fix(1200)

        assert not resource_block.state_of_charge_operating_reserve_up_max[
            modeled_year, dispatch_window, timestamp
        ].expr()

        # Test 2b
        resource_block.provide_reserve[modeled_year, dispatch_window, timestamp].fix(1100)
        assert resource_block.state_of_charge_operating_reserve_up_max[modeled_year, dispatch_window, timestamp].expr()

        # Test 3 : Reserves  <= Power_Input + SOC * discharge_efficiency
        # No charging_efficiency required (see Storage formulation)
        resource_block.power_output[modeled_year, dispatch_window, timestamp].fix(0)
        resource_block.power_input[modeled_year, dispatch_window, timestamp].fix(100)
        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp].fix(1000)
        resource_block.provide_reserve[modeled_year, dispatch_window, timestamp].fix(1500)

        assert not resource_block.state_of_charge_operating_reserve_up_max[
            modeled_year, dispatch_window, timestamp
        ].expr()

        resource_block.provide_reserve[modeled_year, dispatch_window, timestamp].fix(
            100 + storage_resource.discharging_efficiency.data.at[timestamp] * 1000
        )

        assert (
            resource_block.state_of_charge_operating_reserve_up_max[modeled_year, dispatch_window, timestamp].body()
            == 0
        )

    def test_power_output_min_constraint(self, make_dispatch_model_copy, first_index):
        dispatch_model = make_dispatch_model_copy()
        block = dispatch_model.blocks["Example_Resource"]
        model_year, dispatch_window_id, timestamp = first_index

        block.power_output[model_year, dispatch_window_id, timestamp].fix(160.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(120.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(30.0)
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].body() == 160
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].lower() == 100
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].expr()

        block.power_output[model_year, dispatch_window_id, timestamp].fix(50.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(120.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(0.0)
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].body() == 50
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].lower() == 100
        assert not block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].expr()

        block.power_output[model_year, dispatch_window_id, timestamp].fix(0.0)
        block.power_input[model_year, dispatch_window_id, timestamp].fix(0.0)
        block.provide_reserve[model_year, dispatch_window_id, timestamp].fix(0.0)
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].body() == 0
        assert block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].lower() == 100
        assert not block.power_output_min_constraint[model_year, dispatch_window_id, timestamp].expr()

    @pytest.mark.skip()
    def test_annual_energy_budget_constraint(self):
        pass

    @pytest.mark.skip()
    def test_monthly_energy_budget_constraint(self):
        pass

    @pytest.mark.skip()
    def test_daily_energy_budget_constraint(self):
        pass
