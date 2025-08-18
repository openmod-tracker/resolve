from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
from line_profiler import LineProfiler

from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.system.electric.resources import StorageResource
from tests.system.electric.resources import test_generic


class TestStorageResource(test_generic.TestGenericResource):
    _RESOURCE_CLASS = StorageResource
    _RESOURCE_PATH = Path("resources/storage/Battery_Storage.csv")
    _RESOURCE_GROUP_PATH = Path("resource_groups/Storage.csv")
    _SYSTEM_COMPONENT_DICT_NAME = "storage_resources"

    _RESOURCE_INIT_KWARGS = dict(
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
        duration=ts.NumericTimeseries(
            name="storage_duration",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[4.0, 4.0],
                name="value",
            ),
        ),
        # power_input_max and power_input_min inherited from generic resource tests
        # are not representative of a true storage resource. However, methods related to power_input_max
        # and power_output_min were not re-written in the Storage class, so attributes were kept as-is
    )

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
        resource: StorageResource = make_resource_copy()
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

    def test_dispatch(self, make_resource_copy):
        resource = make_resource_copy()
        resource.charging_efficiency = ts.FractionalTimeseries(
            name="charging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.85, 0.85, 0.85, 0.85],
                name="value",
            ),
        )
        resource.discharging_efficiency = ts.FractionalTimeseries(
            name="discharging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.75, 0.75, 0.75, 0.75],
                name="value",
            ),
        )
        resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=1.0,
            ),
        )
        resource.power_input_max = ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=1.0,
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

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, 50.0, 600.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2030)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-400.0, -200.0, 50.0, 332.5],
            ),
        )
        pd.testing.assert_series_equal(
            resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 340.0, 510.0, 443.33333333333],
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-100.0, 0.0, 0.0, 267.5],
            ),
        )

    def test_dispatch_pmax_imax(self, make_resource_copy):
        resource = make_resource_copy()

        resource.charging_efficiency = ts.FractionalTimeseries(
            name="charging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.85, 0.85, 0.85, 0.85],
                name="value",
            ),
        )
        resource.discharging_efficiency = ts.FractionalTimeseries(
            name="discharging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.75, 0.75, 0.75, 0.75],
                name="value",
            ),
        )

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
                data=[-200.0, -0.0, 100.0, 27.5],
            ),
        )
        pd.testing.assert_series_equal(
            resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.0, 170.0, 170.0, 36.66666666666],
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-300.0, -200.0, 0.0, 572.5],
            ),
        )

    def test_dispatch_second_model_year(self, make_resource_copy):
        resource = make_resource_copy()
        resource.charging_efficiency = ts.FractionalTimeseries(
            name="charging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.85, 0.85, 0.85, 0.85],
                name="value",
            ),
        )
        resource.discharging_efficiency = ts.FractionalTimeseries(
            name="discharging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.75, 0.75, 0.75, 0.75],
                name="value",
            ),
        )
        resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=1.0,
            ),
        )
        resource.power_input_max = ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=1.0,
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
                data=[0.0, 170.0, 340.0, 273.333333333],
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

    def test_dispatch_all_negative(self, make_resource_copy):
        resource = make_resource_copy()
        resource.charging_efficiency = ts.FractionalTimeseries(
            name="charging_efficiency",
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
                data=[0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
                name="value",
            ),
        )
        resource.discharging_efficiency = ts.FractionalTimeseries(
            name="discharging_efficiency",
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
                data=[0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75],
                name="value",
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
                data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
                data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
                data=[-500.0, -200.0, -50.0, -150.0, -300.0, -600.0, -500.0],
            ),
        )

    def test_dispatch_all_positive(self, make_resource_copy):
        resource = make_resource_copy()
        resource.charging_efficiency = ts.FractionalTimeseries(
            name="charging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.85, 0.85, 0.85, 0.85],
                name="value",
            ),
        )
        resource.discharging_efficiency = ts.FractionalTimeseries(
            name="discharging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.75, 0.75, 0.75, 0.75],
                name="value",
            ),
        )
        resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=1.0,
            ),
        )
        resource.power_input_max = ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=1.0,
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

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[500.0, 200.0, 50.0, 150.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

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

    def test_dispatch_all_zero(self, make_resource_copy):
        resource = make_resource_copy()
        resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=1.0,
            ),
        )
        resource.power_input_max = ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=1.0,
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

        resource.charging_efficiency = ts.FractionalTimeseries(
            name="charging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.85, 0.85, 0.85, 0.85],
                name="value",
            ),
        )
        resource.discharging_efficiency = ts.FractionalTimeseries(
            name="discharging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.75, 0.75, 0.75, 0.75],
                name="value",
            ),
        )

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[0.0, 0.0, 0.0, 0.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

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

    def test_dispatch_time_varying_efficiency(self, make_resource_copy):
        resource = make_resource_copy()
        resource.charging_efficiency = ts.FractionalTimeseries(
            name="charging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.8, 0.85, 0.9, 0.95],
                name="value",
            ),
        )
        resource.discharging_efficiency = ts.FractionalTimeseries(
            name="discharging_efficiency",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0.75, 0.75, 0.85, 0.85],
                name="value",
            ),
        )
        resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=1.0,
            ),
        )
        resource.power_input_max = ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=1.0,
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

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-500.0, -200.0, 50.0, 600.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2030)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-400.0, -200.0, 50.0, 366.5],
            ),
        )
        pd.testing.assert_series_equal(
            resource.heuristic_storage_SOC_mwh,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[0, 320.0, 490.0, 431.17647],
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-100.0, 0.0, 0.0, 233.5],
            ),
        )

    def test_set_initial_SOC_for_optimization(self, make_resource_copy):
        resource = make_resource_copy()

        resource.charging_efficiency = ts.FractionalTimeseries(
            name="charging_efficiency",
            data=pd.Series(
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
                data=[0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
                name="value",
            ),
        )
        resource.discharging_efficiency = ts.FractionalTimeseries(
            name="discharging_efficiency",
            data=pd.Series(
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
                data=[0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75],
                name="value",
            ),
        )

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

    def test_operational_block_time(self, make_resource_copy, dispatch_model_generator):
        resource = make_resource_copy()
        resource.resample_ts_attributes([2030, 2030], [2010, 2010])
        resource.initial_storage_SOC = pd.Series(index=[float(x) for x in range(0, 8761)]).fillna(0)
        dispatch_model = dispatch_model_generator.get()
        del dispatch_model.blocks_index
        dispatch_model.blocks = pyo.Block(["Example_Resource"])
        lp = LineProfiler()
        lp.add_function(resource.construct_operational_block)

        # Profile the function
        lp_wrapper = lp(resource.construct_operational_block)
        result = lp_wrapper(dispatch_model)

        lp.print_stats()

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

        # first time step
        modeled_year, dispatch_window, timestamp = first_index
        initial_soc = storage_resource.initial_storage_SOC.loc[dispatch_window]

        # update efficiencies in second hour to be different from those in the first hour
        storage_resource.charging_efficiency.data.at[timestamp + pd.DateOffset(hour=1)] = 0.9
        storage_resource.discharging_efficiency.data.at[timestamp + pd.DateOffset(hour=1)] = 0.8

        # create resource block
        resource_block = storage_dispatch_model.blocks[storage_resource.name]

        # test first time step: initial state_of_charge must be initial_soc
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

        # conservation of SOC when charging and discharging efficiency change over time
        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=1)].fix(
            initial_soc - 2.3
        )
        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=2)].fix(
            initial_soc + 5
        )
        resource_block.power_input[modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=1)].fix(2)
        resource_block.power_output[modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=1)].fix(3)
        assert not resource_block.state_of_charge_tracking[
            modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=2)
        ].expr()

        resource_block.state_of_charge[modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=2)].fix(
            initial_soc - 4.6  # 4.25
        )
        assert resource_block.state_of_charge_tracking[
            modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=2)
        ].expr()

        # When constraint is satisfied - the difference in the constraint is zero
        assert (
            resource_block.state_of_charge_tracking[
                modeled_year, dispatch_window, timestamp + pd.DateOffset(hour=1)
            ].body()
            == 0
        )

    def test_state_of_charge_operating_reserve_up_max(self, make_dispatch_model_copy, resource_block, first_index):
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
