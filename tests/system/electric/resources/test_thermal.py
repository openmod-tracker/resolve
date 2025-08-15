import copy
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.system.electric.resources import ThermalResource
from tests.system.electric.resources import test_generic


class TestThermalResource(test_generic.TestGenericResource):
    _RESOURCE_CLASS = ThermalResource
    _RESOURCE_GROUP_PATH = Path("resource_groups/Thermal.csv")
    _SYSTEM_COMPONENT_DICT_NAME = "thermal_resources"
    _RESOURCE_INIT_KWARGS = dict(
        capacity_planned=ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[200.0, 100.0],
                name="value",
            ),
        )
    )

    @pytest.fixture(scope="class")
    def resource_for_budget_tests(self, make_resource_copy):
        resource = make_resource_copy()
        resource.capacity_planned = ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[200.0, 400.0],
                name="value",
            ),
        )
        return resource

    def test_scaled_annual_energy_budget(self, resource_for_budget_tests):
        resource = copy.deepcopy(resource_for_budget_tests)
        resource.energy_budget_annual = ts.FractionalTimeseries(
            name="energy_budget_annual",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-01-01", "2019-01-01"], name="timestamp"), data=[0.000856164, 0.001141553]
            ),
            freq_="YS",
            weather_year=True,
        )

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-01-01 12:00", freq="H", name="timestamp")
        resource.upsample(load_calendar)
        expected_energy_budget_annual = pd.Series(
            index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="YS", name="timestamp"),
            data=0.000856164,
        )
        pd.testing.assert_series_equal(
            resource.energy_budget_annual.data,
            expected_energy_budget_annual,
        )

    def test_scaled_daily_energy_budget(self, resource_for_budget_tests):
        resource = copy.deepcopy(resource_for_budget_tests)
        resource.energy_budget_daily = ts.FractionalTimeseries(
            name="energy_budget_daily",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-07-01", "2010-07-02"], name="timestamp"), data=[0.104166667, 0.114583333]
            ),
            freq_="D",
            weather_year=True,
        )

        expected_profile = {
            2020: pd.Series(
                index=pd.DatetimeIndex(["2010-07-01", "2010-07-02"], name="timestamp"),
                data=[500.0, 550.0],
            ),
            2030: pd.Series(
                index=pd.DatetimeIndex(["2010-07-01", "2010-07-02"], name="timestamp"),
                data=[1000.0, 1100.0],
            ),
        }
        assert resource.scaled_daily_energy_budget.keys() == expected_profile.keys()
        for model_year, profile in resource.scaled_daily_energy_budget.items():
            pd.testing.assert_series_equal(profile, expected_profile[model_year])

    def test_scaled_monthly_energy_budget(self, resource_for_budget_tests):
        resource = copy.deepcopy(resource_for_budget_tests)
        resource.energy_budget_monthly = ts.FractionalTimeseries(
            name="energy_budget_monthly",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-07-01", "2010-08-01"], name="timestamp"), data=[0.00672043, 0.008400538]
            ),
            freq_="MS",
            weather_year=True,
        )

        expected_profile = {
            2020: pd.Series(
                index=pd.DatetimeIndex(["2010-07-01", "2010-08-01"], name="timestamp"),
                data=[1000.0, 1250.0],
            ),
            2030: pd.Series(
                index=pd.DatetimeIndex(["2010-07-01", "2010-08-01"], name="timestamp"),
                data=[2000.0, 2500.0],
            ),
        }
        assert resource.scaled_monthly_energy_budget.keys() == expected_profile.keys()
        for model_year, profile in resource.scaled_monthly_energy_budget.items():
            pd.testing.assert_series_equal(profile, expected_profile[model_year])

    def test_rescale_incremental(self, make_resource_copy):
        """Test the `rescale()` function with `incremental=True`."""
        resource = make_resource_copy()

        resource.rescale(model_year=2020, capacity=100, incremental=True)
        pd.testing.assert_series_equal(
            resource.capacity_planned.data,
            pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[300.0, 100.0],
                name="value",
            ),
        )

    def test_scaled_pmax_profile(self, make_resource_copy):
        resource = make_resource_copy()

        expected_profile = {
            2020: pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"], name="timestamp"
                ),
                data=[200.0, 0.0, 50.0, 100.0],
                name="value",
            ),
            2030: pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"], name="timestamp"
                ),
                data=[100.0, 0.0, 25.0, 50.0],
                name="value",
            ),
        }

        assert resource.scaled_pmax_profile.keys() == expected_profile.keys()
        for model_year, profile in resource.scaled_pmax_profile.items():
            pd.testing.assert_series_equal(profile, expected_profile[model_year])

    def test_dispatch_second_model_year(self, make_resource_copy):
        resource = make_resource_copy()

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[50.0, -20.0, 0, 150.0],
        )
        updated_net_load = resource.dispatch(net_load=net_load, model_year=2030)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"], name="timestamp"
                ),
                data=[100.0, 0.0, 25.0, 50.0],
                name="value",
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-50.0, -20.0, -25.0, 100.0],
            ),
        )

    def test_construct_operational_block(self, make_dispatch_model_copy, monkeypatch):
        construct_operational_block_mock = mock.Mock()
        monkeypatch.setattr(ThermalResource, "construct_operational_block", construct_operational_block_mock)
        model = make_dispatch_model_copy()
        assert len(model.blocks) == 0
        construct_operational_block_mock.assert_not_called()

    def test_variable_bounds(self):
        pass

    def test_power_output_max_constraint(self):
        pass

    def test_power_output_min_constraint(self):
        pass

    def test_power_input_max_constraint(self):
        pass

    def test_adjust_budgets_for_optimization(self):
        pass

    def test_annual_energy_budget_constraint(self):
        pass

    def test_daily_energy_budget_constraint(self):
        pass

    def test_monthly_energy_budget_constraint(self):
        pass
