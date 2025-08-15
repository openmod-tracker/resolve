import copy
from pathlib import Path

import pandas as pd
import pytest
from pandas import Timestamp

from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.system.electric.resources import ShedDrResource
from tests.system.electric.resources import test_unit_commitment


class TestShedDrResource(test_unit_commitment.TestUnitCommitmentResource):
    _RESOURCE_CLASS = ShedDrResource
    _RESOURCE_GROUP_PATH = Path("resource_groups/Generic.csv")
    _SYSTEM_COMPONENT_DICT_NAME = "generic_resources"

    _RESOURCE_INIT_KWARGS = dict(
        name="Example_Resource",
        max_annual_calls=ts.NumericTimeseries(
            name="max_annual_calls",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[5, 5],
                name="value",
            ),
            freq_="YS",
        ),
        max_monthly_calls=ts.NumericTimeseries(
            name="max_monthly_calls",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[1, 1],
                name="value",
            ),
        ),
        max_daily_calls=ts.NumericTimeseries(
            name="max_daily_calls",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[1, 1],
                name="value",
            ),
        ),
    )

    @pytest.fixture(scope="class")
    def resource_for_dispatch_tests(self, make_resource_copy):
        resource = make_resource_copy()
        resource.capacity_planned = ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[10.0, 10.0],
                name="value",
            ),
        )

        resource.unit_size = ts.NumericTimeseries(
            name="unit_size",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[10.0, 10.0],
                name="value",
            ),
        )

        resource.power_output_max = ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-12-31 23:00"], name="timestamp"),
                data=[1.0, 1.0],
                name="value",
            ),
        )
        resource.power_output_min = ts.FractionalTimeseries(
            name="power_output_min",
            data=pd.Series(index=pd.DatetimeIndex(["2010-01-01 00:00"], name="timestamp"), data=0, name="value"),
        )
        resource.power_input_max = ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(index=pd.DatetimeIndex(["2010-01-01 00:00"], name="timestamp"), data=0, name="value"),
        )
        resource.outage_profile = ts.FractionalTimeseries(
            name="outage_profile",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-12-31 23:00"], name="timestamp"
                ),
                data=[1, 1, 1, 1],
                name="value",
            ),
        )

        resource.upsample(pd.DatetimeIndex(["2030-01-01", "2030-12-31"]))
        return resource

    @pytest.fixture(scope="class")
    def test_net_load_for_dispatch(self):
        net_load = pd.Series(
            index=pd.DatetimeIndex(
                [
                    "2030-01-01 00:00",
                    "2030-06-21 00:00",
                    "2030-06-21 10:00",
                    "2030-06-21 15:00",
                    "2030-06-21 22:00",
                    "2030-06-21 23:00",
                    "2030-09-21 00:00",
                    "2030-09-21 10:00",
                    "2030-09-21 15:00",
                    "2030-09-21 22:00",
                    "2030-09-21 23:00",
                    "2030-06-22 00:00",
                    "2030-06-22 10:00",
                    "2030-06-22 15:00",
                    "2030-06-22 22:00",
                    "2030-06-22 23:00",
                    "2030-10-21 00:00",
                    "2030-10-21 10:00",
                    "2030-10-21 15:00",
                    "2030-10-21 22:00",
                    "2030-10-21 23:00",
                    "2030-08-21 00:00",
                    "2030-08-21 10:00",
                    "2030-08-21 15:00",
                    "2030-08-21 22:00",
                    "2030-08-21 23:00",
                    "2030-07-21 00:00",
                    "2030-07-21 10:00",
                    "2030-07-21 15:00",
                    "2030-07-21 22:00",
                    "2030-07-21 23:00",
                    "2030-10-22 00:00",
                    "2030-12-31 23:00",
                ],
                name="timestamp",
            ),
            data=[
                0,
                -15,
                -25,
                25,
                -25,
                -15,
                -15,
                -25,
                25,
                -25,
                -15,
                -15,
                -25,
                25,
                -25,
                -15,
                -15,
                -25,
                25,
                -25,
                -15,
                -15,
                -25,
                25,
                -25,
                -15,
                -15,
                -25,
                25,
                -25,
                -15,
                0,
                0,
            ],
            name="value",
        )
        net_load = net_load.resample("H").ffill()
        return net_load

    def test_dispatch(self, resource_for_dispatch_tests, test_net_load_for_dispatch):
        net_load = test_net_load_for_dispatch
        resource = copy.deepcopy(resource_for_dispatch_tests)

        # test with all daily, monthly, and annual calls
        resource.max_daily_calls.data.loc[:] = 1
        resource.max_monthly_calls.data.loc[:] = 2
        resource.max_annual_calls.data.loc[:] = 5
        resource.dispatch(net_load, 2030)
        new_ts = resource.heuristic_provide_power_mw
        assert new_ts.groupby(new_ts.index.date).sum().max() == 40
        assert new_ts.max() == 10
        assert new_ts.sum() == 200

    def test_dispatch_monthly_annual(self, resource_for_dispatch_tests, test_net_load_for_dispatch):
        net_load = test_net_load_for_dispatch
        resource = copy.deepcopy(resource_for_dispatch_tests)

        # test with all daily, monthly, and annual calls
        resource.validate_assignment = False
        resource.max_daily_calls = None
        resource.max_monthly_calls.data.loc[:] = 2
        resource.max_annual_calls.data.loc[:] = 5
        resource.dispatch(net_load, 2030)
        new_ts = resource.heuristic_provide_power_mw
        assert new_ts.groupby(new_ts.index.date).sum().max() == 40
        assert new_ts.max() == 10
        assert new_ts.sum() == 200

    def test_dispatch_daily_annual(self, resource_for_dispatch_tests, test_net_load_for_dispatch):
        net_load = test_net_load_for_dispatch
        resource = copy.deepcopy(resource_for_dispatch_tests)

        # test with all daily and annual calls
        resource.max_daily_calls.data.loc[:] = 1
        resource.max_monthly_calls = None
        resource.max_annual_calls.data.loc[:] = 5
        resource.dispatch(net_load, 2030)
        new_ts = resource.heuristic_provide_power_mw
        assert new_ts.groupby(new_ts.index.date).sum().max() == 40
        assert new_ts.max() == 10
        assert new_ts.sum() == 200

    def test_dispatch_daily_monthly(self, resource_for_dispatch_tests, test_net_load_for_dispatch):
        net_load = test_net_load_for_dispatch
        resource = copy.deepcopy(resource_for_dispatch_tests)
        # test with all daily, monthly, and annual calls
        resource.max_daily_calls.data.loc[:] = 1
        resource.max_monthly_calls.data.loc[:] = 2
        resource.max_annual_calls = None
        resource.dispatch(net_load, 2030)
        new_ts = resource.heuristic_provide_power_mw
        assert new_ts.groupby(new_ts.index.date).sum().max() == 40
        assert new_ts.max() == 10
        assert new_ts.sum() == 240

    def test_dispatch_daily_only(self, resource_for_dispatch_tests, test_net_load_for_dispatch):
        net_load = test_net_load_for_dispatch
        resource = copy.deepcopy(resource_for_dispatch_tests)

        # test with all daily, monthly, and annual calls
        resource.max_daily_calls.data.loc[:] = 1
        resource.max_monthly_calls = None
        resource.max_annual_calls = None
        resource.dispatch(net_load, 2030)
        new_ts = resource.heuristic_provide_power_mw
        assert new_ts.groupby(new_ts.index.date).sum().max() == 40
        assert new_ts.max() == 10
        assert new_ts.sum() == 240

    def test_dispatch_monthly_only(self, resource_for_dispatch_tests, test_net_load_for_dispatch):
        net_load = test_net_load_for_dispatch
        resource = copy.deepcopy(resource_for_dispatch_tests)

        # test with all daily, monthly, and annual calls
        resource.max_daily_calls = None
        resource.max_monthly_calls.data.loc[:] = 2
        resource.max_annual_calls = None
        resource.dispatch(net_load, 2030)
        new_ts = resource.heuristic_provide_power_mw
        assert new_ts.groupby(new_ts.index.date).sum().max() == 40
        assert new_ts.max() == 10
        assert new_ts.sum() == 240

    def test_dispatch_annual_only(self, resource_for_dispatch_tests, test_net_load_for_dispatch):
        net_load = test_net_load_for_dispatch
        resource = copy.deepcopy(resource_for_dispatch_tests)

        # test with all daily, monthly, and annual calls
        resource.max_daily_calls = None
        resource.max_monthly_calls = None
        resource.max_annual_calls.data.loc[:] = 5
        resource.dispatch(net_load, 2030)
        new_ts = resource.heuristic_provide_power_mw
        assert new_ts.groupby(new_ts.index.date).sum().max() == 40
        assert new_ts.max() == 10
        assert new_ts.sum() == 200

    def test_dispatch_no_call_limit(self, resource_for_dispatch_tests, test_net_load_for_dispatch):
        net_load = test_net_load_for_dispatch
        resource = copy.deepcopy(resource_for_dispatch_tests)

        # test with all daily, monthly, and annual calls
        resource.max_daily_calls = None
        resource.max_monthly_calls = None
        resource.max_annual_calls = None
        resource.dispatch(net_load, 2030)
        new_ts = resource.heuristic_provide_power_mw
        assert new_ts.sum() == 87600

    def test_dispatch_all_negative(self, resource_for_dispatch_tests, test_net_load_for_dispatch):
        net_load = test_net_load_for_dispatch.clip(upper=-1)
        resource = copy.deepcopy(resource_for_dispatch_tests)
        resource.dispatch(net_load, 2030)
        new_ts = resource.heuristic_provide_power_mw
        assert new_ts.sum() == 0

    def test_dispatch_all_positive(self, resource_for_dispatch_tests, test_net_load_for_dispatch):
        net_load = test_net_load_for_dispatch.clip(lower=1)
        resource = copy.deepcopy(resource_for_dispatch_tests)
        resource.dispatch(net_load, 2030)
        new_ts = resource.heuristic_provide_power_mw
        assert new_ts.sum() == 200

    def test_dispatch_second_model_year(self, make_resource_copy):
        resource = make_resource_copy()

        net_load = pd.Series(
            index=pd.DatetimeIndex(
                ["2030-01-01 00:00", "2030-01-01 01:00", "2030-01-01 02:00", "2030-01-01 03:00", "2030-01-01 04:00"]
            ),
            data=[50.0, -20.0, 0, 10.0, 150.0],
        )

        resource.upsample(pd.DatetimeIndex(["2030-01-01", "2030-12-31"]))
        updated_net_load = resource.dispatch(net_load=net_load, model_year=2030)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2030-01-01 00:00", "2030-01-01 01:00", "2030-01-01 02:00", "2030-01-01 03:00", "2030-01-01 04:00"]
                ),
                data=[0, 320, 200, 200, 200],
            ),
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2030-01-01 00:00", "2030-01-01 01:00", "2030-01-01 02:00", "2030-01-01 03:00", "2030-01-01 04:00"]
                ),
                data=[50, -340.0, -200.0, -190.0, -50.0],
            ),
        )

    def test_dispatch_all_zero(self, resource_for_dispatch_tests, test_net_load_for_dispatch):
        net_load = test_net_load_for_dispatch * 0
        resource = copy.deepcopy(resource_for_dispatch_tests)
        resource.dispatch(net_load, 2030)
        new_ts = resource.heuristic_provide_power_mw
        assert new_ts.sum() == 0

    def test_annual_dr_call_limit_constraint(self, resource_block, first_index):
        weather_year = Timestamp("2010-01-01 00:00:00")
        model_year, dispatch_window, timestamp = first_index
        resource_block.start_units.fix(0)
        resource_block.start_units[model_year, dispatch_window, timestamp + pd.DateOffset(hours=2)].fix(1)
        assert resource_block.annual_dr_call_limit_constraint[model_year, weather_year].upper() == 5
        assert resource_block.annual_dr_call_limit_constraint[model_year, weather_year].body() == 1
        assert resource_block.annual_dr_call_limit_constraint[model_year, weather_year].expr()

        resource_block.start_units[model_year, dispatch_window, timestamp + pd.DateOffset(hours=3)].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp + pd.DateOffset(hours=4)].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp + pd.DateOffset(hours=5)].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp + pd.DateOffset(hours=6)].fix(1)
        assert resource_block.annual_dr_call_limit_constraint[model_year, weather_year].body() == 5
        assert resource_block.annual_dr_call_limit_constraint[model_year, weather_year].expr()

        resource_block.start_units[model_year, dispatch_window, timestamp + pd.DateOffset(hours=7)].fix(1)
        assert resource_block.annual_dr_call_limit_constraint[model_year, weather_year].body() == 6
        assert not resource_block.annual_dr_call_limit_constraint[model_year, weather_year].expr()

    def test_annual_dr_call_limit_power_output_constraint(self, resource_block, first_index):
        weather_year = Timestamp("2010-01-01 00:00:00")
        model_year, dispatch_window, timestamp = first_index
        resource_block.start_units_power_output.fix(0)
        resource_block.start_units_power_output[model_year, dispatch_window, timestamp + pd.DateOffset(hours=2)].fix(1)
        assert resource_block.annual_dr_call_limit_power_output_constraint[model_year, weather_year].upper() == 5
        assert resource_block.annual_dr_call_limit_power_output_constraint[model_year, weather_year].body() == 1
        assert resource_block.annual_dr_call_limit_power_output_constraint[model_year, weather_year].expr()

        resource_block.start_units_power_output[model_year, dispatch_window, timestamp + pd.DateOffset(hours=3)].fix(1)
        resource_block.start_units_power_output[model_year, dispatch_window, timestamp + pd.DateOffset(hours=4)].fix(1)
        resource_block.start_units_power_output[model_year, dispatch_window, timestamp + pd.DateOffset(hours=5)].fix(1)
        resource_block.start_units_power_output[model_year, dispatch_window, timestamp + pd.DateOffset(hours=6)].fix(1)
        assert resource_block.annual_dr_call_limit_power_output_constraint[model_year, weather_year].body() == 5
        assert resource_block.annual_dr_call_limit_power_output_constraint[model_year, weather_year].expr()

        resource_block.start_units_power_output[model_year, dispatch_window, timestamp + pd.DateOffset(hours=7)].fix(1)
        assert resource_block.annual_dr_call_limit_power_output_constraint[model_year, weather_year].body() == 6
        assert not resource_block.annual_dr_call_limit_power_output_constraint[model_year, weather_year].expr()

    def test_monthly_dr_call_limit_constraint(self, resource_block, first_index):
        model_year, dispatch_window, timestamp = first_index
        month = Timestamp("2010-06-01 00:00:00")
        resource_block.start_units.fix(0)
        assert resource_block.monthly_dr_call_limit_constraint[model_year, month].upper() == 1
        assert resource_block.monthly_dr_call_limit_constraint[model_year, month].body() == 0
        assert resource_block.monthly_dr_call_limit_constraint[model_year, month].expr()

        resource_block.start_units[model_year, dispatch_window, timestamp + pd.DateOffset(hours=2)].fix(1)
        assert resource_block.monthly_dr_call_limit_constraint[model_year, month].body() == 1
        assert resource_block.monthly_dr_call_limit_constraint[model_year, month].expr()

        resource_block.start_units[model_year, dispatch_window, timestamp + pd.DateOffset(hours=3)].fix(1)
        assert resource_block.monthly_dr_call_limit_constraint[model_year, month].body() == 2
        assert not resource_block.monthly_dr_call_limit_constraint[model_year, month].expr()

    def test_monthly_dr_call_limit_power_output_constraint(self, resource_block, first_index):
        model_year, dispatch_window, timestamp = first_index
        month = Timestamp("2010-06-01 00:00:00")
        resource_block.start_units_power_output.fix(0)
        assert resource_block.monthly_dr_call_limit_power_output_constraint[model_year, month].upper() == 1
        assert resource_block.monthly_dr_call_limit_power_output_constraint[model_year, month].body() == 0
        assert resource_block.monthly_dr_call_limit_power_output_constraint[model_year, month].expr()

        resource_block.start_units_power_output[model_year, dispatch_window, timestamp + pd.DateOffset(hours=2)].fix(1)
        assert resource_block.monthly_dr_call_limit_power_output_constraint[model_year, month].body() == 1
        assert resource_block.monthly_dr_call_limit_power_output_constraint[model_year, month].expr()

        resource_block.start_units_power_output[model_year, dispatch_window, timestamp + pd.DateOffset(hours=3)].fix(1)
        assert resource_block.monthly_dr_call_limit_power_output_constraint[model_year, month].body() == 2
        assert not resource_block.monthly_dr_call_limit_power_output_constraint[model_year, month].expr()

    def test_daily_dr_call_limit_constraint(self, resource_block, first_index):
        day = Timestamp("2010-06-21 00:00:00")
        model_year, dispatch_window, timestamp = first_index
        resource_block.start_units.fix(0)
        assert resource_block.daily_dr_call_limit_constraint[model_year, day].upper() == 1
        assert resource_block.daily_dr_call_limit_constraint[model_year, day].body() == 0
        assert resource_block.daily_dr_call_limit_constraint[model_year, day].expr()

        resource_block.start_units[model_year, dispatch_window, timestamp + pd.DateOffset(hours=2)].fix(1)
        assert resource_block.daily_dr_call_limit_constraint[model_year, day].body() == 1
        assert resource_block.daily_dr_call_limit_constraint[model_year, day].expr()

        resource_block.start_units[model_year, dispatch_window, timestamp + pd.DateOffset(hours=3)].fix(1)
        assert resource_block.daily_dr_call_limit_constraint[model_year, day].body() == 2
        assert not resource_block.daily_dr_call_limit_constraint[model_year, day].expr()

        resource_block.start_units[model_year, dispatch_window, timestamp + pd.DateOffset(hours=4)].fix(1)
        assert resource_block.daily_dr_call_limit_constraint[model_year, day].body() == 3
        assert not resource_block.daily_dr_call_limit_constraint[model_year, day].expr()

    def test_daily_dr_call_limit_power_output_constraint(self, resource_block, first_index):
        day = Timestamp("2010-06-21 00:00:00")
        model_year, dispatch_window, timestamp = first_index
        resource_block.start_units_power_output.fix(0)
        assert resource_block.daily_dr_call_limit_power_output_constraint[model_year, day].upper() == 1
        assert resource_block.daily_dr_call_limit_power_output_constraint[model_year, day].body() == 0
        assert resource_block.daily_dr_call_limit_power_output_constraint[model_year, day].expr()

        resource_block.start_units_power_output[model_year, dispatch_window, timestamp + pd.DateOffset(hours=2)].fix(1)
        assert resource_block.daily_dr_call_limit_power_output_constraint[model_year, day].body() == 1
        assert resource_block.daily_dr_call_limit_power_output_constraint[model_year, day].expr()

        resource_block.start_units_power_output[model_year, dispatch_window, timestamp + pd.DateOffset(hours=3)].fix(1)
        assert resource_block.daily_dr_call_limit_power_output_constraint[model_year, day].body() == 2
        assert not resource_block.daily_dr_call_limit_power_output_constraint[model_year, day].expr()

        resource_block.start_units_power_output[model_year, dispatch_window, timestamp + pd.DateOffset(hours=4)].fix(1)
        assert resource_block.daily_dr_call_limit_power_output_constraint[model_year, day].body() == 3
        assert not resource_block.daily_dr_call_limit_power_output_constraint[model_year, day].expr()

    def test_max_dr_call_duration(self, resource_block, first_index):
        model_year, dispatch_window, timestamp = first_index
        resource_block.start_units_power_output.fix(0)
        timestamp = timestamp + pd.DateOffset(hours=5)

        resource_block.committed_units_power_output[model_year, dispatch_window, timestamp].fix(1)
        assert resource_block.max_dr_call_constraint[model_year, dispatch_window, timestamp].body() == 1
        assert resource_block.max_dr_call_constraint[model_year, dispatch_window, timestamp].upper() == 0
        assert not resource_block.max_dr_call_constraint[model_year, dispatch_window, timestamp].expr()

        resource_block.start_units_power_output[model_year, dispatch_window, timestamp - pd.DateOffset(hours=2)].fix(1)
        assert resource_block.max_dr_call_constraint[model_year, dispatch_window, timestamp].body() == 0
        assert resource_block.max_dr_call_constraint[model_year, dispatch_window, timestamp].upper() == 0
        assert resource_block.max_dr_call_constraint[model_year, dispatch_window, timestamp].expr()

        resource_block.start_units_power_output[model_year, dispatch_window, timestamp - pd.DateOffset(hours=3)].fix(1)
        assert resource_block.max_dr_call_constraint[model_year, dispatch_window, timestamp].body() == -1
        assert resource_block.max_dr_call_constraint[model_year, dispatch_window, timestamp].upper() == 0
        assert resource_block.max_dr_call_constraint[model_year, dispatch_window, timestamp].expr()

    def test_upsample(self, make_resource_copy):
        """Test the `upsample()` method."""
        resource = make_resource_copy()
        # ensure year is repeated

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2011-06-30 23:00", freq="H")
        resource.upsample(load_calendar=load_calendar)
        assert len(resource.power_output_max.data) == 17520
        assert (
            resource.power_output_max.data.loc[resource.power_output_max.data.index.year == 2010].sum()
            == resource.power_output_max.data.loc[resource.power_output_max.data.index.year == 2011].sum()
        )

    def test_upsample_no_overlap(self, make_resource_copy):
        """Test the `upsample()` method in scenarios where the target index has no overlap with the input data."""
        resource = make_resource_copy()

        load_calendar = pd.date_range(start="2008-01-01 00:00", end="2009-06-30 23:00", freq="H")
        resource.upsample(load_calendar=load_calendar)
        assert len(resource.power_output_max.data) == 17544
        assert resource.power_output_max.data.loc[resource.power_output_max.data.index.year == 2008].sum() == 4392.8
        assert resource.power_output_max.data.loc[resource.power_output_max.data.index.year == 2009].sum() == 4380.8

    def test_adjust_remaining_calls_with_none(self, resource_for_dispatch_tests):
        resource = copy.deepcopy(resource_for_dispatch_tests)
        resource.max_annual_calls = None
        resource.max_monthly_calls = None
        resource.max_daily_calls = None
        timestamps_to_include = pd.Series(
            index=pd.date_range(start="2030-07-01 00:00", end="2030-08-31 23:00", freq="H"), data=0.0
        )
        timestamps_to_include.loc["2030-07-03 00:00":"2030-08-25 23:00"] = 1.0
        resource.call_starts = pd.Series(
            index=pd.DatetimeIndex(
                ["2030-06-21", "2030-06-22", "2030-07-21", "2030-08-21", "2030-09-21"], name="timestamp"
            ),
            data=[1, 1, 1, 1, 1],
            name="value",
        )
        resource.adjust_remaining_calls_for_optimization(timestamps_to_include)
        assert resource.max_annual_calls is None
        assert resource.max_monthly_calls is None
        assert resource.max_daily_calls is None

    def test_adjust_remaining_calls_for_optimization(self, resource_for_dispatch_tests):
        """Test the `adjust_remaining_calls_for_optimization()` method."""
        resource = copy.deepcopy(resource_for_dispatch_tests)
        resource.heuristic_provide_power_mw = pd.Series(
            index=pd.date_range(start="2030-07-01 00:00", end="2030-08-31 23:00", freq="H"), data=1
        )

        timestamps_to_include = pd.Series(
            index=pd.date_range(start="2030-07-01 00:00", end="2030-08-31 23:00", freq="H"), data=0.0
        )
        timestamps_to_include.loc["2030-07-03 00:00":"2030-08-25 23:00"] = 1.0

        resource.call_starts = pd.Series(
            index=pd.DatetimeIndex(
                ["2030-06-21", "2030-06-22", "2030-07-21", "2030-08-21", "2030-09-21"], name="timestamp"
            ),
            data=[1, 1, 1, 1, 1],
            name="value",
        )

        resource.adjust_remaining_calls_for_optimization(timestamps_to_include=timestamps_to_include)

        pd.testing.assert_series_equal(
            resource.max_annual_calls.data,
            pd.Series(index=pd.DatetimeIndex(["2030-01-01 00:00"]), data=2.0),
            check_names=False,
            check_freq=False,
        )

        pd.testing.assert_series_equal(
            resource.max_monthly_calls.data,
            pd.Series(
                index=pd.DatetimeIndex(
                    [
                        "2030-01-01 00:00",
                        "2030-02-01 00:00",
                        "2030-03-01 00:00",
                        "2030-04-01 00:00",
                        "2030-05-01 00:00",
                        "2030-06-01 00:00",
                        "2030-07-01 00:00",
                        "2030-08-01 00:00",
                        "2030-09-01 00:00",
                        "2030-10-01 00:00",
                        "2030-11-01 00:00",
                        "2030-12-01 00:00",
                    ]
                ),
                data=[
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                ],
            ),
            check_names=False,
            check_freq=False,
        )

        assert resource.max_daily_calls.data.sum() == 362
