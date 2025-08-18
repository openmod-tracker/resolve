import numpy as np
import pandas as pd

import new_modeling_toolkit.system.electric.resources.hydro as hydro
from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.system.electric.resources import HydroResource
from tests.system.electric.resources import test_generic


def mock_cantor_pairing_function(a, b):
    return 1

class TestHydroResource(test_generic.TestGenericResource):
    _RESOURCE_CLASS = HydroResource
    _SYSTEM_COMPONENT_DICT_NAME = "hydro_resources"

    _RESOURCE_INIT_KWARGS = dict(
        name="Example_Hydro",
        capacity_planned=ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[45.0, 50.0],
                name="value",
            ),
        ),
        power_output_max=ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
                ),
                data=0.9,
                name="value",
            ),
        ),
        power_output_min=ts.FractionalTimeseries(
            name="power_output_min",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
                ),
                data=0.1,
                name="value",
            ),
        ),
        power_input_max=ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
                ),
                data=0.0,
                name="value",
            ),
        ),
        power_input_min=ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
                ),
                data=0.0,
                name="value",
            ),
        ),
        outage_profile=ts.FractionalTimeseries(
            name="outage_profile",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
                ),
                data=1.0,
                name="value",
            ),
        ),
        energy_budget_daily=None,
        energy_budget_monthly=ts.FractionalTimeseries(
            name="energy_budget_monthly",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01", end="2012-12-01", freq="MS"), name="timestamp"
                ),
                data=[
                    0.537634409,
                    0.595238095,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.672043011,
                    0.744047619,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.537634409,
                    0.574712644,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                ],
                name="value",
            ),
        ),
        energy_budget_annual=None,
    )

    def monkey_patch(self, monkeypatch):
        monkeypatch.setattr(hydro, "cantor_pairing_function", mock_cantor_pairing_function)

    def test_adjust_budgets_for_optimization(self, make_resource_copy):
        resource = make_resource_copy()
        original_monthly_energy_budget = resource.energy_budget_monthly.data
        original_scaled_energy_budget = resource.scaled_monthly_energy_budget[2030]

        resource.heuristic_provide_power_mw = pd.Series(
            index=pd.date_range(start="2010-07-01 00:00", end="2010-08-31 23:00", freq="H"), data=10
        )

        timestamps_to_include = pd.Series(
            index=pd.date_range(start="2010-07-01 00:00", end="2010-08-31 23:00", freq="H"), data=0.0
        )
        timestamps_to_include.loc["2010-07-03 00:00":"2010-08-25 23:00"] = 1.0

        resource.adjust_budgets_for_optimization(
            model_year=2030, timestamps_included_in_optimization_flags=timestamps_to_include
        )

        # check that original energy budget does not change
        pd.testing.assert_series_equal(original_monthly_energy_budget, resource.energy_budget_monthly.data)

        # Check that scaled energy budget is reduced by sum of heuristic_provide_power within window
        power_provided_in_window = resource._calculate_heuristic_provide_power_in_window(
            heuristic_provide_power=resource.heuristic_provide_power_mw,
            timestamps_included_in_optimization_flags=timestamps_to_include,
            freq="M",
            target_index=resource.scaled_monthly_energy_budget[2030].index,
        )
        new_scaled_energy_budget = original_scaled_energy_budget - power_provided_in_window
        pd.testing.assert_series_equal(new_scaled_energy_budget, resource.scaled_monthly_energy_budget[2030])

    # TODO (cgulian) 2023-06-16: Come back to this later when writing new tests for those inherited from
    #  TestGenericResource
    # @pytest.fixture(scope="function")
    # def resource_for_upsampling_and_dispatch_tests(self, make_resource_copy):
    #     resource = make_resource_copy()
    #     resource.name="Example_Hydro"
    #     resource.capacity_planned=ts.NumericTimeseries(
    #         name="capacity_planned",
    #         data=pd.Series(
    #             index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
    #             data=[45.0, 50.0],
    #             name="value",
    #         ),
    #     )
    #     resource.power_output_max=ts.NumericTimeseries(
    #         name="power_output_max",
    #         data=pd.Series(
    #             index=pd.DatetimeIndex(
    #                 pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
    #             ),
    #             data=0.9,
    #             name="value",
    #         ),
    #     )
    #     resource.power_output_min=ts.NumericTimeseries(
    #         name="power_output_min",
    #         data=pd.Series(
    #             index=pd.DatetimeIndex(
    #                 pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
    #             ),
    #             data=0.1,
    #             name="value",
    #         ),
    #     )
    #     resource.power_input_max=ts.NumericTimeseries(
    #         name="power_input_max",
    #         data=pd.Series(
    #             index=pd.DatetimeIndex(
    #                 pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
    #             ),
    #             data=0.0,
    #             name="value",
    #         ),
    #     )
    #     resource.power_input_min = ts.NumericTimeseries(
    #         name="power_input_max",
    #         data=pd.Series(
    #             index=pd.DatetimeIndex(
    #                 pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
    #             ),
    #             data=0.0,
    #             name="value",
    #         ),
    #     )
    #     resource.outage_profile=ts.NumericTimeseries(
    #         name="outage_profile",
    #         data=pd.Series(
    #             index=pd.DatetimeIndex(
    #                 pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
    #             ),
    #             data=1.0,
    #             name="value",
    #         ),
    #     )
    #     resource.energy_budget_daily=None
    #     resource.energy_budget_monthly=ts.NumericTimeseries(
    #         name="energy_budget_monthly",
    #         data=pd.Series(
    #             index=pd.DatetimeIndex(
    #                 pd.date_range(start="2010-01-01", end="2012-12-01", freq="MS"), name="timestamp"
    #             ),
    #             data=[
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 25000.0,
    #                 25000.0,
    #                 25000.0,
    #                 25000.0,
    #                 25000.0,
    #                 25000.0,
    #                 25000.0,
    #                 25000.0,
    #                 25000.0,
    #                 25000.0,
    #                 25000.0,
    #                 25000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #                 20000.0,
    #             ],
    #             name="value",
    #         ),
    #     )
    #     resource.energy_budget_annual = None
    #
    #     return resource

    def test_rescale(self, make_resource_copy):
        resource = make_resource_copy()

        # Do re-scaling
        resource.rescale(model_year=2030, capacity=100, incremental=False)

        # Assert that data matches expectations after re-scaling
        pd.testing.assert_series_equal(
            resource.capacity_planned.data,
            pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[45.0, 100.0],
                name="value",
            ),
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            resource.energy_budget_monthly.data,
            pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01", end="2012-12-01", freq="MS"), name="timestamp"
                ),
                data=[
                    0.537634409,
                    0.595238095,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.672043011,
                    0.744047619,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.537634409,
                    0.574712644,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                ],
                name="value",
            ),
            check_dtype=False,
        )

    def test_rescale_incremental(self, make_resource_copy):
        resource = make_resource_copy()

        resource.rescale(model_year=2030, capacity=100, incremental=True)

        pd.testing.assert_series_equal(
            resource.capacity_planned.data,
            pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[45.0, 150.0],
                name="value",
            ),
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            resource.energy_budget_monthly.data,
            pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01", end="2012-12-01", freq="MS"), name="timestamp"
                ),
                data=[
                    0.537634409,
                    0.595238095,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.672043011,
                    0.744047619,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.537634409,
                    0.574712644,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                    0.555555556,
                    0.537634409,
                ],
                name="value",
            ),
            check_dtype=False,
        )

    def test_upsample(self, make_resource_copy, monkeypatch):
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H")

        # Test repeat year dictionary
        hydro_years = list(resource.power_output_max.data.copy().index.year.unique())
        np.random.seed(int(resource.random_seed))
        repeat_year_dict = dict(
            zip(
                list(np.arange(min(load_calendar.year), max(load_calendar.year) + 1)),
                np.random.choice(hydro_years, size=(max(load_calendar.year) - min(load_calendar.year) + 1)),
            )
        )
        assert repeat_year_dict == {2010: 2011}

        # to test that you aren't required to input a full year of data
        resource.power_output_min = ts.FractionalTimeseries(
            name="power_output_min",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01 00:00:00", end="2010-02-01 23:59:59", freq="H"), name="timestamp"
                ),
                data=0.1,
                name="value",
            ),
        )

        resource.upsample(load_calendar=load_calendar, random_seed=0)

        pd.testing.assert_series_equal(
            resource.power_output_max.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=0.9,
                name="value",
            ),
            check_names=False,
        )

        pd.testing.assert_series_equal(
            resource.power_output_min.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=0.1,
                name="value",
            ),
            check_names=False,
        )

        pd.testing.assert_series_equal(
            resource.energy_budget_monthly.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="MS", name="timestamp"),
                data=[
                    0.672043011,
                    0.744047619,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                ],
                name="value",
            ),
            check_names=False,
        )

    def test_upsample_no_overlap(self, make_resource_copy, monkeypatch):
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        load_calendar = pd.date_range(start="2018-01-01 00:00", end="2018-12-31 23:00", freq="H")

        # Test repeat year dictionary
        hydro_years = list(resource.power_output_max.data.copy().index.year.unique())
        np.random.seed(int(resource.random_seed))
        repeat_year_dict = dict(
            zip(
                list(np.arange(min(load_calendar.year), max(load_calendar.year) + 1)),
                np.random.choice(hydro_years, size=(max(load_calendar.year) - min(load_calendar.year) + 1)),
            )
        )
        assert repeat_year_dict == {2018: 2011}

        resource.upsample(load_calendar=load_calendar, random_seed=0)

        pd.testing.assert_series_equal(
            resource.power_output_max.data,
            pd.Series(
                index=pd.date_range(start="2018-01-01 00:00", end="2018-12-31 23:00", freq="H", name="timestamp"),
                data=0.9,
                name="value",
            ),
            check_names=False,
        )

        pd.testing.assert_series_equal(
            resource.power_output_min.data,
            pd.Series(
                index=pd.date_range(start="2018-01-01 00:00", end="2018-12-31 23:00", freq="H", name="timestamp"),
                data=0.1,
                name="value",
            ),
            check_names=False,
        )

        pd.testing.assert_series_equal(
            resource.energy_budget_monthly.data,
            pd.Series(
                index=pd.date_range(start="2018-01-01 00:00", end="2018-12-31 23:00", freq="MS", name="timestamp"),
                data=[
                    0.672043011,
                    0.744047619,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                    0.694444444,
                    0.672043011,
                ],
                name="value",
            ),
            check_names=False,
        )

    def test_upsample_power_output_min(self, make_resource_copy, monkeypatch):
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H")
        resource.upsample(load_calendar, random_seed=0)
        pd.testing.assert_series_equal(
            resource.power_output_min.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=0.1,
                name="value",
            ),
            check_names=False,
        )

    def test_upsample_power_input_max(self, make_resource_copy, monkeypatch):
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H")
        resource.upsample(load_calendar, random_seed=0)
        pd.testing.assert_series_equal(
            resource.power_input_max.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=0.0,
                name="value",
            ),
        )

    def test_upsample_power_input_min(self, make_resource_copy, monkeypatch):
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H")
        resource.upsample(load_calendar, random_seed=0)
        pd.testing.assert_series_equal(
            resource.power_input_min.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=0.0,
                name="value",
            ),
            check_names=False,
        )

    def test_upsample_outage_profile(self, make_resource_copy, monkeypatch):
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp")
        resource.upsample(load_calendar, random_seed=0)
        pd.testing.assert_series_equal(
            resource.outage_profile.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=1.0,
                name="value",
            ),
            check_names=False,
        )

    def test_upsample_energy_budget_daily(self, make_resource_copy, monkeypatch):
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        resource.energy_budget_daily = ts.FractionalTimeseries(
            name="energy_budget_daily",
            data=pd.Series(index=pd.DatetimeIndex(["2010-01-01"], name="timestamp"), data=1.0),
            freq_="D",
        )
        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp")
        resource.upsample(load_calendar, random_seed=0)
        pd.testing.assert_series_equal(
            resource.energy_budget_daily.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="D", name="timestamp"),
                data=1.0,
                name="value",
            ),
            check_names=False,
        )

    def test_upsample_energy_budget_monthly(self, make_resource_copy, monkeypatch):
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp")
        expected_energy_budget_monthly = resource.energy_budget_monthly.data.loc[
            resource.energy_budget_monthly.data.index.year == 2011
        ]
        expected_energy_budget_monthly.index = resource.energy_budget_monthly.data.loc[
            resource.energy_budget_monthly.data.index.year == 2010
        ].index
        resource.upsample(load_calendar, random_seed=0)
        pd.testing.assert_series_equal(
            resource.energy_budget_monthly.data, expected_energy_budget_monthly, check_names=False
        )

    def test_upsample_energy_budget_annual(self, make_resource_copy, monkeypatch):
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        resource.energy_budget_annual = ts.FractionalTimeseries(
            name="energy_budget_annual",
            data=pd.Series(index=pd.DatetimeIndex(["2012-01-01"], name="timestamp"), data=1.0),
            freq_="YS",
        )
        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp")
        resource.upsample(load_calendar, random_seed=0)
        pd.testing.assert_series_equal(
            resource.energy_budget_annual.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="YS", name="timestamp"),
                data=1.0,
                name="value",
            ),
            check_names=False,
        )

    def test_scaled_pmax_profile(self, make_resource_copy):
        resource = make_resource_copy()

        expected_profile = {
            2020: pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
                ),
                data=0.9 * 45.0,
                name="value",
            ),
            2030: pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
                ),
                data=0.9 * 50.0,
                name="value",
            ),
        }

        assert resource.scaled_pmax_profile.keys() == expected_profile.keys()
        for model_year, profile in resource.scaled_pmax_profile.items():
            pd.testing.assert_series_equal(profile, expected_profile[model_year])

    def test_scaled_pmin_profile(self, make_resource_copy):
        resource = make_resource_copy()

        expected_profile = {
            2020: pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
                ),
                data=0.1 * 45.0,
                name="value",
            ),
            2030: pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01 00:00:00", end="2012-12-31 23:59:59", freq="H"), name="timestamp"
                ),
                data=0.1 * 50.0,
                name="value",
            ),
        }

        assert resource.scaled_pmin_profile.keys() == expected_profile.keys()
        for model_year, profile in resource.scaled_pmin_profile.items():
            pd.testing.assert_series_equal(profile, expected_profile[model_year])

    def test_scaled_daily_energy_budget(self, make_resource_copy):
        resource = make_resource_copy()
        resource.energy_budget_daily = ts.FractionalTimeseries(
            name="energy_budget_daily",
            data=pd.Series(index=pd.DatetimeIndex(["2010-01-01"], name="timestamp"), data=1.0),
            freq_="D",
        )

        expected_profile = {
            2020: pd.Series(
                index=pd.DatetimeIndex(["2010-01-01"], name="timestamp"),
                data=45.0 * 24,
            ),
            2030: pd.Series(
                index=pd.DatetimeIndex(["2010-01-01"], name="timestamp"),
                data=50.0 * 24,
            ),
        }
        assert resource.scaled_daily_energy_budget.keys() == expected_profile.keys()
        for model_year, profile in resource.scaled_daily_energy_budget.items():
            pd.testing.assert_series_equal(profile, expected_profile[model_year])

    def test_scaled_monthly_energy_budget(self, make_resource_copy):
        resource = make_resource_copy()

        expected_profile = {
            2030: pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01", end="2012-12-01", freq="MS"), name="timestamp"
                ),
                data=[
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    25000.0,
                    25000.0,
                    25000.0,
                    25000.0,
                    25000.0,
                    25000.0,
                    25000.0,
                    25000.0,
                    25000.0,
                    25000.0,
                    25000.0,
                    25000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                    20000.0,
                ],
                name="value",
            ),
        }
        pd.testing.assert_series_equal(expected_profile[2030], expected_profile[2030])

    def test_scaled_annual_energy_budget(self, make_resource_copy):
        resource = make_resource_copy()
        resource.energy_budget_annual = ts.FractionalTimeseries(
            name="energy_budget_annual",
            data=pd.Series(index=pd.DatetimeIndex(["2010-01-01"], name="timestamp"), data=1.0),
            freq_="YS",
        )

        expected_profile = {
            2020: pd.Series(
                index=pd.DatetimeIndex(["2010-01-01"], name="timestamp"),
                data=45.0 * 24 * 365,
            ),
            2030: pd.Series(
                index=pd.DatetimeIndex(["2010-01-01"], name="timestamp"),
                data=50.0 * 24 * 365,
            ),
        }
        assert resource.scaled_annual_energy_budget.keys() == expected_profile.keys()
        for model_year, profile in resource.scaled_annual_energy_budget.items():
            pd.testing.assert_series_equal(profile, expected_profile[model_year])

    def test_dispatch(self, make_resource_copy, monkeypatch):
        """Test the `dispatch()` function."""
        model_year = 2030
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp")
        resource.upsample(load_calendar, random_seed=0)

        # Run dispatch against net load
        net_load = pd.Series(
            index=pd.DatetimeIndex(load_calendar, name="timestamp"),
            data=-100,
        )
        critical_period = pd.Timestamp(year=2010, month=1, day=15, hour=17)
        net_load.loc[critical_period] = 100

        updated_net_load = resource.dispatch(net_load=net_load, model_year=model_year)

        # Calculate expected heuristic provide power series
        expected_heuristic_provide_power = pd.Series(
            index=pd.DatetimeIndex(load_calendar, name="timestamp"),
            data=5.0,
            name="value",
        )
        expected_heuristic_provide_power.loc[load_calendar.month == 1] = 33.58644485473633
        expected_heuristic_provide_power.loc[critical_period] = 45.0

        # Calculate expected updated net load
        expected_updated_net_load = net_load - expected_heuristic_provide_power

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw, expected_heuristic_provide_power, check_names=False
        )
        pd.testing.assert_series_equal(updated_net_load, expected_updated_net_load, check_names=False)

        # Assert that dispatch respects budget constraints
        # Note: the dispatch function allows the budget to be exceeded by up to 1 MWh, so the upper bound in this test
        #  is increased by 1
        power_provided = resource.heuristic_provide_power_mw.groupby(load_calendar.month).sum()
        power_provided.index = power_provided.index.to_series().apply(lambda x: pd.to_datetime(f"{x}-01-{model_year}"))
        budget_in_mwh = resource.scaled_monthly_energy_budget[model_year]
        assert (power_provided.values < budget_in_mwh.values + 1).all()

    def test_dispatch_second_model_year(self, make_resource_copy, monkeypatch):
        """Test the `dispatch()` function."""
        model_year = 2020
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        # re-define monthly energy budget in terms of 2020 planned capacity
        resource.energy_budget_monthly = ts.FractionalTimeseries(
            name="energy_budget_monthly",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    pd.date_range(start="2010-01-01", end="2012-12-01", freq="MS"), name="timestamp"
                ),
                data=[
                    0.597371565,
                    0.661375661,
                    0.597371565,
                    0.617283951,
                    0.597371565,
                    0.617283951,
                    0.597371565,
                    0.597371565,
                    0.617283951,
                    0.597371565,
                    0.617283951,
                    0.597371565,
                    0.746714456,
                    0.826719577,
                    0.746714456,
                    0.771604938,
                    0.746714456,
                    0.771604938,
                    0.746714456,
                    0.746714456,
                    0.771604938,
                    0.746714456,
                    0.771604938,
                    0.746714456,
                    0.597371565,
                    0.638569604,
                    0.597371565,
                    0.617283951,
                    0.597371565,
                    0.617283951,
                    0.597371565,
                    0.597371565,
                    0.617283951,
                    0.597371565,
                    0.617283951,
                    0.597371565,
                ],
                name="value",
            ),
        )

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp")
        resource.upsample(load_calendar, random_seed=0)

        # Run dispatch against net load
        net_load = pd.Series(
            index=pd.DatetimeIndex(load_calendar, name="timestamp"),
            data=-100,
        )
        critical_period = pd.Timestamp(year=2010, month=1, day=15, hour=17)
        net_load.loc[critical_period] = 100

        updated_net_load = resource.dispatch(net_load=net_load, model_year=model_year)

        # Calculate expected heuristic provide power series
        expected_heuristic_provide_power = pd.Series(
            index=pd.DatetimeIndex(load_calendar, name="timestamp"),
            data=4.5,
            name="value",
        )
        expected_heuristic_provide_power.loc[load_calendar.month == 1] = 33.593379974365234
        expected_heuristic_provide_power.loc[critical_period] = 40.5

        # Calculate expected updated net load
        expected_updated_net_load = net_load - expected_heuristic_provide_power

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw, expected_heuristic_provide_power, check_names=False
        )
        pd.testing.assert_series_equal(updated_net_load, expected_updated_net_load, check_names=False)

        # Assert that dispatch respects budget constraints
        # Note: the dispatch function allows the budget to be exceeded by up to 1 MWh, so the upper bound in this test
        #  is increased by 1
        power_provided = resource.heuristic_provide_power_mw.groupby(load_calendar.month).sum()
        power_provided.index = power_provided.index.to_series().apply(lambda x: pd.to_datetime(f"{x}-01-{model_year}"))
        budget_in_mwh = resource.scaled_monthly_energy_budget[model_year]
        assert (power_provided.values < budget_in_mwh.values + 1).all()

    def test_dispatch_all_positive(self, make_resource_copy, monkeypatch):
        """Test the `dispatch()` function."""
        model_year = 2030
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp")
        resource.upsample(load_calendar, random_seed=0)

        # Run dispatch against net load
        net_load = pd.Series(
            index=pd.DatetimeIndex(load_calendar, name="timestamp"),
            data=20.0,
        )

        critical_periods = pd.date_range(start="2010-01-15 17:00:00", end="2010-01-15 21:00:00", freq="H")
        critical_periods = critical_periods.union(
            pd.date_range(start="2010-08-15 17:00:00", end="2010-08-15 21:00:00", freq="H")
        )
        net_load.loc[critical_periods] = 100

        updated_net_load = resource.dispatch(net_load=net_load, model_year=model_year)

        # Calculate expected heuristic provide power series
        expected_heuristic_provide_power = pd.Series(
            index=pd.DatetimeIndex(load_calendar, name="timestamp"),
            data=33.602280,
            name="value",
        )
        expected_heuristic_provide_power.loc[load_calendar.month == 1] = 33.524818
        expected_heuristic_provide_power.loc[load_calendar.month == 2] = 37.202042
        expected_heuristic_provide_power.loc[load_calendar.month == 8] = 33.524818
        expected_heuristic_provide_power.loc[load_calendar.month.isin([4, 6, 9, 11])] = 34.722122
        expected_heuristic_provide_power.loc[critical_periods] = 45.0

        # Calculate expected updated net load
        expected_updated_net_load = net_load - expected_heuristic_provide_power

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw, expected_heuristic_provide_power, check_names=False
        )
        pd.testing.assert_series_equal(updated_net_load, expected_updated_net_load, check_names=False)

        # Assert that dispatch respects budget constraints
        # Note: the dispatch function allows the budget to be exceeded by up to 1 MWh, so the upper bound in this test
        #  is increased by 1
        power_provided = resource.heuristic_provide_power_mw.groupby(load_calendar.month).sum()
        power_provided.index = power_provided.index.to_series().apply(lambda x: pd.to_datetime(f"{x}-01-{model_year}"))
        budget_in_mwh = resource.scaled_monthly_energy_budget[model_year]
        assert (power_provided.values < budget_in_mwh.values + 1).all()

    def test_dispatch_all_negative(self, make_resource_copy, monkeypatch):
        """Test the `dispatch()` function."""
        model_year = 2030
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp")
        resource.upsample(load_calendar, random_seed=0)

        # Run dispatch against net load
        net_load = pd.Series(index=pd.DatetimeIndex(load_calendar, name="timestamp"), data=-100)

        updated_net_load = resource.dispatch(net_load=net_load, model_year=model_year)

        # Calculate expected heuristic provide power series
        expected_heuristic_provide_power = pd.Series(
            index=pd.DatetimeIndex(load_calendar, name="timestamp"),
            data=5.0,
            name="value",
        )

        # Calculate expected updated net load
        expected_updated_net_load = net_load - expected_heuristic_provide_power

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw, expected_heuristic_provide_power, check_names=False
        )
        pd.testing.assert_series_equal(updated_net_load, expected_updated_net_load, check_names=False)

        # Assert that dispatch respects budget constraints
        # Note: the dispatch function allows the budget to be exceeded by up to 1 MWh, so the upper bound in this test
        #  is increased by 1
        power_provided = resource.heuristic_provide_power_mw.groupby(load_calendar.month).sum()
        power_provided.index = power_provided.index.to_series().apply(lambda x: pd.to_datetime(f"{x}-01-{model_year}"))
        budget_in_mwh = resource.scaled_monthly_energy_budget[model_year]
        assert (power_provided.values < budget_in_mwh.values + 1).all()

    def test_dispatch_all_zero(self, make_resource_copy, monkeypatch):
        """Test the `dispatch()` function."""
        model_year = 2030
        resource = make_resource_copy()
        self.monkey_patch(monkeypatch)

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp")
        resource.upsample(load_calendar, random_seed=0)

        # Run dispatch against net load
        net_load = pd.Series(index=pd.DatetimeIndex(load_calendar, name="timestamp"), data=0)

        updated_net_load = resource.dispatch(net_load=net_load, model_year=model_year)

        # Calculate expected heuristic provide power series
        expected_heuristic_provide_power = pd.Series(
            index=pd.DatetimeIndex(load_calendar, name="timestamp"),
            data=5.0,
            name="value",
        )

        # Calculate expected updated net load
        expected_updated_net_load = net_load - expected_heuristic_provide_power

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw, expected_heuristic_provide_power, check_names=False
        )
        pd.testing.assert_series_equal(updated_net_load, expected_updated_net_load, check_names=False)

        # Assert that dispatch respects budget constraints
        # Note: the dispatch function allows the budget to be exceeded by up to 1 MWh, so the upper bound in this test
        #  is increased by 1
        power_provided = resource.heuristic_provide_power_mw.groupby(load_calendar.month).sum()
        power_provided.index = power_provided.index.to_series().apply(lambda x: pd.to_datetime(f"{x}-01-{model_year}"))
        budget_in_mwh = resource.scaled_monthly_energy_budget[model_year]
        assert (power_provided.values < budget_in_mwh.values + 1).all()

    def test_variable_bounds(self, make_dispatch_model_copy):
        pass

    def test_power_output_max_constraint(self, make_dispatch_model_copy, first_index):
        pass

    def test_power_output_min_constraint(self, make_dispatch_model_copy, first_index):
        pass

    def test_power_input_max_constraint(self, make_dispatch_model_copy, first_index):
        pass

    def test_daily_energy_budget_constraint(self, make_dispatch_model_copy):
        pass

    def test_monthly_energy_budget_constraint(self, make_dispatch_model_copy):
        pass

    def test_annual_energy_budget_constraint(self, make_dispatch_model_copy):
        pass
