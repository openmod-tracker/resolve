import logging
from unittest import mock

import pandas as pd

from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.core.temporal.new_temporal import DispatchWindowEdgeEffects
from new_modeling_toolkit.recap.monte_carlo_draw import _identify_positive_net_load_windows
from new_modeling_toolkit.recap.monte_carlo_draw import _PERIOD_CONSECUTIVE_DELTA
from new_modeling_toolkit.recap.monte_carlo_draw import _PERIOD_END_BUFFER
from new_modeling_toolkit.recap.monte_carlo_draw import _PERIOD_MIN_SEPARATION_WINDOW
from new_modeling_toolkit.recap.monte_carlo_draw import _PERIOD_START_BUFFER
from new_modeling_toolkit.recap.monte_carlo_draw import MonteCarloDraw
from new_modeling_toolkit.recap.recap_case_settings import ResourceGrouping


class TestMonteCarloDraw:
    def test_constants(self):
        assert _PERIOD_START_BUFFER == pd.Timedelta(hours=-168)
        assert _PERIOD_END_BUFFER == pd.Timedelta(hours=168)
        assert _PERIOD_MIN_SEPARATION_WINDOW == pd.Timedelta(hours=168 * 2)
        assert _PERIOD_CONSECUTIVE_DELTA == pd.Timedelta(hours=24)

    def test_gross_load(self, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()
        mc_draw.system.loads["CAISO_load"].profile = ts.NumericTimeseries(
            name="profile",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[100, 110, 120, 130],
            ),
        )
        mc_draw.system.loads["CAISO_load"].annual_energy_forecast = ts.NumericTimeseries(
            name="annual_energy_forecast", data=pd.Series(index=pd.DatetimeIndex(["2030-01-01"]), data=3000000)
        )
        mc_draw.system.loads["CAISO_load"].forecast_load((2030, 2030))
        mc_draw._gross_load = None
        pd.testing.assert_series_equal(
            mc_draw.gross_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[652173.91304348, 717391.30434783, 782608.69565217, 847826.08695652],
            ),
        )
        pd.testing.assert_series_equal(
            mc_draw._gross_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[652173.91304348, 717391.30434783, 782608.69565217, 847826.08695652],
            ),
        )

    def test_reserves(self, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()
        mc_draw._reserves = None
        pd.testing.assert_series_equal(
            mc_draw.reserves,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=19.5,
            ),
            check_freq=False,
        )
        pd.testing.assert_series_equal(
            mc_draw._reserves,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=19.5,
            ),
            check_freq=False,
        )

    def test_reserves_no_reserves(self, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()
        mc_draw.system.reserves = {}
        mc_draw.system.zones["CAISO"].reserves = {}
        mc_draw._reserves = None
        pd.testing.assert_series_equal(
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=0.0,
            ),
            mc_draw.reserves,
            check_freq=False,
        )

    def test_create_day_draw_map_for_resource_group(self, tmp_path, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()

        mc_draw.system.resource_groups["Solar"].draw_days_by_group = mock.MagicMock()
        mc_draw.system.resource_groups["Solar"].day_draw_map = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04"]),
            data=[
                pd.Timestamp("2015-01-03"),
                pd.Timestamp("2016-01-28"),
                pd.Timestamp("2013-01-14"),
                pd.Timestamp("2013-01-01"),
            ],
            name="Solar",
        )
        mc_draw.system.resource_groups["Wind"].draw_days_by_group = mock.MagicMock()
        mc_draw.system.resource_groups["Wind"].day_draw_map = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04"]),
            data=[
                pd.Timestamp("2013-01-18"),
                pd.Timestamp("2013-01-21"),
                pd.Timestamp("2013-01-02"),
                pd.Timestamp("2013-01-06"),
            ],
            name="Wind",
        )

        mc_draw.case_settings.draw_settings = "random"
        mc_draw.day_draw_dir = tmp_path

        mc_draw.create_day_draw_map_for_resource_group()

        assert tmp_path.joinpath("day_draw_map_0.csv").exists()

        output_df = pd.read_csv(tmp_path.joinpath("day_draw_map_0.csv"), infer_datetime_format=True, index_col=[0])
        pd.testing.assert_frame_equal(
            pd.DataFrame(
                index=pd.Index(
                    ["2010-01-01 00:00:00", "2010-01-02 00:00:00", "2010-01-03 00:00:00", "2010-01-04 00:00:00"]
                ),
                data={
                    "Wind_date": ["2013-01-18", "2013-01-21", "2013-01-02", "2013-01-06"],
                    "Solar_date": ["2015-01-03", "2016-01-28", "2013-01-14", "2013-01-01"],
                },
            ),
            output_df,
        )

    def test_create_day_draw_map_for_resource_group_existing_draw(self, monkeypatch, monte_carlo_draw_generator):
        def _read_csv_patch(*args, **kwargs):
            return pd.DataFrame(
                index=pd.Index(
                    ["2010-01-01 00:00:00", "2010-01-02 00:00:00", "2010-01-03 00:00:00", "2010-01-04 00:00:00"]
                ),
                data={
                    "Wind_date": ["2013-01-18", "2013-01-21", "2013-01-02", "2013-01-06"],
                    "Solar_date": ["2015-01-03", "2016-01-28", "2013-01-14", "2013-01-01"],
                },
            )

        monkeypatch.setattr(pd, "read_csv", _read_csv_patch)

        mc_draw = monte_carlo_draw_generator.get()

        mc_draw.create_day_draw_map_for_resource_group()

        expected_day_draw_maps = {
            "Solar": pd.Series(
                index=pd.DatetimeIndex(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04"]),
                data=[
                    pd.Timestamp("2015-01-03"),
                    pd.Timestamp("2016-01-28"),
                    pd.Timestamp("2013-01-14"),
                    pd.Timestamp("2013-01-01"),
                ],
                name="Solar",
            ),
            "Wind": pd.Series(
                index=pd.DatetimeIndex(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04"]),
                data=[
                    pd.Timestamp("2013-01-18"),
                    pd.Timestamp("2013-01-21"),
                    pd.Timestamp("2013-01-02"),
                    pd.Timestamp("2013-01-06"),
                ],
                name="Wind",
            ),
        }

        for resource_group in mc_draw.system.resource_groups.values():
            if resource_group.name in expected_day_draw_maps:
                pd.testing.assert_series_equal(expected_day_draw_maps[resource_group.name], resource_group.day_draw_map)
            else:
                assert not hasattr(resource_group, "day_draw_map")

    def test_net_load(self, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()
        mc_draw._gross_load = None
        mc_draw._reserves = None

        index = pd.date_range(start="2010-01-01 00:00", end="2010-01-01 03:00", freq="H")

        heuristic_provide_power = pd.Series(index=index, data=[10.0, 10.0, 10.0, 10.0])

        for resource_name in [
            "CA_Hydro",
            "CA_DR",
            "Battery_Storage",
            "CA_Thermal_1",
            "CA_Thermal_2",
            "DER_Solar",
            "Arizona_Solar",
            "CA_Wind_for_CA",
            "EV_V2G",
        ]:
            mc_draw.system.resources[resource_name].heuristic_provide_power_mw = heuristic_provide_power

        mc_draw.system.loads["CAISO_load"].profile = ts.NumericTimeseries(
            name="profile", data=pd.Series(index=index, data=[100, 110, 120, 130])
        )
        mc_draw.system.loads["CAISO_load"].annual_energy_forecast = ts.NumericTimeseries(
            name="annual_energy_forecast", data=pd.Series(index=pd.DatetimeIndex(["2030-01-01"]), data=[460.0])
        )
        mc_draw.system.loads["CAISO_load"].forecast_load((2030, 2030))

        pd.testing.assert_series_equal(mc_draw.net_load(), pd.Series(index=index, data=[50.0, 60.0, 70.0, 80.0]))
        pd.testing.assert_series_equal(
            mc_draw.net_load(["thermal_resources", "variable_resources"]),
            pd.Series(index=index, data=[50.0, 60.0, 70.0, 80.0]),
        )

        pd.testing.assert_series_equal(
            mc_draw.net_load(["storage_resources"]), pd.Series(index=index, data=[40.0, 50.0, 60.0, 70.0])
        )
        pd.testing.assert_series_equal(
            mc_draw.net_load(["storage_resources", "hydro_resources"]),
            pd.Series(index=index, data=[30.0, 40.0, 50.0, 60.0]),
        )
        pd.testing.assert_series_equal(
            mc_draw.net_load(["storage_resources", "hydro_resources", "shed_dr_resources"]),
            pd.Series(index=index, data=[20.0, 30.0, 40.0, 50.0]),
        )
        pd.testing.assert_series_equal(
            mc_draw.net_load(["storage_resources", "hydro_resources", "shed_dr_resources", "flex_load_resources"]),
            pd.Series(index=index, data=[10.0, 20.0, 30.0, 40.0]),
        )
        pd.testing.assert_series_equal(
            mc_draw.net_load(
                [
                    "storage_resources",
                    "hydro_resources",
                    "shed_dr_resources",
                    "flex_load_resources",
                    "generic_resources",
                ]
            ),
            pd.Series(index=index, data=[10.0, 20.0, 30.0, 40.0]),
        )

    @mock.patch.object(MonteCarloDraw, "create_day_draw_map_for_resource_group")
    def test_upsample(self, create_day_draw_map_mock, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()
        mc_draw.upsample_called = False  # Need to do this to re-upsample MC draw

        for component_dict_name, resource_name in [
            ("hydro_resources", "CA_Hydro"),
            ("shed_dr_resources", "CA_DR"),
            ("storage_resources", "Battery_Storage"),
            ("hybrid_storage_resources", "Hybrid_Battery_Storage"),
            ("thermal_resources", "CA_Thermal_1"),
            ("thermal_resources", "CA_Thermal_2"),
            ("variable_resources", "DER_Solar"),
            ("variable_resources", "Arizona_Solar"),
            ("variable_resources", "CA_Wind_for_CA"),
            ("hybrid_variable_resources", "CA_Solar_Hybrid"),
            ("flex_load_resources", "EV_V2G"),
        ]:
            getattr(mc_draw.system, component_dict_name)[resource_name] = mock.MagicMock()

        mc_draw.upsample(deterministic_upsampling=True)

        create_day_draw_map_mock.assert_called_once()
        for resource_mock in mc_draw.system.resources.values():
            resource_mock.upsample.assert_called_once()

    def test_simulate_outages(self, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()

        for component_dict_name, resource_name in [
            ("hydro_resources", "CA_Hydro"),
            ("shed_dr_resources", "CA_DR"),
            ("storage_resources", "Battery_Storage"),
            ("hybrid_storage_resources", "Hybrid_Battery_Storage"),
            ("thermal_resources", "CA_Thermal_1"),
            ("thermal_resources", "CA_Thermal_2"),
            ("variable_resources", "DER_Solar"),
            ("variable_resources", "Arizona_Solar"),
            ("variable_resources", "CA_Wind_for_CA"),
            ("hybrid_variable_resources", "CA_Solar_Hybrid"),
            ("flex_load_resources", "EV_V2G"),
        ]:
            getattr(mc_draw.system, component_dict_name)[resource_name] = mock.MagicMock(random_seed=1)

        mc_draw.simulate_outages()

        for resource_mock in mc_draw.system.resources.values():
            resource_mock.simulate_outages.assert_called_once()

    def test_rescale(self, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()

        # Mock all the resources, since we don't need to test their individual rescaling behaviors here
        for component_dict_name, resource_name in [
            ("hydro_resources", "CA_Hydro"),
            ("shed_dr_resources", "CA_DR"),
            ("storage_resources", "Battery_Storage"),
            ("hybrid_storage_resources", "Hybrid_Battery_Storage"),
            ("thermal_resources", "CA_Thermal_1"),
            ("thermal_resources", "CA_Thermal_2"),
            ("variable_resources", "DER_Solar"),
            ("variable_resources", "Arizona_Solar"),
            ("variable_resources", "CA_Wind_for_CA"),
            ("hybrid_variable_resources", "CA_Solar_Hybrid"),
            ("flex_load_resources", "EV_V2G"),
        ]:
            getattr(mc_draw.system, component_dict_name)[resource_name] = mock.MagicMock(random_seed=1)

        portfolio_vector = pd.Series(
            {
                "CA_Hydro": 10,
                "CA_DR": 20,
                "Battery_Storage": 30,
                "CA_Thermal_1": 40,
                "CA_Thermal_2": 50,
                "DER_Solar": 60,
            }
        )

        mc_draw.rescale(portfolio_vector=portfolio_vector, incremental=False)

        # Check that the `rescale()` method for each mock was called only once with the expected arguments
        mc_draw.system.resources["CA_Hydro"].rescale.assert_called_once_with(
            model_year=2030, capacity=10, incremental=False
        )
        mc_draw.system.resources["CA_DR"].rescale.assert_called_once_with(
            model_year=2030, capacity=20, incremental=False
        )
        mc_draw.system.resources["Battery_Storage"].rescale.assert_called_once_with(
            model_year=2030, capacity=30, incremental=False
        )
        mc_draw.system.resources["CA_Thermal_1"].rescale.assert_called_once_with(
            model_year=2030, capacity=40, incremental=False
        )
        mc_draw.system.resources["CA_Thermal_2"].rescale.assert_called_once_with(
            model_year=2030, capacity=50, incremental=False
        )
        mc_draw.system.resources["DER_Solar"].rescale.assert_called_once_with(
            model_year=2030, capacity=60, incremental=False
        )

        # Check that the resources not in the porfolio vector were not affected
        mc_draw.system.resources["Arizona_Solar"].rescale.assert_not_called()
        mc_draw.system.resources["CA_Wind_for_CA"].rescale.assert_not_called()
        mc_draw.system.resources["EV_V2G"].rescale.assert_not_called()

        # Test the remaining resources with incremental=True
        portfolio_vector_2 = pd.Series({"Arizona_Solar": 70, "CA_Wind_for_CA": 80, "EV_V2G": 90})
        mc_draw.rescale(portfolio_vector=portfolio_vector_2, incremental=True)

        mc_draw.system.resources["Arizona_Solar"].rescale.assert_called_once_with(
            model_year=2030, capacity=70, incremental=True
        )
        mc_draw.system.resources["CA_Wind_for_CA"].rescale.assert_called_once_with(
            model_year=2030, capacity=80, incremental=True
        )
        mc_draw.system.resources["EV_V2G"].rescale.assert_called_once_with(
            model_year=2030, capacity=90, incremental=True
        )

    def test_split(self, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()
        mc_draw._gross_load = None
        mc_draw._reserves = None
        mc_draw.upsample_called = False

        mc_draw.system.loads["CAISO_load"].scale_by_capacity = True

        # Resample the system load and reserves to cover 2010-2011 so that `MonteCarloDraw.upsample()` will create
        #   2 years of data for all components
        mc_draw.system.loads["CAISO_load"].resample_ts_attributes(
            modeled_years=(2030, 2030),
            weather_years=(2010, 2011),
            resample_weather_year_attributes=True,
            resample_non_weather_year_attributes=True,
        )
        mc_draw.system.loads["CAISO_load"].forecast_load(modeled_years=(2030, 2030))
        mc_draw.system.reserves["upward_reg_mw"].resample_ts_attributes(
            modeled_years=(2030, 2030),
            weather_years=(2010, 2011),
            resample_weather_year_attributes=True,
            resample_non_weather_year_attributes=True,
        )

        # Set a fake monthly budget on the Hydro resource to test splitting of monthly-frequency timeseries data
        mc_draw.system.hydro_resources["CA_Hydro"].energy_budget_monthly = ts.NumericTimeseries(
            name="energy_budget_monthly",
            data=pd.Series(index=pd.date_range(start="2010-01-01", end="2010-12-31", freq="MS"), data=1.0),
        )

        mc_draw.upsample(deterministic_upsampling=True)

        split_mc_draws = mc_draw.split(max_num_years=1)

        expected_hourly_indices = {
            "MC_draw_0_0": pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H"),
            "MC_draw_0_1": pd.date_range(start="2011-01-01 00:00", end="2011-12-31 23:00", freq="H"),
        }
        expected_daily_indices = {
            "MC_draw_0_0": pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="D"),
            "MC_draw_0_1": pd.date_range(start="2011-01-01 00:00", end="2011-12-31 23:00", freq="D"),
        }
        expected_monthly_indices = {
            "MC_draw_0_0": pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="MS"),
            "MC_draw_0_1": pd.date_range(start="2011-01-01 00:00", end="2011-12-31 23:00", freq="MS"),
        }

        assert split_mc_draws.keys() == expected_hourly_indices.keys()
        for split_draw_id, split_mc_draw in split_mc_draws.items():
            expected_hourly_index = expected_hourly_indices[split_draw_id]
            expected_daily_index = expected_daily_indices[split_draw_id]
            expected_monthly_index = expected_monthly_indices[split_draw_id]

            # Check that load attributes have been reindexed correctly
            pd.testing.assert_index_equal(
                expected_hourly_index, split_mc_draw.system.loads["CAISO_load"].profile.data.index, check_names=False
            )
            for model_year_profile in split_mc_draw.system.loads["CAISO_load"].model_year_profiles.values():
                pd.testing.assert_index_equal(
                    expected_hourly_index,
                    model_year_profile.data.index,
                    check_names=False,
                )

            # Check that reserves have been reindexed correctly
            pd.testing.assert_index_equal(
                expected_hourly_index,
                split_mc_draw.system.reserves["upward_reg_mw"].requirement.data.index,
                check_names=False,
            )

            # Check that resources have been reindexed correctly
            for resource in split_mc_draw.system.resources.values():
                for expected_weather_year_attr in [
                    "power_output_max",
                    "power_output_min",
                    "power_input_max",
                    "power_input_min",
                    "outage_profile",
                ]:
                    pd.testing.assert_index_equal(
                        expected_hourly_index,
                        getattr(resource, expected_weather_year_attr).data.index,
                        check_names=False,
                    )

                pd.testing.assert_index_equal(
                    pd.DatetimeIndex(["2030-01-01"]),
                    resource.capacity_planned.data.index,
                    check_names=False,
                )

            pd.testing.assert_index_equal(
                expected_daily_index,
                split_mc_draw.system.resources["CA_Hydro"].energy_budget_daily.data.index,
                check_names=False,
            )
            pd.testing.assert_index_equal(
                expected_monthly_index,
                split_mc_draw.system.resources["CA_Hydro"].energy_budget_monthly.data.index,
                check_names=False,
            )
            pd.testing.assert_index_equal(
                expected_daily_index,
                split_mc_draw.system.resources["EV_V2G"].energy_budget_daily.data.index,
                check_names=False,
            )

            # Check that MC draw gross load and reserves were not changed
            pd.testing.assert_index_equal(expected_hourly_index, split_mc_draw.gross_load.index, check_names=False)
            pd.testing.assert_series_equal(
                mc_draw.gross_load.loc[expected_hourly_index], split_mc_draw.gross_load, check_names=False
            )
            pd.testing.assert_index_equal(expected_hourly_index, split_mc_draw.reserves.index, check_names=False)
            pd.testing.assert_series_equal(
                mc_draw.reserves.loc[expected_hourly_index], split_mc_draw.reserves, check_names=False
            )

    def test_split_fewer_years(self, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()
        mc_draw._gross_load = None
        mc_draw._reserves = None
        mc_draw.upsample_called = False

        mc_draw.system.loads["CAISO_load"].scale_by_capacity = True

        # Resample the system load and reserves to cover 2010-2011 so that `MonteCarloDraw.upsample()` will create
        #   2 years of data for all components
        mc_draw.system.loads["CAISO_load"].resample_ts_attributes(
            modeled_years=(2030, 2030),
            weather_years=(2010, 2011),
            resample_weather_year_attributes=True,
            resample_non_weather_year_attributes=True,
        )
        mc_draw.system.loads["CAISO_load"].forecast_load(modeled_years=(2030, 2030))
        mc_draw.system.reserves["upward_reg_mw"].resample_ts_attributes(
            modeled_years=(2030, 2030),
            weather_years=(2010, 2011),
            resample_weather_year_attributes=True,
            resample_non_weather_year_attributes=True,
        )

        # Set a fake monthly budget on the Hydro resource to test splitting of monthly-frequency timeseries data
        mc_draw.system.hydro_resources["CA_Hydro"].energy_budget_monthly = ts.NumericTimeseries(
            name="energy_budget_monthly",
            data=pd.Series(index=pd.date_range(start="2010-01-01", end="2010-12-31", freq="MS"), data=1.0),
        )

        mc_draw.upsample(deterministic_upsampling=True)

        split_mc_draws = mc_draw.split(max_num_years=10)

        expected_hourly_indices = {
            "MC_draw_0_0": pd.date_range(start="2010-01-01 00:00", end="2011-12-31 23:00", freq="H"),
        }
        expected_daily_indices = {
            "MC_draw_0_0": pd.date_range(start="2010-01-01 00:00", end="2011-12-31 23:00", freq="D"),
        }
        expected_monthly_indices = {
            "MC_draw_0_0": pd.date_range(start="2010-01-01 00:00", end="2011-12-31 23:00", freq="MS"),
        }

        assert split_mc_draws.keys() == expected_hourly_indices.keys()
        for split_draw_id, split_mc_draw in split_mc_draws.items():
            expected_hourly_index = expected_hourly_indices[split_draw_id]
            expected_daily_index = expected_daily_indices[split_draw_id]
            expected_monthly_index = expected_monthly_indices[split_draw_id]

            # Check that load attributes have been reindexed correctly
            pd.testing.assert_index_equal(
                expected_hourly_index, split_mc_draw.system.loads["CAISO_load"].profile.data.index, check_names=False
            )
            for model_year_profile in split_mc_draw.system.loads["CAISO_load"].model_year_profiles.values():
                pd.testing.assert_index_equal(
                    expected_hourly_index,
                    model_year_profile.data.index,
                    check_names=False,
                )

            # Check that reserves have been reindexed correctly
            pd.testing.assert_index_equal(
                expected_hourly_index,
                split_mc_draw.system.reserves["upward_reg_mw"].requirement.data.index,
                check_names=False,
            )

            # Check that resources have been reindexed correctly
            for resource in split_mc_draw.system.resources.values():
                for expected_weather_year_attr in [
                    "power_output_max",
                    "power_output_min",
                    "power_input_max",
                    "power_input_min",
                    "outage_profile",
                ]:
                    pd.testing.assert_index_equal(
                        expected_hourly_index,
                        getattr(resource, expected_weather_year_attr).data.index,
                        check_names=False,
                    )

                pd.testing.assert_index_equal(
                    pd.DatetimeIndex(["2030-01-01"]),
                    resource.capacity_planned.data.index,
                    check_names=False,
                )

            pd.testing.assert_index_equal(
                expected_daily_index,
                split_mc_draw.system.resources["CA_Hydro"].energy_budget_daily.data.index,
                check_names=False,
            )
            pd.testing.assert_index_equal(
                expected_monthly_index,
                split_mc_draw.system.resources["CA_Hydro"].energy_budget_monthly.data.index,
                check_names=False,
            )
            pd.testing.assert_index_equal(
                expected_daily_index,
                split_mc_draw.system.resources["EV_V2G"].energy_budget_daily.data.index,
                check_names=False,
            )

            # Check that MC draw gross load and reserves were not changed
            pd.testing.assert_index_equal(expected_hourly_index, split_mc_draw.gross_load.index, check_names=False)
            pd.testing.assert_series_equal(
                mc_draw.gross_load.loc[expected_hourly_index], split_mc_draw.gross_load, check_names=False
            )
            pd.testing.assert_index_equal(expected_hourly_index, split_mc_draw.reserves.index, check_names=False)
            pd.testing.assert_series_equal(
                mc_draw.reserves.loc[expected_hourly_index], split_mc_draw.reserves, check_names=False
            )

    def test_identify_positive_net_load_windows(self):
        net_load = pd.Series(
            index=pd.date_range(start="2010-01-01 00:00", end="2011-12-31 23:00", freq="H"), data=-100.0
        )
        net_load.loc["2010-06-05 08:00":"2010-09-23 15:00"] = 100.0

        dates_to_include = _identify_positive_net_load_windows(net_load)
        pd.testing.assert_index_equal(dates_to_include, pd.date_range(start="2010-05-29", end="2010-09-30", freq="D"))

    def test_identify_positive_net_load_windows_multiple(self):
        net_load = pd.Series(
            index=pd.date_range(start="2010-01-01 00:00", end="2011-12-31 23:00", freq="H"), data=-100.0
        )
        net_load.loc["2010-06-05 08:00":"2010-07-08 15:00"] = 100.0
        net_load.loc["2010-08-25 11:00":"2010-08-28 07:00"] = 100.0

        dates_to_include = _identify_positive_net_load_windows(net_load)
        expected_dates = pd.date_range(start="2010-05-29", end="2010-07-15", freq="D").append(
            pd.date_range("2010-08-18", "2010-09-04", freq="D")
        )
        pd.testing.assert_index_equal(expected_dates, dates_to_include)

    def test_identify_positive_net_load_windows_multiple_grouped(self):
        net_load = pd.Series(
            index=pd.date_range(start="2010-01-01 00:00", end="2011-12-31 23:00", freq="H"), data=-100.0
        )
        net_load.loc["2010-03-14 12:00":"2010-03-17 20:00"] = 100.0
        net_load.loc["2010-06-05 08:00":"2010-07-08 15:00"] = 100.0
        net_load.loc["2010-07-25 11:00":"2010-08-28 07:00"] = 100.0

        dates_to_include = _identify_positive_net_load_windows(net_load)
        expected_dates = pd.date_range(start="2010-03-07", end="2010-03-24", freq="D").append(
            pd.date_range(start="2010-05-29", end="2010-09-04", freq="D")
        )
        pd.testing.assert_index_equal(expected_dates, dates_to_include)

    def test_identify_positive_net_load_windows_wrap(self):
        net_load = pd.Series(
            index=pd.date_range(start="2010-01-01 00:00", end="2011-12-31 23:00", freq="H"), data=-100.0
        )
        net_load.loc["2010-01-09 08:00":"2010-01-09 15:00"] = 100.0
        net_load.loc["2011-12-15 11:00":"2011-12-20 07:00"] = 100.0

        dates_to_include = _identify_positive_net_load_windows(net_load)
        expected_dates = pd.date_range(start="2010-01-01", end="2010-01-16", freq="D").append(
            pd.date_range(start="2011-12-08", end="2011-12-31", freq="D")
        )
        pd.testing.assert_index_equal(expected_dates, dates_to_include)

    def test_identify_positive_net_load_windows_all_negative(self):
        net_load = pd.Series(
            index=pd.date_range(start="2010-01-01 00:00", end="2011-12-31 23:00", freq="H"), data=-100.0
        )
        dates_to_include = _identify_positive_net_load_windows(net_load)
        pd.testing.assert_index_equal(pd.DatetimeIndex([]), dates_to_include)

    def test_identify_positive_net_load_windows_all_positive(self):
        net_load = pd.Series(index=pd.date_range(start="2010-01-01 00:00", end="2011-12-31 23:00", freq="H"), data=50.0)
        dates_to_include = _identify_positive_net_load_windows(net_load)
        pd.testing.assert_index_equal(
            pd.date_range(start="2010-01-01 00:00", end="2011-12-31 23:00", freq="D"), dates_to_include
        )

    @mock.patch(
        "new_modeling_toolkit.system.electric.resources.storage.StorageResource.set_initial_SOC_for_optimization"
    )
    @mock.patch(
        "new_modeling_toolkit.system.electric.resources.hybrid.HybridStorageResource.set_initial_SOC_for_optimization"
    )
    @mock.patch("new_modeling_toolkit.system.electric.resources.hydro.HydroResource.adjust_budgets_for_optimization")
    def test_compress(
        self,
        hydro_resources_adjust_budget_mock,
        hybrid_storage_resources_set_SOC_mock,
        storage_resources_set_SOC_mock,
        monte_carlo_draw_generator,
    ):
        mc_draw = monte_carlo_draw_generator.get()

        # Set a fake monthly budget on the Hydro resource to test splitting of monthly-frequency timeseries data
        mc_draw.system.hydro_resources["CA_Hydro"].energy_budget_monthly = ts.NumericTimeseries(
            name="energy_budget_monthly",
            data=pd.Series(index=pd.date_range(start="2010-01-01", end="2010-12-31", freq="MS"), data=1.0),
        )
        mc_draw.heuristic_dispatch(perfect_capacity=0)

        mc_draw.compress(perfect_capacity=0, heuristic_net_load_subclasses=["thermal_resources", "variable_resources"])

        expected_dispatch_windows_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "window_label": 1.0,
                        "index": pd.date_range(start="2010-01-01 00:00", end="2010-01-10 23:00", freq="H"),
                        "include": 1.0,
                        "weight": None,
                    }
                ),
                pd.DataFrame(
                    {
                        "window_label": 2.0,
                        "index": pd.date_range(start="2010-02-15 00:00", end="2010-03-02 23:00", freq="H"),
                        "include": 1.0,
                        "weight": None,
                    }
                ),
                pd.DataFrame(
                    {
                        "window_label": 3.0,
                        "index": pd.date_range(start="2010-05-17 00:00", end="2010-06-13 23:00", freq="H"),
                        "include": 1.0,
                        "weight": None,
                    }
                ),
                pd.DataFrame(
                    {
                        "window_label": 4.0,
                        "index": pd.date_range(start="2010-07-08 00:00", end="2010-10-07 23:00", freq="H"),
                        "include": 1.0,
                        "weight": None,
                    }
                ),
                pd.DataFrame(
                    {
                        "window_label": 5.0,
                        "index": pd.date_range(start="2010-11-12 00:00", end="2010-11-30 23:00", freq="H"),
                        "include": 1.0,
                        "weight": None,
                    }
                ),
            ],
            axis=0,
        )
        expected_dispatch_windows_df = expected_dispatch_windows_df.set_index(["window_label", "index"])

        pd.testing.assert_frame_equal(expected_dispatch_windows_df, mc_draw.temporal_settings.dispatch_windows_df)

        assert mc_draw.temporal_settings.name == "recap"
        assert mc_draw.temporal_settings.dispatch_window_edge_effects == DispatchWindowEdgeEffects.FIXED_INITIAL_SOC
        assert mc_draw.temporal_settings.modeled_years == [2030]

        expected_hydro_budget_adjustment_series = pd.Series(
            index=pd.date_range(start="2010-12-01 00:00", end="2010-12-31 23:00", freq="H").append(
                pd.date_range(start="2010-01-01", end="2010-11-30 23:00", freq="H")
            ),
            data=0.0,
            name="include",
        )
        expected_hydro_budget_adjustment_series.loc[
            pd.date_range(start="2010-01-01 00:00", end="2010-01-10 23:00", freq="H")
        ] = 1.0
        expected_hydro_budget_adjustment_series.loc[
            pd.date_range(start="2010-02-15 00:00", end="2010-03-02 23:00", freq="H")
        ] = 1.0
        expected_hydro_budget_adjustment_series.loc[
            pd.date_range(start="2010-05-17 00:00", end="2010-06-13 23:00", freq="H")
        ] = 1.0
        expected_hydro_budget_adjustment_series.loc[
            pd.date_range(start="2010-07-08 00:00", end="2010-10-07 23:00", freq="H")
        ] = 1.0
        expected_hydro_budget_adjustment_series.loc[
            pd.date_range(start="2010-11-12 00:00", end="2010-11-30 23:00", freq="H")
        ] = 1.0
        for hydro_resource in mc_draw.system.hydro_resources.values():
            hydro_resource.adjust_budgets_for_optimization.assert_called_once()
            pd.testing.assert_series_equal(
                expected_hydro_budget_adjustment_series,
                hydro_resource.adjust_budgets_for_optimization.call_args[1][
                    "timestamps_included_in_optimization_flags"
                ],
            )

        expected_storage_resource_df_in = pd.concat(
            [
                pd.DataFrame(
                    index=pd.date_range(start="2010-12-01 00:00", end="2010-12-31 23:00", freq="H"),
                    data={
                        "window_label": 1.0,
                        "include": 0.0,
                    },
                ),
                pd.DataFrame(
                    index=pd.date_range(start="2010-01-01 00:00", end="2010-01-10 23:00", freq="H"),
                    data={
                        "window_label": 1.0,
                        "include": 1.0,
                    },
                ),
                pd.DataFrame(
                    index=pd.date_range(start="2010-01-11 00:00", end="2010-02-14 23:00", freq="H"),
                    data={
                        "window_label": 2.0,
                        "include": 0.0,
                    },
                ),
                pd.DataFrame(
                    index=pd.date_range(start="2010-02-15 00:00", end="2010-03-02 23:00", freq="H"),
                    data={
                        "window_label": 2.0,
                        "include": 1.0,
                    },
                ),
                pd.DataFrame(
                    index=pd.date_range(start="2010-03-03 00:00", end="2010-05-16 23:00", freq="H"),
                    data={
                        "window_label": 3.0,
                        "include": 0.0,
                    },
                ),
                pd.DataFrame(
                    index=pd.date_range(start="2010-05-17 00:00", end="2010-06-13 23:00", freq="H"),
                    data={
                        "window_label": 3.0,
                        "include": 1.0,
                    },
                ),
                pd.DataFrame(
                    index=pd.date_range(start="2010-06-14 00:00", end="2010-07-07 23:00", freq="H"),
                    data={
                        "window_label": 4.0,
                        "include": 0.0,
                    },
                ),
                pd.DataFrame(
                    index=pd.date_range(start="2010-07-08 00:00", end="2010-10-07 23:00", freq="H"),
                    data={
                        "window_label": 4.0,
                        "include": 1.0,
                    },
                ),
                pd.DataFrame(
                    index=pd.date_range(start="2010-10-08 00:00", end="2010-11-11 23:00", freq="H"),
                    data={
                        "window_label": 5.0,
                        "include": 0.0,
                    },
                ),
                pd.DataFrame(
                    index=pd.date_range(start="2010-11-12 00:00", end="2010-11-30 23:00", freq="H"),
                    data={
                        "window_label": 5.0,
                        "include": 1.0,
                    },
                ),
            ],
            axis=0,
        )

        expected_storage_resource_df_in = expected_storage_resource_df_in.loc[:, ["include", "window_label"]]

        for storage_resource in list(mc_draw.system.storage_resources.values()) + list(
            mc_draw.system.hybrid_storage_resources.values()
        ):
            storage_resource.set_initial_SOC_for_optimization.assert_called_once()
            pd.testing.assert_series_equal(
                expected_storage_resource_df_in.loc[:, "include"],
                storage_resource.set_initial_SOC_for_optimization.call_args[1][
                    "timestamps_included_in_optimization_flags"
                ],
            )
            pd.testing.assert_series_equal(
                expected_storage_resource_df_in.loc[:, "window_label"],
                storage_resource.set_initial_SOC_for_optimization.call_args[1]["window_labels"],
            )

    def test_dispatch(self, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()

        with mock.patch(
            "new_modeling_toolkit.recap.monte_carlo_draw.MonteCarloDraw.gross_load", new_callable=mock.PropertyMock
        ) as mc_draw_gross_load_mock:
            with mock.patch(
                "new_modeling_toolkit.recap.monte_carlo_draw.MonteCarloDraw.reserves", new_callable=mock.PropertyMock
            ) as mc_draw_reserves_mock:
                mc_draw_gross_load_mock.return_value = pd.Series(
                    index=pd.DatetimeIndex(
                        ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                    ),
                    data=[80.0, 90.0, 100.0, 110.0],
                )

                mc_draw_reserves_mock.return_value = pd.Series(
                    index=pd.DatetimeIndex(
                        ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                    ),
                    data=[5.0, 5.0, 5.0, 5.0],
                )

                for resource in mc_draw.system.resources.values():
                    resource.dispatch = mock.MagicMock()
                    resource.dispatch.side_effect = lambda net_load, model_year: net_load - pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=10.0,
                    )

                mc_draw.heuristic_dispatch(perfect_capacity=0, mode=ResourceGrouping.DEFAULT)

                pd.testing.assert_series_equal(
                    mc_draw.unserved_energy_and_reserve,
                    pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[0.0, 0.0, 0.0, 5.0],
                    ),
                )

                expected_resource_dispatch_net_load_inputs = {
                    "CA_Thermal_1": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[85.0, 95.0, 105.0, 115.0],
                    ),
                    "CA_Thermal_2": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[75.0, 85.0, 95.0, 105.0],
                    ),
                    "DER_Solar": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[65.0, 75.0, 85.0, 95.0],
                    ),
                    "Arizona_Solar": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[55.0, 65.0, 75.0, 85.0],
                    ),
                    "CA_Wind_for_CA": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[45.0, 55.0, 65.0, 75.0],
                    ),
                    "CA_Hydro": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[35.0, 45.0, 55.0, 65.0],
                    ),
                    "CA_Solar_Hybrid": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[25.0, 35.0, 45.0, 55.0],
                    ),
                    "Hybrid_Battery_Storage": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[15.0, 25.0, 35.0, 45.0],
                    ),
                    "Battery_Storage": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[5.0, 15.0, 25.0, 35.0],
                    ),
                    "EV_V2G": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[-5.0, 5.0, 15.0, 25.0],
                    ),
                    "CA_DR": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[-15.0, -5.0, 5.0, 15.0],
                    ),
                }
                for resource_name, resource in mc_draw.system.resources.items():
                    logging.info(f"Resource: {resource_name}")
                    pd.testing.assert_series_equal(
                        expected_resource_dispatch_net_load_inputs[resource_name],
                        resource.dispatch.call_args[1]["net_load"],
                    )

    def test_dispatch_no_ELRs(self, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()

        with mock.patch(
            "new_modeling_toolkit.recap.monte_carlo_draw.MonteCarloDraw.gross_load", new_callable=mock.PropertyMock
        ) as mc_draw_gross_load_mock:
            with mock.patch(
                "new_modeling_toolkit.recap.monte_carlo_draw.MonteCarloDraw.reserves", new_callable=mock.PropertyMock
            ) as mc_draw_reserves_mock:
                mc_draw_gross_load_mock.return_value = pd.Series(
                    index=pd.DatetimeIndex(
                        ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                    ),
                    data=[80.0, 90.0, 100.0, 110.0],
                )

                mc_draw_reserves_mock.return_value = pd.Series(
                    index=pd.DatetimeIndex(
                        ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                    ),
                    data=[5.0, 5.0, 5.0, 5.0],
                )

                for resource in mc_draw.system.resources.values():
                    resource.dispatch = mock.MagicMock()
                    resource.dispatch.side_effect = lambda net_load, model_year: net_load - pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=10.0,
                    )

                mc_draw.heuristic_dispatch(perfect_capacity=0, mode=ResourceGrouping.NO_ELRS)
                pd.testing.assert_series_equal(
                    mc_draw.unserved_energy_and_reserve,
                    pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[35.0, 45.0, 55.0, 65.0],
                    ),
                )

                expected_resource_dispatch_net_load_inputs = {
                    "CA_Thermal_1": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[85.0, 95.0, 105.0, 115.0],
                    ),
                    "CA_Thermal_2": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[75.0, 85.0, 95.0, 105.0],
                    ),
                    "DER_Solar": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[65.0, 75.0, 85.0, 95.0],
                    ),
                    "Arizona_Solar": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[55.0, 65.0, 75.0, 85.0],
                    ),
                    "CA_Wind_for_CA": pd.Series(
                        index=pd.DatetimeIndex(
                            ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                        ),
                        data=[45.0, 55.0, 65.0, 75.0],
                    ),
                }

                # Note: CA_Solar_Hybrid grouped w Hybrid_Battery_Storage (considered an ELR)
                expected_resources_dispatch_not_called = [
                    "CA_Hydro",
                    "CA_Solar_Hybrid",
                    "Hybrid_Battery_Storage",
                    "Battery_Storage",
                    "EV_V2G",
                    "CA_DR",
                ]

                for resource_name, resource in mc_draw.system.resources.items():
                    if resource_name in expected_resource_dispatch_net_load_inputs:
                        pd.testing.assert_series_equal(
                            expected_resource_dispatch_net_load_inputs[resource_name],
                            resource.dispatch.call_args[1]["net_load"],
                        )
                    elif resource_name in expected_resources_dispatch_not_called:
                        resource.dispatch.assert_not_called()
                    else:
                        raise AssertionError(f"Unexpected resource `{resource_name}` found in test")

    def test_calculate_dispatch_results(self, monte_carlo_draw_generator):
        mc_draw = monte_carlo_draw_generator.get()

        mc_draw.unserved_energy_and_reserve = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[45.0, 55.0, 65.0, 75.0],
        )

        mc_draw.perfect_capacity = 0.0

        for resource in mc_draw.system.resources.values():
            # Note that in reality CA_Solar_Hybrid heuristic_provide_power will actually be 0
            resource.heuristic_provide_power_mw = pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[10.0, 25.0, 2.0, 3.0],
            )

        mc_draw.system.resources["CA_Hydro"].optimized_provide_power_mw = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 01:00", "2010-01-01 02:00"]), data=[30.0, 32.0]
        )
        mc_draw.system.resources["CA_DR"].optimized_provide_power_mw = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 01:00", "2010-01-01 02:00"]), data=[10.0, 0.0]
        )
        expected_dispatch_results = pd.DataFrame(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data={
                "unserved_energy_and_reserve": [45.0, 55.0, 65.0, 75.0],
                "gross_load": [312.400824, 299.539684, 290.621152, 285.700582],
                "reserves": [19.5, 19.5, 19.5, 19.5],
                "perfect_capacity": [0.0, 0.0, 0.0, 0.0],
                "EV_V2G": [10.0, 25.0, 2.0, 3.0],
                "CA_Hydro": [10.0, 30.0, 32.0, 3.0],
                "CA_DR": [10.0, 10.0, 0.0, 3.0],
                "Battery_Storage": [10.0, 25.0, 2.0, 3.0],
                "Hybrid_Battery_Storage": [10.0, 25.0, 2.0, 3.0],
                "CA_Thermal_1": [10.0, 25.0, 2.0, 3.0],
                "CA_Thermal_2": [10.0, 25.0, 2.0, 3.0],
                "DER_Solar": [10.0, 25.0, 2.0, 3.0],
                "Arizona_Solar": [10.0, 25.0, 2.0, 3.0],
                "CA_Wind_for_CA": [10.0, 25.0, 2.0, 3.0],
                "CA_Solar_Hybrid": [10.0, 25.0, 2.0, 3.0],
            },
        )

        pd.testing.assert_frame_equal(expected_dispatch_results, mc_draw.calculate_dispatch_results())
