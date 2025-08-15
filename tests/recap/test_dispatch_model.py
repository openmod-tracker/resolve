import copy

import pandas as pd
import pytest

from new_modeling_toolkit.core.temporal.new_temporal import DispatchWindowEdgeEffects
from new_modeling_toolkit.recap.dispatch_model import DispatchModel
from new_modeling_toolkit.recap.recap_case_settings import DispatchObjective


class TestDispatchModel:
    @pytest.fixture(scope="class")
    def dispatch_model_generator(self, monte_carlo_draw_generator):
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
                        "hybrid_variable_resources",
                        "hybrid_storage_resources",
                        "storage_resources",
                        "flex_load_resources",
                        "shed_dr_resources",
                    ],
                )

                mc_draw._gross_load.loc[:] = 100
                mc_draw._reserves.loc[:] = 50.0
                mc_draw.system.resources["CA_Thermal_1"].heuristic_provide_power_mw.loc[:] = 50.0
                mc_draw.system.resources["CA_Thermal_2"].heuristic_provide_power_mw.loc[:] = 50.0
                mc_draw.system.resources["DER_Solar"].heuristic_provide_power_mw.loc[:] = 50.0
                mc_draw.system.resources["Arizona_Solar"].heuristic_provide_power_mw.loc[:] = 50.0
                mc_draw.system.resources["CA_Wind_for_CA"].heuristic_provide_power_mw.loc[:] = 50.0
                mc_draw.system.resources["CA_Solar_Hybrid"].heuristic_provide_power_mw.loc[:] = 50.0

                self.dispatch_model = DispatchModel(monte_carlo_draw=mc_draw, perfect_capacity=0)

            def get(self):
                return copy.deepcopy(self.dispatch_model)

        return DispatchModelGenerator()

    @pytest.fixture(scope="class")
    def dispatch_model_EUE_generator(self, monte_carlo_draw_generator):
        class DispatchModelGeneratorEUE:
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

                mc_draw._gross_load.loc[mc_draw._gross_load.index] = 100
                mc_draw._reserves.loc[mc_draw._gross_load.index] = 50.0
                mc_draw.system.resources["CA_Thermal_1"].heuristic_provide_power_mw.loc[
                    mc_draw._gross_load.index
                ] = 50.0
                mc_draw.system.resources["CA_Thermal_2"].heuristic_provide_power_mw.loc[
                    mc_draw._gross_load.index
                ] = 50.0
                mc_draw.system.resources["DER_Solar"].heuristic_provide_power_mw.loc[mc_draw._gross_load.index] = 50.0
                mc_draw.system.resources["Arizona_Solar"].heuristic_provide_power_mw.loc[
                    mc_draw._gross_load.index
                ] = 50.0
                mc_draw.system.resources["CA_Wind_for_CA"].heuristic_provide_power_mw.loc[
                    mc_draw._gross_load.index
                ] = 50.0

                mc_draw.case_settings.dispatch_objective = DispatchObjective.EUE

                self.dispatch_model = DispatchModel(monte_carlo_draw=mc_draw, perfect_capacity=0)

            def get(self):
                return copy.deepcopy(self.dispatch_model)

        return DispatchModelGeneratorEUE()

    @pytest.fixture(scope="class")
    def make_dispatch_model_copy(self, dispatch_model_generator):
        """Factory fixture for generating a copy of a dispatch model containing one thermal resource and the resource
        being tested.
        """
        dispatch_model = dispatch_model_generator.get()

        def _dispatch_model_factory():
            return copy.deepcopy(dispatch_model)

        return _dispatch_model_factory

    @pytest.fixture(scope="class")
    def make_dispatch_model_EUE_copy(self, dispatch_model_EUE_generator):
        """Factory fixture for generating a copy of a dispatch model containing one thermal resource and the resource
        being tested.
        """
        dispatch_model = dispatch_model_EUE_generator.get()

        def _dispatch_model_factory():
            return copy.deepcopy(dispatch_model)

        return _dispatch_model_factory

    @pytest.fixture(scope="class")
    def in_periods(self):
        in_periods = (
            pd.DatetimeIndex([])
            .union(
                pd.date_range(start="2010-01-01 00:00", end="2010-01-10 23:00", freq="H"),
            )
            .union(
                pd.date_range(start="2010-02-15 00:00", end="2010-03-02 23:00", freq="H"),
            )
            .union(
                pd.date_range(start="2010-05-17 00:00", end="2010-06-13 23:00", freq="H"),
            )
            .union(
                pd.date_range(start="2010-07-09 00:00", end="2010-10-07 23:00", freq="H"),
            )
            .union(
                pd.date_range(start="2010-11-12 00:00", end="2010-11-30 23:00", freq="H"),
            )
        )
        in_periods.names = ["index"]
        return in_periods

    def test_temporal_settings(self, make_dispatch_model_copy, in_periods):
        dispatch_model = make_dispatch_model_copy()

        # Test temporal settings
        assert dispatch_model.dispatch_window_edge_effects == DispatchWindowEdgeEffects.FIXED_INITIAL_SOC
        assert dispatch_model.temporal_settings.modeled_years == [2030]

        pd.testing.assert_index_equal(
            dispatch_model.temporal_settings.dispatch_windows_df.reset_index(level=0).index, in_periods
        )

        assert (dispatch_model.temporal_settings.dispatch_windows_df["include"] == 1).all()

    def test_pyomo_sets(self, make_dispatch_model_copy):
        dispatch_model = make_dispatch_model_copy()

        # Test pyomo sets
        assert dispatch_model.MODELED_YEARS == [2030]
        assert dispatch_model.WEATHER_YEARS == [pd.Timestamp(year=2010, month=1, day=1, hour=0)]
        assert dispatch_model.DISPATCH_WINDOWS == [1, 2, 3, 4, 5]

        assert (
            dispatch_model.TIMESTAMPS_IN_DISPATCH_WINDOWS[1].data()
            == pd.date_range(start="2010-01-01 00:00", end="2010-01-10 23:00", freq="H")
        ).all()
        assert (
            dispatch_model.TIMESTAMPS_IN_DISPATCH_WINDOWS[2].data()
            == pd.date_range(start="2010-02-15 00:00", end="2010-03-02 23:00", freq="H")
        ).all()
        assert (
            dispatch_model.TIMESTAMPS_IN_DISPATCH_WINDOWS[3].data()
            == pd.date_range(start="2010-05-17 00:00", end="2010-06-13 23:00", freq="H")
        ).all()
        assert (
            dispatch_model.TIMESTAMPS_IN_DISPATCH_WINDOWS[4].data()
            == pd.date_range(start="2010-07-09 00:00", end="2010-10-07 23:00", freq="H")
        ).all()
        assert (
            dispatch_model.TIMESTAMPS_IN_DISPATCH_WINDOWS[5].data()
            == pd.date_range(start="2010-11-12 00:00", end="2010-11-30 23:00", freq="H")
        ).all()

        assert (
            dispatch_model.DAYS_IN_DISPATCH_WINDOWS[1].data()
            == pd.date_range(start="2010-01-01 00:00", end="2010-01-10 23:00", freq="D")
        ).all()
        assert (
            dispatch_model.DAYS_IN_DISPATCH_WINDOWS[2].data()
            == pd.date_range(start="2010-02-15 00:00", end="2010-03-02 23:00", freq="D")
        ).all()
        assert (
            dispatch_model.DAYS_IN_DISPATCH_WINDOWS[3].data()
            == pd.date_range(start="2010-05-17 00:00", end="2010-06-13 23:00", freq="D")
        ).all()
        assert (
            dispatch_model.DAYS_IN_DISPATCH_WINDOWS[4].data()
            == pd.date_range(start="2010-07-09 00:00", end="2010-10-07 23:00", freq="D")
        ).all()
        assert (
            dispatch_model.DAYS_IN_DISPATCH_WINDOWS[5].data()
            == pd.date_range(start="2010-11-12 00:00", end="2010-11-30 23:00", freq="D")
        ).all()

        assert dispatch_model.MONTHS_IN_DISPATCH_WINDOWS[1].data() == (pd.Timestamp(year=2010, month=1, day=1),)
        assert dispatch_model.MONTHS_IN_DISPATCH_WINDOWS[2].data() == (
            pd.Timestamp(year=2010, month=2, day=1),
            pd.Timestamp(year=2010, month=3, day=1),
        )
        assert dispatch_model.MONTHS_IN_DISPATCH_WINDOWS[3].data() == (
            pd.Timestamp(year=2010, month=5, day=1),
            pd.Timestamp(year=2010, month=6, day=1),
        )
        assert dispatch_model.MONTHS_IN_DISPATCH_WINDOWS[4].data() == (
            pd.Timestamp(year=2010, month=7, day=1),
            pd.Timestamp(year=2010, month=8, day=1),
            pd.Timestamp(year=2010, month=9, day=1),
            pd.Timestamp(year=2010, month=10, day=1),
        )
        assert dispatch_model.MONTHS_IN_DISPATCH_WINDOWS[5].data() == (pd.Timestamp(year=2010, month=11, day=1),)

        assert dispatch_model.WEATHER_YEARS_IN_DISPATCH_WINDOWS[1].data() == (pd.Timestamp(year=2010, month=1, day=1),)
        assert dispatch_model.WEATHER_YEARS_IN_DISPATCH_WINDOWS[2].data() == (pd.Timestamp(year=2010, month=1, day=1),)
        assert dispatch_model.WEATHER_YEARS_IN_DISPATCH_WINDOWS[3].data() == (pd.Timestamp(year=2010, month=1, day=1),)
        assert dispatch_model.WEATHER_YEARS_IN_DISPATCH_WINDOWS[4].data() == (pd.Timestamp(year=2010, month=1, day=1),)
        assert dispatch_model.WEATHER_YEARS_IN_DISPATCH_WINDOWS[5].data() == (pd.Timestamp(year=2010, month=1, day=1),)

    def test_resources_to_construct_and_pyomo_blocks(self, make_dispatch_model_copy):
        dispatch_model = make_dispatch_model_copy()

        # Get full list of resources
        assert list(dispatch_model.system.resources.keys()) == [
            "EV_V2G",
            "CA_Hydro",
            "CA_DR",
            "Battery_Storage",
            "Hybrid_Battery_Storage",
            "CA_Thermal_1",
            "CA_Thermal_2",
            "DER_Solar",
            "Arizona_Solar",
            "CA_Wind_for_CA",
            "CA_Solar_Hybrid",
        ]

        # Get names of resources in resources_to_construct
        resources_to_construct_names = [resource.name for resource in dispatch_model.resources_to_construct]
        assert resources_to_construct_names == [
            "CA_Hydro",
            "CA_Solar_Hybrid",
            "Hybrid_Battery_Storage",
            "Battery_Storage",
            "EV_V2G",
            "CA_DR",
        ]

        # Test pyomo blocks
        assert list(dispatch_model.blocks.keys()) == resources_to_construct_names

    def test_net_load(self, make_dispatch_model_copy):
        dispatch_model = make_dispatch_model_copy()

        target_net_load = pd.Series(index=dispatch_model.full_index, data=-150.0, name="load")

        pd.testing.assert_series_equal(dispatch_model.net_load, target_net_load)

    def test_EUE_and_100xLOLE_objective_function(self, make_dispatch_model_copy):
        dispatch_model = make_dispatch_model_copy()

        assert dispatch_model.objective_fn == DispatchObjective.EUE_and_100xLOLE
        assert dispatch_model.Big_M == 1e6

        for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
            dispatch_model.Unserved_Energy_MW[dispatch_window, timestamp].fix(0)
            dispatch_model.Unserved_Reserve_MW[dispatch_window, timestamp].fix(0)

        for day in dispatch_model.DAYS:
            dispatch_model.LOLE_Count[day].fix(0)

        for day in dispatch_model.DAYS:
            assert dispatch_model.LOLE_Counter_Constraint[day].body() == 0
            assert dispatch_model.LOLE_Counter_Constraint[day].upper() == 0
            assert dispatch_model.LOLE_Counter_Constraint[day].expr()

        dispatch_model.Unserved_Energy_MW[3.0, pd.Timestamp(year=2010, month=6, day=1, hour=8)].fix(100)
        dispatch_model.Unserved_Reserve_MW[3.0, pd.Timestamp(year=2010, month=6, day=1, hour=8)].fix(200)
        assert not dispatch_model.LOLE_Counter_Constraint[pd.Timestamp(year=2010, month=6, day=1)].expr()

        dispatch_model.LOLE_Count[pd.Timestamp(year=2010, month=6, day=1)].fix(1)
        assert dispatch_model.LOLE_Counter_Constraint[pd.Timestamp(year=2010, month=6, day=1)].expr()

        dispatch_model.LOLE_Count[pd.Timestamp(year=2010, month=6, day=2)].fix(1)
        dispatch_model.LOLE_Count[pd.Timestamp(year=2010, month=6, day=3)].fix(1)
        dispatch_model.LOLE_Count[pd.Timestamp(year=2010, month=6, day=4)].fix(1)

        assert dispatch_model.Total_Unserved_Energy_and_100x_Unserved_Energy_Events.expr() == 700

    def test_EUE_objective_function(self, make_dispatch_model_EUE_copy):
        dispatch_model = make_dispatch_model_EUE_copy()

        assert dispatch_model.objective_fn == DispatchObjective.EUE

        for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
            dispatch_model.Unserved_Energy_MW[dispatch_window, timestamp].fix(0)
            dispatch_model.Unserved_Reserve_MW[dispatch_window, timestamp].fix(0)

        dispatch_model.Unserved_Energy_MW[3.0, pd.Timestamp(year=2010, month=6, day=1, hour=8)].fix(100)
        dispatch_model.Unserved_Reserve_MW[3.0, pd.Timestamp(year=2010, month=6, day=1, hour=8)].fix(200)

        assert dispatch_model.Total_Unserved_Energy.expr() == 300

    def test_unserved_energy_and_reserve_constraint(self, make_dispatch_model_copy):
        dispatch_model = make_dispatch_model_copy()

        for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
            dispatch_model.Unserved_Energy_MW[dispatch_window, timestamp].fix(0)
            dispatch_model.Unserved_Reserve_MW[dispatch_window, timestamp].fix(0)
            dispatch_model.Provide_Power_System_MW[dispatch_window, timestamp].fix(100)
            dispatch_model.Provide_Reserve_System_MW[dispatch_window, timestamp].fix(50)

        for resource in dispatch_model.resources_to_construct:
            block = dispatch_model.blocks[resource.name]
            for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
                block.power_output[2030, dispatch_window, timestamp].fix(100.0)
                block.power_input[2030, dispatch_window, timestamp].fix(50.0)
                block.provide_reserve[2030, dispatch_window, timestamp].fix(75.0)

        for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
            assert dispatch_model.Unserved_Energy_Constraint[2030, dispatch_window, timestamp].upper() == 0.0
            assert dispatch_model.Unserved_Energy_Constraint[2030, dispatch_window, timestamp].body() == -400.0
            assert dispatch_model.Unserved_Energy_Constraint[2030, dispatch_window, timestamp].expr()
            assert dispatch_model.Unserved_Reserve_Constraint[2030, dispatch_window, timestamp].upper() == 0.0
            assert dispatch_model.Unserved_Reserve_Constraint[2030, dispatch_window, timestamp].body() == -450.0
            assert dispatch_model.Unserved_Reserve_Constraint[2030, dispatch_window, timestamp].expr()

    def test_max_system_output_constraint(self, make_dispatch_model_copy):
        dispatch_model = make_dispatch_model_copy()

        for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
            dispatch_model.Unserved_Energy_MW[dispatch_window, timestamp].fix(0)
            dispatch_model.Unserved_Reserve_MW[dispatch_window, timestamp].fix(0)
            dispatch_model.Provide_Power_System_MW[dispatch_window, timestamp].fix(100)
            dispatch_model.Provide_Reserve_System_MW[dispatch_window, timestamp].fix(50)

        for resource in dispatch_model.resources_to_construct:
            block = dispatch_model.blocks[resource.name]
            for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
                block.power_output[2030, dispatch_window, timestamp].fix(100.0)
                block.power_input[2030, dispatch_window, timestamp].fix(50.0)
                block.provide_reserve[2030, dispatch_window, timestamp].fix(75.0)

        for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
            assert dispatch_model.Max_System_Output_Constraint[dispatch_window, timestamp].body() == 150.0
            assert dispatch_model.Max_System_Output_Constraint[dispatch_window, timestamp].upper() == 150.0
            assert dispatch_model.Max_System_Output_Constraint[dispatch_window, timestamp].expr()

    def test_calculate_dispatch_results(self, make_dispatch_model_copy, in_periods):
        dispatch_model = make_dispatch_model_copy()

        for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
            dispatch_model.Unserved_Energy_MW[dispatch_window, timestamp].fix(0)
            dispatch_model.Unserved_Reserve_MW[dispatch_window, timestamp].fix(0)
            dispatch_model.Provide_Power_System_MW[dispatch_window, timestamp].fix(100)
            dispatch_model.Provide_Reserve_System_MW[dispatch_window, timestamp].fix(50)

        for resource in dispatch_model.resources_to_construct:
            block = dispatch_model.blocks[resource.name]
            for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
                block.power_output[2030, dispatch_window, timestamp].fix(100.0)
                block.power_input[2030, dispatch_window, timestamp].fix(0.0)
                block.provide_reserve[2030, dispatch_window, timestamp].fix(75.0)

        # Run calculate_dispatch_results
        dispatch_model.calculate_resource_dispatch_results()

        # Test
        expected_unserved_energy_and_reserve = pd.Series(index=dispatch_model.full_index, data=0.0, name=0)
        pd.testing.assert_series_equal(
            dispatch_model.unserved_energy_and_reserve.squeeze(), expected_unserved_energy_and_reserve
        )

        for resource in dispatch_model.resources_to_construct:
            expected_optimized_provide_power_mw = pd.Series(
                index=in_periods, data=100.0, name=f"blocks[{resource.name}].power_output"
            )
            expected_optimized_provide_power_mw.index.names = ["TIMESTAMPS"]
            pd.testing.assert_series_equal(
                resource.optimized_provide_power_mw.squeeze(), expected_optimized_provide_power_mw
            )
