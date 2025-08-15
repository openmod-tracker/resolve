import copy
import shutil

import numpy as np
import pandas as pd
import pytest

import new_modeling_toolkit.recap.recap_case as recap_case
from new_modeling_toolkit.recap.monte_carlo_draw import MonteCarloDraw
from new_modeling_toolkit.recap.recap_case import RecapCase
from new_modeling_toolkit.recap.recap_case_settings import DispatchMode
from new_modeling_toolkit.recap.recap_case_settings import ReliabilityMetric
from new_modeling_toolkit.recap.recap_case_settings import ResourceGrouping

class TestRecapCase:
    def pass_function(self, *args, **kwargs):
        """
        For use in monkeypatching
        """
        return self

    def _mock_reliability_func(self, perfect_capacity, tuned_flag):
        """
        For use in monkeypatching
        """
        if perfect_capacity > 50:
            return 0.05
        else:
            return 0.15

    def monkey_patch(self, monkeypatch):
        monkeypatch.setattr(recap_case, "_dispatch", self.dispatch_tuple)
        monkeypatch.setattr(RecapCase, "_create_and_start_gurobi_pool", self.pass_function)
        monkeypatch.setattr(pd.DataFrame, "to_csv", self.pass_function)
        monkeypatch.setattr(shutil, "copy", self.pass_function)
        monkeypatch.setattr(
            RecapCase,
            "_reliability_func_optimized_dispatch",
            self._mock_reliability_func,
        )
        monkeypatch.setattr(RecapCase, "_reliability_func_no_ELRs", self._mock_reliability_func)
        monkeypatch.setattr(
            RecapCase,
            "_reliability_func_heuristic_dispatch",
            self._mock_reliability_func,
        )

    @pytest.fixture(scope="class")
    def unserved_energy_series(self):
        unserved_energy = pd.Series(
            index=pd.MultiIndex.from_tuples(
                [
                    ("draw_1", 0, pd.Timestamp("2010-01-01 12:00")),
                    ("draw_1", 1, pd.Timestamp("2011-06-01 16:00")),
                    ("draw_2", 0, pd.Timestamp("2010-01-01 08:00")),
                    ("draw_2", 1, pd.Timestamp("2012-08-01 02:00")),
                    ("draw_2", 1, pd.Timestamp("2012-08-01 03:00")),
                    ("draw_2", 1, pd.Timestamp("2012-08-01 04:00")),
                ],
                names=("MC_draw", "subproblem", "timestamp"),
            ),
            data=[-100.3, 0, 28, -50, -12.85, 2.26],
        )
        return unserved_energy

    @pytest.fixture(scope="class")
    def empty_portfolio_vector(self, system_generator):
        system = system_generator.get()
        portfolio_vector = pd.Series(index=system.resources.keys(), data=np.zeros(len(system.resources)))
        return portfolio_vector

    @pytest.fixture(scope="class")
    def portfolio_vector(self, empty_portfolio_vector):
        portfolio_vector = copy.deepcopy(empty_portfolio_vector)
        portfolio_vector += 5
        return portfolio_vector

    @pytest.fixture(scope="class")
    def elcc_matrix(self, recap_case_generator):
        recap_case = recap_case_generator.get()
        marginal_ELCC_points_matrix = pd.read_csv(
            recap_case.dir_str.recap_settings_dir / recap_case.case_name / "ELCC_surfaces" / "marginal_ELCC.csv"
        )
        return marginal_ELCC_points_matrix

    @pytest.fixture(scope="class")
    def unserved_energy_result(self):
        date = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H")
        unserved_energy = pd.DataFrame(index=date, data=[0.1] * 8760, columns=["unserved_energy_and_reserve"])
        unserved_energy.index.name = "timestamp"
        unserved_energy["MC_draw"] = "MC_draw_0"
        unserved_energy["subproblem"] = 0
        unserved_energy.reset_index(inplace=True)
        unserved_energy.set_index(["MC_draw", "subproblem", "timestamp"], inplace=True)
        return unserved_energy.squeeze()

    def dispatch_tuple(self, *args, **kwargs):
        date = pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H")
        new_series = pd.DataFrame(index=date, data=[0.1] * 8760, columns=["unserved_energy_and_reserve"])
        return ("MC_draw_0", 0, new_series)

    def test_setup_monte_carlo_draws(self, recap_case_generator):
        recap_case = recap_case_generator.get()
        recap_case.monte_carlo_draws = None
        recap_case.case_settings.number_of_monte_carlo_draws = 2
        recap_case.case_settings.draw_settings = "random"

        recap_case.setup_monte_carlo_draws()
        assert type(recap_case.monte_carlo_draws) == dict
        assert len(recap_case.monte_carlo_draws) == 2
        assert type(recap_case.monte_carlo_draws["MC_draw_1"]) == MonteCarloDraw

    def test_print_resource_portfolio(self, recap_case_generator, monkeypatch):
        recap_case = recap_case_generator.get()
        self.monkey_patch(monkeypatch)
        df = recap_case._print_resource_portfolio(recap_case.dir_str.recap_output_dir)
        test_df = pd.DataFrame(
            data=[
                ["Flex", 1.0, np.nan],
                ["Hydro", 10.0, np.nan],
                ["DR", 10.0, np.nan],
                ["Storage", 5.0, 20.0],
                ["HybridBatteryStorage", 5.0, 20.0],
                ["Thermal", 500.0, np.nan],
                ["Firm", 5.0, np.nan],
                ["Solar", 100.0, np.nan],
                ["Solar", 100.0, np.nan],
                ["Wind", 100.0, np.nan],
                ["Solar", 100.0, np.nan],
            ],
            index=[
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
            ],
            columns=["group", "capacity_planned", "storage_capacity_planned"],
        )

        assert type(df) == pd.DataFrame
        pd.testing.assert_frame_equal(df, test_df)

    def test_ELR_capacity(self, recap_case_generator):
        recap_case = recap_case_generator.get()
        assert recap_case.ELR_capacity == 131.0

    @pytest.mark.parametrize(
        "metric,expected",
        [
            (ReliabilityMetric.LOLE, 365.25 * 2 / 4),
            (ReliabilityMetric.LOLP, 2 / 6),
            (ReliabilityMetric.LOLH, 8766 * 2 / 6),
            (ReliabilityMetric.EUE, sum([-100.3, 0, 28, -50, -12.85, 2.26]) / 6 * 8766),
        ],
    )
    def test_calculate_reliability(self, recap_case_generator, unserved_energy_series, metric, expected):
        recap_case = recap_case_generator.get()
        assert (
            recap_case.calculate_reliability(unserved_energy_and_reserve=unserved_energy_series, metric=metric)
            == expected
        )

    @pytest.mark.parametrize(
        "target,lower_bound,upper_bound,bisection_xtol,max_iter,expected",
        [(100, 0, 1000, 0.1, 50, 100.03662109375), (100, -1000, -100, 0.1, 50, -100.054931640625)],
    )
    def test_bisection_method(
        self, recap_case_generator, target, lower_bound, upper_bound, bisection_xtol, max_iter, expected
    ):
        recap_case = recap_case_generator.get()

        def _test_reliability_func(x, tuned_flag):
            return 200 - x

        bisection_result = recap_case.bisection_method(
            reliability_func=_test_reliability_func,
            target=target,
            LB=lower_bound,
            UB=upper_bound,
            bisection_xtol=bisection_xtol,
            max_iter=max_iter,
        )
        assert bisection_result == expected

    @pytest.mark.parametrize(
        "dispatch_mode, metric, expected",
        [
            (DispatchMode.HEURISTICS_ONLY, ReliabilityMetric.LOLE, 38.4997347601947),
            (DispatchMode.SEMI_OPTIMIZED, ReliabilityMetric.LOLE, 30.8239535101947),
            (DispatchMode.FULLY_OPTIMIZED, ReliabilityMetric.LOLE, 30.8239535101947),
            (DispatchMode.HEURISTICS_ONLY, ReliabilityMetric.EUE, 38.4997347601947),
            (DispatchMode.SEMI_OPTIMIZED, ReliabilityMetric.EUE, 30.8239535101947),
            (DispatchMode.FULLY_OPTIMIZED, ReliabilityMetric.EUE, 30.8239535101947),
        ],
    )
    def test_calculate_perfect_capacity_shortfall(
        self, recap_case_generator, dispatch_mode, expected, metric, monkeypatch
    ):
        recap_case = recap_case_generator.get()
        recap_case.dispatch_results = pd.Series([0])
        recap_case.case_settings.target_metric = metric
        self.monkey_patch(monkeypatch)
        recap_case.case_settings.dispatch_mode = dispatch_mode
        recap_case.calculate_perfect_capacity_shortfall()
        assert recap_case.perfect_capacity_shortfall == expected

    @pytest.mark.parametrize("incremental, expected_elr", [(True, 161.0), (False, 30.0)])
    def test_rescale_portfolio(self, recap_case_generator, portfolio_vector, incremental, expected_elr):
        recap_case = recap_case_generator.get()
        print("ELR Capacty: ", recap_case.ELR_capacity)
        print("Rescaling")
        recap_case.rescale_portfolio(portfolio_vector, incremental)
        print("ELR Capacty: ", recap_case.ELR_capacity)
        mc_draw = recap_case.monte_carlo_draws["MC_draw_0"]
        assert recap_case.ELR_capacity == expected_elr
        for resource_name, resource in recap_case.system.resources.items():
            assert resource.capacity_planned.slice_by_year(2030) == mc_draw.system.resources[
                resource_name
            ].capacity_planned.slice_by_year(2030)

    @pytest.mark.parametrize("incremental, expected_elr", [(True, 131.0), (False, 0.0)])
    def test_rescale_empty_portfolio(self, recap_case_generator, empty_portfolio_vector, incremental, expected_elr):
        recap_case = recap_case_generator.get()
        recap_case.rescale_portfolio(empty_portfolio_vector, incremental)
        mc_draw = recap_case.monte_carlo_draws["MC_draw_0"]
        assert recap_case.ELR_capacity == expected_elr
        for resource_name, resource in recap_case.system.resources.items():
            assert resource.capacity_planned.slice_by_year(2030) == mc_draw.system.resources[
                resource_name
            ].capacity_planned.slice_by_year(2030)

    @pytest.mark.parametrize(
        "perfect_capacity, dispatch_mode, heuristic_dispatch_mode, expected_lole",
        [
            (0, DispatchMode.HEURISTICS_ONLY, ResourceGrouping.DEFAULT, 365.25),
            (0, DispatchMode.SEMI_OPTIMIZED, ResourceGrouping.DEFAULT, 365.25),
            (0, DispatchMode.FULLY_OPTIMIZED, ResourceGrouping.DEFAULT, 365.25),
            (0, DispatchMode.FULLY_OPTIMIZED, ResourceGrouping.NO_ELRS, 365.25),
            (50, DispatchMode.FULLY_OPTIMIZED, ResourceGrouping.NO_ELRS, 365.25),
        ],
    )
    def test_run_dispatch(
        self,
        recap_case_generator,
        unserved_energy_result,
        monkeypatch,
        perfect_capacity,
        dispatch_mode,
        heuristic_dispatch_mode,
        expected_lole,
    ):
        recap_case = recap_case_generator.get()
        self.monkey_patch(monkeypatch)
        unserved_energy = recap_case.run_dispatch(
            perfect_capacity=perfect_capacity,
            dispatch_mode=dispatch_mode,
            bootstrap=False,
        )
        assert recap_case.reliability_results.loc[perfect_capacity, "LOLE"] == expected_lole
        pd.testing.assert_series_equal(unserved_energy, unserved_energy_result)

    def test_calculate_ELCC_points(self, recap_case_generator, elcc_matrix, monkeypatch):
        recap_case = recap_case_generator.get()
        recap_case.perfect_capacity_shortfall = 20.0
        recap_case.dispatch_results = pd.Series([0])

        def _mock_create_copy():
            recap_case_copy = recap_case_generator.get()
            recap_case_copy.dispatch_results = pd.Series([0])
            return recap_case_copy

        monkeypatch.setattr(recap_case, "create_copy", _mock_create_copy)

        self.monkey_patch(monkeypatch)
        elcc_results = recap_case.calculate_ELCC_points(elcc_matrix)
        results_df = pd.DataFrame(
            columns=[
                "CA_Thermal_1",
                "DER_Solar",
                "Arizona_Solar",
                "ELCC_case_perfect_capacity_shortfall_MW",
            ],
            index=[0, 1, 2],
            data=[
                [100, 0, 0, 30.8239535101947],
                [0, 100, 0, 30.8239535101947],
                [0, 0, 100, 30.8239535101947],
            ],
        )
        pd.testing.assert_frame_equal(elcc_results, results_df)

    def test_upsample_monte_carlo_draws(self, recap_case_generator, monkeypatch):

        recap_case = recap_case_generator.get()
        recap_case.monte_carlo_draws = None
        recap_case.case_settings.number_of_monte_carlo_draws = 2
        recap_case.case_settings.draw_settings = "RECAP_test"

        for setting in [
            "calculate_reliability",
            "calculate_perfect_capacity_shortfall",
            "calculate_total_resource_need",
            "calculate_marginal_ELCC",
            "calculate_incremental_last_in_ELCC",
            "calculate_decremental_last_in_ELCC",
            "calculate_ELCC_surface",
        ]:
            setattr(recap_case.case_settings, setting, False)

        # With all run settings set to False, this will only upsample MC draws
        recap_case.setup_monte_carlo_draws()
        recap_case.run_case()

        # Test that variable resource profiles are different
        for variable_resource in recap_case.system.variable_resources:
            # Get copy of resource from each MC draw
            resource0 = recap_case.monte_carlo_draws["MC_draw_0"].system.resources[variable_resource]
            resource1 = recap_case.monte_carlo_draws["MC_draw_1"].system.resources[variable_resource]
            assert (resource0.power_output_max.data != resource1.power_output_max.data).any()

        # Test that hybrid variable resource profiles are different
        for hybrid_variable_resource in recap_case.system.hybrid_variable_resources:
            # Get copy of resource from each MC draw
            resource0 = recap_case.monte_carlo_draws["MC_draw_0"].system.resources[hybrid_variable_resource]
            resource1 = recap_case.monte_carlo_draws["MC_draw_1"].system.resources[hybrid_variable_resource]
            assert (resource0.power_output_max.data != resource1.power_output_max.data).any()

        # Currently, test case is not set up to easily test hydro upsampling (need > 1 year of data)

    def test_run_case(self, recap_case_generator, monkeypatch):
        # just test that it takes in all the inputs and runs through
        RecapCase.run_dispatch.__defaults__ = (ResourceGrouping.DEFAULT, False, False)
        recap_case = recap_case_generator.get()
        self.monkey_patch(monkeypatch)

        def _mock_create_copy():
            recap_case_copy = recap_case_generator.get()
            recap_case_copy.dispatch_results = pd.Series([0])
            return recap_case_copy

        monkeypatch.setattr(recap_case, "create_copy", _mock_create_copy)

        for setting in [
            "calculate_reliability",
            "calculate_perfect_capacity_shortfall",
            "calculate_marginal_ELCC",
            "calculate_incremental_last_in_ELCC",
            "calculate_decremental_last_in_ELCC",
            "calculate_ELCC_surface",
        ]:
            setattr(recap_case.case_settings, setting, True)

        self.monkey_patch(monkeypatch)
        recap_case.run_case()

    def test_calculate_total_resource_need(self, recap_case_generator, monkeypatch):
        recap_case = recap_case_generator.get()
        recap_case.dispatch_results = pd.Series([0])

        def _mock_create_copy():
            recap_case_copy = recap_case_generator.get()
            recap_case_copy.dispatch_results = pd.Series([0])
            return recap_case_copy

        monkeypatch.setattr(recap_case, "create_copy", _mock_create_copy)

        self.monkey_patch(monkeypatch)
        recap_case.calculate_total_resource_need()
        assert recap_case.total_resource_need == 46.6872347601947
