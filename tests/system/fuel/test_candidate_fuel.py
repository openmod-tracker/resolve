import copy

import numpy as np
import pandas as pd
import pytest

from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.system.pollutant import Pollutant


class TestCandidateFuel:
    _TEST_SYSTEM_NAME: str = "TEST_fuels"
    _START_YEAR: int = 2020
    _END_YEAR: int = 2050

    def test_final_fuel_name(self, mini_fuels_system):
        """
        Test final_fuel_name property. Assert that:
        - the final_fuel_name is expected for every candidate fuel in the mini_fuels_system
        - the final_fuel_name is type str
        """
        candidate_fuel = mini_fuels_system.candidate_fuels["Fossil Natural Gas"]
        assert candidate_fuel.final_fuel_name == "Natural Gas"

        for final_fuel in mini_fuels_system.final_fuels.values():
            cfuel_final_fuel_name_list = [cfuel.final_fuel_name for cfuel in final_fuel.candidate_fuels_list]
            assert all(element == final_fuel.name for element in cfuel_final_fuel_name_list)
            assert all(isinstance(name, str) for name in cfuel_final_fuel_name_list)

    def test_pollutant_list(self, mini_fuels_system):
        """
        Test the pollutant_list property. Assert that:
        - it returns a list of the expected Pollutant type
        """
        for cfuel_name, cfuel in mini_fuels_system.candidate_fuels.items():
            pollutant_list = cfuel.pollutant_list
            expected_list = [linkage.instance_to for linkage in list(cfuel.pollutants.values())]

            assert isinstance(pollutant_list, list)
            assert all(isinstance(p, Pollutant) for p in pollutant_list)
            assert all([a == b for a, b in zip(expected_list, pollutant_list)])

    def test_candidate_fuel_blend(self, mini_fuels_system):
        """
        Test the candidate_fuel_blend function. Assert that:
        - it returns a pd Series type
        - it returns expected values based on hard-coded attributes in the mini_fuels_system class fixture
        """
        sector = "Industry"
        candidate_fuel = mini_fuels_system.candidate_fuels["Fossil Natural Gas"]

        result = candidate_fuel.candidate_fuel_blend(sector)
        assert isinstance(result, pd.Series)

        result_array = result.values
        assert result_array == pytest.approx(1.0)

    @pytest.fixture(scope="class")
    def energy_demand_for_tests(
        self,
        pathways_series_index,
    ) -> pd.Series:
        data = pd.Series(
            index=pathways_series_index,
            data=np.array([2, 4, 6, 8, 10]),
            name="value",
        )
        return data

    def test_calc_energy_demand(self, mini_fuels_system, energy_demand_for_tests):
        """
        Test the calc_energy_demand function. Assert that:
        - the output is a pd.Series type
        - the output is assigned to the out_energy_demand attribute of the final fuel linkage
        - the output matches expected values based on hard-coded attributes in the mini_fuels_system class fixture
        """
        system = copy.deepcopy(mini_fuels_system)
        system.candidate_fuels["Fossil Natural Gas"].sector_candidate_fuel_blending[
            ("Industry", "Natural Gas")
        ].blend_override.data[
            pd.DatetimeIndex(
                [
                    "2031-01-01",
                    "2032-01-01",
                    "2033-01-01",
                    "2034-01-01",
                    "2035-01-01",
                ]
            )
        ] = np.array(
            [0.75, 0.5, 0.25, 1, 0]
        )

        sector = "Industry"
        final_fuel_name = "Natural Gas"
        candidate_fuel = system.candidate_fuels["Fossil Natural Gas"]
        ed = candidate_fuel.calc_energy_demand(sector, energy_demand_for_tests)

        assert isinstance(ed, pd.Series)
        assert candidate_fuel.final_fuels[final_fuel_name].out_energy_demand.data.values == pytest.approx(ed)
        assert ed.values == pytest.approx(np.array([2 * 0.75, 4 * 0.5, 6 * 0.25, 8.0, 0.0]))

    def test_calc_fuel_cost(self, mini_fuels_system, energy_demand_for_tests):
        """
        Test the calc_fuel_cost function. Assert that:
        - the output is assigned to the out_fuel_cost attribute of the final fuel linkage
        - the output matches expected values based on hard-coded attributes in the mini_fuels_system class fixture
        """
        system = copy.deepcopy(mini_fuels_system)
        candidate_fuel = system.candidate_fuels["Fossil Natural Gas"]
        candidate_fuel.fuel_price_per_mmbtu.data[
            pd.DatetimeIndex(
                [
                    "2031-01-01",
                    "2032-01-01",
                    "2033-01-01",
                    "2034-01-01",
                    "2035-01-01",
                ]
            )
        ] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        candidate_fuel.calc_fuel_cost(energy_demand_for_tests)
        fuel_linkage = candidate_fuel.final_fuels["Natural Gas"]

        assert isinstance(fuel_linkage.out_fuel_cost, ts.NumericTimeseries)
        assert isinstance(fuel_linkage.out_fuel_cost.data, pd.Series)
        assert fuel_linkage.out_fuel_cost.data.values == pytest.approx(
            np.array([2 * 0.1, 4 * 0.2, 6 * 0.3, 8 * 0.4, 10 * 0.5])
        )

    def test_check_fuel_price(self, mini_fuels_system):
        """
        Test the check_fuel_price validator. Assert that:
        - Error is raised when fuel_is_commodity_bool is True and fuel_price_per_mmbtu is not specified
        """
        candidate_fuel = copy.deepcopy(mini_fuels_system).candidate_fuels["Fossil Natural Gas"]
        candidate_fuel.fuel_is_commodity_bool = 1
        candidate_fuel.fuel_price_per_mmbtu = None

        with pytest.raises(ValueError):
            result = candidate_fuel.check_fuel_price()
            assert result is None

    @pytest.fixture(scope="class")
    def timeseries_for_tests(
        self,
        pathways_series_index,
    ) -> ts.NumericTimeseries:
        series = ts.NumericTimeseries(
            name="random data for tests",
            data=pd.Series(
                index=pathways_series_index,
                data=np.ones(len(pathways_series_index)),
                name="value",
            ),
        )

        return series

    def test_validate_or_calculate_hourly_fuel_prices(self, mini_fuels_system, timeseries_for_tests):
        pass

    def test_fuel_production_plants(self, mini_fuels_system):
        # TODO (2/29/24) - revisit when function is updated
        pass
