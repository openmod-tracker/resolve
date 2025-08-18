import copy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Type

import numpy as np
import pandas as pd
import pytest

from new_modeling_toolkit.core.linkage import ResourceToOutageDistribution
from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.core.temporal.new_temporal import NewTemporalSettings
from new_modeling_toolkit.recap.dispatch_model import DispatchModel
from new_modeling_toolkit.system.electric.resource_group import ResourceGroup
from new_modeling_toolkit.system.electric.resources import GenericResource
from new_modeling_toolkit.system.outage_distribution import OutageDistribution


class ResourceTestTemplate:
    """Template class to be used for implementing unit tests for classes that inherit from GenericResource."""

    _RESOURCE_CLASS: Type[GenericResource]
    _RESOURCE_GROUP_PATH: Path
    _SYSTEM_COMPONENT_DICT_NAME: str

    _RESOURCE_INIT_KWARGS: Dict[str, Any] = dict()

    @property
    def resource_init_kwargs(self):
        init_kwargs = dict()
        for parent_class in list(reversed(self.__class__.mro()))[1:]:
            init_kwargs.update(parent_class._RESOURCE_INIT_KWARGS)
        return init_kwargs

    @pytest.fixture(scope="class")
    def _create_resource(self, dir_structure):
        """Creates an instance of the resource class to be tested.

        This method is not intended to be called directly. Instead, use the `make_resource_copy()` fixture,
        which is a factory for generating a clean copy of the resource.
        """
        resource = self._RESOURCE_CLASS(**self.resource_init_kwargs)

        return resource

    @pytest.fixture(scope="class")
    def _create_resource_group(self, dir_structure):
        """Creates an instance of the resource group that is associated with the class to be tested.

        This method is not intended to be called directly. Instead, use the `make_resource_group_copy()` fixture,
        which is a factory for generating a clean copy of the resource group.
        """
        _, group = ResourceGroup.from_csv(
            dir_structure.data_interim_dir.joinpath(self._RESOURCE_GROUP_PATH), return_type=tuple
        )

        return group

    @pytest.fixture(scope="class")
    def make_resource_copy(self, _create_resource):
        """Factory fixture for generating a copy of the resource to be used in tests and other fixtures."""

        def _resource_factory():
            resource = copy.deepcopy(_create_resource)

            return resource

        return _resource_factory

    @pytest.fixture(scope="class")
    def make_resource_group_copy(self, _create_resource_group):
        """Factory fixture for generating a copy of the resource group to be used in tests and other fixtures."""

        def _resource_group_factory():
            resource = copy.deepcopy(_create_resource_group)

            return resource

        return _resource_group_factory

    @pytest.fixture(scope="class")
    def make_dispatch_model_copy(
        self, single_resource_dispatch_model_generator, make_resource_copy, make_resource_group_copy
    ):
        """Factory fixture for generating a copy of a dispatch model containing one thermal resource and the resource
        being tested.
        """
        dispatch_model = single_resource_dispatch_model_generator.get(
            component_dict_name=self._SYSTEM_COMPONENT_DICT_NAME,
            resource=make_resource_copy(),
            resource_group=make_resource_group_copy(),
            perfect_capacity=0,
        )

        def _dispatch_model_factory():
            return copy.deepcopy(dispatch_model)

        return _dispatch_model_factory

    @pytest.fixture(scope="function")
    def resource_block(self, make_dispatch_model_copy, make_resource_copy):
        """Fixture that returns the pyomo block for the resource being tested."""
        model = make_dispatch_model_copy()
        block = model.blocks[make_resource_copy().name]

        return block

    @pytest.fixture(scope="class")
    def first_index(self, make_dispatch_model_copy):
        """Fixture that returns an example index for the dispatch model, for use in testing."""
        model = make_dispatch_model_copy()
        modeled_year = 2030
        dispatch_window, timestamp = model.DISPATCH_WINDOWS_AND_TIMESTAMPS.first()

        return modeled_year, dispatch_window, timestamp

    @pytest.fixture(scope="class")
    def last_index(self, make_dispatch_model_copy):
        """Fixture that returns an example index for the dispatch model, for use in testing."""
        model = make_dispatch_model_copy()
        modeled_year = 2030
        dispatch_window, timestamp = model.DISPATCH_WINDOWS_AND_TIMESTAMPS.first()
        last_tp = model.TIMESTAMPS_IN_DISPATCH_WINDOWS[dispatch_window].prevw(timestamp)

        return modeled_year, dispatch_window, last_tp

    @pytest.fixture(scope="class")
    def add_energy_budget_to_resource(self, make_dispatch_model_copy):
        dispatch_model = make_dispatch_model_copy()
        resource = copy.deepcopy(dispatch_model.system.resources["Example_Resource"])
        resource.energy_budget_daily = ts.NumericTimeseries(
            name="energy_budget_daily",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-07-01", "2010-07-02"], name="timestamp"),
                data=[1000 / (400 * 24), 1100 / (400 * 24)],
            ),
            freq_="D",
        )

        resource.energy_budget_monthly = ts.NumericTimeseries(
            name="energy_budget_monthly",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-07-01", "2010-08-01"], name="timestamp"),
                data=[2000.0 / (400 * 24 * 31), 2500.0 / (400 * 24 * 31)],
            ),
            freq_="M",
        )

        resource.energy_budget_annual = ts.NumericTimeseries(
            name="energy_budget_annual",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-01-01", "2019-01-01"], name="timestamp"),
                data=[3000.0 / (400 * 8760), 4000.0 / (400 * 8760)],
            ),
            freq_="YS",
        )

        return resource

    @pytest.fixture(scope="class")
    def temporal_settings(self, recap_df_in):

        # Save df_in as temporal settings
        temporal_settings = NewTemporalSettings(
            name="recap",
            dispatch_window_edge_effects="fixed_initial_soc",
            dispatch_windows_df=recap_df_in,
            modeled_years=[2030],
        )
        return temporal_settings

    @pytest.fixture(scope="class")
    def resource_block_with_budgets(self, make_dispatch_model_copy, add_energy_budget_to_resource, temporal_settings):
        dispatch_model = make_dispatch_model_copy()
        resource = copy.deepcopy(add_energy_budget_to_resource)
        resource.resample_ts_attributes([2030, 2030], [2010, 2010])

        dispatch_model.temporal_settings = temporal_settings

        super(DispatchModel, dispatch_model).__init__(temporal_settings, dispatch_model.system)

        del dispatch_model.blocks[resource.name]
        resource.construct_operational_block(dispatch_model)

        return dispatch_model


class TestGenericResource(ResourceTestTemplate):
    _RESOURCE_CLASS = GenericResource
    _RESOURCE_GROUP_PATH = "resource_groups/Generic.csv"
    _SYSTEM_COMPONENT_DICT_NAME = "generic_resources"

    _RESOURCE_INIT_KWARGS = dict(
        name="Example_Resource",
        random_seed=1,
        capacity_planned=ts.NumericTimeseries(
            name="capacity_planned",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[200.0, 400.0],
                name="value",
            ),
        ),
        power_output_max=ts.FractionalTimeseries(
            name="power_output_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"], name="timestamp"
                ),
                data=[1.0, 0.8, 0.5, 0.5],
                name="value",
            ),
            weather_year=True,
        ),
        power_output_min=ts.FractionalTimeseries(
            name="power_output_min",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"], name="timestamp"
                ),
                data=0.25,
                name="value",
            ),
            weather_year=True,
        ),
        power_input_max=ts.FractionalTimeseries(
            name="power_input_max",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"], name="timestamp"
                ),
                data=0.5,
                name="value",
            ),
            weather_year=True,
        ),
        outage_profile=ts.FractionalTimeseries(
            name="outage_profile",
            data=pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"], name="timestamp"
                ),
                data=[1.0, 0.0, 0.5, 1.0],
                name="value",
            ),
        ),
    )

    def test_rescale(self, make_resource_copy):
        """Test the `rescale()` function with `incremental=False`."""
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

    def test_rescale_incremental(self, make_resource_copy):
        """Test the `rescale()` function with `incremental=True`."""
        resource = make_resource_copy()
        resource.rescale(model_year=2020, capacity=100, incremental=True)
        pd.testing.assert_series_equal(
            resource.capacity_planned.data,
            pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[300.0, 400.0],
                name="value",
            ),
        )

    def test_upsample(self, make_resource_copy):
        """Test the `upsample()` method."""
        resource = make_resource_copy()

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2011-06-30 23:00", freq="H")
        resource.upsample(load_calendar=load_calendar)
        pd.testing.assert_series_equal(
            resource.power_output_max.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2011-12-31 23:00", freq="H", name="timestamp"),
                data=[1.0, 0.8, 0.5, 0.5] + [0.5] * (8760 * 2 - 4),
                name="value",
            ),
        )

    def test_upsample_no_overlap(self, make_resource_copy):
        """Test the `upsample()` method in scenarios where the target index has no overlap with the input data."""
        resource = make_resource_copy()

        load_calendar = pd.date_range(start="2008-01-01 00:00", end="2009-06-30 23:00", freq="H")
        resource.upsample(load_calendar=load_calendar)
        pd.testing.assert_series_equal(
            resource.power_output_max.data,
            pd.Series(
                index=pd.date_range(start="2008-01-01 00:00", end="2009-12-31 23:00", freq="H", name="timestamp"),
                data=1.0,
                name="value",
            ),
        )

        load_calendar = pd.date_range(start="2021-01-01 00:00", end="2022-06-30 23:00", freq="H")
        resource.upsample(load_calendar=load_calendar)
        pd.testing.assert_series_equal(
            resource.power_output_max.data,
            pd.Series(
                index=pd.date_range(start="2021-01-01 00:00", end="2022-12-31 23:00", freq="H", name="timestamp"),
                data=1.0,
                name="value",
            ),
        )

    def test_upsample_outage_profile(self, make_resource_copy):
        resource = make_resource_copy()

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-01-01 12:00", freq="H", name="timestamp")
        resource.upsample(load_calendar)
        pd.testing.assert_series_equal(
            resource.outage_profile.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=[1.0, 0.0, 0.5, 1.0] + [1.0] * (8760 - 4),
                name="value",
            ),
        )

    def test_upsample_power_output_min(self, make_resource_copy):
        resource = make_resource_copy()

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-01-01 12:00", freq="H", name="timestamp")
        resource.upsample(load_calendar)
        pd.testing.assert_series_equal(
            resource.power_output_min.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=0.25,
                name="value",
            ),
            check_names=False,
        )

    def test_upsample_power_input_max(self, make_resource_copy):
        resource = make_resource_copy()

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-01-01 12:00", freq="H", name="timestamp")
        resource.upsample(load_calendar)
        pd.testing.assert_series_equal(
            resource.power_input_max.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H", name="timestamp"),
                data=0.5,
                name="value",
            ),
            check_names=False,
        )

    def test_upsample_power_input_min(self, make_resource_copy):
        resource = make_resource_copy()

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-01-01 12:00", freq="H", name="timestamp")
        resource.upsample(load_calendar)
        pd.testing.assert_series_equal(
            resource.power_input_min.data,
            pd.Series(
                index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="H"),
                data=0.0,
            ),
        )

    def test_upsample_energy_budget_daily(self, make_resource_copy):
        resource = make_resource_copy()
        resource.energy_budget_daily = ts.FractionalTimeseries(
            name="energy_budget_daily",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-07-01", "2010-07-02"], name="timestamp"), data=[0.104166667, 0.114583333]
            ),
            freq_="D",
            weather_year=True,
        )

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-01-01 12:00", freq="H", name="timestamp")
        resource.upsample(load_calendar)
        expected_energy_budget_daily = pd.Series(
            index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="D", name="timestamp"),
            data=np.concatenate([np.repeat(0.104166667, 182), np.repeat(0.114583333, 183)]).tolist(),
        )
        pd.testing.assert_series_equal(
            resource.energy_budget_daily.data,
            expected_energy_budget_daily,
        )

    def test_upsample_energy_budget_monthly(self, make_resource_copy):
        resource = make_resource_copy()
        resource.energy_budget_monthly = ts.FractionalTimeseries(
            name="energy_budget_monthly",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-07-01", "2010-08-01"], name="timestamp"), data=[0.00672043, 0.008400538]
            ),
            freq_="MS",
            weather_year=True,
        )

        load_calendar = pd.date_range(start="2010-01-01 00:00", end="2010-01-01 12:00", freq="H", name="timestamp")
        resource.upsample(load_calendar)
        expected_energy_budget_monthly = pd.Series(
            index=pd.date_range(start="2010-01-01 00:00", end="2010-12-31 23:00", freq="MS", name="timestamp"),
            data=np.concatenate([np.repeat(0.00672043, 7), np.repeat(0.008400538, 5)]).tolist(),
        )
        pd.testing.assert_series_equal(
            resource.energy_budget_monthly.data,
            expected_energy_budget_monthly,
        )

    def test_upsample_energy_budget_annual(self, make_resource_copy):
        resource = make_resource_copy()
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
                data=[400.0, 0.0, 100.0, 200.0],
                name="value",
            ),
        }

        assert resource.scaled_pmax_profile.keys() == expected_profile.keys()
        for model_year, profile in resource.scaled_pmax_profile.items():
            pd.testing.assert_series_equal(profile, expected_profile[model_year])

    def test_scaled_daily_energy_budget(self, make_resource_copy):
        resource = make_resource_copy()
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

    def test_scaled_monthly_energy_budget(self, make_resource_copy):
        resource = make_resource_copy()
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

    def test_scaled_annual_energy_budget(self, make_resource_copy):
        resource = make_resource_copy()
        resource.energy_budget_annual = ts.FractionalTimeseries(
            name="energy_budget_annual",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-01-01", "2019-01-01"], name="timestamp"), data=[0.000856164, 0.001141553]
            ),
            freq_="YS",
            weather_year=True,
        )

        expected_profile = {
            2020: pd.Series(
                index=pd.DatetimeIndex(["2010-01-01", "2019-01-01"], name="timestamp"),
                data=[1500.0, 2000.0],
            ),
            2030: pd.Series(
                index=pd.DatetimeIndex(["2010-01-01", "2019-01-01"], name="timestamp"),
                data=[3000.0, 4000.0],
            ),
        }
        assert resource.scaled_annual_energy_budget.keys() == expected_profile.keys()
        for model_year, profile in resource.scaled_annual_energy_budget.items():
            pd.testing.assert_series_equal(profile, expected_profile[model_year])

    def test_simulate_outages(self, make_resource_copy, monkeypatch):
        """Test the `simulate_outages()` method."""
        # Initialize the resource copy and set outage-related attributes
        resource: GenericResource = make_resource_copy()
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
                data=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
        )

        # Note: the value of these attributes have no effect because the scipy exponential random number generator is
        #   mocked below. If these values are not set, then the outage simulation function will not run.
        resource.mean_time_to_repair = 1
        resource.stochastic_outage_rate = ts.FractionalTimeseries(
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
                data=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ),
        )  # Has no effect because the random generator is mocked
        resource.random_seed = 5  # Has no effect because the random generator is mocked

        # Mock the outage distribution function that returns outage percentage random variables
        def _get_random_derate_frac_mock(self, seed, size):
            return np.array([0.5, 0.8, 0.6, 1.0, 0.4, 0.9, 0.8])

        monkeypatch.setattr(OutageDistribution, "get_random_derate_fraction_arr", _get_random_derate_frac_mock)

        # Create the outage distribution instance and link it to the resource
        outage_distribution = OutageDistribution(name="Outage_Distribution_1")
        linkage = ResourceToOutageDistribution(
            name=("Example_Resource", "Outage_Distribution_1"),
            instance_from=resource,
            instance_to=outage_distribution,
        )
        ResourceToOutageDistribution.announce_linkage_to_instances()

        # Run the outage simulation
        resource.simulate_outages(model_year=2020, random_seed=2)

        # Check that the simulated outage profile matches expectations
        pd.testing.assert_series_equal(
            resource.outage_profile.data,
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
                data=[0.5, 1.0, 0.2, 1.0, 0.4, 1.0, 0.0],
            ),
        )

    def test_simulate_outages_time_varying(self, make_resource_copy, monkeypatch):
        """Test the `simulate_outages()` method."""
        # Initialize the resource copy and set outage-related attributes
        resource: GenericResource = make_resource_copy()

        resource.power_output_max = ts.NumericTimeseries(
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
                data=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
        )

        # Note: the value of these attributes have no effect because the scipy exponential random number generator is
        #   mocked below. If these values are not set, then the outage simulation function will not run.
        resource.mean_time_to_repair = 1
        resource.stochastic_outage_rate = ts.FractionalTimeseries(
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
                data=[0.5, 0.1, 0.5, 0.9, 0.5, 0.1, 0.5],
            ),
        )  # Has no effect because the random generator is mocked
        resource.random_seed = 5  # Has no effect because the random generator is mocked

        # Mock the outage distribution function that returns outage percentage random variables
        def _get_random_derate_frac_mock(self, seed, size):
            return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        monkeypatch.setattr(OutageDistribution, "get_random_derate_fraction_arr", _get_random_derate_frac_mock)

        # Create the outage distribution instance and link it to the resource
        outage_distribution = OutageDistribution(name="Outage_Distribution_1")
        linkage = ResourceToOutageDistribution(
            name=("Example_Resource", "Outage_Distribution_1"),
            instance_from=resource,
            instance_to=outage_distribution,
        )
        ResourceToOutageDistribution.announce_linkage_to_instances()

        # Run the outage simulation
        resource.simulate_outages(model_year=2020, random_seed=2)

        # Check that the simulated outage profile matches expectations
        pd.testing.assert_series_equal(
            resource.outage_profile.data,
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
                data=[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            ),
        )

    def test_simulate_outages_multi_unit(self, make_resource_copy, monkeypatch):
        """Test the `simulate_outages()` method."""
        # Initialize the resource copy and set outage-related attributes
        resource: GenericResource = make_resource_copy()

        # Set unit size (10% of resource nameplate capacity)
        resource.unit_size = ts.NumericTimeseries(
            name="unit_size",
            data=0.1 * resource.capacity_planned.data,
        )

        # Test num units
        assert resource.num_units == {2020: 10, 2030: 10}

        resource.power_output_max = ts.NumericTimeseries(
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
                data=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
        )

        # Note: the value of these attributes have no effect because the scipy exponential random number generator is
        #   mocked below. If these values are not set, then the outage simulation function will not run.
        resource.mean_time_to_repair = 1
        resource.stochastic_outage_rate = ts.FractionalTimeseries(
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
                data=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
            ),
        )  # Has no effect because the random generator is mocked
        resource.random_seed = 5  # Has no effect because the random generator is mocked

        # Mock the outage distribution function that returns outage percentage random variables
        def _get_random_derate_frac_mock(self, seed, size):
            return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        monkeypatch.setattr(OutageDistribution, "get_random_derate_fraction_arr", _get_random_derate_frac_mock)

        # Create the outage distribution instance and link it to the resource
        outage_distribution = OutageDistribution(name="Outage_Distribution_1")
        linkage = ResourceToOutageDistribution(
            name=("Example_Resource", "Outage_Distribution_1"),
            instance_from=resource,
            instance_to=outage_distribution,
        )
        ResourceToOutageDistribution.announce_linkage_to_instances()

        # Run the outage simulation
        resource.simulate_outages(model_year=2020, random_seed=2)

        # Check that the simulated outage profile matches expectations
        pd.testing.assert_series_equal(
            resource.outage_profile.data,
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
                data=[0.8, 1.0, 0.7, 0.8, 0.7, 0.8, 0.8],
            ),
        )

    def test_num_units(self, make_resource_copy):

        # Initialize the resource copy
        resource: GenericResource = make_resource_copy()

        # Assert unit_size is 0 (by default)
        assert (resource.unit_size.data == 0).all()

        # Assert default number of units is 1
        assert resource.num_units == {2020: 1, 2030: 1}

        # Set unit size (10% of resource nameplate capacity)
        resource.unit_size = ts.NumericTimeseries(
            name="unit_size",
            data=0.1 * resource.capacity_planned.data,
        )
        # Clear calculated properties including num_units
        resource.clear_calculated_properties()

        # Assert number of units is 10
        assert resource.num_units == {2020: 10, 2030: 10}

    def test_dispatch(self, make_resource_copy):
        """Test the `dispatch()` function."""
        resource = make_resource_copy()

        # Run dispatch against net load
        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[50.0, -20.0, 0, 150.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"], name="timestamp"
                ),
                data=[200.0, 0.0, 50.0, 100.0],
                name="value",
            ),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-150.0, -20.0, -50.0, 50.0],
            ),
            check_names=False,
        )

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
                data=[400.0, 0.0, 100.0, 200.0],
                name="value",
            ),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-350.0, -20.0, -100.0, -50.0],
            ),
            check_names=False,
        )

    def test_dispatch_all_positive(self, make_resource_copy):
        """Test the `dispatch()` function when all net load hours are positive."""
        resource = make_resource_copy()

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[50.0, 100.0, 10.0, 300.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"], name="timestamp"
                ),
                data=[200.0, 0.0, 50.0, 100.0],
                name="value",
            ),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-150.0, 100.0, -40.0, 200.0],
            ),
            check_names=False,
        )

    def test_dispatch_all_negative(self, make_resource_copy):
        """Test the dispatch() function when all net load hours are negative."""
        resource = make_resource_copy()

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[-50.0, -100.0, -10.0, -300.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"], name="timestamp"
                ),
                data=[200.0, 0.0, 50.0, 100.0],
                name="value",
            ),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-250.0, -100.0, -60.0, -400.0],
            ),
            check_names=False,
        )

    def test_dispatch_all_zero(self, make_resource_copy):
        """Test the `dispatch()` function when all net load hours are zero."""
        resource = make_resource_copy()

        net_load = pd.Series(
            index=pd.DatetimeIndex(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]),
            data=[0.0, 0.0, 0.0, 0.0],
        )

        updated_net_load = resource.dispatch(net_load=net_load, model_year=2020)

        pd.testing.assert_series_equal(
            resource.heuristic_provide_power_mw,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"], name="timestamp"
                ),
                data=[200.0, 0.0, 50.0, 100.0],
                name="value",
            ),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            updated_net_load,
            pd.Series(
                index=pd.DatetimeIndex(
                    ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"]
                ),
                data=[-200.0, 0.0, -50.0, -100.0],
            ),
            check_names=False,
        )

    def test_variable_bounds(self, make_dispatch_model_copy):
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

    def test_adjust_budgets_for_optimization(self, make_resource_copy):
        resource = copy.deepcopy(make_resource_copy())

        resource.energy_budget_monthly = ts.NumericTimeseries(
            name="energy_budget_monthly",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-07-01", "2010-08-01"], name="timestamp"),
                data=[2000.0 / (400 * 24 * 31), 2500.0 / (400 * 24 * 31)],
            ),
            freq_="M",
        )
        original_monthly_energy_budget = resource.energy_budget_monthly.data
        original_scaled_monthly_energy_budget = resource.scaled_monthly_energy_budget[2030]

        resource.energy_budget_annual = ts.NumericTimeseries(
            name="energy_budget_annual",
            data=pd.Series(
                index=pd.DatetimeIndex(["2010-01-01", "2019-01-01"], name="timestamp"),
                data=[3000.0 / (400 * 8760), 4000.0 / (400 * 8760)],
            ),
            freq_="YS",
        )
        original_annual_energy_budget = resource.energy_budget_annual.data
        original_scaled_annual_energy_budget = resource.scaled_annual_energy_budget[2030]

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

        # Check that original energy budgets do not change
        pd.testing.assert_series_equal(original_monthly_energy_budget, resource.energy_budget_monthly.data)
        pd.testing.assert_series_equal(original_annual_energy_budget, resource.energy_budget_annual.data)

        # Check that scaled energy budget is reduced by sum of heuristic_provide_power within window
        monthly_power_provided_in_window = resource._calculate_heuristic_provide_power_in_window(
            heuristic_provide_power=resource.heuristic_provide_power_mw,
            timestamps_included_in_optimization_flags=timestamps_to_include,
            freq="M",
            target_index=resource.scaled_monthly_energy_budget[2030].index,
        )
        new_scaled_monthly_energy_budget = original_scaled_monthly_energy_budget - monthly_power_provided_in_window
        pd.testing.assert_series_equal(new_scaled_monthly_energy_budget, resource.scaled_monthly_energy_budget[2030])

        annual_power_provided_in_window = resource._calculate_heuristic_provide_power_in_window(
            heuristic_provide_power=resource.heuristic_provide_power_mw,
            timestamps_included_in_optimization_flags=timestamps_to_include,
            freq="Y",
            target_index=resource.scaled_annual_energy_budget[2030].index,
        )
        new_scaled_annual_energy_budget = original_scaled_annual_energy_budget - annual_power_provided_in_window
        pd.testing.assert_series_equal(new_scaled_annual_energy_budget, resource.scaled_annual_energy_budget[2030])

    def test_annual_energy_budget_constraint(self, resource_block_with_budgets):
        dispatch_model = copy.deepcopy(resource_block_with_budgets)

        block = dispatch_model.blocks["Example_Resource"]

        for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
            block.power_output[2030, dispatch_window, timestamp].fix(0)
        block.power_output[2030, 1.0, pd.Timestamp("2010-07-01 00:00")].fix(400)
        block.power_output[2030, 2.0, pd.Timestamp("2010-10-01 00:00")].fix(500)

        assert block.annual_energy_budget_constraint[2030, pd.Timestamp("2010-01-01")].body() == 900

        assert block.annual_energy_budget_constraint[2030, pd.Timestamp("2010-01-01")].upper() == pytest.approx(
            3000 + 1
        )  # Tolerance

        assert block.annual_energy_budget_constraint[2030, pd.Timestamp("2010-01-01")].expr()

        block.power_output[2030, 1.0, pd.Timestamp("2010-07-16 12:00")].fix(2500)
        assert block.annual_energy_budget_constraint[2030, pd.Timestamp("2010-01-01")].body() == 3400

        assert block.annual_energy_budget_constraint[2030, pd.Timestamp("2010-01-01")].upper() == pytest.approx(
            3000 + 1
        )  # Tolerance

        assert not block.annual_energy_budget_constraint[2030, pd.Timestamp("2010-01-01")].expr()

    def test_monthly_energy_budget_constraint(self, resource_block_with_budgets):
        dispatch_model = copy.deepcopy(resource_block_with_budgets)

        block = dispatch_model.blocks["Example_Resource"]

        for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
            block.power_output[2030, dispatch_window, timestamp].fix(0)

        block.power_output[2030, 1.0, pd.Timestamp("2010-07-01 00:00")].fix(400)
        block.power_output[2030, 1.0, pd.Timestamp("2010-07-15 12:00")].fix(500)
        assert block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].body() == 900

        assert block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].upper() == pytest.approx(
            2000 + 1
        )  # Tolerance

        assert block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].expr()

        block.power_output[2030, 1.0, pd.Timestamp("2010-07-01 12:00")].fix(2550)
        assert block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].body() == 3450

        assert block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].upper() == pytest.approx(
            2000 + 1
        )  # Tolerance

        assert not block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].expr()

        block.power_output[2030, 2.0, pd.Timestamp("2010-09-03 09:00")].fix(2200)
        block.power_output[2030, 2.0, pd.Timestamp("2010-09-05 22:00")].fix(600)
        assert block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-09-01")].body() == 2800

        assert block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-09-01")].upper() == pytest.approx(
            2419.3548 + 1
        )  # Tolerance, September budget is 30-day budget

        assert not block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-09-01")].expr()

        block.power_output[2030, 2.0, pd.Timestamp("2010-09-03 09:00")].fix(1500)
        assert block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-09-01")].body() == 2100

        assert block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-09-01")].upper() == pytest.approx(
            2419.3548 + 1
        )  # September budget is 30-day budget  # Tolerance

        assert block.monthly_energy_budget_constraint[2030, pd.Timestamp("2010-09-01")].expr()

    def test_daily_energy_budget_constraint(self, resource_block_with_budgets):
        dispatch_model = copy.deepcopy(resource_block_with_budgets)

        block = dispatch_model.blocks["Example_Resource"]

        for dispatch_window, timestamp in dispatch_model.DISPATCH_WINDOWS_AND_TIMESTAMPS:
            block.power_output[2030, dispatch_window, timestamp].fix(0)

        block.power_output[2030, 1.0, pd.Timestamp("2010-07-01 00:00")].fix(400)
        block.power_output[2030, 1.0, pd.Timestamp("2010-07-01 12:00")].fix(500)
        assert block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].body() == 900

        assert block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].upper() == pytest.approx(
            1000 + 1
        )  # Tolerance

        assert block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].expr()

        block.power_output[2030, 1.0, pd.Timestamp("2010-07-01 13:00")].fix(250)
        assert block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].body() == 1150

        assert block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].upper() == pytest.approx(
            1000 + 1
        )  # Tolerance

        assert not block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-07-01")].expr()

        block.power_output[2030, 2.0, pd.Timestamp("2010-09-03 09:00")].fix(2200)
        assert block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-09-03")].body() == 2200

        assert block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-09-03")].upper() == pytest.approx(
            1100 + 1
        )  # Tolerance

        assert not block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-09-03")].expr()

        block.power_output[2030, 2.0, pd.Timestamp("2010-09-03 09:00")].fix(800)
        assert block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-09-03")].body() == 800

        assert block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-09-03")].upper() == pytest.approx(
            1100 + 1
        )  # Tolerance

        assert block.daily_energy_budget_constraint[2030, pd.Timestamp("2010-09-03")].expr()
