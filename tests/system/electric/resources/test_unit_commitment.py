from pathlib import Path

import pandas as pd

from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.system.electric.resources import UnitCommitmentResource
from tests.system.electric.resources import test_generic


class TestUnitCommitmentResource(test_generic.TestGenericResource):
    _RESOURCE_CLASS = UnitCommitmentResource
    _RESOURCE_GROUP_PATH = Path("resource_groups/Generic.csv")
    _SYSTEM_COMPONENT_DICT_NAME = "generic_resources"

    _RESOURCE_INIT_KWARGS = dict(
        name="Example_Resource",
        unit_commitment="integer",
        unit_size=ts.NumericTimeseries(
            name="unit_size",
            data=pd.Series(
                index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], name="timestamp"),
                data=[200.0, 400.0],
                name="value",
            ),
        ),
        max_call_duration=4,
        min_stable_level=0.5,
        min_up_time=4,
        min_down_time=4,
    )

    def test_num_units(self, make_resource_copy):

        # Initialize the resource copy
        resource: UnitCommitmentResource = make_resource_copy()

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

    def test_operational_units_in_timepoint(self, resource_block, first_index):
        operational_units = resource_block.operational_units_in_timepoint[first_index]
        assert operational_units.expr() == 1.0

    def test_committed_capacity_mw_power_output(self, resource_block, first_index):
        resource_block.committed_units_power_output[first_index].fix(0)
        assert resource_block.committed_capacity_mw_power_output[first_index].expr() == 0

        resource_block.committed_units_power_output[first_index].fix(1)
        assert resource_block.committed_capacity_mw_power_output[first_index].expr() == 400.0

    def test_plant_increase_load_capacity_in_timepoint_mw(self, resource_block, first_index):
        resource_block.committed_units_power_input[first_index].fix(0)
        assert resource_block.plant_increase_load_capacity_in_timepoint_mw[first_index].expr() == 0

        resource_block.committed_units_power_input[first_index].fix(1)
        assert resource_block.plant_increase_load_capacity_in_timepoint_mw[first_index].expr() == 200.0

    def test_plant_provide_power_capacity_in_timepoint_mw(self, resource_block, first_index):
        resource_block.committed_units_power_output[first_index].fix(0)
        assert resource_block.plant_provide_power_capacity_in_timepoint_mw[first_index].expr() == 0

        resource_block.committed_units_power_output[first_index].fix(1)
        assert resource_block.plant_provide_power_capacity_in_timepoint_mw[first_index].expr() == 200.0

    def test_plant_power_min_capacity_in_timepoint_mw(self, resource_block, first_index):
        resource_block.committed_units_power_output[first_index].fix(0)
        assert resource_block.plant_power_min_capacity_in_timepoint_mw[first_index].expr() == 0

        resource_block.committed_units_power_output[first_index].fix(1)
        assert resource_block.plant_power_min_capacity_in_timepoint_mw[first_index].expr() == 100.0

    def test_zero_committed_units_in_first_timepoint(self, resource_block, first_index):
        model_year, dispatch_window, timestamp = first_index
        resource_block.committed_units[first_index].fix(0)

        assert resource_block.zero_committed_units_in_first_timepoint[(model_year, dispatch_window)].upper() == 0
        assert resource_block.zero_committed_units_in_first_timepoint[(model_year, dispatch_window)].lower() == 0
        assert resource_block.zero_committed_units_in_first_timepoint[(model_year, dispatch_window)].body() == 0
        assert resource_block.zero_committed_units_in_first_timepoint[(model_year, dispatch_window)].expr()

        resource_block.committed_units[first_index].fix(1)
        assert not resource_block.zero_committed_units_in_first_timepoint[(model_year, dispatch_window)].expr()

    def test_zero_start_units_in_first_timepoint(self, resource_block, first_index):
        model_year, dispatch_window, timestamp = first_index
        resource_block.start_units[first_index].fix(0)
        assert resource_block.zero_start_units_in_first_timepoint[(model_year, dispatch_window)].upper() == 0
        assert resource_block.zero_start_units_in_first_timepoint[(model_year, dispatch_window)].lower() == 0
        assert resource_block.zero_start_units_in_first_timepoint[(model_year, dispatch_window)].body() == 0
        assert resource_block.zero_start_units_in_first_timepoint[(model_year, dispatch_window)].expr()

        resource_block.start_units[first_index].fix(1)
        assert not resource_block.zero_start_units_in_first_timepoint[(model_year, dispatch_window)].expr()

    def test_zero_committed_units_in_last_timepoint(self, resource_block, last_index):
        model_year, dispatch_window, timestamp = last_index
        resource_block.committed_units[last_index].fix(0)

        assert resource_block.zero_committed_units_in_last_timepoint[(model_year, dispatch_window)].upper() == 0
        assert resource_block.zero_committed_units_in_last_timepoint[(model_year, dispatch_window)].lower() == 0
        assert resource_block.zero_committed_units_in_last_timepoint[(model_year, dispatch_window)].body() == 0
        assert resource_block.zero_committed_units_in_last_timepoint[(model_year, dispatch_window)].expr()

        resource_block.committed_units[last_index].fix(1)
        assert not resource_block.zero_committed_units_in_last_timepoint[(model_year, dispatch_window)].expr()

    def test_max_provide_power_domain_constraint(self, resource_block, first_index):
        resource_block.committed_units_power_output[first_index].fix(1)
        resource_block.power_output[first_index].fix(0)
        assert resource_block.max_provide_power_domain_constraint[first_index].upper() == 0
        assert resource_block.max_provide_power_domain_constraint[first_index].body() == -200.0
        assert resource_block.max_provide_power_domain_constraint[first_index].expr()

        resource_block.power_output[first_index].fix(100)
        assert resource_block.max_provide_power_domain_constraint[first_index].body() == -100.0
        assert resource_block.max_provide_power_domain_constraint[first_index].expr()

        resource_block.power_output[first_index].fix(200)
        assert resource_block.max_provide_power_domain_constraint[first_index].body() == 0.0
        assert resource_block.max_provide_power_domain_constraint[first_index].expr()

        resource_block.power_output[first_index].fix(300)
        assert resource_block.max_provide_power_domain_constraint[first_index].body() == 100.0
        assert not resource_block.max_provide_power_domain_constraint[first_index].expr()

    def test_min_provide_power_domain_constraint(self, resource_block, first_index):
        resource_block.power_output[first_index].fix(0)
        resource_block.committed_units_power_output[first_index].fix(1)
        assert resource_block.min_provide_power_domain_constraint[first_index].upper() == 0.0
        assert resource_block.min_provide_power_domain_constraint[first_index].body() == 100.0
        assert not resource_block.min_provide_power_domain_constraint[first_index].expr()

        resource_block.power_output[first_index].fix(50)
        assert resource_block.min_provide_power_domain_constraint[first_index].body() == 50.0
        assert not resource_block.min_provide_power_domain_constraint[first_index].expr()

        resource_block.power_output[first_index].fix(100)
        assert resource_block.min_provide_power_domain_constraint[first_index].body() == 0.0
        assert resource_block.min_provide_power_domain_constraint[first_index].expr()

        resource_block.power_output[first_index].fix(200)
        assert resource_block.min_provide_power_domain_constraint[first_index].body() == -100
        assert resource_block.min_provide_power_domain_constraint[first_index].expr()

    def test_max_increase_load_domain_constraint(self, resource_block, first_index):
        resource_block.power_input[first_index].fix(0)
        resource_block.committed_units_power_input[first_index].fix(1)
        assert resource_block.min_provide_power_domain_constraint[first_index].upper() == 0.0
        assert resource_block.max_increase_load_domain_constraint[first_index].body() == -200.0
        assert resource_block.max_increase_load_domain_constraint[first_index].expr()

        resource_block.power_input[first_index].fix(100)
        assert resource_block.max_increase_load_domain_constraint[first_index].body() == -100.0
        assert resource_block.max_increase_load_domain_constraint[first_index].expr()

        resource_block.power_input[first_index].fix(200)
        assert resource_block.max_increase_load_domain_constraint[first_index].body() == 0.0
        assert resource_block.max_increase_load_domain_constraint[first_index].expr()

        resource_block.power_input[first_index].fix(400)
        assert resource_block.max_increase_load_domain_constraint[first_index].body() == 200.0
        assert not resource_block.max_increase_load_domain_constraint[first_index].expr()

    def test_committed_units_ub_constraint(self, resource_block, first_index):
        resource_block.committed_units[first_index].fix(0)
        assert resource_block.committed_units_ub_constraint[first_index].upper() == 0.0
        assert resource_block.committed_units_ub_constraint[first_index].body() == -1.0
        assert resource_block.committed_units_ub_constraint[first_index].expr()

        resource_block.committed_units[first_index].fix(1)
        assert resource_block.committed_units_ub_constraint[first_index].body() == 0.0
        assert resource_block.committed_units_ub_constraint[first_index].expr()

        resource_block.committed_units[first_index].fix(2)
        assert resource_block.committed_units_ub_constraint[first_index].body() == 1.0
        assert not resource_block.committed_units_ub_constraint[first_index].expr()

    def test_start_units_ub_constraint(self, resource_block, first_index):
        resource_block.start_units[first_index].fix(0)
        assert resource_block.start_units_ub_constraint[first_index].upper() == 0.0
        assert resource_block.start_units_ub_constraint[first_index].body() == -1.0
        assert resource_block.start_units_ub_constraint[first_index].expr()

        resource_block.start_units[first_index].fix(1)
        assert resource_block.start_units_ub_constraint[first_index].body() == 0.0
        assert resource_block.start_units_ub_constraint[first_index].expr()

        resource_block.start_units[first_index].fix(2)
        assert resource_block.start_units_ub_constraint[first_index].body() == 1.0
        assert not resource_block.start_units_ub_constraint[first_index].expr()

    def test_shutdown_units_ub_constraint(self, resource_block, first_index):
        resource_block.shutdown_units[first_index].fix(0)
        assert resource_block.shutdown_units_ub_constraint[first_index].upper() == 0.0
        assert resource_block.shutdown_units_ub_constraint[first_index].body() == -1.0
        assert resource_block.shutdown_units_ub_constraint[first_index].expr()

        resource_block.shutdown_units[first_index].fix(1)
        assert resource_block.shutdown_units_ub_constraint[first_index].body() == 0.0
        assert resource_block.shutdown_units_ub_constraint[first_index].expr()

        resource_block.shutdown_units[first_index].fix(2)
        assert resource_block.shutdown_units_ub_constraint[first_index].body() == 1.0
        assert not resource_block.shutdown_units_ub_constraint[first_index].expr()

    def test_commitment_tracking_constraint(self, resource_block, first_index):
        model_year, dispatch_window, timestamp = first_index
        next_index = (model_year, dispatch_window, timestamp + pd.DateOffset(hours=1))
        resource_block.committed_units[first_index].fix(0)
        resource_block.committed_units[next_index].fix(0)
        resource_block.start_units[next_index].fix(1)
        resource_block.shutdown_units[next_index].fix(0)
        assert resource_block.commitment_tracking_constraint[first_index].upper() == 0.0
        assert resource_block.commitment_tracking_constraint[first_index].lower() == 0.0
        assert resource_block.commitment_tracking_constraint[first_index].body() == -1.0
        assert not resource_block.commitment_tracking_constraint[first_index].expr()

        resource_block.committed_units[next_index].fix(1)
        resource_block.committed_units[first_index].fix(0)
        resource_block.start_units[next_index].fix(1)
        resource_block.shutdown_units[next_index].fix(0)
        assert resource_block.commitment_tracking_constraint[first_index].body() == 0.0
        assert resource_block.commitment_tracking_constraint[first_index].expr()

        resource_block.committed_units[next_index].fix(0)
        resource_block.committed_units[first_index].fix(0)
        resource_block.start_units[next_index].fix(1)
        resource_block.shutdown_units[next_index].fix(1)
        assert resource_block.commitment_tracking_constraint[first_index].body() == 0.0
        assert resource_block.commitment_tracking_constraint[first_index].expr()

        resource_block.committed_units[next_index].fix(1)
        resource_block.committed_units[first_index].fix(1)
        resource_block.start_units[next_index].fix(0)
        resource_block.shutdown_units[next_index].fix(0)
        assert resource_block.commitment_tracking_constraint[first_index].body() == 0.0
        assert resource_block.commitment_tracking_constraint[first_index].expr()

        resource_block.committed_units[next_index].fix(0)
        resource_block.committed_units[first_index].fix(1)
        resource_block.start_units[next_index].fix(0)
        resource_block.shutdown_units[next_index].fix(1)
        assert resource_block.commitment_tracking_constraint[first_index].body() == 0.0
        assert resource_block.commitment_tracking_constraint[first_index].expr()

        resource_block.committed_units[next_index].fix(1)
        resource_block.committed_units[first_index].fix(1)
        resource_block.start_units[next_index].fix(1)
        resource_block.shutdown_units[next_index].fix(1)
        assert resource_block.commitment_tracking_constraint[first_index].body() == 0.0
        assert resource_block.commitment_tracking_constraint[first_index].expr()

        resource_block.committed_units[next_index].fix(1)
        resource_block.committed_units[first_index].fix(0)
        resource_block.start_units[next_index].fix(1)
        resource_block.shutdown_units[next_index].fix(1)
        assert resource_block.commitment_tracking_constraint[first_index].body() == 1.0
        assert not resource_block.commitment_tracking_constraint[first_index].expr()

    def test_commitment_power_output_tracking_constraint(self, resource_block, first_index):
        model_year, dispatch_window, timestamp = first_index
        next_index = (model_year, dispatch_window, timestamp + pd.DateOffset(hours=1))
        resource_block.committed_units_power_output[first_index].fix(0)
        resource_block.committed_units_power_output[next_index].fix(0)
        resource_block.start_units_power_output[next_index].fix(1)
        resource_block.shutdown_units_power_output[next_index].fix(0)
        assert resource_block.commitment_power_output_tracking_constraint[first_index].upper() == 0.0
        assert resource_block.commitment_power_output_tracking_constraint[first_index].lower() == 0.0
        assert resource_block.commitment_power_output_tracking_constraint[first_index].body() == -1.0
        assert not resource_block.commitment_power_output_tracking_constraint[first_index].expr()

        resource_block.committed_units_power_output[next_index].fix(1)
        resource_block.committed_units_power_output[first_index].fix(0)
        resource_block.start_units_power_output[next_index].fix(1)
        resource_block.shutdown_units_power_output[next_index].fix(0)
        assert resource_block.commitment_power_output_tracking_constraint[first_index].body() == 0.0
        assert resource_block.commitment_power_output_tracking_constraint[first_index].expr()

        resource_block.committed_units_power_output[next_index].fix(0)
        resource_block.committed_units_power_output[first_index].fix(0)
        resource_block.start_units_power_output[next_index].fix(1)
        resource_block.shutdown_units_power_output[next_index].fix(1)
        assert resource_block.commitment_power_output_tracking_constraint[first_index].body() == 0.0
        assert resource_block.commitment_power_output_tracking_constraint[first_index].expr()

        resource_block.committed_units_power_output[next_index].fix(1)
        resource_block.committed_units_power_output[first_index].fix(1)
        resource_block.start_units_power_output[next_index].fix(0)
        resource_block.shutdown_units_power_output[next_index].fix(0)
        assert resource_block.commitment_power_output_tracking_constraint[first_index].body() == 0.0
        assert resource_block.commitment_power_output_tracking_constraint[first_index].expr()

        resource_block.committed_units_power_output[next_index].fix(0)
        resource_block.committed_units_power_output[first_index].fix(1)
        resource_block.start_units_power_output[next_index].fix(0)
        resource_block.shutdown_units_power_output[next_index].fix(1)
        assert resource_block.commitment_power_output_tracking_constraint[first_index].body() == 0.0
        assert resource_block.commitment_power_output_tracking_constraint[first_index].expr()

        resource_block.committed_units_power_output[next_index].fix(1)
        resource_block.committed_units_power_output[first_index].fix(1)
        resource_block.start_units_power_output[next_index].fix(1)
        resource_block.shutdown_units_power_output[next_index].fix(1)
        assert resource_block.commitment_power_output_tracking_constraint[first_index].body() == 0.0
        assert resource_block.commitment_power_output_tracking_constraint[first_index].expr()

        resource_block.committed_units_power_output[next_index].fix(1)
        resource_block.committed_units_power_output[first_index].fix(0)
        resource_block.start_units_power_output[next_index].fix(1)
        resource_block.shutdown_units_power_output[next_index].fix(1)
        assert resource_block.commitment_power_output_tracking_constraint[first_index].body() == 1.0
        assert not resource_block.commitment_power_output_tracking_constraint[first_index].expr()

    def test_unit_commitment_pmin_constraint(self, resource_block, first_index):
        resource_block.power_output[first_index].fix(0)
        resource_block.committed_units_power_output[first_index].fix(1)
        assert resource_block.unit_commitment_pmin_constraint[first_index].upper() == 0
        assert resource_block.unit_commitment_pmin_constraint[first_index].body() == 200
        assert not resource_block.unit_commitment_pmin_constraint[first_index].expr()

        resource_block.power_output[first_index].fix(200)
        resource_block.committed_units_power_output[first_index].fix(1)
        assert resource_block.unit_commitment_pmin_constraint[first_index].body() == 0
        assert resource_block.unit_commitment_pmin_constraint[first_index].expr()

        resource_block.power_output[first_index].fix(100)
        resource_block.committed_units_power_output[first_index].fix(1)
        assert resource_block.unit_commitment_pmin_constraint[first_index].body() == 100
        assert not resource_block.unit_commitment_pmin_constraint[first_index].expr()

        resource_block.power_output[first_index].fix(400)
        resource_block.committed_units_power_output[first_index].fix(1)
        assert resource_block.unit_commitment_pmin_constraint[first_index].body() == -200
        assert resource_block.unit_commitment_pmin_constraint[first_index].expr()

    def test_min_uptime_constraint(self, resource_block, first_index):
        model_year, dispatch_window, timestamp = first_index
        timestamp = timestamp + pd.DateOffset(hours=5)
        first_index = (model_year, dispatch_window, timestamp)

        resource_block.committed_units[first_index].fix(4)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=1)].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=2)].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=3)].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=4)].fix(1)
        assert resource_block.min_uptime_constraint[first_index].upper() == 0.0
        assert resource_block.min_uptime_constraint[first_index].body() == 0.0
        assert resource_block.min_uptime_constraint[first_index].expr()

        resource_block.committed_units[first_index].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=1)].fix(0)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=2)].fix(0)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=3)].fix(0)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=4)].fix(1)
        assert resource_block.min_uptime_constraint[first_index].body() == 0.0
        assert resource_block.min_uptime_constraint[first_index].expr()

        resource_block.committed_units[first_index].fix(3)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=1)].fix(0)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=2)].fix(0)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=3)].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=4)].fix(1)
        assert resource_block.min_uptime_constraint[first_index].body() == -1.0
        assert resource_block.min_uptime_constraint[first_index].expr()

        resource_block.committed_units[first_index].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=1)].fix(0)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=2)].fix(0)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=3)].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=4)].fix(1)
        assert resource_block.min_uptime_constraint[first_index].body() == 1.0
        assert not resource_block.min_uptime_constraint[first_index].expr()

        resource_block.committed_units[first_index].fix(3)
        resource_block.start_units[first_index].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=1)].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=2)].fix(1)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=3)].fix(0)
        resource_block.start_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=4)].fix(0)
        assert resource_block.min_uptime_constraint[first_index].body() == -1.0
        assert resource_block.min_uptime_constraint[first_index].expr()

    def test_min_downtime_constraint(self, resource_block, first_index):
        model_year, dispatch_window, timestamp = first_index
        timestamp = timestamp + pd.DateOffset(hours=5)
        first_index = (model_year, dispatch_window, timestamp)

        resource_block.committed_units[first_index].fix(1)
        # operational will always be 0 or 1 for integer unit commitment
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=1)].fix(1)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=2)].fix(0)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=3)].fix(0)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=4)].fix(0)
        assert resource_block.min_downtime_constraint[first_index].upper() == 0.0
        assert resource_block.min_downtime_constraint[first_index].body() == 1.0
        assert not resource_block.min_downtime_constraint[first_index].expr()

        resource_block.committed_units[first_index].fix(1)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=1)].fix(1)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=2)].fix(1)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=3)].fix(1)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=4)].fix(1)
        assert resource_block.min_downtime_constraint[first_index].body() == 4.0
        assert not resource_block.min_downtime_constraint[first_index].expr()

        resource_block.committed_units[first_index].fix(0)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=1)].fix(0)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=2)].fix(0)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=3)].fix(0)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=4)].fix(1)
        assert resource_block.min_downtime_constraint[first_index].body() == 0.0
        assert resource_block.min_downtime_constraint[first_index].expr()

        resource_block.committed_units[first_index].fix(0)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=1)].fix(1)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=2)].fix(0)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=3)].fix(0)
        resource_block.shutdown_units[model_year, dispatch_window, timestamp - pd.DateOffset(hours=4)].fix(0)
        assert resource_block.min_downtime_constraint[first_index].body() == 0.0
        assert resource_block.min_downtime_constraint[first_index].expr()
