import numpy as np
import pandas as pd
from pyomo import environ as pyo

from new_modeling_toolkit.core import linkage
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.system.electric.resources.storage import StorageResource
from new_modeling_toolkit.system.electric.resources.variable import VariableResource


class HybridVariableResource(VariableResource):
    hybrid_storage_resources: dict[str, linkage.HybridStorageResourceToHybridVariableResource] = {}

    @property
    def hybrid_linkage(self):
        hybrid_linkage_name = list(self.hybrid_storage_resources.keys())[0]
        hybrid_linkage = self.hybrid_storage_resources[hybrid_linkage_name]

        return hybrid_linkage

    def upsample(self, load_calendar: pd.DatetimeIndex, random_seed: int = None):
        super().upsample(load_calendar, random_seed)

    @timer
    def dispatch(self, net_load: pd.Series, model_year: int) -> pd.Series:
        """This method currently just returns in the input net load. All heuristic dispatch logic for hybrid variable
        resources is in the HybridStorageResource.dispatch() method because hybrid variable and hybrid storage must be
        dispatched simultaneously.

        Args:
            net_load: the net load to dispatch against
            model_year: which model year to use for the resource profile (not used in the function)

        Returns:
            the input net_load argument
        """
        return net_load

    @timer
    def construct_operational_block(self, model: pyo.ConcreteModel):
        super().construct_operational_block(model)


class HybridStorageResource(StorageResource):
    hybrid_variable_resources: dict[str, linkage.HybridStorageResourceToHybridVariableResource] = {}

    def revalidate(self):
        if len(self.hybrid_variable_resources) > 1:
            raise ValueError(
                f"A HybridStorageResource can only be linked to one other HybridVariableResource, but multiple linkages "
                f"were found: `{[link.instance_to.name for link in self.hybrid_variable_resources.values()]}`"
            )

    # TODO: Only one paired storage-variable
    @property
    def hybrid_linkage(self):
        hybrid_linkage_name = list(self.hybrid_variable_resources.keys())[0]
        hybrid_linkage = self.hybrid_variable_resources[hybrid_linkage_name]

        return hybrid_linkage

    @property
    def paired_variable_resource(self) -> HybridVariableResource:
        return self.hybrid_linkage.instance_to

    def dispatch(self, net_load: pd.Series, model_year: int) -> pd.Series:
        """Dispatches a hybrid storage and hybrid variable resource against a net load for a given model year.

        The storage resource is dispatched with a greedy algorithm. When net load is negative, the resource charges
        as much as possible from the paired variable resource and the grid (if permitted) until it is full.
        When net load is positive, first the variable output is subtracted from net load
        until net load is 0. If variable output is insufficient to meet positive net load, the battery is discharged.

        The dispatch of the paired resources must respect the interconnection_limit and grid_charging_allowed.

        Current implementation assumes there is no SOC_min or P_min.

        Current implementation only supports one storage resource to one variable resource paired resources.

        Args:
            net_load: the net load to dispatch against
            model_year: which model year to use for nameplate capacity and duration

        Returns:
            the load net of the storage dispatch
        """
        # Skip dispatch if insufficient capacity
        if (
            self.capacity_planned.data.at[f"01-01-{model_year}"] < 0.1
            and self.paired_variable_resource.capacity_planned.data.at[f"01-01-{model_year}"] < 0.1
        ):
            self.heuristic_provide_power_mw = 0.0 * self.scaled_pmax_profile[model_year].copy()
            self.paired_variable_resource.heuristic_provide_power_mw = self.heuristic_provide_power_mw.copy()
            self.heuristic_storage_SOC_mwh = 0.0 * self.heuristic_provide_power_mw.copy()
            return net_load

        # This represents the combined output of the paired renewable and storage behind interconnection limit

        # Attributes unique to hybrid storage in linkage
        # Interconnection limit is for MW_variable_grid + MW_storage_grid
        interconnection_limit_mw = self.hybrid_linkage.interconnection_limit_mw.data.at[f"{model_year}-01-01"]
        grid_charging_allowed = self.hybrid_linkage.grid_charging_allowed

        # paired variable profile
        paired_variable_unclipped_scaled_pmax_profile = self.paired_variable_resource.scaled_pmax_profile[model_year]

        # Initialize relevant variables
        net_load_mw_numpy = net_load.copy().to_numpy()
        net_provide_power_mw_numpy_storage = np.zeros(len(net_load_mw_numpy))
        net_provide_power_mw_numpy_variable = np.zeros(len(net_load_mw_numpy))
        SOC_mwh_numpy = np.zeros(len(net_load_mw_numpy))

        SOC_max = self.storage_capacity_planned.slice_by_year(model_year)
        pmax_numpy = self.scaled_pmax_profile[model_year].to_numpy()
        imax_numpy = self.scaled_imax_profile[model_year].to_numpy()

        SOC = 0
        timestamps = net_load.copy().index.to_numpy()
        for i, ts in enumerate(timestamps):
            # Calculate net provide power and SOC for each timestep
            curr_pmax = pmax_numpy[i]
            curr_imax = imax_numpy[i]
            curr_net_load = net_load_mw_numpy[i]
            SOC_mwh_numpy[i] = SOC  # hour beginning
            curr_variable_unclipped_scaled_pmax = paired_variable_unclipped_scaled_pmax_profile[ts]
            curr_available_charging_headroom = SOC_max - SOC
            charging_efficiency = self.charging_efficiency.data.at[ts]
            discharging_efficiency = self.discharging_efficiency.data.at[ts]

            # If net load is negative:
            # - Charge the battery as much as possible from the variable resource
            # - If there is still room in the battery and interconnection to charge the battery from the grid, charge
            #     from the grid
            # - If not, send any remaining variable power to the grid
            if curr_net_load <= 0:
                curr_storage_net_power_from_variable = min(
                    curr_variable_unclipped_scaled_pmax,
                    curr_imax,
                    curr_available_charging_headroom / charging_efficiency,
                )

                curr_storage_net_power_to_grid = (
                    -1
                    * grid_charging_allowed
                    * min(
                        -1 * curr_net_load,
                        curr_imax - curr_storage_net_power_from_variable,
                        curr_available_charging_headroom / charging_efficiency - curr_storage_net_power_from_variable,
                        interconnection_limit_mw,
                    )
                )

                SOC += charging_efficiency * (curr_storage_net_power_from_variable - curr_storage_net_power_to_grid)

                curr_variable_net_power_to_grid = min(
                    curr_variable_unclipped_scaled_pmax - curr_storage_net_power_from_variable,
                    interconnection_limit_mw,
                )

            # If net load is positive:
            # - Send as much power from the variable resource as possible, up to the interconnection limit
            # - If there is still positive net load, discharge the battery to meet it
            # - If there is no more net load and leftover variable power, charge the battery
            else:
                curr_variable_net_power_to_grid = min(
                    curr_variable_unclipped_scaled_pmax, interconnection_limit_mw, curr_net_load
                )

                if curr_net_load - curr_variable_net_power_to_grid > 0:
                    curr_storage_net_power_to_grid = min(
                        curr_net_load - curr_variable_net_power_to_grid,
                        interconnection_limit_mw - curr_variable_net_power_to_grid,
                        curr_pmax,
                        SOC * discharging_efficiency,
                    )
                    SOC -= curr_storage_net_power_to_grid / discharging_efficiency
                    curr_storage_net_power_from_variable = 0
                else:
                    curr_storage_net_power_to_grid = 0
                    curr_storage_net_power_from_variable = min(
                        curr_variable_unclipped_scaled_pmax - curr_variable_net_power_to_grid,
                        curr_imax,
                        curr_available_charging_headroom / charging_efficiency,
                    )
                    SOC += curr_storage_net_power_from_variable * charging_efficiency

            # Save net provide power and SOC
            net_provide_power_mw_numpy_storage[i] = curr_storage_net_power_to_grid
            net_provide_power_mw_numpy_variable[i] = curr_variable_net_power_to_grid

        self.paired_variable_resource.heuristic_provide_power_mw = pd.Series(
            index=net_load.index, data=net_provide_power_mw_numpy_variable
        )
        self.heuristic_provide_power_mw = pd.Series(index=net_load.index, data=net_provide_power_mw_numpy_storage)
        self.heuristic_storage_SOC_mwh = pd.Series(index=net_load.index, data=SOC_mwh_numpy)

        return net_load - net_provide_power_mw_numpy_storage - net_provide_power_mw_numpy_variable

    def construct_operational_block(self, model: pyo.ConcreteModel):
        # This writes constraints for the storage resource and
        # hybrid-specific constraints (interconnection limit, grid charging allowed)

        # Constraints for paired variable resource are already automatically written; hybrid_variable_resources
        # is included in self.resources_to_construct in updated_dispatch_model_v2.py as of 2023-06-20

        # IMPORTANT NOTE: The heuristic dispatch considers the hybrid resource as a single resource
        # and determines the output of the collective paired resource. HOWEVER, the constraints for the optimization
        # treat the two resouces' power_output as separate.
        # They still enforce constraints that exist between the two resources.

        # Write constraints for the storage resource
        super().construct_operational_block(model)

        # Paired variable resource blocks
        paired_variable_block = model.blocks[self.paired_variable_resource.name]
        block = model.blocks[self.name]

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def hybrid_resource_interconnection_limit_constraint(
            block, modeled_year: int, dispatch_window: int, timestamp: pd.Timestamp
        ):
            """The sum of the storage and paired variable resources' output power and reserves may
            not exceed the interconnection limit"""
            return (
                block.power_output[modeled_year, dispatch_window, timestamp]
                + block.provide_reserve[modeled_year, dispatch_window, timestamp]
                + paired_variable_block.power_output[modeled_year, dispatch_window, timestamp]
                + paired_variable_block.provide_reserve[modeled_year, dispatch_window, timestamp]
                <= self.hybrid_linkage.interconnection_limit_mw.data.at[f"{modeled_year}-01-01"]
            )

        @block.Constraint(model.MODELED_YEARS, model.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def hybrid_charging_constraint(block, modeled_year: int, dispatch_window: int, timestamp: pd.Timestamp):
            """Neither the power_input nor power_output may exceed the interconnection limit"""
            return (
                block.power_input[modeled_year, dispatch_window, timestamp]
                - paired_variable_block.power_output[modeled_year, dispatch_window, timestamp]
                <= self.hybrid_linkage.interconnection_limit_mw.data.at[f"{modeled_year}-01-01"]
                * self.hybrid_linkage.grid_charging_allowed
            )
