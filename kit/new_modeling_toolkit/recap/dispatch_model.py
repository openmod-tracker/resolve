import pathlib
import time
import warnings

import loguru as logger
import pandas as pd
import pyomo.environ as pyo
from loguru import logger
from pyomo.opt import TerminationCondition

from new_modeling_toolkit.core.model import ModelTemplate
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.core.utils.pyomo_utils import convert_pyomo_object_to_dataframe
from new_modeling_toolkit.recap.recap_case_settings import DispatchObjective
from new_modeling_toolkit.recap.recap_case_settings import ResourceGrouping


warnings.simplefilter(action="ignore", category=FutureWarning)

# Index level names for net load series used in dispatch and perfect capacity tuning
_NET_LOAD_MC_DRAW_INDEX_NAME = "MC_draw"
_NET_LOAD_SUBPROBLEM_INDEX_NAME = "subproblem"
_NET_LOAD_TIMESTAMP_INDEX_NAME = "timestamp"


class DispatchModel(ModelTemplate):
    """
    Optimization model that dispatches resources to minimize unserved energy for RECAP modeling.
    """

    def __init__(
        self,
        monte_carlo_draw: "MonteCarloDraw",
        perfect_capacity: float,
        calculate_duals_flag: bool = False,
        **kwargs,
    ):
        """
        Creates the dispatch model.
        """
        logger.debug("Initializing model...")
        start = time.time()
        super().__init__(monte_carlo_draw.temporal_settings, monte_carlo_draw.system, **kwargs)
        end = time.time()
        logger.debug(f"Dispatch model initialized. Took {end-start:.0f}s.")

        self.name = monte_carlo_draw.name
        self.net_load = monte_carlo_draw.net_load(ResourceGrouping.NO_ELRS.resource_subclasses)
        self.reserves = monte_carlo_draw.reserves.copy()
        self.objective_fn = monte_carlo_draw.case_settings.dispatch_objective
        self.full_index = self.net_load.index

        # Save MonteCarloDraw, net load as attribute for testing
        self.monte_carlo_draw = monte_carlo_draw

        # Create a dual suffix component so that we can print dual values
        self.calculate_duals_flag = calculate_duals_flag
        if self.calculate_duals_flag:
            self.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # Initialize and construct resource blocks
        self.resources_to_construct = [
            resource
            for resource_subclass in (ResourceGrouping.ELRS.resource_subclasses)
            for resource in getattr(self.system, resource_subclass).values()
            if resource.capacity_planned.data.at[f"01-01-{monte_carlo_draw.model_year}"] > 0.1
        ]
        self.blocks = pyo.Block([resource.name for resource in self.resources_to_construct])
        for resource in self.resources_to_construct:
            resource.construct_operational_block(self)

        # Auxiliary variables for unserved energy tracking
        self.Unserved_Energy_MW = pyo.Var(self.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=pyo.NonNegativeReals)
        self.Unserved_Reserve_MW = pyo.Var(self.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=pyo.NonNegativeReals)

        # Decision Variables
        self.Provide_Power_System_MW = pyo.Var(self.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=pyo.NonNegativeReals)
        self.Provide_Reserve_System_MW = pyo.Var(self.DISPATCH_WINDOWS_AND_TIMESTAMPS, within=pyo.NonNegativeReals)

        #################################
        ########### OBJECTIVE ###########
        #################################
        # fmt: off
        if self.objective_fn == DispatchObjective.EUE:

            @self.Objective(sense=pyo.minimize)
            def Total_Unserved_Energy(model: "DispatchModel"):
                total_unserved_energy = sum(
                    model.Unserved_Energy_MW[dispatch_window, timestamp] + model.Unserved_Reserve_MW[dispatch_window, timestamp]
                    for dispatch_window, timestamp in model.DISPATCH_WINDOWS_AND_TIMESTAMPS
                )
                return total_unserved_energy

        elif self.objective_fn in [DispatchObjective.LOLE, DispatchObjective.EUE_and_100xLOLE]:

            self.LOLE_Count = pyo.Var(self.DAYS, within=pyo.Binary)
            self.Big_M = pyo.Param(initialize=1e6, within=pyo.NonNegativeReals)

            @self.Constraint(self.DAYS)
            def LOLE_Counter_Constraint(model: "DispatchModel", day: pd.Timestamp):
                """Count the number of days with unserved energy or reserve"""
                lole_counter_constraint = (
                    sum(
                        model.Unserved_Energy_MW[dispatch_window, timestamp] + model.Unserved_Reserve_MW[dispatch_window, timestamp]
                        for dispatch_window, timestamp in model.DAY_TO_TIMESTAMPS_MAPPING[day]
                    )
                    <= model.LOLE_Count[day] * model.Big_M
                )
                return lole_counter_constraint

            if self.objective_fn == DispatchObjective.LOLE:
                @self.Objective(sense=pyo.minimize)
                def Total_Unserved_Energy_Events(model: "DispatchModel"):
                    total_unserved_energy_events = sum(model.LOLE_Count[day] for day in model.DAYS)
                    return total_unserved_energy_events

            elif self.objective_fn == DispatchObjective.EUE_and_100xLOLE:
                @self.Objective(sense=pyo.minimize)
                def Total_Unserved_Energy_and_100x_Unserved_Energy_Events(model):
                    total_unserved_energy = sum(
                        model.Unserved_Energy_MW[dispatch_window, timestamp] + model.Unserved_Reserve_MW[
                            dispatch_window, timestamp]
                        for dispatch_window, timestamp in model.DISPATCH_WINDOWS_AND_TIMESTAMPS
                    )
                    total_unserved_energy_events = sum(model.LOLE_Count[day] for day in model.DAYS)
                    return total_unserved_energy + 100 * total_unserved_energy_events

        elif self.objective_fn in [DispatchObjective.LOLH]:
            self.LOLH_Count = pyo.Var(self.TIMESTAMPS, within=pyo.Binary)
            self.Big_M = pyo.Param(initialize=1e6, within=pyo.NonNegativeReals)
            @self.Constraint(self.DISPATCH_WINDOWS_AND_TIMESTAMPS)
            def LOLH_Counter_Constraint(model: "DispatchModel", dispatch_window: int, timestamp: pd.Timestamp):
                """Count the number of days with unserved energy or reserve"""
                lolh_counter_constraint = (
                    model.Unserved_Energy_MW[dispatch_window, timestamp] + model.Unserved_Reserve_MW[dispatch_window, timestamp]
                    <= model.LOLH_Count[timestamp] * model.Big_M
                )
                return lolh_counter_constraint

            @self.Objective(sense=pyo.minimize)
            def Total_Unserved_Energy_Hours(model: "DispatchModel"):
                total_unserved_energy_hours = sum(model.LOLH_Count[hour] for hour in model.TIMESTAMPS)
                return total_unserved_energy_hours

        else:
            raise ValueError(f"Dispatch objective function `{self.objective_fn}` is currently not implemented.")

        #################################
        ## UNSERVED ENERGY CONSTRAINTS ##
        #################################

        @self.Constraint(self.MODELED_YEARS, self.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def Unserved_Energy_Constraint(model: "DispatchModel", modeled_year: int, dispatch_window: int, timestamp: pd.Timestamp):
            """Defining the calculation of unserved energy."""
            unserved_energy_constraint = (
                model.Unserved_Energy_MW[dispatch_window, timestamp] >=
                sum(
                model.blocks[resource.name].power_input[modeled_year, dispatch_window, timestamp]
                for resource in self.resources_to_construct
            ) - sum(
                model.blocks[resource.name].power_output[modeled_year, dispatch_window, timestamp]
                for resource in self.resources_to_construct
            ) - model.Provide_Power_System_MW[dispatch_window, timestamp] + max(0, self.net_load.loc[timestamp] - perfect_capacity)
            )
            return unserved_energy_constraint

        @self.Constraint(self.MODELED_YEARS, self.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def Unserved_Reserve_Constraint(model: "DispatchModel", modeled_year: int,  dispatch_window: int, timestamp: pd.Timestamp):
            """Defining the calculation of unserved reserve."""
            unserved_reserve_constraint = (
                model.Unserved_Reserve_MW[dispatch_window, timestamp]
                >= self.reserves.loc[timestamp]
                - sum(
                model.blocks[resource.name].provide_reserve[modeled_year, dispatch_window, timestamp]
                for resource in self.resources_to_construct
            ) - model.Provide_Reserve_System_MW[dispatch_window, timestamp]
            )
            return unserved_reserve_constraint

        @self.Constraint(self.DISPATCH_WINDOWS_AND_TIMESTAMPS)
        def Max_System_Output_Constraint(model: "DispatchModel", dispatch_window: int, timestamp: pd.Timestamp):
            max_system_output_constraint = (
                    model.Provide_Power_System_MW[dispatch_window, timestamp]
                    + model.Provide_Reserve_System_MW[dispatch_window, timestamp]
                    <= max(0, perfect_capacity - self.net_load.loc[timestamp])
            )
            return max_system_output_constraint
        # fmt: on

    @property
    def return_int_components(self):
        """
        Return a list of all integer or binary variable components
        """
        int_list = []
        for comp in list(self.component_objects(pyo.Var, active=True)):
            if comp[comp.index_set()[1]].domain in [pyo.NonNegativeIntegers, pyo.Binary]:
                int_list.append(comp)
        return int_list

    def solve(self, output_dir: pathlib.Path, calculate_duals_flag: bool = False):
        """Solves the dispatch model with the specified solver.
        Args:
            solver_name: which solver to use
        Returns:
            solution: a pyomo solution object
        """

        # Get solver and set solver options
        solver = pyo.SolverFactory("gurobi", solver_io="lp")

        if output_dir is not None:
            solver.options["ResultFile"] = str(output_dir / "infeasibility.ilp")
            solver.options["LogFile"] = str(output_dir / "recap.log")

        # Solve model
        solution = solver.solve(self, tee=True)

        if solution.solver.termination_condition == TerminationCondition.infeasible:
            logger.warning(
                "Model was infeasible; re-solving model with `symbolic_solver_labels=True` to provide better Irreducible Infeasible Set (IIS) information."
            )
            solver.solve(self, keepfiles=True, tee=True, symbolic_solver_labels=True)

            raise RuntimeError(
                f"Model was infeasible; check ILP file for infeasible constraints {((output_dir) / 'infeasibility.ilp').absolute()}"
            )

        if (
            calculate_duals_flag
            and self.monte_carlo_draw.case_settings.print_duals
            and len(self.return_int_components) > 0
        ):
            # fix all integer variables
            for comp in self.return_int_components:
                for x, index in enumerate(comp.index_set()):
                    comp[index].fix(comp[index]())

            logger.info("Re-solving to calculate duals...")
            solution = solver.solve(self, tee=True)

        return solution

    @timer
    def calculate_resource_reserve_results(self):
        # Save optimized reserves dispatch back to resources
        for resource in self.resources_to_construct:
            reserves = convert_pyomo_object_to_dataframe(self.blocks[resource.name].provide_reserve).squeeze()
            # Reset index to just timestamps and drop model year, dispatch window labels
            reserves = (
                reserves.reset_index().set_index("TIMESTAMPS").drop(columns=["MODELED_YEARS", "DISPATCH_WINDOWS"])
            )
            # Save back to resource
            resource.optimized_provide_reserves = reserves

    @timer
    def calculate_resource_dispatch_results(self):
        # Save optimized dispatch back to resources
        for resource in self.resources_to_construct:
            power_output = convert_pyomo_object_to_dataframe(self.blocks[resource.name].power_output).squeeze()
            if (
                resource.name in self.system.storage_resources.keys()
                or resource.name in self.system.flex_load_resources.keys()
            ):
                power_input = convert_pyomo_object_to_dataframe(self.blocks[resource.name].power_input).squeeze()
                power_output -= power_input
            # Reset index to just timestamps and drop model year, dispatch window labels
            power_output = (
                power_output.reset_index().set_index("TIMESTAMPS").drop(columns=["MODELED_YEARS", "DISPATCH_WINDOWS"])
            )
            # Save back to resource
            resource.optimized_provide_power_mw = power_output

        # Get total unserved energy and reserve
        unserved_energy = convert_pyomo_object_to_dataframe(self.Unserved_Energy_MW).squeeze()
        unserved_reserve = convert_pyomo_object_to_dataframe(self.Unserved_Reserve_MW).squeeze()
        unserved_energy_and_reserve = unserved_energy + unserved_reserve

        # Reset index to just timestamps and drop dispatch window labels
        unserved_energy_and_reserve = (
            unserved_energy_and_reserve.reset_index().set_index("TIMESTAMPS").drop(columns=["DISPATCH_WINDOWS"])
        )
        unserved_energy_and_reserve_full = pd.DataFrame(index=self.full_index)
        unserved_energy_and_reserve = unserved_energy_and_reserve_full.join(unserved_energy_and_reserve).fillna(0)

        # Save unserved energy and reserve back to model
        self.unserved_energy_and_reserve = unserved_energy_and_reserve
        self.calculate_resource_reserve_results()

        # Save duals of max system output constraint if
        if self.monte_carlo_draw.case_settings.print_duals:
            max_system_output_constraint = convert_pyomo_object_to_dataframe(self.Max_System_Output_Constraint)
            unserved_energy_and_reserve_duals = (
                max_system_output_constraint["Dual"]
                .reset_index()
                .set_index("TIMESTAMPS")
                .drop(columns=["DISPATCH_WINDOWS"])
            )
            unserved_energy_and_reserve_duals_full = pd.DataFrame(index=self.full_index)
            unserved_energy_and_reserve_duals = unserved_energy_and_reserve_duals_full.join(
                unserved_energy_and_reserve_duals
            ).fillna(0)
            self.unserved_energy_and_reserve_duals = unserved_energy_and_reserve_duals

        if self.monte_carlo_draw.case_settings.print_raw_results:
            self._print_opt_outputs()

    def _print_opt_outputs(self):
        """
        Print optimization constraints, variables, parameters, and expressions by split mc draw to report folder.
        Prints out Unserved Energy, Unserved Reserve and Max System Output constraints always, but will print all others if print_raw_results toggle is TRUE in case settings
        """
        save_dir = self.system.dir_str.recap_output_dir / f"{self.name}_raw_results"
        save_dir.mkdir(exist_ok=True)
        components_to_print = []
        for component in [pyo.Constraint, pyo.Var, pyo.Param, pyo.Expression]:
            components_to_print.extend(list(self.component_objects(component, active=True)))
        for cp in components_to_print:
            df = convert_pyomo_object_to_dataframe(cp)
            df.to_csv(save_dir / f"{cp.name}.csv")
