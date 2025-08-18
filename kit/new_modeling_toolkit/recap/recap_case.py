import copy
import gc
import os
import shutil
from tempfile import TemporaryDirectory
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import bootstrap
from tqdm.auto import tqdm

import new_modeling_toolkit.recap.dispatch_model as dispatch_model
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.core.utils.gurobi_utils import GurobiCredentials
from new_modeling_toolkit.core.utils.gurobi_utils import (
    set_license_file_environment_variable,
)
from new_modeling_toolkit.core.utils.parallelization_utils import parallelize
from new_modeling_toolkit.core.utils.string_utils import convert_snake_to_lower_camel_case
from new_modeling_toolkit.core.utils.util import DirStructure
from new_modeling_toolkit.recap.monte_carlo_draw import MonteCarloDraw
from new_modeling_toolkit.recap.recap_case_settings import DispatchMode
from new_modeling_toolkit.recap.recap_case_settings import RecapCaseSettings
from new_modeling_toolkit.recap.recap_case_settings import ReliabilityMetric
from new_modeling_toolkit.recap.recap_case_settings import ResourceGrouping
from new_modeling_toolkit.system import System

_GUROBI_DEFAULT_NUM_DISTRIBUTED_WORKERS = 0
_GUROBI_DEFAULT_INSTANCE_TYPE = "r6i.8xlarge"

# TODO: move helper functions into a separate script?
### Helper functions
def _dispatch(
    subproblem_number: int,
    split_mc_draw: MonteCarloDraw,
    perfect_capacity: float,
    heuristic_dispatch_mode: ResourceGrouping,
    dispatch_mode: DispatchMode,
    heuristic_net_load_subclasses=None,
    calculate_duals_flag: bool = False,
) -> Tuple[str, int, pd.DataFrame]:
    """
    Dispatches the resources in a single subproblem. Calculate heuristics, compress and optimize if necessary.
    """
    # make sure log files are set for parallelization
    i = logger.add(split_mc_draw.dir_str.recap_output_dir / "recap.log", level="DEBUG")
    t = logger.add(split_mc_draw.dir_str.recap_output_dir / "timing_log.log", level="SUCCESS", format="{message}")
    # Calculate heuristic dispatch for all resources
    split_mc_draw.heuristic_dispatch(perfect_capacity=perfect_capacity, mode=heuristic_dispatch_mode)

    if dispatch_mode == DispatchMode.SEMI_OPTIMIZED or dispatch_mode == DispatchMode.FULLY_OPTIMIZED:
        # Compress MC draws
        split_mc_draw.compress(
            perfect_capacity=perfect_capacity, heuristic_net_load_subclasses=heuristic_net_load_subclasses
        )
        # Create and solve dispatch model
        split_mc_draw.optimized_dispatch(perfect_capacity=perfect_capacity, calculate_duals_flag=calculate_duals_flag)

    logger.remove(i)
    logger.remove(t)

    return (split_mc_draw.name, subproblem_number, split_mc_draw.calculate_dispatch_results())


def _combine_dispatch_results(subproblem_dispatch_results: List[Tuple[str, int, pd.DataFrame]]) -> pd.DataFrame:
    """
    Combine the dispatch results from all subproblems into one DataFrame
    """
    dispatch_results = pd.concat(
        {
            (draw_name, subproblem_number): dispatch_results
            for draw_name, subproblem_number, dispatch_results in subproblem_dispatch_results
        },
        axis=0,
        names=(
            dispatch_model._NET_LOAD_MC_DRAW_INDEX_NAME,
            dispatch_model._NET_LOAD_SUBPROBLEM_INDEX_NAME,
            dispatch_model._NET_LOAD_TIMESTAMP_INDEX_NAME,
        ),
    )
    dispatch_results = dispatch_results.sort_index(
        level=(
            dispatch_model._NET_LOAD_MC_DRAW_INDEX_NAME,
            dispatch_model._NET_LOAD_SUBPROBLEM_INDEX_NAME,
            dispatch_model._NET_LOAD_TIMESTAMP_INDEX_NAME,
        )
    )

    return dispatch_results


class RecapCase:
    ###############################
    ##### Managerial Functions ####
    ###############################

    def __init__(
        self,
        case_name: str,
        dir_str: DirStructure,
        gurobi_credentials: GurobiCredentials,
        case_settings: RecapCaseSettings,
        system: System,
        monte_carlo_draws: Optional[Dict[str, MonteCarloDraw]] = None,
    ):

        # Initialize RecapCase object
        self.case_name = case_name
        self.dir_str = dir_str
        self.gurobi_credentials = gurobi_credentials
        self.case_settings = case_settings
        self.system = system
        self.monte_carlo_draws = monte_carlo_draws

        # Get model year from case settings
        self.model_year = self.case_settings.analysis_year

        # Set minimum resource nameplate capacity to 1e-5 to avoid scaling/numerical issues throughout RECAP code
        for resource in self.system.resources.values():
            resource.capacity_planned.data = resource.capacity_planned.data.clip(lower=1e-5)

        # Initialize results dataframes
        self.reliability_results = pd.DataFrame()
        self.ELCC_results = pd.DataFrame(columns=["perfect_capacity_shortfall", "incremental", "bisection_xtol"])

    # TODO: usage feels limited enough that this can be phased out...? Can do that during tuning clean-up
    @property
    def ELR_capacity(self):
        """
        Energy limited resource: an ELR is an energy supplier that is unable to operate
        at a level that represents its ICAP obligation for all hours of the day, but can
        operate at that level for at least four consecutive hours each day
        """
        elr_capacity = sum(
            sum(resource.capacity_planned.slice_by_year(self.model_year) for resource in resource_dict.values())
            for resource_dict in [
                getattr(self.system, resource_type) for resource_type in ResourceGrouping.ELRS.resource_subclasses
            ]
        )
        return elr_capacity

    @classmethod
    def from_dir(
        cls,
        case_name: str,
        dir_str: DirStructure,
        gurobi_credentials: GurobiCredentials,
    ) -> "RecapCase":

        # Read in case settings
        case_settings = RecapCaseSettings.from_csv(dir_str.recap_settings_dir / case_name / "case_settings.csv")

        # Get case-specific scenarios
        if (dir_str.recap_settings_dir / case_name / "scenarios.csv").is_file():
            logger.info("Reading scenario settings")
            scenarios = pd.read_csv(dir_str.recap_settings_dir / case_name / "scenarios.csv")["scenarios"].tolist()
        else:
            scenarios = []

        # Get System class instance
        system = System(
            name=case_settings.system_name,
            dir_str=dir_str,
            scenarios=scenarios,
        )

        # Instantiate RECAP case
        case_instance = cls(
            case_name=case_name,
            dir_str=dir_str,
            gurobi_credentials=gurobi_credentials,
            case_settings=case_settings,
            system=system,
        )

        return case_instance

    def setup_monte_carlo_draws(self):
        """
        Set up MonteCarloDraw instances for current case from case settings
        Defines monte_carlo_draws attribute (dictionary of MonteCarloDraw instances)
        """
        logger.info("Setting up Monte Carlo draws...")

        # Assert monte_carlo_draws attribute is defined and is None, then set to empty dictionary
        assert hasattr(self, "monte_carlo_draws")
        assert self.monte_carlo_draws is None
        self.monte_carlo_draws = {}

        # Scale loads in system to model year (do this once before creating Monte Carlo draws)
        for load in self.system.loads.values():
            load.forecast_load((self.model_year, self.model_year))

        # Initialize load and reserves once for all Monte Carlo draws
        gross_load = self.system.zones[self.case_settings.zone_to_analyze].get_aggregated_load_profile(self.model_year)
        # TODO: pass gross load and reserves into MonteCarloDraws
        # reserves = self.system.zones[self.case_settings.zone_to_analyze].get_aggregated_up_reserves(self.model_year)

        # Upsample resources with non-randomized data once for all Monte Carlo draws
        for subclass in ResourceGrouping.DETERMINISTIC_UPSAMPLING.resource_subclasses:
            resources = getattr(self.system, subclass).values()
            for resource in resources:
                resource.upsample(load_calendar=gross_load.index.copy())

        # Setup Monte Carlo draws
        for k in range(self.case_settings.number_of_monte_carlo_draws):
            logger.info(f"Setting up MC_draw_{k}")
            # Instantiate Monte Carlo draw
            mc_draw_k = MonteCarloDraw(
                dir_str=copy.deepcopy(self.dir_str),
                case_settings=self.case_settings,
                system=copy.deepcopy(self.system),
                random_seed=k,
            )

            # Upsample + simulate outages for Monte Carlo draw instance (note: this should always happen in this order)
            mc_draw_k.upsample()  # Upsample data first
            mc_draw_k.simulate_outages()  # Simulate outages second

            # Save Monte Carlo draw to dictionary
            self.monte_carlo_draws[mc_draw_k.name] = mc_draw_k

    # TODO: move this into the helper functions for ELCC calculations below?
    @timer
    def _calculate_elcc(self, elcc_type: str):
        """
        Read in ELCC points matrix and call calculate_ELCC_points method
        """
        logger.info(f"Calculating {elcc_type} of resources in case {self.case_name}")
        ELCC_points_matrix = pd.read_csv(
            self.dir_str.recap_settings_dir / self.case_name / "ELCC_surfaces" / f"{elcc_type}.csv"
        )
        # TODO: do we still want ELCC results for specific run types?
        setattr(
            self, f"{elcc_type}_results", self.calculate_ELCC_points(ELCC_points_matrix, ELCC_surface_name=elcc_type)
        )

    @timer
    def run_case(self):
        """Runs full RECAP case"""

        if self.case_settings.calculate_reliability:
            logger.info(f"Calculating reliability metrics for case {self.case_name}")
            self.run_dispatch(
                perfect_capacity=0,
                dispatch_mode=self.case_settings.dispatch_mode,
                calculate_duals_flag=self.case_settings.print_duals,
            )
            self.untuned_dispatch_results = self.dispatch_results

        if self.case_settings.calculate_reliability_w_incremental_pcap:
            logger.info(
                f"Calculating reliability metrics for case {self.case_name} with {self.case_settings.incremental_pcap:.0f} MW of incremental perfect capacity"
            )
            if not self.case_settings.incremental_pcap:
                logger.warning(
                    "Warning: calculate reliability with incremental perfect capacity is selected, but no incremental "
                    "perfect capacity is specified ('incremental_pcap' row of 'case_settings.csv')"
                )
            self.run_dispatch(
                perfect_capacity=self.case_settings.incremental_pcap,
                dispatch_mode=self.case_settings.dispatch_mode,
                calculate_duals_flag=self.case_settings.print_duals,
            )
            if hasattr(self, "untuned_dispatch_results"):
                logger.warning(
                    "Over-writing untuned dispatch results with dispatch results from case with incremental perfect capacity"
                )
            self.untuned_dispatch_results = self.dispatch_results

        if self.case_settings.calculate_perfect_capacity_shortfall:
            logger.info(f"Calculating perfect capacity shortfall for case {self.case_name}")
            self.calculate_perfect_capacity_shortfall()

        if self.case_settings.calculate_total_resource_need:
            logger.info(f"Calculating total resource need for case {self.case_name}")
            self.calculate_total_resource_need()

        if self.case_settings.calculate_marginal_ELCC:
            self._calculate_elcc("marginal_ELCC")

        if self.case_settings.calculate_incremental_last_in_ELCC:
            self._calculate_elcc("incremental_last_in_ELCC")

        if self.case_settings.calculate_decremental_last_in_ELCC:
            self._calculate_elcc("decremental_last_in_ELCC")

        if self.case_settings.calculate_ELCC_surface:
            self._calculate_elcc("custom_ELCC_surface")

    def _print_resource_portfolio(self, output_dir):
        # Save portfolio resource nameplate capacities
        portfolio_nameplate_df = pd.DataFrame(index=self.system.resources.keys())
        for resource in self.system.resources.values():
            # Get resource group
            group = resource.resource_group.name
            portfolio_nameplate_df.loc[resource.name, "group"] = group
            # Get nameplate capacity (MW)
            portfolio_nameplate_df.loc[resource.name, "capacity_planned"] = resource.capacity_planned.slice_by_year(
                self.model_year
            )
            # Optionally get storage capacity (MWh)
            if resource.name in (self.system.storage_resources.keys() | self.system.hybrid_storage_resources.keys()):
                portfolio_nameplate_df.loc[
                    resource.name, "storage_capacity_planned"
                ] = resource.storage_capacity_planned.slice_by_year(self.model_year)
        # Print out to csv
        portfolio_nameplate_df.to_csv(output_dir / "resource_portfolio.csv")
        return portfolio_nameplate_df

    def _print_reliability_results(self, output_dir):
        self.reliability_results.index.name = "perfect_capacity"
        self.reliability_results.to_csv(output_dir / "reliability_results.csv")

    def _print_ELCC_results(self, output_dir):
        # Get total resource need (if available)
        if hasattr(self, "total_resource_need"):
            total_resource_need = self.total_resource_need
        else:
            total_resource_need = np.nan

        # Save total resource need, portfolio ELCC
        self.ELCC_results["total_resource_need"] = total_resource_need
        self.ELCC_results["portfolio_ELCC"] = (
            self.ELCC_results["total_resource_need"] - self.ELCC_results["perfect_capacity_shortfall"]
        )

        # Get base case perfect capacity shortfall (if available)
        if "base_case" in self.ELCC_results.index:
            base_case_perfect_capacity_shortfall = self.ELCC_results.loc["base_case", "perfect_capacity_shortfall"]
        else:
            base_case_perfect_capacity_shortfall = np.nan

        # Save incremental ELCC
        self.ELCC_results["base_case_perfect_capacity_shortfall"] = base_case_perfect_capacity_shortfall
        self.ELCC_results["incremental_ELCC"] = (
            self.ELCC_results["base_case_perfect_capacity_shortfall"] - self.ELCC_results["perfect_capacity_shortfall"]
        )

        # Re-order columns
        columns = self.ELCC_results.columns
        ELCC_columns = [
            "total_resource_need",
            "perfect_capacity_shortfall",
            "portfolio_ELCC",
            "base_case_perfect_capacity_shortfall",
            "incremental_ELCC",
            "incremental",
            "bisection_xtol",
        ]
        other_columns = [col for col in columns if col not in ELCC_columns]
        reordered_columns = ELCC_columns + other_columns
        self.ELCC_results = self.ELCC_results[reordered_columns]

        # Write out ELCC results
        self.ELCC_results.to_csv(output_dir / "ELCC_results.csv")

    def _print_untuned_dispatch_results(self, output_dir):
        if self.case_settings.output_dispatch_results:
            logger.debug("Writing untuned dispatch results")
            self.untuned_dispatch_results.to_parquet(output_dir / "untuned_dispatch_results.parquet")

    def _print_tuned_dispatch_results(self, output_dir):
        if self.case_settings.output_dispatch_results:
            logger.debug("Writing tuned dispatch results")
            self.tuned_dispatch_results.to_parquet(output_dir / "tuned_dispatch_results.parquet")

    def print_timing_df(self):

        # TODO: add comments to make it clear what this is doing

        output_dir = self.dir_str.recap_output_dir
        df = pd.read_csv(output_dir / "timing_log.log", header=None)
        df = df[0].str.split(": ", expand=True)
        df[3] = df[2].str.split("'", expand=True)[1].fillna(0)
        df[2] = df[2].str.split(" ", expand=True)[2].fillna(0)
        df[2] = df[2].replace("", np.nan).astype(float)
        df = df[[0, 3, 2]]

        mean = df.dropna(axis=0).groupby([0, 3]).mean()
        avg = df.dropna(axis=0).groupby([0, 3]).max()
        count = df.dropna(axis=0).groupby([0, 3]).count()
        total = df.dropna(axis=0).groupby([0, 3]).sum()

        df = pd.concat([mean, avg, count, total], axis=1)
        df.columns = ["Average time per hit", "Max time per hit", "# of hits", "Total time (s)"]
        df["Total time (min)"] = df["Total time (s)"] / 60
        df.index.names = ("Component Type", "Function Fame")
        df.sort_index(level=1).to_csv(output_dir / "timing_df.csv")

    @timer
    def report_results(self):
        """Saves results for full RECAP case"""
        # Create output directory for case results
        output_dir = self.dir_str.recap_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing results for `{self.case_name}` to `{output_dir}`")

        # Save case settings / scenarios files to output directory
        for f in ["case_settings.csv", "scenarios.csv"]:
            shutil.copy(self.dir_str.recap_settings_dir / self.case_name / f, output_dir / f)

        # TODO: print out median peak load / load statistics somewhere here

        for report_results_func in [
            self._print_resource_portfolio,
            self._print_reliability_results,
            self._print_ELCC_results,
            self._print_tuned_dispatch_results,
            self._print_untuned_dispatch_results,
        ]:
            try:
                report_results_func(output_dir)
            except Exception as e:
                logger.warning(e)

    #############################
    ##### Dispatch Functions ####
    #############################
    def _create_and_start_gurobi_pool(self, num_instances: int, num_jobs: int):
        """
        Start the Gurobi pool and set solver options for optimized and semi-optimized dispatch
        """
        pool_id = convert_snake_to_lower_camel_case(self.case_name)
        self.gurobi_credentials.pool_id = pool_id
        if not self.gurobi_credentials.check_if_pool_exists():
            logger.info("Creating Gurobi pool...")
            self.gurobi_credentials.create_pool(
                machine_type=_GUROBI_DEFAULT_INSTANCE_TYPE,
                num_instances=num_instances,
                job_limit=num_jobs,
                num_distributed_workers=_GUROBI_DEFAULT_NUM_DISTRIBUTED_WORKERS,
            )
        logger.info("Starting Gurobi pool...")
        self.gurobi_credentials.scale_pool(num_instances=num_instances)
        self.gurobi_credentials.start_pool(wait_time_seconds=1)

        license_path = self.dir_str.recap_output_dir.joinpath("gurobi.lic")
        self.gurobi_credentials.to_license_file(license_path)
        set_license_file_environment_variable(license_path)

        return pool_id

    def run_dispatch(
        self,
        perfect_capacity: float,
        dispatch_mode: DispatchMode,
        heuristic_dispatch_mode: ResourceGrouping = ResourceGrouping.DEFAULT,
        calculate_duals_flag: bool = False,
        bootstrap: bool = True,
    ) -> pd.Series:
        """
        Runs dispatch optimization across all MC draws.

        Args:
            perfect_capacity: the amount of perfect capacity to add or remove from the system in dispatch
            dispatch_mode: heuristics_only, semi-optimized, or fully-optimized
            heuristic_dispatch_mode: resource list with or without ELRs
            parallel: If true, calculate split_mc_draws in parallel
            untuned_flag: If true, save results as "untuned_dispatch_results"

        Returns:
            unserved_energy_and_reserve: pd.Series of hourly unserved energy and reserves across all MC draws

        """
        logger.info("Dispatching resources across all Monte Carlo draws")

        # Save dispatch settings to reliability results
        self.reliability_results.loc[perfect_capacity, "dispatch_mode"] = dispatch_mode.name
        self.reliability_results.loc[perfect_capacity, "heuristic_dispatch_mode"] = heuristic_dispatch_mode.name

        # Initialize subproblem dispatch results list
        # List of 3-tuples: (MC draw name, subproblem number, dispatch results dataframe)
        subproblem_dispatch_results = []

        if dispatch_mode == DispatchMode.SEMI_OPTIMIZED or dispatch_mode == DispatchMode.HEURISTICS_ONLY:
            heuristic_net_load_subclasses = ResourceGrouping.DEFAULT.resource_subclasses
        elif dispatch_mode == DispatchMode.FULLY_OPTIMIZED:
            heuristic_net_load_subclasses = ResourceGrouping.NO_ELRS.resource_subclasses
        else:
            raise NotImplementedError(
                f"Mapping between dispatch model `{dispatch_mode}` and heuristic dispatch net "
                f"load subclasses is not defined."
            )

        if dispatch_mode == DispatchMode.HEURISTICS_ONLY and heuristic_dispatch_mode == ResourceGrouping.NO_ELRS:
            # Heuristic without ELRs

            for mc_draw in tqdm(list(self.monte_carlo_draws.values()), desc="Dispatching MC Draws:"):
                mc_dispatch_results = _dispatch(
                    subproblem_number=mc_draw.name,
                    split_mc_draw=mc_draw,
                    perfect_capacity=perfect_capacity,
                    heuristic_dispatch_mode=heuristic_dispatch_mode,
                    dispatch_mode=dispatch_mode,
                    heuristic_net_load_subclasses=heuristic_net_load_subclasses,
                    calculate_duals_flag=calculate_duals_flag,
                )
                subproblem_dispatch_results.append(mc_dispatch_results)

        else:
            # Heuristic with ELRs or Optimized

            # split mc_draws into subproblems
            split_mc_draws = {}
            for mc_draw in self.monte_carlo_draws.values():
                mc_split = mc_draw.split(mc_draw.case_settings.maximum_subproblem_length)
                split_mc_draws.update(mc_split)

            # Determine number of subproblems
            num_processes = min(len(split_mc_draws), os.cpu_count())

            if dispatch_mode in [DispatchMode.SEMI_OPTIMIZED, DispatchMode.FULLY_OPTIMIZED]:
                self._create_and_start_gurobi_pool(
                    num_instances=int(np.ceil(num_processes / len(split_mc_draws))), num_jobs=len(split_mc_draws)
                )

            subproblem_kwargs = [
                dict(
                    subproblem_number=subproblem_number,
                    split_mc_draw=copy.deepcopy(split_mc_draw),
                    perfect_capacity=perfect_capacity,
                    heuristic_dispatch_mode=heuristic_dispatch_mode,
                    dispatch_mode=dispatch_mode,
                    heuristic_net_load_subclasses=heuristic_net_load_subclasses,
                    calculate_duals_flag=calculate_duals_flag,
                )
                for subproblem_number, split_mc_draw in split_mc_draws.items()
            ]

            with TemporaryDirectory() as tempdir:
                mc_dispatch_results = parallelize(
                    _dispatch,
                    kwargs_list=subproblem_kwargs,
                    progress_bar_description="Dispatch subproblems",
                    num_processes=num_processes,
                    temp_folder=tempdir,
                    backend="threading",
                )
            subproblem_dispatch_results.extend(mc_dispatch_results)

        # Combine dispatch results
        dispatch_results = _combine_dispatch_results(subproblem_dispatch_results)

        # Store dispatch results dataframe as attribute of RecapCase object
        self.dispatch_results = dispatch_results

        # Calculate reliability / uncertainty results and save to reliability results dataframe

        # Get unserved energy and reserves
        unserved_energy_and_reserve = dispatch_results["unserved_energy_and_reserve"]

        for metric in ReliabilityMetric:
            # Calculate reliability metric and save to reliability results
            self.reliability_results.loc[perfect_capacity, metric.name] = self.calculate_reliability(
                unserved_energy_and_reserve, metric
            )
            if bootstrap:
                # Do bootstrapping to estimate confidence interval for reliability metric and save to reliability results
                bootstrap_result = self.do_annual_bootstrap(unserved_energy_and_reserve, metric)
                CI = bootstrap_result.confidence_interval
                self.reliability_results.loc[perfect_capacity, f"{metric.name}_LCB"] = CI.low
                self.reliability_results.loc[perfect_capacity, f"{metric.name}_UCB"] = CI.high

        return unserved_energy_and_reserve

    ###########################################
    ##### Tuning and Reliability Functions ####
    ###########################################

    @staticmethod
    def calculate_reliability(
        unserved_energy_and_reserve: pd.Series, metric: ReliabilityMetric = ReliabilityMetric.LOLE
    ) -> float:
        """
        Calculates reliability metrics given hourly time series of unserved energy and reserves

        Args:
            unserved_energy_and_reserve: pd.Series of hourly unserved energy and reserves
            metric: reliability metric to calculate
        """

        LOL_threshold = 0.005  # For numerical issues
        if metric == ReliabilityMetric.LOLE:
            unserved_energy_per_draw_and_date = (
                (unserved_energy_and_reserve > LOL_threshold)
                .groupby(
                    [
                        unserved_energy_and_reserve.index.get_level_values(dispatch_model._NET_LOAD_MC_DRAW_INDEX_NAME),
                        unserved_energy_and_reserve.index.get_level_values(
                            dispatch_model._NET_LOAD_TIMESTAMP_INDEX_NAME
                        ).date,
                    ]
                )
                .any()
            )
            reliability_metric_value = 365.25 * unserved_energy_per_draw_and_date.mean()
            # loss-of-load days/year
        elif metric == ReliabilityMetric.LOLH:
            reliability_metric_value = 8766 * (unserved_energy_and_reserve > LOL_threshold).mean()
            # loss-of-load hours/year
        elif metric == ReliabilityMetric.LOLP:
            reliability_metric_value = (unserved_energy_and_reserve > LOL_threshold).mean()
            # loss-of-load hours/hour
        elif metric == ReliabilityMetric.ALOLP:
            unserved_energy_per_draw_and_year = (
                (unserved_energy_and_reserve > LOL_threshold)
                .groupby(
                    [
                        unserved_energy_and_reserve.index.get_level_values(dispatch_model._NET_LOAD_MC_DRAW_INDEX_NAME),
                        unserved_energy_and_reserve.index.get_level_values(
                            dispatch_model._NET_LOAD_TIMESTAMP_INDEX_NAME
                        ).year,
                    ]
                )
                .any()
            )
            reliability_metric_value = unserved_energy_per_draw_and_year.mean()
            # loss-of-load years/year
        elif metric == ReliabilityMetric.EUE:
            reliability_metric_value = 8766 * unserved_energy_and_reserve.mean()
            # loss-of-load MWh/year
        else:
            raise ValueError(f"Invalid reliability metric specified: `{metric}`")

        return reliability_metric_value

    def do_annual_bootstrap(
        self, unserved_energy_and_reserve: pd.Series, metric: ReliabilityMetric, n_resamples: int = 1000
    ):
        def calculate_metric(unserved_energy_and_reserve):
            return self.calculate_reliability(unserved_energy_and_reserve, metric=metric)

        # Calculate metric for every year
        metric_by_year = unserved_energy_and_reserve.groupby(
            [
                unserved_energy_and_reserve.index.get_level_values(dispatch_model._NET_LOAD_MC_DRAW_INDEX_NAME),
                unserved_energy_and_reserve.index.get_level_values(dispatch_model._NET_LOAD_TIMESTAMP_INDEX_NAME).year,
            ]
        ).apply(calculate_metric)

        # TODO: if total simulation length is <= 1 year, avoid doing bootstrapping

        # Do bootstrap
        bootstrap_result = bootstrap(
            metric_by_year.values.reshape(1, -1),
            np.mean,
            n_resamples=n_resamples,
            method="percentile",
        )

        return bootstrap_result

    @staticmethod
    def bisection_method(
        reliability_func: Callable, target: float, LB: float, UB: float, bisection_xtol: float = 10, max_iter: int = 20
    ) -> float:
        """
        Root-finding method for generic reliability / dispatch as functions of perfect capacity
        NOTE: We create a custom bisect method for flexibility and to minimize the number of times the optimization function is called for model speed
        Args:
            reliability_func: system reliability metric as a function of perfect capacity
            target: system reliability metric target value
            LB: lower bound of search interval
            UB: upper bound of search interval
            capacity_tol: (x tolerance) stopping criterion for final interval width
            max_iter: maximum number of bisection method iterations to perform

        Returns:
            perfect_capacity_shortfall: root of generic reliability / dispatch function of perfect capacity
            pcap_metric_dict: Dictionary[perfect_capacity[float], reliability[float]]
        """

        # Initialize bisection method starting interval half-width
        perfect_capacity = (LB + UB) / 2
        diff = (UB - LB) / 2

        # Terminate bi-section method when capacity_tol or max_iter is reached
        iter = 0
        while diff > bisection_xtol and iter <= max_iter:
            # Update perfect capacity addition (use interval midpoint) and interval half-width
            perfect_capacity = (LB + UB) / 2
            diff = (UB - LB) / 2

            # Calculate duals flag to pass to run_dispatch
            tuned_flag = not (diff > bisection_xtol and iter <= max_iter)

            # Calculate reliability of system after perfect capacity addition
            reliability = reliability_func(perfect_capacity, tuned_flag=tuned_flag)

            logger.debug(f"Perfect Capacity Addition: {perfect_capacity:.2f} MW")
            logger.debug(f"Achieved Reliability: {reliability:.2f}")

            if reliability > target:
                # System under-reliable; increase perfect capacity
                LB = perfect_capacity

            else:
                # System over-reliable; decrease perfect capacity
                UB = perfect_capacity

            # Increment iteration
            iter += 1

        return perfect_capacity

    def _reliability_func_optimized_dispatch(self, perfect_capacity: float, tuned_flag: bool = False) -> float:
        """
        Calculate true perfect capacity shortfall using above interval bounds for ELR ELCC
        """
        unserved_energy_and_reserve = self.run_dispatch(
            perfect_capacity=perfect_capacity,
            dispatch_mode=self.case_settings.dispatch_mode,
            calculate_duals_flag=(tuned_flag and self.case_settings.print_duals),
        )
        return self.calculate_reliability(unserved_energy_and_reserve, self.case_settings.target_metric)

    def _reliability_func_no_ELRs(self, perfect_capacity: float, tuned_flag: bool = False) -> float:
        """
        Define reliability as function of perfect capacity when ELRs are effectively removed from system
        """
        unserved_energy_and_reserve = self.run_dispatch(
            perfect_capacity=perfect_capacity,
            dispatch_mode=DispatchMode.HEURISTICS_ONLY,
            heuristic_dispatch_mode=ResourceGrouping.NO_ELRS,
            calculate_duals_flag=(tuned_flag and self.case_settings.print_duals),
        )

        return self.calculate_reliability(unserved_energy_and_reserve, self.case_settings.target_metric)

    def _reliability_func_heuristic_dispatch(self, perfect_capacity: float, tuned_flag: bool = False) -> float:
        """
        Define reliability as function of perfect capacity using heuristic dispatch only
        """
        unserved_energy_and_reserve = self.run_dispatch(
            perfect_capacity=perfect_capacity,
            dispatch_mode=DispatchMode.HEURISTICS_ONLY,
            calculate_duals_flag=(tuned_flag and self.case_settings.print_duals),
        )

        return self.calculate_reliability(unserved_energy_and_reserve, self.case_settings.target_metric)

    def _calculate_perfect_capacity_shortfall_lb(self, lb: float, ub: float):
        """
        Calculate perfect capacity shortfall lower bound. Lower bound is unreliable, adding this amount of perfect capacity makes metric > target.
        Given when we assume ELCC of ELRS are 0% of nameplate capacity.
        """
        # ELCC = TRN - CapShort
        # ELR ELCC upper bound is 100%
        cap_short_lb = self.bisection_method(
            reliability_func=self._reliability_func_no_ELRs,
            target=self.case_settings.target_metric_value,
            LB=lb,
            UB=ub,
            bisection_xtol=self.case_settings.bisection_xtol,
        )
        cap_short_lb -= self.ELR_capacity

        logger.debug(f"Perfect Capacity Shortfall Lower Bound: {cap_short_lb:.2f} MW")
        return cap_short_lb

    def _calculate_perfect_capacity_shortfall_ub(self, lb: float, ub: float):
        """
        Calculate perfect capacity shortfall upper bound. Upper bound is reliable, adding this amount of perfect capacity makes metric < target.
        Given when we tune the system with heuristic dispatch only. I.e. optimization can only improve the reliability relative to heuristic dispatch.
        """
        # ELR ELCC lower bound is ELCC from heuristic dispatch
        cap_short_ub = self.bisection_method(
            reliability_func=self._reliability_func_heuristic_dispatch,
            target=self.case_settings.target_metric_value,
            LB=lb,
            UB=ub,
            bisection_xtol=self.case_settings.bisection_xtol,
        )

        logger.debug(f"Perfect Capacity Shortfall Upper Bound: {cap_short_ub:.2f} MW")
        return cap_short_ub

    @timer
    def calculate_perfect_capacity_shortfall(self):
        """
        Calculate perfect capacity shortfall of system (self.system)
        """

        # First, calculate capacity shortfall counting energy-limited resources at 0% and 100% of nameplate capacity
        # to provide upper and lower bounds for true system capacity shortfall, respectively
        logger.debug(f"ELR Capacity: {self.ELR_capacity} MW")

        # Define search interval upper/lower bounds
        UB = max(self.monte_carlo_draws["MC_draw_0"].gross_load + self.monte_carlo_draws["MC_draw_0"].reserves)
        LB = -UB

        logger.debug(f"Initial Lower Bound: {LB:.2f} MW")
        logger.debug(f"Initial Upper Bound: {UB:.2f} MW")

        # Use bisection method to get perfect capacity shortfall lower and upper bounds
        # Note: it is important that UB is calculated AFTER LB for correct set of dispatch results to be saved
        perfect_capacity_LB = self._calculate_perfect_capacity_shortfall_lb(lb=LB, ub=UB)
        perfect_capacity_UB = self._calculate_perfect_capacity_shortfall_ub(
            lb=perfect_capacity_LB, ub=perfect_capacity_LB + self.ELR_capacity
        )

        if self.case_settings.dispatch_mode == DispatchMode.HEURISTICS_ONLY:
            # Heuristic dispatch only; calculated upper bound IS the final perfect capacity shortfall
            perfect_capacity_shortfall = perfect_capacity_UB
        else:
            # Semi or fully optimized dispatch; search for perfect capacity shortfall with optimization
            perfect_capacity_shortfall = self.bisection_method(
                reliability_func=self._reliability_func_optimized_dispatch,
                target=self.case_settings.target_metric_value,
                LB=perfect_capacity_LB,
                UB=perfect_capacity_UB,
                bisection_xtol=self.case_settings.bisection_xtol,
            )

        # Save tuned dispatch results
        self.tuned_dispatch_results = self.dispatch_results

        # Save perfect capacity shortfall, perfect capacity - metric dictionary
        self.perfect_capacity_shortfall = perfect_capacity_shortfall

        logger.info("Perfect capacity shortfall: {:.2f} MW".format(self.perfect_capacity_shortfall))

        # Save to ELCC results
        self.ELCC_results.loc["base_case", "perfect_capacity_shortfall"] = self.perfect_capacity_shortfall
        self.ELCC_results.loc["base_case", "bisection_xtol"] = self.case_settings.bisection_xtol

    ##############################################
    ##### ELCC Calculation + Helper Functions ####
    ##############################################

    def create_copy(self):
        """
        Creates a copy of case with same system, settings, and MC draws
        Advantage over directly using "deepcopy" of self is that certain case-specific attributes will not persist
        """
        recap_case = RecapCase(
            copy.deepcopy(self.system),
            copy.deepcopy(self.dir_str),
            self.case_name,
            self.case_settings,
            self.gurobi_credentials,
            copy.deepcopy(self.monte_carlo_draws),
        )
        return recap_case

    def rescale_portfolio(self, portfolio_vector: pd.Series, incremental=False):
        """
        Re-scales portfolio in self.system and in all MC draws in self.monte_carlo_draws (including profiles)
        Args:
            portfolio_vector: pd.Series with resource names in the index and (incremental) nameplate capacities in values
            incremental: boolean indicator for whether nameplate capacities are incremental or absolute
            [Arizona Solar | 1000]
            [CA_Wind_for_CA | 500]
        """
        # Check for "blanks" / NaNs / None in portfolio vector; remove those entries from portfolio vector
        portfolio_vector = portfolio_vector.dropna()

        for MC_draw in self.monte_carlo_draws:
            logger.info(f"Rescaling resources in Monte Carlo draw {MC_draw}")
            # Rescale resources in each Monte Carlo draw
            self.monte_carlo_draws[MC_draw].rescale(portfolio_vector, incremental=incremental)

        # Update system
        self.system = self.monte_carlo_draws[MC_draw].system

    @timer
    def calculate_total_resource_need(self):
        """
        Calculate total resource need (TRN) by creating a system with zero installed planned capacity for all resource
        and calculating the  empty system perfect capacity shortfall
        """

        # Create RECAP case with zero planned installed capacity in model year for all resources
        recap_case_empty_portfolio = self.create_copy()
        portfolio_vector = pd.Series(index=self.system.resources.keys(), data=np.zeros(len(self.system.resources)))
        recap_case_empty_portfolio.rescale_portfolio(portfolio_vector)

        # Calculate capacity shortfall of empty portfolio
        recap_case_empty_portfolio.calculate_perfect_capacity_shortfall()

        # Save to ELCC results
        self.ELCC_results.loc["empty_case", "perfect_capacity_shortfall"] = (
            recap_case_empty_portfolio.perfect_capacity_shortfall
        )
        for resource_name in portfolio_vector.index:
            self.ELCC_results.loc["empty_case", resource_name] = portfolio_vector.loc[resource_name]
        self.ELCC_results.loc[f"empty_case", "incremental"] = False
        self.ELCC_results.loc["empty_case", "bisection_xtol"] = recap_case_empty_portfolio.case_settings.bisection_xtol

        # Save empty portfolio capacity shortfall
        self.total_resource_need = recap_case_empty_portfolio.perfect_capacity_shortfall
        del recap_case_empty_portfolio  # Save memory

    @timer
    def calculate_ELCC_points(
        self,
        ELCC_points_matrix: pd.DataFrame,
        ELCC_surface_name: str = "ELCC_surface",
    ) -> pd.DataFrame:
        """
        Iteratively calculates ELCCs of portfolio vectors defined in rows of ELCC_points_matrix

        Args:
            ELCC_points_matrix: DataFrame with resource name columns and rows of portfolio nameplate vectors

        Returns:
            ELCC_results: DataFrame similar to ELCC_points_matrix but appended with ELCC results columns

        """

        # Get incremental flags for surface points and remove incremental flag column
        incremental = ELCC_points_matrix["incremental"]
        ELCC_points_matrix.drop(columns=["incremental"], inplace=True)

        # Initialize results dataframe
        ELCC_results = ELCC_points_matrix.copy()

        # Loop through ELCC surface points and calculate portfolio capacity shortfall
        for k, portfolio_vector in ELCC_points_matrix.iterrows():
            # Create copy of base case for ELCC calculation
            ELCC_case = self.create_copy()
            # Re-scale resource profiles and resource capacity in ELCC case
            ELCC_case.rescale_portfolio(portfolio_vector, incremental=incremental.loc[k])
            # Calculate ELCC case perfect capacity shortfall
            ELCC_case.calculate_perfect_capacity_shortfall()
            # Save ELCC case perfect capacity shortfall
            ELCC_results.loc[k, "ELCC_case_perfect_capacity_shortfall_MW"] = ELCC_case.perfect_capacity_shortfall

            # Save out intermediate ELCC results to temporary results file
            output_dir = self.dir_str.recap_output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            ELCC_results.to_csv(output_dir / "temp_ELCC_results.csv")

            # Save out ELCC case results to sub-directory
            ELCC_case.dir_str.recap_output_dir = (
                self.dir_str.recap_output_dir / f"{ELCC_surface_name}_cases" / f"point_{k}"
            )
            ELCC_case.dir_str.recap_output_dir.mkdir(parents=True, exist_ok=True)
            portfolio_vector.to_csv(ELCC_case.dir_str.recap_output_dir / "portfolio_vector.csv")
            ELCC_case.report_results()

            # Save to ELCC results
            self.ELCC_results.loc[
                f"{ELCC_surface_name}_point_{k}", "perfect_capacity_shortfall"
            ] = ELCC_case.perfect_capacity_shortfall
            for resource_name in portfolio_vector.index:
                self.ELCC_results.loc[f"{ELCC_surface_name}_point_{k}", resource_name] = portfolio_vector.loc[
                    resource_name
                ]
            self.ELCC_results.loc[f"{ELCC_surface_name}_point_{k}", "incremental"] = incremental.loc[k]
            self.ELCC_results.loc[
                f"{ELCC_surface_name}_point_{k}", "bisection_xtol"
            ] = ELCC_case.case_settings.bisection_xtol

            del ELCC_case  # Save memory
            gc.collect()

        return ELCC_results
