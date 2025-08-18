import enum
from pathlib import Path

import pandas as pd
import pydantic

from new_modeling_toolkit.core.custom_model import CustomModel


@enum.unique
class DispatchMode(enum.Enum):
    # Heuristics only: run dispatch with only heuristics; skip all optimization
    HEURISTICS_ONLY = "heuristics_only"
    # Semi-optimized: run dispatch with optimization, but compress over full set of heuristic dispatch results
    # I.e., net load = load - thermal - renewables - heuristics of all energy-limited resources
    SEMI_OPTIMIZED = "semi_optimized"
    # Fully-optimized: run dispatch with optimization, but compress only over net load = load - thermal - renewables)
    FULLY_OPTIMIZED = "fully_optimized"


@enum.unique
class DispatchObjective(enum.Enum):
    EUE = "EUE"
    LOLE = "LOLE"
    EUE_and_100xLOLE = "EUE_and_100xLOLE"
    LOLH = "LOLH"


@enum.unique
class ReliabilityMetric(enum.Enum):
    EUE = "EUE"
    LOLE = "LOLE"
    LOLH = "LOLH"
    LOLP = "LOLP"
    ALOLP = "ALOLP"


@enum.unique
class ResourceGrouping(enum.Enum):
    # Should make the use case of these resource groupings more clear
    DEFAULT = "default"  # Call this DEFAULT_DISPATCH_ORDER
    NO_ELRS = "no_ELRs"  # Call this NON_ENERGY_LIMITED
    ELRS = "ELRs"  # Call this ENERGY_LIMITED
    PROBABILISTIC_UPSAMPLING = "probabilistic_upsampling"
    DETERMINISTIC_UPSAMPLING = "deterministic_upsampling"

    @property
    def resource_subclasses(self):
        subclass_mapping = {
            "no_ELRs": ["thermal_resources", "variable_resources"],
            "default": [
                "thermal_resources",
                "variable_resources",
                "generic_resources",
                "hydro_resources",
                "hybrid_variable_resources",
                "hybrid_storage_resources",
                "storage_resources",
                "flex_load_resources",
                "shed_dr_resources",
            ],
            "ELRs": [
                "generic_resources",
                "hydro_resources",
                "hybrid_variable_resources",
                "hybrid_storage_resources",
                "storage_resources",
                "flex_load_resources",
                "shed_dr_resources",
            ],
            "deterministic_upsampling": [
                "thermal_resources",
                "storage_resources",
                "hybrid_storage_resources",
                "flex_load_resources",
                "shed_dr_resources",
            ],
            "probabilistic_upsampling": [
                "variable_resources",
                "hybrid_variable_resources",
                "hydro_resources",
            ],
        }

        return subclass_mapping[self.value]


class RecapCaseSettings(CustomModel):
    """Pydantic model for storing RECAP case settings."""

    # TODO: add validators for various fields
    system_name: str
    zone_to_analyze: str
    analysis_year: int
    number_of_monte_carlo_draws: int
    output_dispatch_results: bool
    incremental_pcap: float
    print_raw_results: bool
    print_duals: bool
    dispatch_mode: DispatchMode
    dispatch_objective: DispatchObjective
    target_metric: ReliabilityMetric
    target_metric_value: float
    draw_settings: str
    day_window_variable_draws: int
    variable_draws_probability_function: str
    maximum_subproblem_length: int
    bisection_xtol: int

    # Calculation settings
    calculate_reliability: bool
    calculate_reliability_w_incremental_pcap: bool
    calculate_perfect_capacity_shortfall: bool
    calculate_total_resource_need: bool
    calculate_portfolio_ELCC: bool
    calculate_marginal_ELCC: bool
    calculate_incremental_last_in_ELCC: bool
    calculate_decremental_last_in_ELCC: bool
    calculate_ELCC_surface: bool

    @pydantic.validator(
        "output_dispatch_results",
        "calculate_reliability",
        "calculate_reliability_w_incremental_pcap",
        "calculate_portfolio_ELCC",
        "calculate_marginal_ELCC",
        "calculate_incremental_last_in_ELCC",
        "calculate_decremental_last_in_ELCC",
        "calculate_ELCC_surface",
        pre=True,
    )
    def str2bool(cls, v):
        if type(v) == bool:
            bool_value = v
        else:
            bool_value = v.lower() in ("yes", "true", "t", "1")
        return bool_value

    @pydantic.validator(
        "analysis_year",
        "number_of_monte_carlo_draws",
        "day_window_variable_draws",
        pre=True,
    )
    def str2int(cls, v):
        return int(float(v))

    @pydantic.validator(
        "incremental_pcap",
        pre=True,
    )
    def str2float(cls, v):
        return float(v)

    @classmethod
    def from_csv(cls, path: Path):
        """Loads case settings from a CSV file."""
        case_name = path.parent.name
        init_kwargs = pd.read_csv(path, index_col=0).squeeze().to_dict()
        settings = cls(name=case_name, **init_kwargs)

        return settings
