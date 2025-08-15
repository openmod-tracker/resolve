import inspect
import pathlib
import sys
from json import dumps
from typing import Optional

import pandas as pd
from loguru import logger
from pydantic import Field
from tqdm import tqdm

from new_modeling_toolkit.core import component
from new_modeling_toolkit.core import linkage
from new_modeling_toolkit.core import three_way_linkage
from new_modeling_toolkit.core.temporal import timeseries as ts
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.core.utils.util import DirStructure
from new_modeling_toolkit.system import asset
from new_modeling_toolkit.system import outage_distribution
from new_modeling_toolkit.system.electric import load_component
from new_modeling_toolkit.system.electric import reserve
from new_modeling_toolkit.system.electric import resource_group
from new_modeling_toolkit.system.electric import zone
from new_modeling_toolkit.system.fuel.candidate_fuel import CandidateFuel
from new_modeling_toolkit.system.fuel.final_fuel import FinalFuel
from new_modeling_toolkit.system.pollutant import Pollutant
from new_modeling_toolkit.system.sector import Sector


# Mapping between `System` attribute name and component class to construct.


NON_COMPONENT_FIELDS = [
    "attr_path",
    "dir_str",
    "linkages",
    "name",
    "non_modeled_customer_costs",
    "non_modeled_distribution_costs",
    "non_modeled_generation_costs",
    "non_modeled_other_costs",
    "non_modeled_transmission_costs",
    "scenarios",
    "three_way_linkages",
    "year_end",
    "year_start",
    "base_year",
]

# Create a list of linkages to construct
LINKAGE_TYPES = [
    cls_obj
    for cls_name, cls_obj in inspect.getmembers(sys.modules["new_modeling_toolkit.core.linkage"])
    if inspect.isclass(cls_obj)
]

THREE_WAY_LINKAGE_TYPES = [
    cls_obj
    for cls_name, cls_obj in inspect.getmembers(sys.modules["new_modeling_toolkit.core.three_way_linkage"])
    if inspect.isclass(cls_obj)
]


class System(component.Component):
    """Initializes Component and Linkage instances."""

    ####################
    # FIELDS FROM FILE #
    ####################
    # TODO: Tradeoff between using a Timeseries, which will be read automatically, and being able to constrain values.
    #  Think if there's a way to constrain data types--maybe only be subclassing timeseries?
    #  A silly way to do this would be to read them in as Timeseries but then use validators to convert them to dicts/lists

    """
    TODO (5/3):
    1. Make a list of timestamps we want to model (will eventually be more dynamic)
    2. Write methods for Timeseries that return a "view" of the timeseries that matches the timestamps we want to model


    1. Subclasses
    2. Should subclasses have different input folders?

    """
    dir_str: DirStructure

    ##############
    # Components #
    ##############
    # fmt: off
    # TODO 2023-05-05: Probably can come up with a default data_filepath to reduce repeating name of component (e.g., data_filepath="assets")
    # Asset child classes
    generic_assets: dict[str, asset.Asset] = Field({}, data_filepath="assets")
    generic_resources: dict[str, electric.resources.generic.GenericResource] = Field({}, data_filepath="resources/generic")
    flex_load_resources: dict[str, electric.resources.flex_load.FlexLoadResource] = Field({}, data_filepath="resources/flexible_load")
    hydro_resources: dict[str, electric.resources.hydro.HydroResource] = Field({}, data_filepath="resources/hydro")
    shed_dr_resources: dict[str, electric.resources.shed_dr.ShedDrResource] = Field({}, data_filepath="resources/shed_dr")
    storage_resources: dict[str, electric.resources.storage.StorageResource] = Field({}, data_filepath="resources/storage")
    hybrid_storage_resources: dict[str, electric.resources.hybrid.HybridStorageResource] = Field({}, data_filepath="resources/hybrid_storage")
    thermal_resources: dict[str, electric.resources.thermal.ThermalResource] = Field({}, data_filepath="resources/thermal")
    thermal_uc_resources: dict[str, electric.resources.thermal.ThermalUnitCommitmentResource] = Field({}, data_filepath="resources/thermal")
    tx_paths: dict[str, electric.tx_path.TxPath] = Field({}, data_filepath="tx_paths")
    variable_resources: dict[str, electric.resources.variable.VariableResource] = Field({},data_filepath="resources/variable")
    hybrid_variable_resources: dict[str, electric.resources.hybrid.HybridVariableResource] = Field({}, data_filepath="resources/hybrid_variable")


    # Other component classes
    final_fuels: dict[str, FinalFuel] = Field({}, data_filepath="final_fuels")
    candidate_fuels: dict[str, CandidateFuel] = Field({}, data_filepath="candidate_fuels")
    pollutants: dict[str, Pollutant] = Field({}, data_filepath="pollutants")
    loads: dict[str, load_component.Load] = Field({}, data_filepath="loads")
    outage_distributions: dict[str, outage_distribution.OutageDistribution] = Field({}, data_filepath="outage_distributions")
    reserves: dict[str, reserve.Reserve] = Field({}, data_filepath="reserves")
    resource_groups: dict[str, resource_group.ResourceGroup] = Field({}, data_filepath="resource_groups")
    sectors: dict[str, Sector] = Field({}, data_filepath="sectors")
    zones: dict[str, zone.Zone] = Field({}, data_filepath="zones")
    # fmt: on

    ############
    # Linkages #
    ############
    linkages: dict[str, list[linkage.Linkage]] = {}
    three_way_linkages: dict[str, list[three_way_linkage.ThreeWayLinkage]] = {}

    ##########
    # FIELDS #
    ##########
    # TODO 2023-05-05: year_start and year_end ideally would come from New
    year_start: Optional[int] = None
    year_end: Optional[int] = None

    non_modeled_generation_costs: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="YS", up_method="interpolate", down_method="annual"
    )
    non_modeled_transmission_costs: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="YS", up_method="interpolate", down_method="annual"
    )
    non_modeled_distribution_costs: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="YS", up_method="interpolate", down_method="annual"
    )
    non_modeled_customer_costs: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="YS", up_method="interpolate", down_method="annual"
    )
    non_modeled_other_costs: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="YS", up_method="interpolate", down_method="annual"
    )

    scenarios: list = []

    @property
    def assets(self):
        """Superset of all `Asset` child classes."""
        return self.resources | self.tx_paths | self.generic_assets

    @property
    def resources(self):
        """Superset of all `Resource` child classes."""
        return (
            self.generic_resources
            | self.flex_load_resources
            | self.hydro_resources
            | self.shed_dr_resources
            | self.storage_resources
            | self.hybrid_storage_resources
            | self.thermal_resources
            | self.thermal_uc_resources  # TODO 2023-05-05: Do we need this one or its sub-classes?
            | self.variable_resources
            | self.hybrid_variable_resources
        )

    @property
    def _component_fields(self):
        """Return list of component FIELDS in `System` (by manually excluding non-`Component` attributes)."""
        return {name: field for name, field in self.model_fields.items() if name not in NON_COMPONENT_FIELDS}

    @property
    def components(self):
        """Return list of component ATTRIBUTES and "virtual" components (i.e., properties that are the union of other components)."""
        return (
            {name: getattr(self, name) for name, field in self.model_fields.items() if name not in NON_COMPONENT_FIELDS}
            | {"assets": self.assets}
            | {"resources": self.resources}
        )

    def __init__(self, **data):
        """
        Initializes a electrical system based on csv inputs. The sequence of initialization can be found in the
        comments of the system class

        Args:
            graph_dict: the dictionary that determines the linkage between different components of the system
        """
        super().__init__(**data)

        ###########################################
        # READ IN COMPONENTS & LINKAGES FROM FILE #
        ###########################################
        self._construct_components()
        self.linkages = self._construct_linkages(
            linkage_subclasses_to_load=LINKAGE_TYPES, linkage_type="linkages", linkage_cls=linkage.Linkage
        )
        self.three_way_linkages = self._construct_linkages(
            linkage_subclasses_to_load=THREE_WAY_LINKAGE_TYPES,
            linkage_type="three_way_linkages",
            linkage_cls=three_way_linkage.ThreeWayLinkage,
        )

        ##########################
        # ADDITIONAL VALIDATIONS #
        ##########################
        logger.info("Revalidating components...")
        for components in self.components.values():
            for instance in components.values():
                instance.revalidate()
        logger.info("Revalidation complete")

    @timer
    def _construct_components(self):
        components_to_load = pd.read_csv(self.dir_str.data_interim_dir / "systems" / self.name / "components.csv")

        components_to_load = components_to_load.groupby("component")

        # Populate component attributes with data from instance CSV files
        # Get field class by introspecting the field info
        for field_name, field_info in self._component_fields.items():
            field_type = self.get_field_type(field_info=field_info)[-1]
            field_data_filepath = self.model_fields[field_name].json_schema_extra["data_filepath"]

            self.update_component_attrs(
                field_name=field_name,
                field_type=field_type,
                field_data_filepath=field_data_filepath,
                components_to_load=components_to_load,
            )

    def update_component_attrs(
        self, *, field_name: str, field_type: "Component", field_data_filepath: str, components_to_load: pd.DataFrame
    ):
        """Load all components of a certain type listed in `components_to_load`."""
        if field_type.__name__ not in components_to_load.groups:
            logger.debug(f"Component type {field_type.__name__} not loaded because component type not recognized")
            # Escape this method
            return
        for component_name in tqdm(
            components_to_load.get_group(field_type.__name__)["instance"],
            desc=f"Loading {field_type.__name__}:".rjust(32),
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
        ):
            # by convention, the dir name is the name of the component dictionary
            # the following line is just a way to get an attribute's (component_dict) name
            vintages = field_type.from_csv(
                self.dir_str.data_interim_dir / field_data_filepath / f"{component_name}.csv", scenarios=self.scenarios
            )
            # TODO 2023-05-28: Sort of weird that this method directly works on the attr instead of returning a dict
            getattr(self, field_name).update(vintages)

    @timer
    def _construct_linkages(self, *, linkage_subclasses_to_load: list, linkage_type: str, linkage_cls):
        """This function now can be used to initialize both two- and three-way linkages."""
        if (self.dir_str.data_interim_dir / "systems" / self.name / f"{linkage_type}.csv").exists():
            linkages_to_load = pd.read_csv(
                self.dir_str.data_interim_dir / "systems" / self.name / f"{linkage_type}.csv"
            )
            linkages_to_load = self._get_scenario_linkages(linkages=linkages_to_load, scenarios=self.scenarios)
            linkages_to_load = linkages_to_load.groupby("linkage")

            for linkage_class in linkage_subclasses_to_load:
                # If no linkages of this class type specified by user, skip this iteration of the loop (`continue` keyword)
                # TODO: update debug statement or load all linkage types at once (two and threeway linkage)
                if linkage_class.__name__ not in linkages_to_load.groups:
                    logger.debug(
                        f"Linkage type {linkage_class.__name__} not loaded because linkage type not recognized"
                    )
                else:
                    if linkage_cls.__name__ == "Linkage":
                        # Assume the data/interim folder has the same name as the file that lists the linkages
                        linkage_class.from_dir(
                            dir_path=self.dir_str.data_interim_dir / f"{linkage_type}",
                            linkages_df=linkages_to_load.get_group(linkage_class.__name__),
                            components_dict=self.components,
                            scenarios=self.scenarios,
                            linkages_csv_path=self.dir_str.data_interim_dir
                            / "systems"
                            / self.name
                            / f"{linkage_type}.csv",
                        )
                    elif linkage_cls.__name__ == "ThreeWayLinkage":
                        linkage_class.from_dir(
                            dir_path=self.dir_str.data_interim_dir / f"{linkage_type}",
                            linkage_pairs=linkages_to_load.get_group(linkage_class.__name__),
                            components_dict=self.components,
                            scenarios=self.scenarios,
                        )
            # Announce linkages
            linkage_cls.announce_linkage_to_instances()

            return linkage_cls._instances
        else:
            return {}

    def _get_scenario_linkages(self, *, linkages: pd.DataFrame, scenarios: list):
        """Filter for the highest priority data based on scenario tags."""

        # Create/fill a dummy (base) scenario tag that has the lowest priority order
        if "scenario" not in linkages.columns:
            linkages["scenario"] = "__base__"
        # Create a dummy (base) scenario tag that has the lowest priority order
        linkages["scenario"] = linkages["scenario"].fillna("__base__")

        # Create a categorical data type in the order of the scenario priority order (lowest to highest)
        linkages["scenario"] = pd.Categorical(linkages["scenario"], ["__base__"] + scenarios)

        # Drop any scenarios that weren't provided in the scenario list (or the default `__base__` tag)
        len_linkages_unfiltered = len(linkages)
        linkages = linkages.sort_values("scenario").dropna(subset="scenario")

        # Log error if scenarios filtered out all data
        if len_linkages_unfiltered != 0 and len(linkages) == 0:
            err = f"No linkages for active scenario(s): {scenarios}. "
            logger.error(err)

        # Keep only highest priority scenario data
        linkages = linkages.groupby([x for x in linkages.columns if x != "scenario"]).last().reset_index()

        return linkages

    @timer
    def resample_ts_attributes(
        self,
        modeled_years: tuple[int, int],
        weather_years: tuple[int, int],
        resample_weather_year_attributes=True,
        resample_non_weather_year_attributes=True,
    ):
        """Interpolate/extrapolate timeseries attributes so that they're all defined for the range of modeled years."""
        # Dictionary of objects & their attributes that were extrapolated (i.e., start/end dates too short)
        extrapolated = {}
        logger.info("Resampling timeseries attributes...")
        for field_name, components in self.components.items():
            logger.debug(f"Resampling timeseries for {field_name}")
            for instance in components.values():
                extrapolated[instance.name] = instance.resample_ts_attributes(
                    modeled_years,
                    weather_years,
                    resample_weather_year_attributes=resample_weather_year_attributes,
                    resample_non_weather_year_attributes=resample_non_weather_year_attributes,
                )

        # Load treated differently: forecast future load
        for instance in self.loads.keys():
            self.loads[instance].forecast_load(modeled_years, weather_years)

        # loads to policies
        for inst in getattr(self, "policies", {}).keys():
            self.policies[inst].update_targets_from_loads()

        # ELCC treated differently
        for inst in getattr(self, "elcc_surfaces", {}).keys():
            for facet in self.elcc_surfaces[inst].facets:
                extrapolated[inst] = (
                    self.elcc_surfaces[inst]
                    .facets[facet]
                    .resample_ts_attributes(
                        modeled_years,
                        weather_years,
                        resample_weather_year_attributes=resample_weather_year_attributes,
                        resample_non_weather_year_attributes=resample_non_weather_year_attributes,
                    )
                )

        # Regularize timeseries attributes, if any, in linkages (same as components above)
        for linkage_class in self.linkages:
            for linkage_inst in self.linkages[linkage_class]:
                extrapolated[", ".join(linkage_inst.name)] = linkage_inst.resample_ts_attributes(
                    modeled_years,
                    weather_years,
                    resample_weather_year_attributes=resample_weather_year_attributes,
                    resample_non_weather_year_attributes=resample_non_weather_year_attributes,
                )

        # Regularize timeseries attributes, if any, in three way linkages (same as components above)
        for three_way_linkage_class in self.three_way_linkages:
            for three_way_linkage_inst in self.three_way_linkages[three_way_linkage_class]:
                extrapolated[three_way_linkage_inst.name] = three_way_linkage_inst.resample_ts_attributes(
                    modeled_years,
                    weather_years,
                    resample_weather_year_attributes=resample_weather_year_attributes,
                    resample_non_weather_year_attributes=resample_non_weather_year_attributes,
                )

        if extrapolated := {str(k): list(v) for k, v in extrapolated.items() if v is not None}:
            logger.debug(
                f"The following timeseries attributes were extrapolated to cover model years: \n{dumps(extrapolated, indent=4)}"
            )

    @classmethod
    def from_csv(cls, filename: pathlib.Path, scenarios: list = [], data: dict = {}):
        input_df = pd.read_csv(filename).sort_index()

        scalar_attrs = cls._parse_scalar_attributes(filename=filename, input_df=input_df, scenarios=scenarios)
        ts_attrs = cls._parse_timeseries_attributes(filename=filename, input_df=input_df, scenarios=scenarios)
        nodate_ts_attrs = cls._parse_nodate_timeseries_attributes(
            filename=filename, input_df=input_df, scenarios=scenarios
        )
        attrs = {
            **{"name": filename.parent.stem, "scenarios": scenarios},
            **scalar_attrs,
            **ts_attrs,
            **nodate_ts_attrs,
            **data,
        }
        return attrs["name"], cls(**attrs)


if __name__ == "__main__":
    system = System(name="test", scenarios=["B"], dir_str=DirStructure(data_folder="data-test"))
    print(system)
