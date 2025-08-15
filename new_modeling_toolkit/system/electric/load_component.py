import calendar
from typing import Optional
from typing import Union

from loguru import logger
from pydantic import Field
from pydantic import root_validator
from typing_extensions import Annotated

from new_modeling_toolkit import get_units
from new_modeling_toolkit.core import component
from new_modeling_toolkit.core import linkage
from new_modeling_toolkit.core.temporal import timeseries as ts

NUM_LEAP_YEAR_HOURS = 366 * 24
NUM_NON_LEAP_YEAR_HOURS = 365 * 24


class Load(component.Component):
    ######################
    # Boolean Attributes #
    ######################
    scale_by_capacity: bool = False
    scale_by_energy: bool = False

    #########################
    # Timeseries Attributes #
    #########################
    profile: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="H", up_method="interpolate", down_method="mean", weather_year=True
    )
    profile_model_years: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="H", up_method="interpolate", down_method="mean"
    )
    annual_peak_forecast: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="YS", up_method="interpolate", down_method="max", units=get_units("annual_peak_forecast")
    )
    annual_energy_forecast: Optional[ts.NumericTimeseries] = Field(
        None, default_freq="YS", up_method="interpolate", down_method="sum", units=get_units("annual_energy_forecast")
    )

    ######################
    # Numeric Attributes #
    ######################
    td_losses_adjustment: Annotated[float, Field(ge=0)] = Field(
        1,
        description="T&D loss adjustment to gross up to system-level loads. For example, a DER may be able to serve "
        "8% more load (i.e., 1.08) than an equivalent bulk system resource due to T&D losses. Adjustment factor is "
        "**directly multiplied** against load (as opposed to 1 / (1 + ``td_losses_adjustment``).",
    )

    ############
    # Linkages #
    ############
    zones: Optional[dict[str, linkage.Linkage]] = None
    devices: Optional[dict[str, linkage.Linkage]] = None
    energy_demand_subsectors: Optional[dict[str, linkage.Linkage]] = None

    @root_validator(skip_on_failure=True)
    @classmethod
    def validate_profile_weather_or_model_year(cls, values):
        """Ensure that users only provide one profile (either ``profile`` or ``profile_weather_years``."""
        assert (values["profile"] and not values["profile_model_years"]) or (
            not values["profile"] and values["profile_model_years"]
        ), f"For {values['name']}: Only one of either `profile` or `profile_model_years` can be defined."
        return values

    def normalize_profile(self, normalize_by):
        """Normalize profile by capacity or by energy"""

        if normalize_by == "capacity":
            logger.info("Normalizing by capacity - setting profile maximum to 1.")
            self.profile.data /= self.profile.data.max()

        elif normalize_by == "energy":
            logger.info("Normalizing by energy - setting annual profile sum to 1.")
            self.profile.data *= (
                len(self.profile.data.index.year.unique()) / self.profile.data.sum()
            )  # Sets profile sum = number of years

    def forecast_load(self, modeled_years: tuple[int, int]):
        """
        Calculate the scaling coefficient and scaling offset for the load series in order to scale them to any future
        between the first model year and last model year. The coefficient and offset is determine by the future peak
        load series and energy series, and the load scaling method defined by the user.
        """
        first_model_year, last_model_year = modeled_years

        self.model_year_profiles = {}
        for model_year in range(first_model_year, last_model_year + 1):
            to_peak = (
                self.annual_peak_forecast.data.loc[str(model_year)].values[0]
                if self.scale_by_capacity
                else self.scale_by_capacity
            )
            to_energy = (
                self.annual_energy_forecast.data.loc[str(model_year)].values[0]
                if self.scale_by_energy
                else self.scale_by_energy
            )
            leap_year = calendar.isleap(model_year)

            if self.profile:
                profile_to_scale = self.profile
            elif self.profile_model_years:
                profile_to_scale = self.profile_model_years.copy(deep=True)
                profile_to_scale.data = self.profile_model_years.data.loc[str(model_year)]

            # Scale profile & save to the ``model_year_profiles`` dictionary
            new_profile = Load.scale_load(profile_to_scale, to_peak, to_energy, self.td_losses_adjustment, leap_year)
            self.model_year_profiles.update({model_year: new_profile})

    @staticmethod
    def scale_load(
        profile: ts.NumericTimeseries,
        to_peak: Union[bool, float],
        to_energy: Union[bool, float],
        td_losses_adjustment: float,
        leap_year: bool,
    ) -> ts.NumericTimeseries:
        """Scale timeseries by energy and/or median peak.

        Scaling to energy assumes ``to_energy`` forecast value will match whether it is/is not a leap year.
        In other words, the energy forecast for a leap day includes an extra day's worth of energy.

        Args:
            profile: Hourly timeseries to be scaled
            to_peak: Median annual peak to be scaled to
            to_energy: Mean annual energy to be scaled to
            td_losses_adjustment: T&D losses adjustment (simple scalar on load profile)
            leap_year: If year being scaled to is a leap year (affecting energy scaling)

        Returns:
            new_profile: Scaled hourly timeseries
        """

        profile_median_peak = profile.data.groupby(profile.data.index.year).max().median()
        profile_mean_annual_energy = profile.data.groupby(profile.data.index.year).sum().mean()

        # calculate the average annual energy of the provided weather years
        n_hours_in_forecast = NUM_LEAP_YEAR_HOURS if leap_year else NUM_NON_LEAP_YEAR_HOURS

        if to_energy is False and to_peak is False:
            scale_multiplier = td_losses_adjustment
            scale_offset = 0
        elif to_energy is False:
            scale_multiplier = to_peak * td_losses_adjustment / profile_median_peak
            scale_offset = 0
            logger.debug(f"Scaling {profile.name} to median peak.")
        elif to_peak is False:
            scale_multiplier = to_energy * td_losses_adjustment / profile_mean_annual_energy
            scale_offset = 0
            logger.debug(f"Scaling {profile.name} to mean annual energy.")
        else:
            scale_multiplier = td_losses_adjustment * (
                (to_peak - (to_energy / n_hours_in_forecast))
                / (profile_median_peak - (profile_mean_annual_energy / n_hours_in_forecast))
            )
            scale_offset = to_peak - scale_multiplier * profile_median_peak
            logger.debug(f"Scaling {profile.name} to median peak & mean annual energy.")
            if to_peak < 0:
                logger.warning("Scaling to peak & energy with a negative peak may not work as intended.")

        new_profile = profile.copy(deep=True)
        new_profile.data = scale_multiplier * profile.data + scale_offset

        new_profile_median_peak = new_profile.data.groupby(profile.data.index.year).max().median()
        new_profile_mean_energy = new_profile.data.mean() * n_hours_in_forecast
        logger.debug("Scaled load profile median peak: {:.0f} MW".format(new_profile_median_peak))
        logger.debug("Scaled load profile mean annual energy: {:.0f} MWh".format(new_profile_mean_energy))

        return new_profile

    def get_load(self, temporal_settings, model_year, period, hour):
        """
        Based on the model year, first find the future load series belonging to that model year. And based on the
        period and hour, query the specific hourly load for that hour in the model year.
        Args:
            system: System.system. current power system
            model_year: int. model year being queried
            period:  model period being queried
            hour: model hour being queried

        Returns:  int. load for the tp under query.

        """

        return self.model_year_profiles[model_year].slice_by_timepoint(temporal_settings, model_year, period, hour)

    def revalidate(self):
        if ((self.devices is not None) or (self.energy_demand_subsectors is not None)) and (
            self.annual_energy_forecast is not None
        ):
            raise ValueError(
                "Error in load component {}: annual energy forecast can not be specified if load component is linked "
                "to a device or energy demand subsector".format(self.name)
            )
