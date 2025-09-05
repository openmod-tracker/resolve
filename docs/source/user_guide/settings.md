## Important Settings in Scenario Tool

### Financial Settings

RESOLVE optimizes net present value for the entire planning horizon. To do so, it relies on discount factors for selected modeled years. The model endogenously calculates the annual discount factors to use for each modeled year based on four pieces 
of information:
1. **Cost dollar year:** The dollar year that costs are input in and should be reported out in. In general, `RESOLVE` is designed 
to be in real dollars for a specific dollar year.
2. **Modeled years:** Which modeled years to include in the `RESOLVE` case.
3. **End-effect years:** The number of years to extrapolate out the last modeled year. In other words, considering 20 years 
of end effects after 2045 would mean that the last model year's annual discount factor would represent the discounted cost 
of 2045-2064, assuming that the 2045 costs are representative of a steady-state future for all end effect years.
4. **Annual discount rate:** Real discount rate in each year

The schematic below shows how net present value is being calculated based on costs in selected modeled years:
![Example of RESOLVE Modeling Years and Financing Timeline](_images/Modeling_Year.jpg)

### Timeseries Data

Timeseries data representing profiles for loads and resources are stored in separate CSV files under the `./data/profiles/` subfolder to keep the Scenario 
Tool spreadsheet filesize manageable; however, the Scenario Tool contains the profile path associated with each resource parameter which point the code to the relevant CSV file. These CSVs must have the following format:

| timestamp                    | value             |
|------------------------------|-------------------|
| [timestamp (hour beginning)] | [attribute value] |
| ...                          | ...               |

Note that for any profile relying on historical data must have at least data for the representative day weather years. The profiles get rescaled in the code to ensure that representative day-based capacity factors for solar and wind match the historical timeseries average capacity factor value. The rescaled profiles are saved in the `./data/processed/resolve/rescaled_profiles`.  

### Representative Period Settings

The Timeseries Clusters worksheet in Scenario Tool contains the representative days and chronological periods covering selected weather years (typically 23 weather years in the CPUC IRP model are represented in 36 days). Note that RESOLVE representative days are created using RESOLVE Day Sampling Script provided in the RESOLVE package. 
In order to model inter-day sharing for storage to shift energy between days in a single modeled year, RESOLVE relies on chronological timeseries mapping with the representative days. 

![Scenario Tool Timeseries Clusters Worksheet](_images/Rep_days.png)

### Temporal Settings
In Resolve Case Setup worksheet, you have the option to choose a set of representative day scenario (if more than one available) to include in your case run. Additionally, you can choose the years to model in RESOLVE, years to model with inter-day sharing. 

In order to include inter-day sharing, you need to choose "inter-period sharing" for dispatch window behavior; otherwise, choose "loopback" to exclude inter-day sharing. Additionally, choose the years that you wish to include inter-day sharing for. And lastly, choose a single weather year from the weather years list from the last table in RESOLVE Case Setup worksheet (all years have the same chronological mapping with representative days, so any single weather year can be chosen in this case).

![Scenario Tool Timeseries Clusters Worksheet](_images/Temporal_Settings.png)

### Scenario tagging functionality





