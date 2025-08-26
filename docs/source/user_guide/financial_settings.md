## Financial Settings

The model will now endogenously calculate the annual discount factors to use for each modeled year based on four pieces 
of information:
1. **Cost dollar year:** The dollar year that costs are input in and should be reported out in. In general, `RESOLVE` is designed 
to be in real dollars for a specific dollar year.
2. **Modeled years:** Which modeled years to include in the `RESOLVE` case.
3. **End effect years:** The number of years to extrapolate out the last modeled year. In other words, considering 20 years 
of end effects after 2045 would mean that the last model year's annual discount factor would represent the discounted cost 
of 2045-2064, assuming that the 2045 costs are representative of a steady-state future for all end effect years.
4. **Annual discount rate:** Real discount rate in each year


**Add picture: discount rate and modeled years
