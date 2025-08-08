# RESOLVE Results Viewing

Note that for any pre-existing `RESOLVE` case results included in the `RESOLVE` package, there is no need to take these steps. You are advised to directly open the `Saved Case Results Viewer` folder to find the pre-loaded results viewer for each case.

Once RESOLVE is done solving your case, it will save a series of CSVs that summarize the portfolio investment and operational decisions. These files are stored in the run’s report folder, which is found in  
`./reports/resolve/[case name]/[timestamp]/`

Once you have a RESOLVE case run optimization completed, the output directory structure will look something similar to this after all case results are saved:

![Example of a RESOLVE case results](_images/b0ea53918fd18611287549e2c85b9431.png)

The output file size will also be dependent on the case settings and components. For example, including hourly results will increase the file size and number of outputs. As there is a large amount of information, it is not convenient to open each file and explore; thus, having a centralized result viewer becomes very important. The RESOLVE package comes with a readily results viewer template to help modelers retrieve and summarize detailed case results. The results viewer is an excel based workbook that has inbuilt formulas and methods that digests the individual CSVs and provides comprehensive information on the entire system. Users can load specific case results into Results Viewer using an interactive Jupyter notebook script to aggregate annual results. An additional script is made available to aggregate hourly results which is totally optional and recommended if hourly data review is needed. Additionally, all publicly shared cases have pre-loaded case results that can be found in the `Saved Case Results Viewer` folder.

## Using the interactive Jupyter Notebook to Load Annual Case Results

Users can find a Jupyter notebook script called “RV_script.ipynb” within the "Notebooks" directory of the repository. It is recommended that users use a Python version greater than v3.7.0 to use this. Once the script is opened using Jupyter Notebook or Jupyterhub, users can work with this without having to reactivate the environment or related dependencies. Note that the script is designed to work independently.

```Tip
To open Jupyter Notrebook, simply run Jupyter Notebook from Pycharm terminal and navigate to the `notebooks` folder to find the script.
```

As you run the first couple blocks of code, you will be asked to enter the file path of results folder and the Results Viewer Workbook itself, which will be a part of the package as well. Please note that any new case that you run, you should make sure the results are saved in the `results` folder. This is what that selection process should look like:

![Path selection for RESOLVE Results Viewer template and Case results folder](_images/2c6f3edd7cda9192962652f8a214a64f.png)

Once you have confirmed the path selections, you would need to add the subfolder names of the case (typically the name of the case and the timestamp) that you want loaded. These inputs will go here:

![Example of case results sufolder names](_images/65aa4f936e8f28af02ced7c79cae48e3.png)

Note that users have the option to select loading one case at a time or a batch of cases. It is recommended that starting out users select just the one case option. Once this is done, users can move ahead to the `generate_rv` function. If you choose to save results viewers for multiple cases at same time, you should specify `generate_multiple_rvs and comment our the single RV option. After this point, the script will start loading in the RV and the script will provide updates as follows:

![Logs shown when a case is successfully loaded in Results Viewer](_images/56291fc65b012f9da9e15988c6cb75e2.png)

After this block completes the run, the users can find the Results viewer in the specified folder. Note that this could be a computationally intense process and depending on the size of the case can take about 30 minutes to load one case.

## Optional Hourly Results Processing Jupyter Notebook

In addition to viewing annual results, one may be interested in viewing and analyzing hourly model results. This is only possible if the “report_hourly_results” argument in the attributes.csv file for the case is set to TRUE at the time of setting up the case, which tells RESOLVE whether to save hourly results to the case reports folder. If hourly results are included in the case reports folder, you can analyze them with the “RESOLVE Hourly Results Viewer.ipynb” Jupyter notebook within the “notebooks” directory of the kit repository. In this case, make sure to open Jupyter Notebook after activating your environment.

![RESOLVE dialog box where user can choose case and hourly results.](_images/8e741d826c6d418510f0afe91d88354f.png)

The hourly results viewer notebook provides a workflow for analyzing and visualizing hourly and chronological dispatch results from RESOLVE model outputs. The user is guided to select a local directory containing RESOLVE case results and a destination folder for output via interactive file choosers. It is recommended that the case results be stored on a local drive rather than cloud to increase the speed of importing the results into the notebook. If you would like to aggregate certain resources into aggregate groups for analysis, the user can optionally select an Excel Results Viewer workbook, which should contain two named ranges on the `Results Groupings` worksheet: one for assigning resources to "Build Groups" and another for defining color settings and chart ordering for those groups. This setup enables streamlined downstream plotting of dispatch data either by grouped resource types or individual resources. The user then has the option to export hourly load-resource balance results (i.e., generation, load, zonal imports and exports, and battery charge and discharge results) for each specified zone and modeled year. Moreover, one can create hourly dispatch plots for a specified zone, model year and dispatch window (a.k.a. representative period) as shown with an example below.
![Options to set for dispatch chart view](_images/f72f141262d258148e732e0ffe021e6d.png)
![Example Dispatch Chart](_images/923796f9979f67abf68108bf8a1bf5c5.png)

For hourly dispatch charts, the resource aggregation and color coding is defined in the Excel Results Viewer workbook.

![Colors and build group setup in Results Viewer](_images/a9e5f769410f85539fc45c298935ad71.png)
