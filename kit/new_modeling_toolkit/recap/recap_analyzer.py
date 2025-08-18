import glob
import io
import os
import pathlib

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import clear_output
from IPython.display import display
from ipywidgets import fixed
from ipywidgets import interact
from ipywidgets import interactive
from plotly.subplots import make_subplots
from scipy.stats import bootstrap

import new_modeling_toolkit.recap.dispatch_model as dispatch_model
from new_modeling_toolkit.recap.recap_case import RecapCase
from new_modeling_toolkit.recap.recap_case_settings import ReliabilityMetric


color_map = {
    "Thermal": "rgba(110,110,110,1)",
    "Imports": "rgba(0,138,55,1)",
    "Nuclear": "rgba(196,189,151,1)",
    "Other": "rgba(196,189,151,1)",
    "Firm Resources": "#B4B4B4",
    "Solar": "rgba(255,175,0,1)",
    "LBW": "rgba(21,181,249,1)",
    "OSW": "rgba(176,230,253,1)",
    "Storage": "#7030A0",
    "LDES": "#BD92DE",
    "Hydro": "rgba(3,78,110,1)",
    "DR": "rgba(255,201,187,1)",
    "FlexLoad": "#0dd106",
    "Perfect Capacity": "rgba(0,0,0,1)",
    "Unserved Energy and Reserve": "rgba(175,34,0,1)",
    "Gross Load": "black",
    "Gross Load + Reserves": "black",
    "Variable plus Hydro": "black",
    "summer": "#FF967D",
    "winter": "#056F9F",
}


def _load_disp_result_parquet(path):
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df = df.reset_index().set_index(["MC_draw", "timestamp"])
    else:
        df = pd.DataFrame()
    return df


def _load_general_result_csv(path):
    df = pd.read_csv(path, index_col=0) if os.path.exists(path) else pd.DataFrame()
    return df


def _get_month_hour_data(disp_data):
    """
    Calculate month-hour loss-of-load statistics based on dispatch results, tuned or untuned.
    """
    disp_data = disp_data.reset_index()
    disp_data.index = pd.to_datetime(disp_data["timestamp"])
    disp_data["year"] = disp_data.index.year
    disp_data["month"] = disp_data.index.month
    disp_data["hour"] = disp_data.index.hour
    month_hour_table = (
        disp_data.groupby(["month", "hour"])["unserved_energy_and_reserve"]
        .agg([np.sum, lambda x: (x > 0).sum(), np.size])
        .rename(columns={"sum": "Total unserved energy", "<lambda_0>": "LOL hours count", "size": "Total hours"})
    )
    month_hour_table["EUE"] = month_hour_table["Total unserved energy"] / disp_data["year"].nunique()
    month_hour_table["LOLP"] = month_hour_table["LOL hours count"] / month_hour_table["Total hours"]
    month_hour_table["LOLH"] = month_hour_table["LOL hours count"] / disp_data["year"].nunique()
    month_hour_table.drop(["Total unserved energy", "LOL hours count", "Total hours"], inplace=True, axis=1)
    month_hour_table.reset_index(inplace=True)

    return month_hour_table


class InputChecks:
    def __init__(
        self,
        dir_str,
    ):
        self.dir_str = dir_str
        self.inputs_dir = dir_str.data_settings_dir.joinpath("recap")

    def case_selection(self):
        """
        Select cases for input comparison.
        """

        case_list = [pathlib.Path(d).name for d in glob.glob(str(self.inputs_dir / "*")) if pathlib.Path(d).is_dir()]

        case_selector = widgets.SelectMultiple(
            options=case_list,
            style={"description_width": "initial"},
            layout={"width": "50%"},
            rows=min(len(case_list), 10),
        )

        butt = widgets.Button(description="Save Case Selection. ")
        outt = widgets.Output()

        def _collect_case_name(a):
            with outt:
                clear_output()
                self.case_to_compare = list(case_selector.value)

        butt.on_click(_collect_case_name)
        display(widgets.VBox([case_selector, butt, outt]))

    def case_selection_from_local(self):
        """
        Another option to load cases - specify case name in a spreadsheet and load from local.
        """

        w_case_to_compare = widgets.FileUpload(multiple=False)
        outt = widgets.Output()

        def _upload_case_name(a):
            with outt:
                clear_output()
                df_compare = pd.read_csv(io.BytesIO(w_case_to_compare.value[0].content))
                self.case_to_compare = list(df_compare["case name"].values)
                print(f"{len(self.case_to_compare)} case names uploaded.")

        w_case_to_compare.observe(_upload_case_name)
        display(widgets.VBox([w_case_to_compare, outt]))

    def load_case_object(self):
        """
        load one of multiple RecapCase(s) from directory.
        """
        self.case_instances = [
            RecapCase.from_dir(
                dir_str=self.dir_str,
                case_name=case,
                gurobi_credentials=None,
                skip_monte_carlo_draw_setup=True,
                skip_creating_results_folder=True,
            )
            for case in self.case_to_compare
        ]
        self.load_case_inputs()

    def load_case_inputs(self):
        """
        Aggregate all type of inputs for case(s) and save in a dictionary.
        """

        def _create_portfolio_vector(case: RecapCase) -> pd.Series:
            """
            Load case resource portfolio for ONE case.
            """
            portfolio = {}
            for resource in case.system.resources.values():
                portfolio[(resource.name, list(resource.resource_groups.values())[0].instance_to.name)] = (
                    resource.capacity_planned.slice_by_year(case.model_year)
                )
            portfolio_series = pd.Series(portfolio, name=case.case_name).rename_axis(
                index=("Resource", "Resource Group")
            )

            return portfolio_series

        def _create_load_vector(case: RecapCase) -> pd.DataFrame:
            """
            Identify case load components and scaling for ONE case.
            """
            scale_by_cap = []
            scale_by_en = []
            for ld in case.system.loads.keys():
                to_peak = (
                    round(case.system.loads[ld].annual_peak_forecast.data.loc[str(case.model_year)].values[0], 0)
                    if case.system.loads[ld].scale_by_capacity
                    else None
                )
                if to_peak is not None:
                    scale_by_cap.append(ld + "-" + str(to_peak))

                to_energy = (
                    round(case.system.loads[ld].annual_energy_forecast.data.loc[str(case.model_year)].values[0], 0)
                    if case.system.loads[ld].scale_by_energy
                    else case.system.loads[ld].scale_by_energy
                )
                if to_energy is not None:
                    scale_by_en.append(ld + "-" + str(to_energy))

            load_scale = pd.DataFrame(
                {case.case_name: [",".join(scale_by_cap), ",".join(scale_by_en)]},
                index=(["Scaled by Capacity (MW)", "Scaled by Energy (MWh)"]),
            )

            return load_scale

        portfolio_vector_frame = pd.concat(
            [_create_portfolio_vector(instance) for instance in self.case_instances], axis=1
        )
        portfolio_vector_frame.loc[("", "Total"), :] = portfolio_vector_frame.sum()

        resource_group_frame = portfolio_vector_frame.groupby(level=["Resource Group"], sort=False).sum()
        load_scaling_frame = pd.concat([_create_load_vector(instance) for instance in self.case_instances], axis=1)
        case_setting_frame = pd.concat(
            [pd.Series(instance.case_settings.dict(), name=instance.case_name) for instance in self.case_instances],
            axis=1,
        )
        case_scenario_frame = pd.concat(
            [
                pd.Series(instance.system.scenarios, name=instance.case_name, dtype="object")
                for instance in self.case_instances
            ],
            axis=1,
        )

        self.input_data_dict = {
            "Resource Portfolio": portfolio_vector_frame,
            "Resource Group": resource_group_frame,
            "Load Component": load_scaling_frame,
            "Case Setting": case_setting_frame,
            "Case Scenarios": case_scenario_frame,
        }

    def create_case_comparison(self):
        """
        Create widget for comparing case(s) inputs, plotting charts, and download csv files. No data return.
        """

        item = widgets.Dropdown(options=list(self.input_data_dict.keys()))
        butt1 = widgets.Button(description="Compare")
        butt2 = widgets.Button(description="Download csv file")
        outt = widgets.Output()

        def action1(a):
            with outt:
                clear_output()
                pd.set_option("display.max_rows", len(self.input_data_dict[item.value]))
                display(self.input_data_dict[item.value])

        def action2(b):
            with outt:
                path = "../analysis/Inputs checker/"
                if not os.path.exists(path):
                    os.mkdir(path)
                self.input_data_dict[item.value].to_csv(path + item.value + ".csv")

        butt1.on_click(action1)
        butt2.on_click(action2)
        buttons = widgets.HBox([butt1, butt2])
        display(widgets.VBox([item, buttons, outt]))

    def upload_customized_resource_group(self):
        """
        Categorize resources into customized groups for visualization purpose.
        """

        w_resource_group = widgets.FileUpload(multiple=False)
        outt = widgets.Output()

        def _upload_disp_group(a):
            with outt:
                clear_output()
                df_group = pd.read_csv(io.BytesIO(w_resource_group.value[0].content))[["resource", "dispatch_group"]]
                self.resource_dispatch_group_map = df_group
                print("Customized dispatch group uploaded.")
                pd.set_option("display.max_rows", len(self.resource_dispatch_group_map))
                display(self.resource_dispatch_group_map)

        w_resource_group.observe(_upload_disp_group)
        display(widgets.VBox([w_resource_group, outt]))

    def create_resource_stack_chart(self, Height, Width, portfolio):
        """
        Creating resource stack chart for case(s).
        """
        global color_map

        portfolio = portfolio.rename_axis(columns="Case").stack(0).rename("Capacity").reset_index()
        portfolio_cust = pd.merge(
            portfolio, self.resource_dispatch_group_map, left_on="Resource", right_on="resource", how="outer"
        ).drop(columns=["resource"])
        portfolio_cust = portfolio_cust.loc[portfolio_cust["Resource Group"] != "Total"]

        portfolio_cust.index = portfolio_cust["dispatch_group"]
        group_order = [
            "Thermal",
            "Nuclear",
            "Imports",
            "Other",
            "Hydro",
            "Solar",
            "LBW",
            "OSW",
            "Storage",
            "LDES",
            "DR",
            "FlexLoad",
        ]
        group_order_cs = [g for g in group_order if g in portfolio_cust.index.values]
        portfolio_cust = portfolio_cust.loc[group_order_cs]

        fig = px.bar(
            portfolio_cust,
            x="Case",
            y="Capacity",
            barmode="stack",
            color="dispatch_group",
            color_discrete_map=color_map,
            opacity=0.9,
            hover_data=["Resource Group"],
        )
        fig.update_layout(
            title_text="Resource Stack (MW)",
            yaxis_title="",
            yaxis_tickformat=",d",
            height=int(Height),
            width=int(Width),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, pad=10),
            font={"family": "Arial", "size": 12},
            legend_traceorder="reversed",
        )
        fig.show("notebook")

    def show_resource_portfolio(self):
        height_input = widgets.IntSlider(value=500, min=200, max=1000, description="Height:")
        width_input = widgets.IntSlider(value=500, min=200, max=1000, description="Width:")
        w = interactive(
            self.create_resource_stack_chart,
            {"manual": True, "manual_name": "Update Chart"},
            Height=height_input,
            Width=width_input,
            portfolio=fixed(self.input_data_dict["Resource Portfolio"]),
        )
        display(w)

    def show_ELCC_input(self, ELCC_case: str):
        """
        Create widget for loading ELCC resource info for selected case. No data return.
        """

        def create_ELCC_dict(case: RecapCase) -> dict[str : pd.DataFrame]:
            """
            Aggregate ELCC resource info of ONE case and save in a dictionary.
            """
            ELCC_dir = self.dir_str.recap_settings_dir / case.case_name / "ELCC_surfaces"
            ELCC_surface = pd.read_csv(ELCC_dir / "custom_ELCC_surface.csv")
            Marg_ELCC = pd.read_csv(ELCC_dir / "marginal_ELCC.csv")
            Incre_ELCC = pd.read_csv(ELCC_dir / "incremental_last_in_ELCC.csv")
            Decre_ELCC = pd.read_csv(ELCC_dir / "decremental_last_in_ELCC.csv")
            data_dict = {
                "ELCC Surface": ELCC_surface,
                "Marginal ELCC Resource": Marg_ELCC,
                "Incremental ELCC Resource": Incre_ELCC,
                "Decremental ELCC Resource": Decre_ELCC,
            }

            return data_dict

        for instance in self.case_instances:
            if instance.case_name == ELCC_case.value:
                ELCC_data_dict = create_ELCC_dict(instance)

        item = widgets.Dropdown(options=list(ELCC_data_dict.keys()))
        butt = widgets.Button(description="Load Input")
        outt = widgets.Output()

        def action(a):
            with outt:
                clear_output()
                pd.set_option("display.max_rows", len(ELCC_data_dict[item.value]))
                display(ELCC_data_dict[item.value])

        butt.on_click(action)
        display(widgets.VBox([item, butt, outt]))


class ProfileChecker:
    """
    Create a ProfileChecker object to QC the profile inputs into the model.
    This can include thermal Pmin/pmax, renewable profiles, as well as hydro budget data, etc
    """

    def __init__(
        self,
        dir_str,
    ):
        self.profile_dir = dir_str.data_dir.joinpath("profiles")

    def get_agg_profile_input(self):
        """
        Read in all profiles in the /profiles folder when possible
        :return: a dataframe with timestamp as index and each column represents one timeseries profile input
        """
        f_list = [pathlib.Path(d).name for d in glob.glob(str(self.profile_dir / "*")) if not pathlib.Path(d).is_dir()]

        df_agg_prof = pd.DataFrame()
        for f in f_list:
            f_name = f.replace("\\", "_").split(".csv")[0]
            df = pd.read_csv(self.profile_dir / f, index_col=0)
            df.index = pd.to_datetime(df.index)
            try:
                if pd.infer_freq(df.index) == "H":
                    df.columns = [f_name]
                    df_agg_prof = pd.concat([df_agg_prof, df], axis=1)
                else:
                    print(f"{f_name} is not an hourly timeseries, data not loaded.")
            except KeyError:
                print(f"Fail to read in profile {f_name}.")

        self.agg_profile = df_agg_prof

    def get_data_summary(self):
        """
        Produce a summary table of some basic inspections on the data.
        """
        dt_summary = pd.DataFrame(index=self.agg_profile.columns)
        dt_summary["Data Start"] = [self.agg_profile[col].notna().idxmax() for col in dt_summary.index]
        dt_summary["Data End"] = [self.agg_profile[col][::-1].notna().idxmax() for col in dt_summary.index]
        dt_summary["Years"] = (dt_summary["Data End"].dt.year - dt_summary["Data Start"].dt.year) + 1
        dt_summary["Data Max"] = [round(self.agg_profile[col].max(), 2) for col in dt_summary.index]
        dt_summary["Average Annual Sum"] = round(
            [self.agg_profile[col].sum() for col in dt_summary.index] / dt_summary["Years"], 2
        )
        dt_summary["Cap Factor"] = round(dt_summary["Average Annual Sum"] / (dt_summary["Data Max"] * 8760), 2)
        dt_summary["Include Leap Day ?"] = [
            True if sum(self.agg_profile[col].index.strftime("%m-%d") == "02-29") > 0 else False
            for col in dt_summary.index
        ]
        dt_summary["Missing Data Input?"] = dt_summary.apply(
            lambda x: (
                True
                if len(
                    pd.date_range(start=x["Data Start"], end=x["Data End"], freq="H").difference(
                        self.agg_profile[x.name].index
                    )
                )
                > 0
                else False
            ),
            axis=1,
        )
        dt_summary["# of NaN Values Between Data Start and End"] = dt_summary.apply(
            lambda x: self.agg_profile[x.name].loc[x["Data Start"] : x["Data End"]].isnull().sum(), axis=1
        )
        dt_summary["# of Consecutive Zeros"] = [
            self.agg_profile[col].ne(0).cumsum()[~(self.agg_profile[col].ne(0))].value_counts().max()
            for col in dt_summary.index
        ]
        dt_summary["% of Values Outside of (Q1-IQR, Q3+IQR)"] = [
            str(
                round(
                    (
                        1
                        - len(
                            self.agg_profile[col][
                                ~(
                                    (
                                        self.agg_profile[col]
                                        < (
                                            self.agg_profile[col].quantile(0.25)
                                            - 1.5
                                            * (
                                                self.agg_profile[col].quantile(0.75)
                                                - self.agg_profile[col].quantile(0.25)
                                            )
                                        )
                                    )
                                    | (
                                        self.agg_profile[col]
                                        > (
                                            self.agg_profile[col].quantile(0.75)
                                            + 1.5
                                            * (
                                                self.agg_profile[col].quantile(0.75)
                                                - self.agg_profile[col].quantile(0.25)
                                            )
                                        )
                                    )
                                )
                            ]
                        )
                        / len(self.agg_profile[col])
                    )
                    * 100,
                    2,
                )
            )
            + "%"
            for col in dt_summary.index
        ]
        dt_summary.style.format({"% of values outside of (Q1-IQR, Q3+IQR)": "{:.2%}"})

        self.data_summary = dt_summary
        pd.set_option("display.max_rows", len(dt_summary))
        display(self.data_summary)

    def print_low_quality_data(self):
        """
        Print the date index of missing values, NaNs, and zeros
        """
        item1 = widgets.Dropdown(
            options=self.data_summary.index, description="Choose an item", style={"description_width": "initial"}
        )
        item2 = widgets.Dropdown(
            options=["Missing data time index", "NaN values", "Longest zero inputs"],
            description="Data Inspection",
            style={"description_width": "initial"},
        )
        butt = widgets.Button(description="Print")
        outt = widgets.Output()

        def action(a):
            with outt:
                clear_output()
                if item2.value == "Missing data time index":
                    missing_mask = pd.date_range(
                        start=self.data_summary.loc[item1.value, "Data Start"],
                        end=self.data_summary.loc[item1.value, "Data End"],
                        freq="H",
                    ).difference(self.agg_profile[item1.value].index)
                    print(pd.Series(missing_mask))
                elif item2.value == "NaN values":
                    nan_mask = (
                        self.agg_profile[item1.value]
                        .loc[
                            self.data_summary.loc[item1.value, "Data Start"] : self.data_summary.loc[
                                item1.value, "Data End"
                            ]
                        ]
                        .isnull()
                    )
                    print(nan_mask.loc[nan_mask])
                else:
                    try:
                        df_zeros = self.agg_profile[item1.value].ne(0).cumsum()[~(self.agg_profile[item1.value].ne(0))]
                        count_zero = df_zeros.value_counts().index[0]
                        df_zeros = pd.DataFrame(df_zeros)
                        idx = df_zeros.loc[df_zeros[item1.value] == count_zero].index
                        print(self.agg_profile.loc[idx, item1.value])
                    except IndexError:
                        print("Selection not valid. ")

        butt.on_click(action)
        items = widgets.VBox([item1, item2])
        display(widgets.VBox([items, butt, outt]))

    # def fill_missing(self):
    #     """
    #     Fill missing values with rolling average / month-hour average
    #     """
    # df_fill = df_raw.copy(deep=True)
    #
    # if option == 'Rolling Average':
    #     df_fill[col + '_adj'] = df_fill[col].values
    #     df_fill[col + '_adj']= df_fill[col + '_adj'].fillna(df_fill[col + '_adj'].rolling(window = window_length, center=True).mean())
    # elif option == 'Month-hour Average':
    #     df_fill[col + '_adj'] = df_fill.groupby([df_fill.index.month, df_fill.index.hour], dropna=True)[col].apply(lambda x: x.fillna(x.mean()))
    # else:
    #     print('Choose a valid filling option.')
    #
    # # Only fill in the NaN values in start-end period in case index date is longer than actual data input
    # df_fill.loc[dt_summary.loc[col, 'Data Start']:dt_summary.loc[col, 'Data End'], col] = df_fill.loc[dt_summary.loc[col, 'Data Start']:dt_summary.loc[col, 'Data End'], col + '_adj']
    # df_fill = df_fill.drop(columns = [col + '_adj'])
    #
    # return df_fill

    def create_raw_data_chart(self):
        fig = go.Figure()
        for l in self.agg_profile.columns:
            fig.add_trace(go.Scatter(x=self.agg_profile.index, y=self.agg_profile[l].values, mode="lines", name=l))
            fig.update_layout(
                height=400,
                width=800,
                yaxis_tickformat=",d",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Arial", "size": 12},
                showlegend=True,
            )
        fig.show()
        self.raw_data_chart = fig

    def get_MH_shape(self):
        """
        Input: long dataframe with multi-year dateindex
        Output: 288-rows month-hour dataframe
        """
        df_mh = self.agg_profile.copy(deep=True)
        df_mh.index = pd.to_datetime(df_mh.index)
        df_mh["month"] = df_mh.index.month
        df_mh["hour"] = df_mh.index.hour
        mh_shape = df_mh.groupby(["month", "hour"]).mean().reset_index()

        self.mh_profile_shape = mh_shape

    def create_month_hour_shape(self, Ylim):
        num_ylim = 1 if Ylim == True else self.mh_profile_shape.max().max()
        fig = go.Figure()
        hour = np.arange(0, 288)
        for l in self.mh_profile_shape.columns:
            fig.add_trace(go.Scatter(x=hour, y=self.mh_profile_shape[l].values, mode="lines", name=l))
        for x, m in zip(
            list(np.arange(0, 288, 24)),
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"],
        ):
            fig.add_vline(
                x=x,
                line_dash="dot",
                line_color="grey",
                annotation_text=m,
                annotation_position="top",
                annotation_font_size=16,
            )
        fig.update_layout(
            title_text="Month-Hour Shape",
            xaxis_title="Hour",
            yaxis_range=[0, num_ylim],
            height=450,
            width=900,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Arial", "size": 12},
            showlegend=True,
        )
        fig.show()
        self.month_hour_chart = fig

    def show_profile_shape(self):
        data_selector = widgets.RadioButtons(
            options=["Raw data shape", "Month-hour shape"],
            vlaue=None,
            description="Choose data to show",
            style={"description_width": "initial"},
            layout={"width": "300px"},
        )
        butt = widgets.Button(description="Print shape", layout={"width": "300px"})
        outt = widgets.Output()

        def action(a):
            with outt:
                clear_output()
                if data_selector.value == "Raw data shape":
                    self.create_raw_data_chart()
                else:
                    self.get_MH_shape()
                    ylim_slider = widgets.Checkbox(
                        value=False, description="Limit month-hour chart yaxis at 1?", disabled=False, indent=False
                    )
                    w = interactive(
                        self.create_month_hour_shape, {"manual": True, "manual_name": "Update chart"}, Ylim=ylim_slider
                    )
                    display(w)

        butt.on_click(action)
        display(widgets.VBox([data_selector, butt, outt]))

    def show_specific_ts_distribution(self):
        item = widgets.Dropdown(options=self.agg_profile.columns)
        butt = widgets.Button(description="Check Specific Data Distribution", layout={"width": "300px"})
        outt = widgets.Output()

        def action(a):
            with outt:
                clear_output()
                self.create_ts_distribution(item.value)

        fig = butt.on_click(action)
        display(widgets.VBox([item, butt, outt]))

        return fig

    def create_ts_distribution(self, item_name):
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=self.agg_profile[item_name].values, name=item_name, marker_color="#3D9970", opacity=0.75)
        )
        fig.update_layout(
            title_text="Distribution of {}".format(item_name),
            yaxis_title="Frequency",
            height=400,
            width=600,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Arial", "size": 12},
            showlegend=True,
        )
        fig.show()


class CaseComparison:
    def __init__(
        self,
        dir_str,
    ):
        self.results_dir = dir_str.results_dir.joinpath("recap")

    def case_selection(self):
        """
        Select several cases manually for comparison. Default to the latest run for each case.
        """

        case_list = [
            pathlib.Path(d).name for d in glob.glob(str(self.results_dir / "*")) if any(pathlib.Path(d).iterdir())
        ]
        run_list_all = [
            [
                pathlib.Path(d).name
                for d in glob.glob(str(self.results_dir / cs / "*"))
                if any(pathlib.Path(d).iterdir())
            ]
            for cs in case_list
        ]
        run_list = [r[-1] if len(r) > 0 else 0 for r in run_list_all]
        self.case_run_dict = dict(zip(case_list, run_list))

        case_selector = widgets.SelectMultiple(
            options=case_list,
            style={"description_width": "initial"},
            layout={"width": "50%"},
            rows=min(len(case_list), 10),
        )

        butt = widgets.Button(description="Save Case Selection. ")
        outt = widgets.Output()

        def _collect_case_name(a):
            with outt:
                clear_output()
                self.case_to_compare = list(case_selector.value)
                self.run_to_compare = [self.case_run_dict[cs] for cs in self.case_to_compare]

        butt.on_click(_collect_case_name)
        display(widgets.VBox([case_selector, butt, outt]))

    def case_selection_from_local(self):
        """
        Another option to load cases - specify case name and paired run name in a spreadsheet and load from local.
        """

        w_case_to_compare = widgets.FileUpload(multiple=False)
        outt = widgets.Output()

        def _upload_case_name(a):
            with outt:
                clear_output()
                df_compare = pd.read_csv(io.BytesIO(w_case_to_compare.value[0].content))
                self.case_to_compare = list(df_compare["case name"].values)
                self.run_to_compare = list(df_compare["run name"].values)
                self.case_run_dict = dict(zip(self.case_to_compare, self.run_to_compare))
                print(f"{len(self.case_to_compare)} case names uploaded.")

        w_case_to_compare.observe(_upload_case_name)
        display(widgets.VBox([w_case_to_compare, outt]))

    def load_case_to_compare(self):
        print(f"Selecting {len(self.case_to_compare)} cases to compare:")

        self.all_case_results = {}
        for cs in self.case_to_compare:
            self.all_case_results[cs] = ResultsViewer(
                result_folder=self.results_dir, case_name=cs, run_name=self.case_run_dict[cs]
            )
            self.all_case_results[cs].read_case_data(load_disp_result=True, msg_box=False)
            self.all_case_results[cs].get_metrics_result()
            if not self.all_case_results[cs].metrics.empty:
                print(f"Case: {cs}-{self.case_run_dict[cs]} loaded. Detailed dispatch results excluded.")
            else:
                print(f"No results / unfinished case for {cs}.")

    def compare_case_metrics(self):
        case_metrics = [self.all_case_results[cs].metrics for cs in self.case_to_compare]
        self.metrics_comparison = pd.concat(case_metrics, axis=0, keys=self.case_to_compare)

        metrics_selector = widgets.Dropdown(
            options=["Tuned System Metrics", "Untuned System Metrics", "Cap Short", "Portfolio ELCC and TRN"],
            value=None,
            description="Select metrics to compare: ",
            style={"description_width": "initial"},
        )

        butt1 = widgets.Button(description="Compare Case")
        butt2 = widgets.Button(description="Download csv file")
        outt = widgets.Output()

        def extract_comparison_data(a):
            with outt:
                clear_output()
                try:
                    if metrics_selector.value == "Tuned System Metrics":
                        df_display = self.metrics_comparison.xs("Tuned System Metrics", level=1)[
                            ["LOLE", "EUE", "LOLH", "LOLP", "ALOLP"]
                        ]
                    elif metrics_selector.value == "Untuned System Metrics":
                        df_display = self.metrics_comparison.xs("Untuned System Metrics", level=1)[
                            ["LOLE", "EUE", "LOLH", "LOLP", "ALOLP"]
                        ]
                    elif metrics_selector.value == "Cap Short":
                        df_display = self.metrics_comparison.xs("Tuned System Metrics", level=1)[
                            ["perfect_capacity_shortfall"]
                        ]
                    elif metrics_selector.value == "Portfolio ELCC and TRN":
                        df_display = self.metrics_comparison.xs("Tuned System Metrics", level=1)[
                            ["total_resource_need", "portfolio_ELCC"]
                        ]
                    else:
                        df_display = pd.DataFrame()
                    pd.set_option("display.max_rows", len(df_display))
                    display(df_display)
                except KeyError as e:
                    print("No data available for selected cases.")

        def downloader(b):
            with outt:
                clear_output()
                path = "../analysis/Result Inspection/"
                self.metrics_comparison.to_csv(path + "/" + "case_comparison_results.csv")
                print("Downloaded. ")

        butt1.on_click(extract_comparison_data)
        butt2.on_click(downloader)
        buttons = widgets.HBox([butt1, butt2])
        display(widgets.VBox([metrics_selector, buttons, outt]))


class ResultsViewer:
    def __init__(self, result_folder, case_name=None, run_name=None):
        self.result_folder = result_folder
        self.case_name = case_name
        self.run_name = run_name

    def case_selection(self):
        """
        Select cases for input comparison.
        """

        case_list = [
            pathlib.Path(d).name
            for d in glob.glob(str(self.result_folder / "*"))
            if pathlib.Path(d + "/" + "reliability_results.csv").exists
        ]

        case_selector = widgets.Dropdown(
            options=case_list,
            description="Select a case:",
            style={"description_width": "initial"},
            layout={"width": "35%"},
        )
        run_selector = widgets.Dropdown(
            value=None, description="Select a run:", style={"description_width": "initial"}, layout={"width": "35%"}
        )

        butt = widgets.Button(description="Save Case Selection. ")
        outt = widgets.Output()

        @interact(case=case_selector, run=run_selector)
        def _set_run_name(case, run):
            run_selector.options = [
                pathlib.Path(d).name
                for d in glob.glob(str(self.result_folder / case_selector.value / "*"))
                if any(pathlib.Path(d).iterdir())
            ]

        def _collect_case_name(a):
            with outt:
                clear_output()
                self.case_name = case_selector.value
                self.run_name = run_selector.value

        butt.on_click(_collect_case_name)
        display(widgets.VBox([butt, outt]))

    def read_case_data(self, load_disp_result=False, msg_box=True) -> dict[str : pd.DataFrame]:
        """
        Read in all csv data files for the case.
        """
        self.results_dir = self.result_folder / self.case_name / self.run_name
        for attr, f_path in [
            ("reliability_results", "reliability_results.csv"),
            ("ELCC_results", "ELCC_results.csv"),
            ("case_settings", "case_settings.csv"),
        ]:
            setattr(self, attr, _load_general_result_csv(self.results_dir / f_path))
            if msg_box:
                if not getattr(self, attr).empty:
                    print(f"{attr} loaded.")
                else:
                    print(f" -- No {attr} results.")

        if load_disp_result:
            for attr, f_path in [
                ("untuned_disp", "untuned_dispatch_results.parquet"),
                ("tuned_disp", "tuned_dispatch_results.parquet"),
            ]:
                setattr(self, attr, _load_disp_result_parquet(self.results_dir / f_path))
                if msg_box:
                    if not getattr(self, attr).empty:
                        print(f"{attr} loaded.")
                    else:
                        print(f" -- No {attr} results.")

            for attr, disp_data in [
                ("month_hour_stats", self.untuned_disp),
                ("tuned_month_hour_stats", self.tuned_disp),
            ]:
                if not disp_data.empty:
                    setattr(self, attr, _get_month_hour_data(disp_data))
                else:
                    setattr(self, attr, pd.DataFrame())

    def get_metrics_result(self):
        """
        Summarize various metrics result for tuned and untuned system.
        Filling NaN values with 'N/A' for ease of identifying missing results.
        """
        self.ELCC_results.fillna("N/A", inplace=True)
        tuned_cap_short_results = self.ELCC_results.filter(items=["base_case"], axis=0)

        if not self.reliability_results.empty:
            self.reliability_results.fillna("N/A", inplace=True)

            untuned_relib_metrics = (
                self.reliability_results.iloc[[0]]
                if self.reliability_results.iloc[0, 1] == "DEFAULT"
                else pd.DataFrame(np.nan, index=[0], columns=self.reliability_results.columns)
            )

            tuned_relib_metrics = (
                self.reliability_results.iloc[[-1]]
                if len(self.reliability_results) > 1
                else pd.DataFrame(np.nan, index=[0], columns=self.reliability_results.columns)
            )
        else:
            untuned_relib_metrics = pd.DataFrame(np.nan, index=[0], columns=["EUE", "LOLE", "LOLH", "LOLP", "ALOLP"])
            tuned_relib_metrics = pd.DataFrame(np.nan, index=[0], columns=["EUE", "LOLE", "LOLH", "LOLP", "ALOLP"])

        self.metrics = (
            pd.concat(
                [
                    untuned_relib_metrics.reset_index().iloc[:, -5:],
                    pd.concat(
                        [
                            tuned_relib_metrics.reset_index().iloc[:, -5:],
                            tuned_cap_short_results.reset_index().iloc[:, 1:4],
                        ],
                        axis=1,
                    ),
                ],
                keys=["Untuned System Metrics", "Tuned System Metrics"],
                axis=0,
            )
            .fillna("N/A")
            .droplevel(1)
        )

    def get_ELCC_results(self):
        for elcc_type in [
            "marginal_ELCC",
            "incremental_last_in_ELCC",
            "decremental_last_in_ELCC",
            "custom_ELCC_surface",
        ]:
            df_ELCC = self.ELCC_results.filter(like=elcc_type, axis=0)
            df_ELCC = df_ELCC.drop(columns=["total_resource_need", "portfolio_ELCC"])
            setattr(self, elcc_type, df_ELCC)
            if not getattr(self, elcc_type).empty:
                print(f"{elcc_type} loaded.")
            else:
                print(f" -- No {elcc_type} results.")

    def show_heat_map(self):
        """
        Show month-hour heat maps for case-run. Default to show LOLE heat map.
        """
        sys_dict = {
            "Tuned System": getattr(self, "tuned_month_hour_stats"),
            "Untuned System": getattr(self, "month_hour_stats"),
        }

        sys_selector = widgets.RadioButtons(
            options=["Tuned System", "Untuned System"],
            vlaue=None,
            description="Tuned or Untuned system? ",
            style={"description_width": "initial"},
            layout={"width": "50%"},
        )
        target_selector = widgets.RadioButtons(
            options=["LOLP", "LOLH", "EUE"],
            vlaue=None,
            description="Select a metric to produce heat map: ",
            style={"description_width": "initial"},
            layout={"width": "50%"},
        )

        butt1 = widgets.Button(description="Produce heat map")
        butt2 = widgets.Button(description="Download csv file")
        outt = widgets.Output()

        def _extract_chart_data(a):
            with outt:
                clear_output()
                if not sys_dict[sys_selector.value].empty:
                    mh_dt = sys_dict[sys_selector.value][target_selector.value].values
                    stepsize = 0.01 if target_selector.value == "LOLP" else 1
                    height_input = widgets.IntSlider(
                        value=400, min=200, max=1000, description="Height:", style={"description_width": "initial"}
                    )
                    width_input = widgets.IntSlider(
                        value=600, min=200, max=1000, description="Width:", style={"description_width": "initial"}
                    )
                    zmax_input = widgets.FloatText(
                        value=round(max(mh_dt), 2),
                        step=stepsize,
                        description="Color Bar Upper Bound - max:",
                        style={"description_width": "initial"},
                        readout_format=".1f",
                    )
                    zmin_input = widgets.FloatText(
                        value=round(min(mh_dt), 2),
                        description="Color Bar Lower Bound - min:",
                        style={"description_width": "initial"},
                        readout_format=".1f",
                    )
                    w = interactive(
                        self.create_heatmap_plot,
                        {"manual": True, "manual_name": "Update Heat Map"},
                        Height=height_input,
                        Width=width_input,
                        MaxRange=zmax_input,
                        MinRange=zmin_input,
                        mh_table=fixed(sys_dict[sys_selector.value]),
                        metric=fixed(target_selector.value),
                    )
                    display(w)
                else:
                    print(f"No dispatch data available to make heat map for {sys_selector.value}.")

        def _downloader(b):
            with outt:
                clear_output()
                path = "../analysis/Result Inspection/" + self.case_name
                if not os.path.exists(path):
                    os.mkdir(path)
                if not sys_dict[sys_selector.value].empty:
                    sys_dict[sys_selector.value].to_csv(path + "/" + f"month_hour_{sys_selector.value}.csv")
                    print("Downloaded. ")
                else:
                    print("No data available. ")

        butt1.on_click(_extract_chart_data)
        butt2.on_click(_downloader)
        items = widgets.VBox([sys_selector, target_selector])
        buttons = widgets.HBox([butt1, butt2])
        display(widgets.VBox([items, buttons, outt]))

    def create_heatmap_plot(self, Height, Width, MaxRange, MinRange, mh_table, metric):
        """
        Visual function for heatmap.
        """
        trace = go.Heatmap(
            x=["{}".format(t) for t in np.arange(24)],
            y=["Dec", "Nov", "Oct", "Sep", "Aug", "July", "June", "May", "Apr", "Mar", "Feb", "Jan"],
            z=np.flip(mh_table[metric].values.reshape(12, 24), axis=0),
            type="heatmap",
            colorscale="Reds",
            colorbar=dict(title=metric),
            xgap=1,
            ygap=1,
            zmin=MinRange,
            zmax=MaxRange,
        )
        fig = go.Figure(data=[trace])

        fig.update_layout(
            xaxis_title="Hour of the day",
            title_text="Month-Hour Heat Map",
            height=Height,
            width=Width,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Arial", "size": 11},
        )
        fig.update_xaxes(showline=True, linewidth=0.5, linecolor="grey", mirror=True)
        fig.update_yaxes(showline=True, linewidth=0.5, linecolor="grey", mirror=True)
        fig.show()

    def show_ELCC_results(self):
        """
        Show ELCC results of the case if applicable.
        """
        ELCC_selector = widgets.Dropdown(
            options=["marginal_ELCC", "incremental_last_in_ELCC", "decremental_last_in_ELCC", "custom_ELCC_surface"]
        )
        butt = widgets.Button(description="Print ELCC results")
        outt = widgets.Output()

        def print_results(a):
            with outt:
                clear_output()
                if not (getattr(self, ELCC_selector.value).empty):
                    display(getattr(self, ELCC_selector.value))
                else:
                    print("No ELCC Results Available.")

        butt.on_click(print_results)
        display(widgets.VBox([ELCC_selector, butt, outt]))

    def calc_lol_stats(self):
        """
        Calculate lol event duration and size in the system.
        """
        sys_dict = {"Tuned System": getattr(self, "tuned_disp"), "Untuned System": getattr(self, "untuned_disp")}

        self.lol_stats = {}
        for sys in sys_dict.keys():
            if not (sys_dict[sys].empty):
                df_unserved_series = sys_dict[sys][["unserved_energy_and_reserve"]].copy()
                df_unserved_series["Outage indicator"] = df_unserved_series["unserved_energy_and_reserve"] > 0
                df_unserved_series["grouping"] = (
                    df_unserved_series["Outage indicator"] != df_unserved_series["Outage indicator"].shift()
                ).cumsum()
                outage_durations = (
                    df_unserved_series[df_unserved_series["Outage indicator"] == True].groupby(["grouping"]).size()
                )
                outage_size = (
                    df_unserved_series[df_unserved_series["Outage indicator"] == True]
                    .groupby(["grouping"])["unserved_energy_and_reserve"]
                    .mean()
                )

                df_unserved_date = (
                    df_unserved_series[df_unserved_series["Outage indicator"] == True]
                    .reset_index()
                    .drop_duplicates(["grouping"])
                )
                df_unserved_date["date"] = df_unserved_date["timestamp"].dt.date

                df_outage = pd.concat([outage_durations, outage_size], axis=1).reset_index()
                df_outage_day = pd.merge(
                    df_unserved_date[["MC_draw", "date", "grouping"]], df_outage, on=(["grouping"]), how="inner"
                ).drop(columns=(["grouping"]))
                df_outage_day.columns = ["MC_draw", "date", "Outage Duration (hr)", "Average Outage Magnitude (MW)"]

                self.lol_stats[sys] = df_outage_day
            else:
                self.lol_stats[sys] = pd.DataFrame()

    def show_lol_stats(self):
        """
        Show system (tuned / untuned) loss-of-load pattern.
        """
        self.calc_lol_stats()

        sys_selector = widgets.RadioButtons(options=list(self.lol_stats.keys()))
        butt1 = widgets.Button(description="Show outage pattern")
        butt2 = widgets.Button(description="Download chart data")
        outt = widgets.Output()

        def _show_stats(a):
            with outt:
                clear_output()
                if not self.lol_stats[sys_selector.value].empty:
                    self.create_lol_stats_plot(self.lol_stats[sys_selector.value])
                else:
                    print("No Unserved Energy Data Available.")

        def _downloader(b):
            with outt:
                clear_output()
                if not self.lol_stats[sys_selector.value].empty:
                    path = "../analysis/Result Inspection/" + self.case_name
                    if not os.path.exists(path):
                        os.mkdir(path)
                    self.lol_stats[sys_selector.value].to_csv(
                        path + "/" + f"{sys_selector.value} loss-of-load event stats.csv"
                    )
                    print("Downloaded. ")
                else:
                    print("No data available. ")

        butt1.on_click(_show_stats)
        butt2.on_click(_downloader)
        buttons = widgets.HBox([butt1, butt2])
        display(widgets.VBox([sys_selector, buttons, outt]))

    def create_lol_stats_plot(self, df_lol):
        """
        Visual function for lol stats chart.
        Create scatter plot and histogram plot to show outage event duration and size
        """
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Loss-of-Load Duration V.S. Magnitude", "Loss-of-Load Duration Histogram")
        )
        fig.add_trace(
            go.Scatter(
                x=df_lol["Outage Duration (hr)"].values,
                y=df_lol["Average Outage Magnitude (MW)"].values,
                mode="markers",
                text=df_lol["date"].values,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=df_lol["Outage Duration (hr)"].values, marker_color="grey", opacity=0.75, bingroup=1),
            row=1,
            col=2,
        )
        fig.update_layout(
            xaxis=dict(title="Outage Duration (hrs)"),
            yaxis=dict(title="Outage Magnitude (MW)"),
            xaxis2=dict(title="Outage Duration (hrs)"),
            yaxis2=dict(title="Frequency"),
            height=400,
            width=800,
            font={"family": "Arial", "size": 11},
            bargap=0.2,
            showlegend=False,
        )

        fig.show()

    def collect_lol_date_EUE(self):
        """
        Collect date of all lol events in the system.
        """
        sys_dict = {"Tuned System": getattr(self, "tuned_disp"), "Untuned System": getattr(self, "untuned_disp")}

        self.lol_date_EUE = {}
        for sys in sys_dict.keys():
            if not sys_dict[sys].empty:
                df = sys_dict[sys].reset_index()
                df["year"] = df["timestamp"].dt.year
                df["month"] = df["timestamp"].dt.month
                df["day"] = df["timestamp"].dt.day
                df["hour"] = df["timestamp"].dt.hour

                outage_day_EUE = pd.DataFrame()
                df_group = (
                    df[["MC_draw", "year", "month", "day", "hour", "unserved_energy_and_reserve"]]
                    .groupby(["MC_draw", "year", "month", "day", "hour"])
                    .sum()
                    .reset_index()
                )
                df_group["time"] = df_group.apply(
                    lambda x: "-".join([x["MC_draw"], str(x["year"]), str(x["month"]), str(x["day"])]), axis=1
                )
                outage_date = df_group.loc[df["unserved_energy_and_reserve"] != 0]

                for date in outage_date["time"].unique():
                    outage_series = df_group.loc[df_group["time"] == date]["unserved_energy_and_reserve"].values
                    outage_day_EUE.loc[:, date] = outage_series
                self.lol_date_EUE[sys] = outage_day_EUE
            else:
                self.lol_date_EUE[sys] = pd.DataFrame()

    def show_lol_date_EUE(self):
        self.collect_lol_date_EUE()

        sys_selector = widgets.RadioButtons(options=list(self.lol_date_EUE.keys()))
        butt1 = widgets.Button(description="Show lol-day EUE")
        butt2 = widgets.Button(description="Download chart data")
        outt = widgets.Output()

        def _show_data(a):
            with outt:
                clear_output()
                if not self.lol_date_EUE[sys_selector.value].empty:
                    self.create_lol_date_EUE_plot(self.lol_date_EUE[sys_selector.value])
                else:
                    print("No Unserved Energy Data Available.")

        def _downloader(b):
            with outt:
                clear_output()
                if not self.lol_date_EUE[sys_selector.value].empty:
                    path = "../analysis/Result Inspection/" + self.case_name
                    if not os.path.exists(path):
                        os.mkdir(path)
                    self.lol_date_EUE[sys_selector.value].to_csv(
                        path + "/" + f"{sys_selector.value} daily EUE series.csv"
                    )
                    print("Downloaded. ")
                else:
                    print("No data available. ")

        butt1.on_click(_show_data)
        butt2.on_click(_downloader)
        buttons = widgets.HBox([butt1, butt2])
        display(widgets.VBox([sys_selector, buttons, outt]))

    def create_lol_date_EUE_plot(self, df_EUE):
        """
        Visual function for LOL day EUE.
        """
        fig = go.Figure()
        for date in df_EUE.columns:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(24),
                    y=df_EUE[date].values,
                    mode="lines",
                    hovertext=date,
                    line=dict(color="grey"),
                    opacity=0.2,
                )
            )
        average_EUE = df_EUE.mean(axis=1)
        fig.add_trace(
            go.Scatter(
                x=np.arange(24),
                y=average_EUE.values,
                mode="lines",
                name="Average",
                line=dict(color="black"),
                opacity=0.8,
            )
        )
        fig.update_layout(
            title_text="Daily EUE Pattern",
            xaxis_title="Hour",
            yaxis_title="EUE (MW)",
            yaxis_tickformat=",d",
            height=400,
            width=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Arial", "size": 12},
            hoverlabel=dict(font_size=10),
            showlegend=False,
        )

        fig.show()

    def _upload_resource_group_csv(self, path):
        df_group = pd.read_csv(path)[["resource", "dispatch_group"]]
        df_group = df_group.set_index("dispatch_group")
        self.resource_dispatch_group_map = df_group["resource"].groupby(level=0).agg(list).to_dict()

    def upload_resource_group(self):
        # upload resource group csv
        w_resource_group = widgets.FileUpload(multiple=False)
        outt = widgets.Output()

        def _upload_resourcegroup(a):
            with outt:
                clear_output()
                self._upload_resource_group_csv(io.BytesIO(w_resource_group.value[0].content))
                print("Dispatch group uploaded.")
                pd.set_option("display.max_rows", len(self.resource_dispatch_group_map))
                display(self.resource_dispatch_group_map)

        w_resource_group.observe(_upload_resourcegroup)
        display(widgets.VBox([w_resource_group, outt]))

    def calc_disp_by_group(self, disp_raw: str):
        disp_by_group = disp_raw.copy(deep=True)
        groups = self.resource_dispatch_group_map.keys()

        for group in groups:
            disp_by_group[group + " group"] = disp_by_group[self.resource_dispatch_group_map[group]].sum(axis=1)

        return disp_by_group

    # Show list of LOL days
    def show_lol_day_list(self):
        sys_dict = {"Tuned System": getattr(self, "tuned_disp"), "Untuned System": getattr(self, "untuned_disp")}

        sys_selector = widgets.RadioButtons(
            options=["Tuned System", "Untuned System"],
            vlaue=None,
            description="Tuned or Untuned system? ",
            style={"description_width": "auto"},
            layout={"width": "300px"},
        )

        butt1 = widgets.Button(
            description="Print LOL-day list", style={"description_width": "initial"}, layout={"width": "300px"}
        )
        butt2 = widgets.Button(
            description="Visualize LOL periods by MC draw",
            style={"description_width": "auto"},
            layout={"width": "300px"},
        )
        butt3 = widgets.Button(
            description="Download LOL-day list", style={"description_width": "initial"}, layout={"width": "300px"}
        )
        outt = widgets.Output()

        self.calc_lol_stats()

        def _extract_lolday(a):
            with outt:
                clear_output()
                if not self.lol_stats[sys_selector.value].empty:
                    pd.set_option("display.max_rows", len(self.lol_stats[sys_selector.value]))
                    display(self.lol_stats[sys_selector.value])
                else:
                    print("No dispatch data available.")

        def _show_mc_lolperiod(b):
            with outt:
                clear_output()
                if not sys_dict[sys_selector.value].empty:
                    df_EUE = sys_dict[sys_selector.value].reset_index().set_index("timestamp")
                    df_EUE = df_EUE.loc[df_EUE["unserved_energy_and_reserve"] > 0]
                    df_EUE.index = pd.to_datetime(df_EUE.index)
                    df_EUE["date"] = df_EUE.index.date
                    df_EUE["dayofyear"] = df_EUE.index.dayofyear

                    df_EUE_plot = df_EUE[["MC_draw", "date", "dayofyear"]].drop_duplicates()
                    self.create_lol_bymc_chart(df_EUE_plot)
                else:
                    print("No dispatch data available.")

        def _downloader(c):
            with outt:
                clear_output()
                if not self.lol_stats[sys_selector.value].empty:
                    path = "../analysis/Result Inspection/" + self.case_name
                    if not os.path.exists(path):
                        os.mkdir(path)
                    self.lol_stats[sys_selector.value].to_csv(
                        path + "/" + f"{sys_selector.value} loss-of-load event stats.csv"
                    )
                    print("Downloaded. ")
                else:
                    print("No data available. ")

        butt1.on_click(_extract_lolday)
        butt2.on_click(_show_mc_lolperiod)
        butt3.on_click(_downloader)
        buttons = widgets.HBox([butt1, butt2])
        display(widgets.VBox([sys_selector, buttons, butt3, outt]))

    def create_lol_bymc_chart(self, df_plot):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_plot["dayofyear"],
                y=df_plot["MC_draw"],
                mode="markers",
                hovertext=df_plot["date"],
                line=dict(color="#AF2200"),
                opacity=0.8,
            )
        )
        fig.update_layout(
            title_text="LOL day distribution by MC draw",
            xaxis_title="Month",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            width=600,
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
                ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                range=[0, 365],
            ),
            font={"family": "Arial", "size": 12},
            hoverlabel=dict(font_size=10),
            showlegend=False,
        )
        fig.update_xaxes(showline=True, linewidth=1.5, linecolor="black", gridcolor="grey", gridwidth=0.5, mirror=True)
        fig.update_yaxes(showline=True, linewidth=1.5, linecolor="black", gridcolor="grey", gridwidth=0.5, mirror=True)
        fig.show()

    def show_dispatch_days(self):
        sys_dict = {"Tuned System": getattr(self, "tuned_disp"), "Untuned System": getattr(self, "untuned_disp")}

        sys_selector = widgets.RadioButtons(
            options=["Tuned System", "Untuned System"],
            vlaue=None,
            description="Tuned or Untuned system? ",
            style={"description_width": "initial"},
            layout={"width": "300px"},
        )
        mc_selector = widgets.Dropdown(
            vlaue=None,
            description="Select a specific MC draw: ",
            style={"description_width": "initial"},
            layout={"width": "300px"},
        )
        start_date_selector = widgets.DatePicker(
            value=None,
            description="Start Date",
            disabled=False,
            style={"description_width": "initial"},
            layout={"width": "300px"},
        )
        end_date_selector = widgets.DatePicker(
            value=None,
            description="End Date",
            disabled=False,
            style={"description_width": "initial"},
            layout={"width": "300px"},
        )
        dispatch_stack_selector = widgets.RadioButtons(
            vlaue=None, description="Dispatch Type: ", style={"description_width": "initial"}, layout={"width": "300px"}
        )

        @interact(
            sys=sys_selector,
            mc=mc_selector,
            start=start_date_selector,
            end=end_date_selector,
            disp_stack=dispatch_stack_selector,
        )
        def determine_option(sys, mc, start, end, disp_stack):
            mc_selector.options = (
                list(sys_dict[sys].index.get_level_values(0).unique()) if not sys_dict[sys].empty else []
            )
            dispatch_stack_selector.options = ["Economic dispatch", "RECAP dispatch"] if not sys_dict[sys].empty else []

        butt = widgets.Button(
            description="Produce dispatch chart", style={"description_width": "initial"}, layout={"width": "300px"}
        )
        outt = widgets.Output()

        def _create_chart(a):
            with outt:
                clear_output()
                if not sys_dict[sys_selector.value].empty:
                    disp_by_group = self.calc_disp_by_group(sys_dict[sys_selector.value])
                    if dispatch_stack_selector.value == "Economic dispatch":
                        self.disp_plot = DispatchPlot_economic(disp_by_group)
                    else:
                        self.disp_plot = DispatchPlot_RECAPstack(disp_by_group)

                    fig = self.disp_plot.create_dispatch_plot(
                        start_date=start_date_selector.value,
                        end_date=end_date_selector.value,
                        mc_draw=mc_selector.value,
                    )
                    fig.show()
                else:
                    print("No dispatch data available to make dispatch chart.")

        butt.on_click(_create_chart)
        display(widgets.VBox([butt, outt]))

    def calculate_LOLE_uncertainties(self, df_EUE):
        LOL_threshold = 0.05

        # LOLE uncertainty calc

        unserved_energy_per_draw_and_date = (
            (df_EUE > LOL_threshold)
            .groupby(
                [
                    df_EUE.index.get_level_values(dispatch_model._NET_LOAD_MC_DRAW_INDEX_NAME),
                    df_EUE.index.get_level_values(dispatch_model._NET_LOAD_TIMESTAMP_INDEX_NAME).date,
                ]
            )
            .any()
        ) * 1

        unserved_energy_per_draw_and_date.index.names = ["MC_draw", "timestamp"]
        unserved_energy_per_draw_and_date = unserved_energy_per_draw_and_date.reset_index()
        unserved_energy_per_draw_and_date["timestamp"] = pd.to_datetime(unserved_energy_per_draw_and_date["timestamp"])
        unserved_energy_per_draw_and_date = unserved_energy_per_draw_and_date.set_index(
            ["MC_draw", "timestamp"], drop=True
        )

        LOL_events_by_year = (
            unserved_energy_per_draw_and_date.groupby(
                [
                    unserved_energy_per_draw_and_date.index.get_level_values("MC_draw"),
                    unserved_energy_per_draw_and_date.index.get_level_values("timestamp").year,
                ]
            )
            .sum()
            .squeeze()
        )

        reliability_metric_pivot_table = LOL_events_by_year.reset_index().pivot(
            index=["MC_draw"], columns=["timestamp"]
        )
        return reliability_metric_pivot_table

    def calculate_EUE_uncertainties(self, df_EUE):
        unserved_energy_by_year = df_EUE.groupby(
            [
                df_EUE.index.get_level_values(dispatch_model._NET_LOAD_MC_DRAW_INDEX_NAME),
                df_EUE.index.get_level_values(dispatch_model._NET_LOAD_TIMESTAMP_INDEX_NAME).year,
            ]
        ).sum()

        reliability_metric_pivot_table = unserved_energy_by_year.reset_index().pivot(
            index=["MC_draw"], columns=["timestamp"]
        )
        return reliability_metric_pivot_table

    def calculate_weatheryear_peaks(self, load_profile):
        global color_map

        weatheryear_loads = (
            load_profile.groupby(
                [
                    load_profile.index.get_level_values(dispatch_model._NET_LOAD_TIMESTAMP_INDEX_NAME).year,
                ]
            )
            .agg(["max", "idxmax"])
            .reset_index()
            .rename(columns={"max": "Gross load peak", "idxmax": "Peak timestamp"})
        )
        weatheryear_loads["month"] = weatheryear_loads.apply(lambda x: x["Peak timestamp"].month, axis=1)
        weatheryear_loads["season"] = weatheryear_loads.apply(
            lambda x: "summer" if (x["month"] < 10 and x["month"] > 4) else "winter", axis=1
        )

        weatheryear_loads["_color"] = weatheryear_loads.apply(lambda x: color_map[x["season"]], axis=1)
        self.weatheryear_grosspeak = weatheryear_loads

    def do_annual_bootstrap(self, unserved_energy_and_reserve, metric, n_resamples=1000):
        def calculate_metric(unserved_energy_and_reserve):
            return RecapCase.calculate_reliability(unserved_energy_and_reserve, metric=metric)

        # Calculate metric for every year
        metric_by_year = unserved_energy_and_reserve.groupby(
            [
                unserved_energy_and_reserve.index.get_level_values(dispatch_model._NET_LOAD_MC_DRAW_INDEX_NAME),
                unserved_energy_and_reserve.index.get_level_values(dispatch_model._NET_LOAD_TIMESTAMP_INDEX_NAME).year,
            ]
        ).apply(calculate_metric)

        # Do bootstrap
        result = bootstrap(metric_by_year.values.reshape(1, -1), np.mean, n_resamples=n_resamples)

        return result

    def calculate_bootstrap_statistics(self, metric, verbose=False):
        # Get mean + confidence interval
        mean = self.bootstrap_result.bootstrap_distribution.mean()
        CI = self.bootstrap_result.confidence_interval
        LCB, UCB = CI.low, CI.high
        if verbose:
            print(f"{metric.name}: {LCB:.2f} - {mean:.2f} - {UCB:.2f}")
        return LCB, mean, UCB

    def draw_tuned_PCAP(self):
        dispatch_mode = self.case_settings.loc["dispatch_mode", "value"]
        metric = self.case_settings.loc["target_metric"]

        capacity_shortfall = self.ELCC_results.loc["base_case", "perfect_capacity_shortfall"]
        bisection_xtol = float(self.case_settings.loc["bisection_xtol"])

        reliability_df = self.reliability_results.copy(deep=True)
        reliability_df = reliability_df.loc[
            (reliability_df["dispatch_mode"] == dispatch_mode.upper())
            & (reliability_df["heuristic_dispatch_mode"] == "DEFAULT")
        ][metric]

        # Get last 5 (?? arbitrary) points? or, perhaps should get points within 8 * bisection_xtol of capacity shortfall
        reliability_df = reliability_df.loc[
            (reliability_df.index >= capacity_shortfall - 8 * bisection_xtol)
            & (reliability_df.index <= capacity_shortfall + 8 * bisection_xtol)
        ]

        beta = np.polyfit(reliability_df.index, reliability_df.values, 1)
        # uncertainty_coeff = abs(1 / beta[0])
        # capacity_shortfall_uncertainty = uncertainty_coeff * LOLE_SE  # Change to EUE SE as needed

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=reliability_df.index, y=reliability_df.values, mode="markers"))
        fig.add_trace(go.Scatter(x=reliability_df.index, y=beta[0] * reliability_df.index + beta[1], mode="lines"))
        fig.add_vline(x=capacity_shortfall, line_width=2, line_color="black")
        fig.update_layout(
            xaxis_title="Perfect Capacity (MW)",
            yaxis_title=metric.values[0],
            height=400,
            width=600,
            xaxis_tickformat=",d",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Arial", "size": 12},
            showlegend=True,
        )
        fig.show()

    def draw_weatheryear_metrics(self, metric_flag):
        fig = go.Figure()

        fig = make_subplots(
            rows=2, cols=1, subplot_titles=("Weather Year Gross Load Peak", "Range of Reliability by Weather Year")
        )

        fig.add_trace(
            go.Bar(
                x=self.weatheryear_grosspeak["timestamp"].values,  # weather years
                y=self.weatheryear_grosspeak["Gross load peak"].values,  # gross load peak
                marker_color=list(self.weatheryear_grosspeak["_color"].values),  # based on peaking season
                hovertext=[
                    pd.to_datetime(t).strftime("%b-%d %Y, %H:%M %p")
                    for t in self.weatheryear_grosspeak["Peak timestamp"].values
                ],
                name="Gross load peak",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=self.reliability_metric_pivot_table.columns.get_level_values(1).values,  # weather years
                y=self.reliability_metric_pivot_table.mean().values,  # average EUE/LOLE for year across each mc draw
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=self.reliability_metric_pivot_table.max().values
                    - self.reliability_metric_pivot_table.mean().values,
                    arrayminus=self.reliability_metric_pivot_table.mean().values
                    - self.reliability_metric_pivot_table.min().values,
                ),  # range of EUE/LOLE by mc draw
                marker_color="grey",
                name="Reliability Metric by year",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            xaxis=dict(
                title="Weather Year",
                tickmode="array",
                tickvals=self.reliability_metric_pivot_table.columns.get_level_values(1).values,
            ),
            yaxis=dict(title="Gross Load Peak", tickformat=",d"),
            xaxis2=dict(
                title="Weather Year",
                tickmode="array",
                tickvals=self.reliability_metric_pivot_table.columns.get_level_values(1).values,
            ),
            yaxis2=dict(title=metric_flag, tickformat=",d"),
            height=600,
            width=800,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Arial", "size": 12},
            showlegend=True,
        )
        fig.show()

    def draw_bootstrap_sampling_distribution(self, LCB, mean, UCB):
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=self.bootstrap_result.bootstrap_distribution,
                nbinsx=30,
                histnorm="probability",
                marker_color="#034E6E",
                opacity=0.75,
            )
        )

        fig.add_vline(
            x=mean,
            annotation_text=f"Bootstrap Mean = {mean:.2f}",
            annotation_position="top right",
            annotation_font_size=12,
            annotation_font_color="black",
            line_color="#034E6E",
        )
        fig.add_vline(
            x=LCB,
            annotation_text=f"95% Confidence Interval Width = {UCB - LCB:.2f}",
            annotation_position="bottom right",
            annotation_font_size=12,
            annotation_font_color="black",
            line_color="#FF8700",
            line_dash="dash",
        )
        fig.add_vline(x=UCB, line_color="#FF8700", line_dash="dash")

        fig.update_layout(
            width=600,
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Arial", "size": 12},
            showlegend=False,
            yaxis={"showticklabels": False},
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor="grey", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="grey", mirror=True)
        fig.show()

    def show_results_uncertainty(self):
        sys_dict = {"Tuned System": getattr(self, "tuned_disp"), "Untuned System": getattr(self, "untuned_disp")}

        sys_selector = widgets.RadioButtons(
            options=["Tuned System", "Untuned System"],
            vlaue=None,
            description="Tuned or Untuned system? ",
            style={"description_width": "initial"},
            layout={"width": "300px"},
        )
        metric_selector = widgets.RadioButtons(
            options=["EUE", "LOLE"],
            vlaue=None,
            description="Select a metric:",
            style={"description_width": "initial"},
            layout={"width": "300px"},
        )
        butt1 = widgets.Button(description="Show metric by MC draw/Weather Year", layout={"width": "300px"})
        butt2 = widgets.Button(description="Download csv file", layout={"width": "300px"})
        butt3 = widgets.Button(description="Visualize Weather Year metrics", layout={"width": "300px"})

        outt = widgets.Output()

        def action(a):
            with outt:
                clear_output()
                if not sys_dict[sys_selector.value].empty:
                    if metric_selector.value == "EUE":
                        self.reliability_metric_pivot_table = self.calculate_EUE_uncertainties(
                            sys_dict[sys_selector.value]["unserved_energy_and_reserve"]
                        )
                    else:
                        self.reliability_metric_pivot_table = self.calculate_LOLE_uncertainties(
                            sys_dict[sys_selector.value]["unserved_energy_and_reserve"]
                        )
                    pd.set_option("display.max_rows", len(self.reliability_metric_pivot_table))
                    display(self.reliability_metric_pivot_table)
                else:
                    print(f"No data available to calculate uncertainty for {sys_selector.value}, please double check.")

        def _downloader(b):
            with outt:
                clear_output()
                if not self.reliability_metric_pivot_table.empty:
                    path = "../analysis/Result Inspection/" + self.case_name
                    if not os.path.exists(path):
                        os.mkdir(path)
                    self.reliability_metric_pivot_table.to_csv(
                        path + "/" + f"{sys_selector.value}_{metric_selector.value}_byyear_and_mcdraw.csv"
                    )
                    print("Downloaded. ")
                else:
                    print("No data available. ")

        def visual_metric(c):
            with outt:
                clear_output()
                if not sys_dict[sys_selector.value].empty:
                    loads = sys_dict[sys_selector.value]["gross_load"].droplevel(level=0)  # discard MC draw index
                    self.calculate_weatheryear_peaks(load_profile=loads)
                    try:
                        self.draw_weatheryear_metrics(metric_flag=metric_selector.value)
                    except AttributeError as e:
                        print(
                            "Please create the reliability metrics table first by clicking "
                            '"Show metric by MC draw/Weather Year" button if data are available.'
                        )

                else:
                    print("No data available. ")

        butt1.on_click(action)
        butt2.on_click(_downloader)
        butt3.on_click(visual_metric)
        buttons = widgets.HBox([butt1, butt2])
        display(widgets.VBox([sys_selector, metric_selector, buttons, butt3, outt]))

    def show_bootstrapping_metric(self):
        sys_dict = {"Tuned System": getattr(self, "tuned_disp"), "Untuned System": getattr(self, "untuned_disp")}

        sys_selector = widgets.RadioButtons(
            options=["Tuned System", "Untuned System"],
            vlaue=None,
            description="Tuned or Untuned system? ",
            style={"description_width": "initial"},
            layout={"width": "300px"},
        )
        metric_selector = widgets.RadioButtons(
            options=list(ReliabilityMetric),
            vlaue=None,
            description="Select a metric:",
            style={"description_width": "initial"},
            layout={"width": "300px"},
        )
        butt1 = widgets.Button(description="Show metric distribution", layout={"width": "300px"})
        outt = widgets.Output()

        def action(a):
            with outt:
                clear_output()
                if not sys_dict[sys_selector.value].empty:
                    self.bootstrap_result = self.do_annual_bootstrap(
                        sys_dict[sys_selector.value]["unserved_energy_and_reserve"], metric_selector.value
                    )
                    LCB, mean, UCB = self.calculate_bootstrap_statistics(metric_selector.value, verbose=True)
                    self.draw_bootstrap_sampling_distribution(LCB, mean, UCB)
                else:
                    print(f"No data available to conduct bootstrapping for {sys_selector.value}, please double check.")

        butt1.on_click(action)
        display(widgets.VBox([sys_selector, metric_selector, butt1, outt]))


class DispatchPlot_RECAPstack:
    global color_map
    plot_colors = color_map.copy()

    def __init__(self, dispatch_results: pd.DataFrame, color_mapping=None):
        self._min_start_date = dispatch_results.index.get_level_values(1).min()
        self._max_end_date = dispatch_results.index.get_level_values(1).max()

        # load and reserves data
        for attr, col, new_name in [
            ("gross_load", "gross_load", "Gross Load"),
            ("reserves", "reserves", "Reserve Requirement"),
            ("perfect_capacity", "perfect_capacity", "Perfect Capacity"),
            ("unserved_energy_and_reserve", "unserved_energy_and_reserve", "Unserved Energy and Reserve"),
            ("thermal", "Thermal group", "Thermal"),
            ("nuclear", "Nuclear group", "Nuclear"),
            ("other", "Other group", "Other"),
            ("imports", "Imports group", "Imports"),
            ("LBW", "LBW group", "LBW"),
            ("OSW", "OSW group", "OSW"),
            ("solar", "Solar group", "Solar"),
            ("hydro", "Hydro group", "Hydro"),
            ("dr", "DR group", "DR"),
            ("flexload", "FlexLoad group", "FlexLoad"),
            ("storage", "Storage group", "Storage"),
            ("LDES", "LDES group", "LDES"),
        ]:
            if col in dispatch_results.columns:
                setattr(self, attr, dispatch_results.loc[:, col].rename(new_name))
            else:
                setattr(self, attr, pd.Series(np.zeros(len(dispatch_results)), index=dispatch_results.index))

        setattr(
            self,
            "gross_load_plus_reserves",
            (self.gross_load + self.reserves - self.perfect_capacity).rename("Gross Load + Reserves"),
        )

        self.gross_load = self.gross_load - self.perfect_capacity.values

        plot_resources = [
            self.thermal,
            self.nuclear,
            self.other,
            self.imports,
            self.solar,
            self.LBW,
            self.OSW,
            self.hydro,
            self.storage,
            self.LDES,
            self.dr,
            self.flexload,
            self.unserved_energy_and_reserve,
            self.gross_load,
            self.gross_load_plus_reserves,
        ]
        plot_resources = [resource for resource in plot_resources if resource.sum() != 0]
        self._plot_data = pd.concat(plot_resources, axis=1)

        if color_mapping is not None:
            self.plot_colors = color_mapping

        self._figure = self._initialize_figure()

    @property
    def _line_trace_names(self):
        return ["Gross Load", "Gross Load + Reserves"]

    def _initialize_figure(self):
        if "Storage" in self._plot_data.columns.tolist():
            self._plot_data["Storage"] = self._plot_data["Storage"].clip(lower=0, axis=0)
        if "LDES" in self._plot_data.columns.tolist():
            self._plot_data["LDES"] = self._plot_data["LDES"].clip(lower=0, axis=0)

        # RL: a very temporary fix for showing storage providing reserves
        up_reserve = (
            self._plot_data["Gross Load + Reserves"] - self._plot_data.drop(self._line_trace_names, axis=1).sum(axis=1)
        ).clip(lower=0)
        if "LDES" in self._plot_data.columns.tolist():
            self._plot_data["LDES"] = self._plot_data["LDES"] + up_reserve

        fig = go.Figure(layout=dict(template="plotly_white", height=600, width=1500))

        for i, col in enumerate(self._plot_data.columns):
            curr_trace = go.Scatter(
                x=[],
                y=[],
                name=col,
                mode="lines" if col in self._line_trace_names else None,
                line=dict(
                    color=self.plot_colors[col],
                    dash="dot" if col == "Gross Load" else None,
                    # shape="hv" if col == "Unserved Energy and Reserve" else "linear",
                ),
            )
            if col not in self._line_trace_names:
                curr_trace.line.width = 0.25
                curr_trace.line.color = "white"
                curr_trace.stackgroup = "one"
                curr_trace.fill = "tonexty" if i else "tozeroy"
            fig.add_trace(curr_trace)

        return fig

    def _get_plot_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp, mc_draw: str) -> pd.DataFrame:
        return self._plot_data.xs(mc_draw, level="MC_draw").loc[start_date:end_date]

    def _update_plot_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp, mc_draw: str):
        plot_data = self._get_plot_data(start_date=start_date, end_date=end_date, mc_draw=mc_draw)

        for col in plot_data:
            self._figure.update_traces(
                selector=dict(name=col),
                patch=dict(
                    x=plot_data.loc[start_date:end_date, col].index,
                    y=plot_data.loc[start_date:end_date, col].values,
                    fillcolor=self.plot_colors[col],
                ),
                overwrite=True,
            )

    def create_dispatch_plot(self, start_date: str = None, end_date: str = None, mc_draw: str = None):
        if start_date is None:
            start_date = self._min_start_date
        if end_date is None:
            end_date = self._max_end_date
        if mc_draw is None:
            mc_draw = self._plot_data.index.unique(level="MC_draw")[0]
        self._update_plot_data(start_date=pd.Timestamp(start_date), end_date=pd.Timestamp(end_date), mc_draw=mc_draw)
        self._figure.update_xaxes(range=[start_date, end_date], ticklabelmode="period")
        self._figure.update_layout(
            xaxis_title="Weather Date",
            yaxis_title="Load or Generation (MWh)",
            yaxis_tickformat=",d",
            font_family="Arial",
        )

        return self._figure


class DispatchPlot_economic:
    global color_map
    plot_colors = color_map.copy()

    def __init__(self, dispatch_results: pd.DataFrame, color_mapping=None):
        self._min_start_date = dispatch_results.index.get_level_values(1).min()
        self._max_end_date = dispatch_results.index.get_level_values(1).max()

        # load and reserves data
        for attr, col, new_name in [
            ("gross_load", "gross_load", "Gross Load"),
            ("reserves", "reserves", "Reserve Requirement"),
            ("perfect_capacity", "perfect_capacity", "Perfect Capacity"),
            ("unserved_energy_and_reserve", "unserved_energy_and_reserve", "Unserved Energy and Reserve"),
            ("thermal", "Thermal group", "Thermal"),
            ("nuclear", "Nuclear group", "Nuclear"),
            ("other", "Other group", "Other"),
            ("imports", "Imports group", "Imports"),
            ("LBW", "LBW group", "LBW"),
            ("OSW", "OSW group", "OSW"),
            ("solar", "Solar group", "Solar"),
            ("hydro", "Hydro group", "Hydro"),
            ("dr", "DR group", "DR"),
            ("flexload", "FlexLoad group", "FlexLoad"),
            ("storage", "Storage group", "Storage"),
            ("LDES", "LDES group", "LDES"),
        ]:
            if col in dispatch_results.columns:
                setattr(self, attr, dispatch_results.loc[:, col].rename(new_name))
            else:
                setattr(self, attr, pd.Series(np.zeros(len(dispatch_results)), index=dispatch_results.index))

        setattr(
            self, "firm_resources", (self.nuclear + self.thermal + self.imports + self.other).rename("Firm Resources")
        )

        setattr(self, "total_variable", (self.LBW + self.OSW + self.solar).rename("Variable Resources"))
        setattr(self, "variable_plus_hydro", (self.total_variable + self.hydro).rename("Variable plus Hydro"))

        setattr(
            self,
            "gross_load_plus_reserves",
            (self.gross_load + self.reserves - self.perfect_capacity).rename("Gross Load + Reserves"),
        )

        self.gross_load = self.gross_load - self.perfect_capacity.values

        plot_resources = [
            self.solar,
            self.LBW,
            self.OSW,
            self.hydro,
            self.variable_plus_hydro,
            self.storage,
            self.LDES,
            self.dr,
            self.flexload,
            self.firm_resources,
            self.unserved_energy_and_reserve,
            self.gross_load,
            self.gross_load_plus_reserves,
        ]
        plot_resources = [resource for resource in plot_resources if resource.sum() != 0]
        self._plot_data = pd.concat(plot_resources, axis=1)

        if color_mapping is not None:
            self.plot_colors = color_mapping

        self._figure = self._initialize_figure()

    @property
    def _line_trace_names(self):
        return ["Gross Load", "Gross Load + Reserves", "Variable plus Hydro"]

    def _initialize_figure(self):
        fig = go.Figure(layout=dict(template="plotly_white", height=600, width=1500))

        self._plot_data["Storage"] = self._plot_data["Storage"].clip(lower=0, axis=0)
        self._plot_data["LDES"] = self._plot_data["LDES"].clip(lower=0, axis=0)

        self._plot_data["Firm Resources"] = self._plot_data["Firm Resources"].clip(
            upper=self._plot_data["Gross Load + Reserves"]
            - self._plot_data.drop(self._line_trace_names + ["Firm Resources"], axis=1).sum(axis=1),
            axis=0,
        )
        self._plot_data["Firm Resources"] = self._plot_data["Firm Resources"].clip(lower=0, axis=0)
        self._plot_data["Variable plus Hydro"] = self._plot_data["Variable plus Hydro"].clip(
            lower=self._plot_data["Gross Load + Reserves"], axis=0
        )

        # RL: a very temporary fix for showing storage providing reserves
        up_reserve = (
            self._plot_data["Gross Load + Reserves"] - self._plot_data.drop(self._line_trace_names, axis=1).sum(axis=1)
        ).clip(lower=0)
        self._plot_data["LDES"] = self._plot_data["LDES"] + up_reserve

        for i, col in enumerate(self._plot_data.columns):
            curr_trace = go.Scatter(
                x=[],
                y=[],
                name=col,
                mode="lines" if col in self._line_trace_names else None,
                line=dict(
                    color=self.plot_colors[col],
                    dash="dot" if col == "Gross Load" else "dash" if col == "Variable plus Hydro" else None,
                    # shape="hv" if col == "Unserved Energy and Reserve" else "linear",
                ),
            )
            if col not in self._line_trace_names:
                curr_trace.line.width = 0.25
                curr_trace.line.color = "white"
                curr_trace.stackgroup = "one"
                curr_trace.fill = "tonexty" if i else "tozeroy"
            fig.add_trace(curr_trace)

        return fig

    def _get_plot_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp, mc_draw: str) -> pd.DataFrame:
        return self._plot_data.xs(mc_draw, level="MC_draw").loc[start_date:end_date]

    def _update_plot_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp, mc_draw: str):
        plot_data = self._get_plot_data(start_date=start_date, end_date=end_date, mc_draw=mc_draw)

        for col in plot_data:
            self._figure.update_traces(
                selector=dict(name=col),
                patch=dict(
                    x=plot_data.loc[start_date:end_date, col].index,
                    y=plot_data.loc[start_date:end_date, col].values,
                    fillcolor=self.plot_colors[col],
                ),
                overwrite=True,
            )

    def create_dispatch_plot(self, start_date: str = None, end_date: str = None, mc_draw: str = None):
        if start_date is None:
            start_date = self._min_start_date
        if end_date is None:
            end_date = self._max_end_date
        if mc_draw is None:
            mc_draw = self._plot_data.index.unique(level="MC_draw")[0]
        self._update_plot_data(start_date=pd.Timestamp(start_date), end_date=pd.Timestamp(end_date), mc_draw=mc_draw)
        self._figure.update_xaxes(range=[start_date, end_date], ticklabelmode="period")
        self._figure.update_layout(
            xaxis_title="Weather Date",
            yaxis_title="Load or Generation (MWh)",
            yaxis_tickformat=",d",
            font_family="Arial",
        )

        return self._figure
