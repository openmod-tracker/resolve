from __future__ import annotations

import importlib
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import xlwings as xw
import yaml
from loguru import logger
from upath import UPath

from new_modeling_toolkit.core.custom_model import CustomModel
from new_modeling_toolkit.core.custom_model import ModelType
from new_modeling_toolkit.core.utils.util import DirStructure


class SectionToWrite(CustomModel):
    """Sheets are sub-divided into sections separated by Header-formatted rows."""

    level: int = 1
    sections: list[SectionToWrite] = []
    components: list[str] = []

    def construct(
        self,
        *,
        sheet: xw.Sheet,
        anchor_range: xw.Range,
        modeled_years: list[Any],
        component_tables: list[str],
        add_doc_hyperlinks: bool,
    ) -> tuple[int, int]:
        logger.debug(f"----Constructing section: {self.name}")

        section_header = sheet.range(anchor_range, anchor_range.end("right"))
        section_header.api.style_object.set(sheet.book.api.styles[f"Heading {self.level}"])
        section_header.resize(1, 1).value = self.name

        column_offset = 0
        row_offset = 4
        for section in self.sections:
            rows_to_add, columns_to_add = section.construct(
                sheet=sheet,
                anchor_range=anchor_range.offset(row_offset=row_offset, column_offset=column_offset),
                modeled_years=modeled_years,
            )
            row_offset += rows_to_add
            column_offset += 0

        row_offset += 3
        for name in self.components:
            module_name, class_name = name.rsplit(".", 1)
            component = getattr(importlib.import_module(f"new_modeling_toolkit.{module_name}"), class_name)

            rows_to_add, columns_to_add, table_name = component.to_excel(
                anchor_range=anchor_range.offset(row_offset=row_offset, column_offset=column_offset),
                section=self.name,
                modeled_years=modeled_years,
                add_doc_hyperlinks=add_doc_hyperlinks,
            )
            row_offset += rows_to_add
            column_offset += 0
            component_tables += [table_name]

        return row_offset + 2, 0, component_tables


class SheetToWrite(CustomModel):
    title: str
    sections: list[SectionToWrite] = []

    def __init__(self, **kwargs):
        """I don't know how to create the recursive heading level, so I'm manually assigning the level (which affects the heading text formatting)"""
        super().__init__(**kwargs)

        if self.title is None:
            self.title = self.name

        for section in self.sections:
            for subsection in section.sections:
                subsection.level = 2
                for subsubsection in subsection.sections:
                    subsubsection.level = 3

    def construct(
        self, *, wb: xw.Book, modeled_years: list[Any], component_tables: list[str], add_doc_hyperlinks: bool
    ):
        logger.info(f"Constructing sheet: {self.title}")
        try:
            wb.sheets.add(self.name)
            wb.app.api.active_window.display_gridlines.set(False)
        except ValueError:
            pass

        wb.sheets[self.name].range("1:1").api.style_object.set(wb.api.styles["Title"])

        wb.sheets[self.name].range("A1").value = self.title

        column_offset = 0
        row_offset = 0
        for section in self.sections:
            rows_to_add, columns_to_add, table_names = section.construct(
                sheet=wb.sheets[self.name],
                anchor_range=wb.sheets[self.name].range("B5").offset(row_offset=row_offset),
                modeled_years=modeled_years,
                component_tables=component_tables,
                add_doc_hyperlinks=add_doc_hyperlinks,
            )
            row_offset += rows_to_add
            column_offset += 0

        return component_tables


class ExcelTemplate(CustomModel):
    template_path: str
    modeled_years: list[Any] = list(range(2030, 2046))
    sheets: list[SheetToWrite]
    component_tables: list[str] = []
    add_doc_hyperlinks: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.modeled_years = pd.to_datetime([f"1/1/{year}" for year in self.modeled_years]).tolist()

    def construct(self):
        """Construct a Scenario Tool"""
        with xw.App(visible=False) as app:
            app.calculation = "manual"
            app.screen_updating = False

            wb = xw.Book(self.template_path)
            wb.save(f"{self.name}.xlsm")

            wb.sheets["Cover"].range("A1").value = self.name

            for sheet in self.sheets:
                self.component_tables = sheet.construct(
                    wb=wb,
                    modeled_years=self.modeled_years,
                    component_tables=self.component_tables,
                    add_doc_hyperlinks=self.add_doc_hyperlinks,
                )

            wb.sheets["Lists"].range("System.ComponentTableFirst").options(transpose=True).value = self.component_tables

            wb.save(pathlib.Path(self.template_path).parent.absolute() / f"{self.name}.xlsm")
            wb.close()

        logger.info("Done!")


class ScenarioTool(CustomModel):
    book: xw.Book

    def __init__(self, *, book: str):
        book = xw.Book(UPath(book))
        name = book.name
        super().__init__(name=name, book=book)

        # TODO 2024-04-30: Clear all filters only once

    @classmethod
    def from_template(cls, *, name: str, schema_file: UPath | None = None, add_doc_hyperlinks: bool):
        if schema_file is None:
            schema_file = UPath(__file__).parent / "ui-config.yml"

        with open(schema_file, "r") as config_file:
            yml_dict = yaml.safe_load(config_file)

            st = ExcelTemplate(name=name, **yml_dict, add_doc_hyperlinks=add_doc_hyperlinks)
            st.construct()

    def export_system(self, model: ModelType):
        # TODO 2024-04-29: Add a version restriction and deprecation warning
        sheet = self.book.sheets[f"{model.value} Case Setup"]
        dir_str = DirStructure(data_folder=sheet.range(f"{model.value}.__DATAFOLDER__").value)
        system_save_path = dir_str.data_interim_dir / "systems" / sheet.range(f"{model.value}.system").value
        system_save_path.mkdir(exist_ok=True, parents=True)

        component_tables = {}
        for sheet in self.book.sheets:
            for table in sheet.tables:
                if table.name in self.book.sheets["Lists"].range("System.ComponentTable").value:
                    class_name = table.name.split(".")[-1]
                    if class_name not in component_tables:
                        component_tables[class_name] = []
                    component_tables[class_name] += [table]

        all_linkages = []
        all_components = []
        for name, components in component_tables.items():
            data = pd.concat(
                [
                    tbl.range.offset(row_offset=-3)
                    .resize(row_size=tbl.range.shape[0] + 3)
                    .options(pd.DataFrame, index=0, header=4)
                    .value
                    for tbl in components
                ],
                axis=0,
                ignore_index=True,
            ).dropna(how="all")

            # If table is empty, skip
            if data.empty:
                continue

            data = data.dropna(how="all").melt(
                id_vars=[
                    ("name", np.nan, np.nan, "Name"),
                    ("scenarios", np.nan, "Units", "Data Scenario Tag"),
                ],
                ignore_index=True,
            )
            data.columns = ["name", "scenario", "attribute", "", "units", "timestamp", "value"]
            data = data.dropna(subset=["name", "scenario"], how="all")

            # Coerce the timestamps & drop unneeded columns
            if data["attribute"].str.contains("monthly_price_multiplier").any():
                tmp = data.loc[data["attribute"] == "monthly_price_multiplier", "timestamp"].copy()
                tmp = pd.to_datetime(tmp, format="%b")
                tmp = tmp.apply(lambda x: x.replace(year=2000))

            data["timestamp"] = pd.to_datetime(
                data["timestamp"].str.split(" ", expand=True).iloc[:, 0], format="%Y", errors="coerce"
            )
            if data["attribute"].str.contains("monthly_price_multiplier").any():
                data.loc[data["attribute"] == "monthly_price_multiplier", "timestamp"] = tmp

            data = data[["name", "attribute", "timestamp", "value", "scenario"]]
            data = data.dropna(subset="value")

            # Split out linkages to linkages.csv
            # TODO 2024-04-30: This is a pretty weak implementation, but should work for now until `Linkage` is replaced.
            linkage_names_in_excel = data.loc[
                (data["attribute"].str.endswith(".after")) | (data["attribute"].str.endswith(".before")), "attribute"
            ].unique()
            linkages = data.loc[data["attribute"].isin(linkage_names_in_excel), :].copy()

            if not linkages.empty:
                linkage_keys = linkages["attribute"].str.split(".", expand=True)
                linkages.loc[:, "linkage"] = linkage_keys.iloc[:, 0]
                linkages["component_from"] = np.where(
                    linkage_keys.iloc[:, 1] == "before", linkages["value"], linkages["name"]
                )
                linkages["component_to"] = np.where(
                    linkage_keys.iloc[:, 1] == "after", linkages["value"], linkages["name"]
                )
                linkages = linkages[["linkage", "component_from", "component_to", "scenario"]]
                all_linkages.append(linkages)

            # Print out each component CSV
            data = data.loc[~data["attribute"].isin(linkage_names_in_excel), :]
            instances = data.groupby(data["name"])

            component_class = getattr(importlib.import_module(f"new_modeling_toolkit.system"), name)
            logger.info(f"Exporting {component_class.__name__}")
            component_class.dfs_to_csv(instances=instances, wb=self.book, dir_str=dir_str)

            # Split out system include flag
            components_to_include = data.loc[data["attribute"] == "include", ["name", "scenario", "value"]].copy()
            components_to_include.columns = ["instance", "scenario", "include"]
            components_to_include["component"] = component_class.__name__
            components_to_include = components_to_include[["component", "instance", "scenario", "include"]]
            all_components.append(components_to_include)

        # Print out system components.csv
        logger.info("Saving System")
        components_df = pd.concat(all_components, axis=0, ignore_index=True)

        components_df.to_csv(system_save_path / "components.csv", index=False)

        # Print out linkages.csv
        linkages_df = pd.concat(all_linkages, axis=0, ignore_index=True)
        linkages_df.to_csv(system_save_path / "linkages.csv", index=False)
