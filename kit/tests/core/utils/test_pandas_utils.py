import pandas as pd
import pytest  # noqa

from new_modeling_toolkit.core.utils.pandas_utils import convert_index_levels_to_datetime


def test_convert_index_levels_to_datetime_dataframe():
    input_frame_1 = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [
                ("2021-01-01 00:00", "resource_1"),
                ("2021-01-01 00:00", "resource_2"),
                ("2025-01-01 00:00", "resource_1"),
                ("2025-01-01 00:00", "resource_2"),
            ],
        ),
        data={"x": [1, 2, 3, 4]},
    )
    pd.testing.assert_frame_equal(
        convert_index_levels_to_datetime(pandas_object=input_frame_1, levels=0),
        pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp(year=2021, month=1, day=1, hour=0), "resource_1"),
                    (pd.Timestamp(year=2021, month=1, day=1, hour=0), "resource_2"),
                    (pd.Timestamp(year=2025, month=1, day=1, hour=0), "resource_1"),
                    (pd.Timestamp(year=2025, month=1, day=1, hour=0), "resource_2"),
                ],
            ),
            data={"x": [1, 2, 3, 4]},
        ),
    )

    input_frame_2 = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [
                (2021, "resource_1"),
                (2025, "resource_2"),
                (2021, "resource_1"),
                (2025, "resource_2"),
            ],
        ),
        data={"x": [1, 2, 3, 4]},
    )
    pd.testing.assert_frame_equal(
        convert_index_levels_to_datetime(pandas_object=input_frame_2, levels=0, format="%Y"),
        pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp(year=2021, month=1, day=1, hour=0), "resource_1"),
                    (pd.Timestamp(year=2025, month=1, day=1, hour=0), "resource_2"),
                    (pd.Timestamp(year=2021, month=1, day=1, hour=0), "resource_1"),
                    (pd.Timestamp(year=2025, month=1, day=1, hour=0), "resource_2"),
                ],
            ),
            data={"x": [1, 2, 3, 4]},
        ),
    )

    input_frame_3 = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [
                (2021, "resource_1", 2030),
                (2025, "resource_2", 2030),
                (2021, "resource_1", 2030),
                (2025, "resource_2", 2030),
            ],
        ),
        data={"x": [1, 2, 3, 4]},
    )
    pd.testing.assert_frame_equal(
        convert_index_levels_to_datetime(pandas_object=input_frame_3, levels=[0, 2], format="%Y"),
        pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [
                    (
                        pd.Timestamp(year=2021, month=1, day=1, hour=0),
                        "resource_1",
                        pd.Timestamp(year=2030, month=1, day=1, hour=0),
                    ),
                    (
                        pd.Timestamp(year=2025, month=1, day=1, hour=0),
                        "resource_2",
                        pd.Timestamp(year=2030, month=1, day=1, hour=0),
                    ),
                    (
                        pd.Timestamp(year=2021, month=1, day=1, hour=0),
                        "resource_1",
                        pd.Timestamp(year=2030, month=1, day=1, hour=0),
                    ),
                    (
                        pd.Timestamp(year=2025, month=1, day=1, hour=0),
                        "resource_2",
                        pd.Timestamp(year=2030, month=1, day=1, hour=0),
                    ),
                ],
            ),
            data={"x": [1, 2, 3, 4]},
        ),
    )

    input_frame_3_with_index_names = input_frame_3.rename_axis(index=("vintage", "resource", None))
    pd.testing.assert_frame_equal(
        convert_index_levels_to_datetime(
            pandas_object=input_frame_3_with_index_names, levels=["vintage", 2], format="%Y"
        ),
        pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [
                    (
                        pd.Timestamp(year=2021, month=1, day=1, hour=0),
                        "resource_1",
                        pd.Timestamp(year=2030, month=1, day=1, hour=0),
                    ),
                    (
                        pd.Timestamp(year=2025, month=1, day=1, hour=0),
                        "resource_2",
                        pd.Timestamp(year=2030, month=1, day=1, hour=0),
                    ),
                    (
                        pd.Timestamp(year=2021, month=1, day=1, hour=0),
                        "resource_1",
                        pd.Timestamp(year=2030, month=1, day=1, hour=0),
                    ),
                    (
                        pd.Timestamp(year=2025, month=1, day=1, hour=0),
                        "resource_2",
                        pd.Timestamp(year=2030, month=1, day=1, hour=0),
                    ),
                ],
                names=("vintage", "resource", None),
            ),
            data={"x": [1, 2, 3, 4]},
        ),
    )
