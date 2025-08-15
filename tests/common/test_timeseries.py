import calendar

import numpy as np
import pandas as pd
import pytest

from new_modeling_toolkit.core.temporal.timeseries import NumericTimeseries
from new_modeling_toolkit.core.temporal.timeseries import Timeseries


@pytest.fixture(scope="session")
def csv_file(tmpdir_factory):
    """Create a temporary CSV file for testing CSV reading.

    Args:
        tmpdir_factory: pytest tmpdir_factory, which creates fixtures accessible for entire pytest session
            See more here: https://docs.pytest.org/en/stable/tmpdir.html#the-tmpdir-factory-fixture

    Returns: Filepath to temporary timeseries CSV

    """
    file = pd.DataFrame(
        {"data": {t: 1.0 for t in pd.date_range(start="1/1/2020 0:00", end="12/31/2020 23:00", freq="1h")}}
    )
    filename = tmpdir_factory.mktemp("data").join("timeseries.csv")
    file.to_csv(filename)
    return filename


def test_from_csv(csv_file):
    """Test timeseries CSV is read from file correctly.
        1. Test CSV is read as a pandas DataFrame
        2. Test shape of DataFrame
        3. Test DataFrame index type

    Args:
        csv_file: pytest temporary CSV fixture

    Returns:

    """
    ts = Timeseries.from_csv("test", csv_file)
    assert isinstance(ts.data, pd.Series)
    assert ts.data.shape == (8784,)
    assert isinstance(ts.data.index, pd.DatetimeIndex)


def test_resample_simple_extend_years():
    # Test 1: exactly 1 year of input data
    ts = NumericTimeseries(
        name="test_data",
        data=pd.Series(
            name="test_data",
            index=pd.date_range(start="2019-01-01 00:00", end="2019-12-31 23:00", freq="H"),
            data=list(range(8760)),
        ),
    )

    ts.resample_simple_extend_years((2020, 2030))

    test_data = pd.Series(
        name="test_data", index=pd.date_range(start="2020-01-01 00:00", end="2030-12-31 23:00", freq="H")
    )
    for year in set(test_data.index.year):
        if calendar.isleap(year):
            test_data.loc[test_data.index.year == year] = list(range(8760)) + list(range(24))
        else:
            test_data.loc[test_data.index.year == year] = list(range(8760))

    pd.testing.assert_series_equal(ts.data, test_data)

    # Test 2: less than 1 year of input data
    ts = NumericTimeseries(
        name="test_data",
        data=pd.Series(
            name="test_data",
            index=pd.date_range(start="2019-01-01 00:00", end="2019-10-31 23:00", freq="H"),
            data=list(range(7296)),
        ),
    )

    ts.resample_simple_extend_years((2020, 2030))

    test_data = pd.Series(
        name="test_data", index=pd.date_range(start="2020-01-01 00:00", end="2030-12-31 23:00", freq="H")
    )
    for year in set(test_data.index.year):
        if calendar.isleap(year):
            test_data.loc[test_data.index.year == year] = list(range(7296)) + list(range(1488))
        else:
            test_data.loc[test_data.index.year == year] = list(range(7296)) + list(range(1464))

    pd.testing.assert_series_equal(ts.data, test_data)

    # Test 3: more than 1 year of input data
    ts = NumericTimeseries(
        name="test_data",
        data=pd.Series(
            name="test_data",
            index=pd.date_range(start="2019-01-01 00:00", end="2020-01-31 23:00", freq="H"),
            data=list(range(9504)),
        ),
    )

    ts.resample_simple_extend_years((2020, 2030))

    test_data = pd.Series(
        name="test_data", index=pd.date_range(start="2020-01-01 00:00", end="2030-12-31 23:00", freq="H")
    )
    for year in set(test_data.index.year):
        if calendar.isleap(year):
            test_data.loc[test_data.index.year == year] = list(range(8784))
        else:
            test_data.loc[test_data.index.year == year] = list(range(8760))

    pd.testing.assert_series_equal(ts.data, pd.Series(test_data))


def test_days_in_year():
    # Test 1: 4 years of hourly data
    ts = NumericTimeseries(
        name="test_data",
        data=pd.Series(
            name="test_data",
            index=pd.date_range(start="2021-01-01 00:00", end="2024-12-31 23:00", freq="H"),
            data=1,
        ),
    )
    expected_days_in_year = pd.Series(
        data=np.concatenate([np.repeat(365, (8760 * 3)), np.repeat(366, 366 * 24)]).tolist(), index=ts.data.index
    )
    pd.testing.assert_series_equal(ts.days_in_year, expected_days_in_year)

    # Test 2: 4 years of daily data
    ts = NumericTimeseries(
        name="test_data",
        data=pd.Series(
            name="test_data",
            index=pd.date_range(start="2021-01-01 00:00", end="2024-12-31 23:00", freq="D"),
            data=1,
        ),
    )
    expected_days_in_year = pd.Series(
        data=np.concatenate([np.repeat(365, (365 * 3)), np.repeat(366, 366)]).tolist(), index=ts.data.index
    )
    pd.testing.assert_series_equal(ts.days_in_year, expected_days_in_year)

    # Test 3: 4 years of monthly data
    ts = NumericTimeseries(
        name="test_data",
        data=pd.Series(
            name="test_data",
            index=pd.date_range(start="2021-01-01 00:00", end="2024-12-31 23:00", freq="MS"),
            data=1,
        ),
    )
    expected_days_in_year = pd.Series(
        data=np.concatenate([np.repeat(365, (12 * 3)), np.repeat(366, 12)]).tolist(), index=ts.data.index
    )
    pd.testing.assert_series_equal(ts.days_in_year, expected_days_in_year)

    # Test 4: 4 years of yearly data
    ts = NumericTimeseries(
        name="test_data",
        data=pd.Series(
            name="test_data",
            index=pd.date_range(start="2021-01-01 00:00", end="2024-12-31 23:00", freq="YS"),
            data=1,
        ),
    )
    expected_days_in_year = pd.Series(
        data=np.concatenate([np.repeat(365, 3), np.array([366])]).tolist(), index=ts.data.index
    )
    pd.testing.assert_series_equal(ts.days_in_year, expected_days_in_year)
