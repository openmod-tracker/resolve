import numpy as np
import pandas as pd
import pytest

from new_modeling_toolkit.core.utils.core_utils import cantor_pairing_function
from new_modeling_toolkit.core.utils.core_utils import filter_not_none
from new_modeling_toolkit.core.utils.core_utils import map_dict
from new_modeling_toolkit.core.utils.core_utils import map_not_none
from new_modeling_toolkit.core.utils.core_utils import sum_not_none


def test_map_dict():
    assert map_dict(sum, {"a": [1, 2, 3], "b": [4, 5, 6]}) == {"a": 6, "b": 15}


def test_filter_not_none():
    assert filter_not_none(["a", None, 1, None, "b"]) == ["a", 1, "b"]
    assert filter_not_none([None, None]) == []
    assert filter_not_none(["a", "b", 1, 2]) == ["a", "b", 1, 2]


def test_map_not_none():
    assert map_not_none(min, [[-5, 2, 3], None, [4, 6, 12]]) == [-5, 4]


def test_sum_not_none():
    assert sum_not_none([1, 4, None, -3, 2, None]) == 4
    pd.testing.assert_series_equal(
        sum_not_none([None, pd.Series(data=[1, 2, 3, 4]), pd.Series([5, 5, np.nan, 5])]),
        pd.Series(data=[6, 7, np.nan, 9]),
    )


def test_cantor_pairing_function():
    # Check that the function produces expected output for simple test cases
    assert cantor_pairing_function(a=2, b=3) == 18
    assert cantor_pairing_function(a=3, b=2) == 17
    assert cantor_pairing_function(0, 0) == 0
    # Check that the function raises errors when appropriate
    with pytest.raises(ValueError):
        cantor_pairing_function(a=3, b=-1)
    with pytest.raises(ValueError):
        cantor_pairing_function(a=-3, b=4)
