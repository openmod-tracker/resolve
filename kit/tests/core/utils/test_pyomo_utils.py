import pyomo.environ as pyo
import pytest  # noqa

from new_modeling_toolkit.core.utils.pyomo_utils import get_index_labels


def test_get_index_labels():
    m = pyo.ConcreteModel()

    # Create various types of sets to test
    m.single_set = pyo.Set(initialize=["a", "b", "c"])
    m.range_set = pyo.RangeSet(1, 10)
    m.tuple_set = pyo.Set(initialize=[(1, "x"), (2, "y"), (3, "z")])
    m.TIMEPOINTS = pyo.Set(initialize=[(2020, 1, 1), (2020, 1, 2), (2020, 1, 3)])
    m.MODEL_YEARS_AND_ADJACENT_REP_PERIODS = pyo.Set(initialize=[(2020, 1, 2), (2020, 1, 3), (2020, 2, 3)])

    m.param_single_index = pyo.Param(m.single_set)
    m.param_range_set = pyo.Param(m.range_set)
    m.param_tuple_set = pyo.Param(m.tuple_set)
    m.param_multi_set = pyo.Param(m.single_set, m.range_set)
    m.param_TIMEPOINTS = pyo.Param(m.TIMEPOINTS)
    m.param_MODEL_YEARS_AND_ADJACENT_REP_PERIODS = pyo.Param(m.MODEL_YEARS_AND_ADJACENT_REP_PERIODS)

    assert get_index_labels(m.param_single_index) == ["single_set"]
    assert get_index_labels(m.param_range_set) == ["range_set"]
    assert get_index_labels(m.param_tuple_set) == ["tuple_set"]
    assert get_index_labels(m.param_multi_set) == ["single_set", "range_set"]
    # m.single_set is not indexed by anything, so its index labels should be [None]
    assert get_index_labels(m.single_set) == [None]
    assert get_index_labels(m.param_TIMEPOINTS) == ["MODEL_YEARS", "REP_PERIODS", "HOURS"]
    assert get_index_labels((m.param_MODEL_YEARS_AND_ADJACENT_REP_PERIODS)) == [
        "MODEL_YEARS",
        "PREV_REP_PERIODS",
        "NEXT_REP_PERIODS",
    ]
