from new_modeling_toolkit.core.utils.parallelization_utils import parallelize


def test_parallelize():
    def _test_func(a: float, b: float, c: float) -> float:
        return a**2 + 2 * b - c

    assert parallelize(
        _test_func,
        args_list=[
            (1, 2, 3),
            (4, 5, 6),
        ],
        show_progress_bar=False,
    ) == [2, 20]

    assert parallelize(
        _test_func,
        args_list=[
            (1, 2),
            (4,),
        ],
        kwargs_list=[{"c": 3}, {"b": 5, "c": 6}],
        show_progress_bar=False,
    ) == [2, 20]
