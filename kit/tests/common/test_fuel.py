# import pytest
# from new_modeling_toolkit.common import fuel
# from new_modeling_toolkit.core.temporal import timeseries as ts
# @pytest.fixture(scope="session")
# def groups():
#     """
#     Returns:  an intialized dictionary of fuel groups
#     """
#     # path to data folder
#     data_path = "../../data/interim"
#     # instantiate fuel objects
#     fuels = fuel.Fuel.from_dir(data_path)
#     # instantiate fuel group objects
#     # TODO: Fix this test
#     fuel_groups = fuel.Fuel_Group.from_dir(fuels, data_path)
#     return fuel_groups
# def test_price_is_ts(groups):
#     for group in groups.values():
#         for member in group.members.values():
#             assert isinstance(member.fuel_price_per_mmbtu, ts.Timeseries)
# if __name__ == '__main__':
#     test_price_is_ts(groups)
