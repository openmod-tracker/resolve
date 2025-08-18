# import collections
# import pytest
# import new_modeling_toolkit.common.pro_forma
# from new_modeling_toolkit.common import resource
# from new_modeling_toolkit.core.temporal import timeseries as ts
# data_path = "../../data/interim"
# @pytest.fixture(scope="session")
# def init_resources():
#     """
#     Returns: a resources dict with all input data initialized
#     """
#     # instantiate proforma
#     proforma = new_modeling_toolkit.common.pro_forma.ProForma.from_csv(data_path)
#     # instantiate resources
#     resources = resource.Resource.from_dir(data_path, proforma)
#     ResourceProforma = collections.namedtuple("ResourceProforma", "proforma resources")
#     rp = ResourceProforma(proforma, resources)
#     return rp
# def test_static_attr(init_resources):
#     assert float(init_resources.resources["CAISO_CT"].start_fuel_use) == float(3)
#     assert init_resources.resources["Distributed_Solar"].proforma_type == "Solar - Residential"
#     assert float(init_resources.resources["CAISO_CCGT_old"].fuel_burn_intercept) == float(50)
# def test_annual_attr(init_resources):
#     assert isinstance(
#         init_resources.resources["CAISO_Battery"].planned_installed_capacity,
#         ts.Timeseries,
#     )
# def test_proforma_attr(init_resources):
#     assert init_resources.resources["CAISO_Battery"].storage_annualized_fixed_cost.data.equals(
#         init_resources.proforma.get_total_lfc("Utility-scale Battery - Li [Energy] - No ITC").data
#     )
