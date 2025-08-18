# import numpy as np
# import pandas as pd
# from new_modeling_toolkit.common.load_component import Load
# from new_modeling_toolkit.core import dir_str
# from new_modeling_toolkit.core.temporal.timeseries import Timeseries
# def test_load():  # noqa: C901
#     test_index = pd.date_range("20170101", "20200101", freq="1H")[:-1]  # 3 years of data
#     test_data = 830 * np.random.random((test_index.size))  # 830 MW peak load
#     df = pd.Series(index=test_index, data=test_data)
#     annual_peak_forecast = Timeseries(
#         name="annual_peak_forecast",
#         data=pd.Series(index=[2020, 2025, 2030, 2035], data=[6000, 6500, 6750, 7200]),
#     )
#     annual_energy_forecast = Timeseries(
#         name="annual_energy_forecast",
#         data=pd.Series(index=[2020, 2025, 2030, 2035], data=[63000, 65040, 67550, 72800]),
#     )
#     ts = Load(
#         name="random_profile",
#         profile=Timeseries(name="profile", data=df, timezone="Etc/GMT-7"),
#         annual_peak_forecast=annual_peak_forecast,
#         annual_energy_forecast=annual_energy_forecast,
#         scale_by_energy=True,
#         scale_by_capacity=True,
#     )
#     load_2035 = ts.get_load(2035)
#     assert np.isclose(load_2035.max(), annual_peak_forecast.data[2035])
#     assert np.isclose(load_2035.mean() * ts.n_timesteps_per_year, annual_energy_forecast.data[2035])
# def test_from_csv():
#     """Test that loads can be read from CSV correctly."""
#     loads = Load.from_dir(data_path=dir_str.data_dir / "interim" / "loads")
#     assert loads["CAISO"].name == "CAISO"
#     assert loads["CAISO"].scale_by_capacity
