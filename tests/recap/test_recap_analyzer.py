import copy

import pytest

from new_modeling_toolkit.recap.recap_analyzer import DispatchPlot_RECAPstack
from new_modeling_toolkit.recap.recap_analyzer import ResultsViewer


class TestResultsViewer:
    @pytest.fixture(scope="class")
    def test_results_viewer(self, dir_structure):
        test_results_viewer = ResultsViewer(
            result_folder=dir_structure.results_dir / "recap",
            case_name="integration_fully_optimized",
            run_name="test_results",
        )

        test_results_viewer._upload_resource_group_csv(dir_structure.data_dir / "raw" / "CUSTOMIZED_disp_group.csv")
        return test_results_viewer

    # @pytest.fixture
    # def mock_resource_group_upload(self,dir_structure,monkeypatch):
    #     # A fixture to mock the file upload widget
    #     monkeypatch.setattr(widgets, "FileUpload")
    #
    # @pytest.fixture
    # def mock_resource_group_output(self,dir_structure,mock_resource_group_upload,monkeypatch):
    #     def mock_file_upload_function(multiple=False):
    #         return pd.read_csv(dir_structure.data_dir/"raw"/"CUSTOMIZED_disp_group.csv")
    #     monkeypatch.setattr(widgets,"Outputs",mock_file_upload_function)

    def test_read_case_data(self, test_results_viewer):
        results_viewer = copy.deepcopy(test_results_viewer)
        results_viewer.read_case_data(load_disp_result=True)
        results_viewer.untuned_disp.loc["MC_draw_0"].plot()

    def test_calc_disp_by_group(self, test_results_viewer):
        test_results_viewer.read_case_data(load_disp_result=True)
        test_results_viewer.calc_disp_by_group(test_results_viewer.untuned_disp)

    def test_create_dispatch_plot(self, test_results_viewer):
        test_results_viewer.read_case_data(load_disp_result=True)
        disp_by_group = test_results_viewer.calc_disp_by_group(test_results_viewer.untuned_disp)
        dispatch_plot = DispatchPlot_RECAPstack(disp_by_group)
        dispatch_plot.create_dispatch_plot()
        dispatch_plot._figure.show()
