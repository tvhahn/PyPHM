import unittest
import numpy as np
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
from pyphm.datasets.milling import MillingPrepMethodA


class TestMilling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass


    def setUp(self):
        # path to mill_truncated.mat
        self.root = (
            Path(__file__).parent / "fixtures"
        )

        # path to milling_labels_with_tool_class_truncated.csv
        self.labels_path = (
            self.root
            / "milling/milling_labels_with_tool_class_truncated.csv"
        )

        # path to milling_truncated_results.csv.gz
        self.results_path = (
            self.root / "milling/milling_truncated_results.csv.gz"
        )

    def test_load_run_as_df(self):
        """Test the loading of an individual run as a dataframe."""

        # load the data and instantiate the data prep class
        mill = MillingPrepMethodA(
            self.root,
            window_len=64,
            stride=64,
            cut_drop_list=[],
            path_csv_labels=self.labels_path,
            download=False,
        )

        # create the results dataframe
        df = mill.create_xy_dataframe()

        # load the ground truth results dataframe
        col_names_ordered = [
            "cut_id",
            "cut_no",
            "case",
            "time",
            "ae_spindle",
            "ae_table",
            "vib_spindle",
            "vib_table",
            "smcdc",
            "smcac",
            "tool_class",
        ]

        col_dtype = [
            str,
            int,
            int,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            int,
        ]

        col_dtype_dict = dict(zip(col_names_ordered, col_dtype))

        # load the ground truth results dataframe
        df_gt = pd.read_csv(
            self.results_path,
            compression="gzip",
        ).astype(col_dtype_dict)

        # compare the results
        assert_frame_equal(df, df_gt)


if __name__ == "__main__":

    unittest.main()
