import unittest
import numpy as np
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
from src.datasets.milling import MillingDataPrep


class TestMilling(unittest.TestCase):
    def setUp(self):
        # path to mill_truncated.mat
        self.mill_data_path = Path(__file__).parent / "fixtures/mill_truncated.mat"
        print("mill_data_path:", self.mill_data_path)

        # path to labels_with_tool_class_truncated.csv
        self.labels_path = (
            Path(__file__).parent / "fixtures/labels_with_tool_class_truncated.csv"
        )

        # path to milling_truncated_results.csv.gz
        self.results_path = (
            Path(__file__).parent / "fixtures/milling_truncated_results.csv.gz"
        )

    def test_milling_data_prep(self):
        """Test that the milling data prep works as expected."""
        
        # load the data and instantiate the data prep class
        milldata = MillingDataPrep(
            self.mill_data_path,
            path_df_labels=self.labels_path,
            window_size=64,
            stride=64,
            cut_drop_list=None,
        )

        # create the results dataframe
        df = milldata.create_xy_dataframe()
        print("df.shape:", df.shape)
        print("df.columns:", df.columns)

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

        print("df_gt.shape:", df_gt.shape)
        print("df_gt.columns:", df_gt.columns)

        # compare the results
        assert_frame_equal(df, df_gt)


if __name__ == "__main__":

    unittest.main()
