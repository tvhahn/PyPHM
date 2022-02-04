import unittest
import numpy as np
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
from pyphm.datasets.ims import ImsDataLoad


class TestIms(unittest.TestCase):
    def setUp(self):
        # path to mill_truncated.mat
        self.root = (
            Path(__file__).parent / "fixtures"
        )
        print("mill_data_path:", self.root)


        # path to ims_truncated_results.csv.gz
        self.results_path = (
            self.root / "ims/ims_truncated_results.csv.gz"
        )

    def test_milling_data_prep(self):
        """Test that the milling data prep works as expected."""

        # load the data and instantiate the data prep class
        ims = ImsDataLoad(self.root, download=False)

        # create the results dataframe
        df = ims.load_run_as_df(1, n_jobs=1)

        # load the ground truth results dataframe
        col_names_ordered = ["id", "run", "file", "time_step"] + ims.col_1st_names

        col_dtype = [
            str,
            int,
            str,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
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
