import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
from .pyphm import PHMDataset
from typing import Any, Callable, List, Optional, Tuple
import pkg_resources
from .utils import (
    download_and_extract_archive,
    extract_archive,
    check_integrity,
)
import os
from urllib.error import URLError

"""
Contains the data prep class for the UC-Berkely milling data set.

Also contains helper functions associated with the milling data set.
"""


###############################################################################
# Data Prep Classes
###############################################################################
class MillingDataLoad(PHMDataset):
    """
    Load the UC Berkely milling data set from .mat file, and download if necessary.

    Args:
        root (string): Root directory to place all the  data sets.

        dataset_folder_name (string): Name of folder containing raw data.
            This folder will be created in the root directory if not present.

        download (bool): If True, the data will be downloaded from the NASA Prognostics Repository.

    """

    mirrors = [
        # "https://github.com/tvhahn/ml-tool-wear/raw/master/data/raw/",
        "https://drive.google.com/file/d/1_4Hm8RO_7Av1LzGtFnhx6cIN-zi-W40j/view?usp=sharing",
        # "https://ti.arc.nasa.gov/m/project/prognostic-repository/"     
    ]

    resources = [
        ("mill.zip", "81d821fdef812183a7d38b6f83f7cefa"),
    ]

    def __init__(
        self,
        root: Path,
        dataset_folder_name: str = "milling",
        data_file_name: str = "mill.mat",
        download: bool = False,
        data: np.ndarray = None,
    ) -> None:
        super().__init__(root, dataset_folder_name)

        self.dataset_folder_path = self.root / self.dataset_folder_name
        self.data_file_name = data_file_name

        if download:
            self.download()

        data_file_path = self.dataset_folder_path / self.data_file_name
        # assert that data_file_path exists
        assert data_file_path.exists(), f"{data_file_path} does not exist."

        self.data = self.load_mat()

    def _check_exists(self) -> bool:
        return all(
            check_integrity(self.dataset_folder_path / file_name)
            for file_name, _ in self.resources
        )

    def download(self) -> None:
        """Download the UC Berkeley milling data if it doesn't exist already."""

        if self._check_exists():
            return

        # pathlib makdir if not exists
        self.dataset_folder_path.mkdir(parents=True, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(
                        url,
                        download_root=self.dataset_folder_path,
                        filename=filename,
                        md5=md5,
                    )
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def load_mat(self) -> np.ndarray:
        """Load the mat file and return the data as a numpy array."""
        data = sio.loadmat(
            self.dataset_folder_path / self.data_file_name, struct_as_record=True
        )
        return data["mill"]


class MillingPrepMethodA(MillingDataLoad):
    """
    Class used to prepare the UC Berkeley milling dataset before feature engining or machine learning.
    Method is described in the paper:

    `Self-supervised learning for tool wear monitoring with a disentangled-variational-autoencoder`
    by von Hahn and Mechefkse, 2021

    Args:
        root (string): Root directory to place all the  data sets. (likely the raw data folder)

        dataset_folder_name (string): Name of folder (within root) containing raw data.
            This folder will be created in the root directory if not present.

        download (bool): If True, the data will be downloaded from the NASA Prognostics Repository.

        path_csv_labels (Path, optional): Path to the csv of the label dataframe.
            If not provided, the 'milling_labels_with_tool_class.csv' will be used, provided in the
            PyPHM package.

        window_len (int): Length of the window to be used for the sliding window.

        stride (int): Amount to move (stride) between individual windows of data.

        cut_drop_list (list, optional): List of cut numbers to drop. cut_no 17 and 94 are erroneous and
            will be dropped as default.
    """

    def __init__(
        self,
        root: Path,
        dataset_folder_name: str = "milling",
        dataset_folder_path: Path = None,
        data_file_name: str = "mill.mat",
        download: bool = False,
        data: np.ndarray = None,
        path_csv_labels: Path = None,
        window_len: int = 64,
        stride: int = 64,
        cut_drop_list: List[int] = [17, 94],
    ) -> None:
        super().__init__(root, dataset_folder_name, data_file_name, download, data)

        self.window_len = window_len  # size of the window
        self.stride = stride  # stride between windows
        self.cut_drop_list = cut_drop_list  # list of cut numbers to be dropped

        if path_csv_labels is not None:
            self.path_csv_labels = path_csv_labels
        else:
            # path of pyphm source directory using pathlib
            self.path_csv_labels = Path(pkg_resources.resource_filename('pyphm', 'datasets/auxilary_metadata/milling_labels_with_tool_class.csv'))

        # load the labels dataframe
        self.df_labels = pd.read_csv(self.path_csv_labels)

        if self.cut_drop_list is not None:
            self.df_labels.drop(
                self.cut_drop_list, inplace=True
            )  # drop the cuts that are bad

        self.df_labels.reset_index(drop=True, inplace=True)  # reset the index

        self.field_names = self.data.dtype.names

        self.signal_names = self.field_names[7:][::-1]

    def create_labels(self):
        """Function that will create the label dataframe from the mill data set

        Only needed if the dataframe with the labels is not provided.
        """

        # create empty dataframe for the labels
        df_labels = pd.DataFrame()

        # get the labels from the original .mat file and put in dataframe
        for i in range(7):
            # list for storing the label data for each field
            x = []

            # iterate through each of the unique cuts
            for j in range(167):
                x.append(self.data[0, j][i][0][0])
            x = np.array(x)
            df_labels[str(i)] = x

        # add column names to the dataframe
        df_labels.columns = self.field_names[0:7]

        # create a column with the unique cut number
        df_labels["cut_no"] = [i for i in range(167)]

        def tool_state(cols):
            """Add the label to the cut.

            Categories are:
            Healthy Sate (label=0): 0~0.2mm flank wear
            Degredation State (label=1): 0.2~0.7mm flank wear
            Failure State (label=2): >0.7mm flank wear
            """
            # pass in the tool wear, VB, column
            vb = cols

            if vb < 0.2:
                return 0
            elif vb >= 0.2 and vb < 0.7:
                return 1
            elif pd.isnull(vb):
                pass
            else:
                return 2

        # apply the label to the dataframe
        df_labels["tool_class"] = df_labels["VB"].apply(tool_state)

        return df_labels

    def create_data_array(self, cut_no):
        """Create an array from an individual cut sample.

        Parameters
        ===========
        cut_no : int
            Index of the cut to be used.

        Returns
        ===========
        sub_cut_array : np.array
            Array of the cut samples. Shape of [no. samples, sample len, features/sample]

        sub_cut_labels : np.array
            Array of the labels for the cut samples. Shape of [# samples, # features/sample]

        """

        assert (
            cut_no in self.df_labels["cut_no"].values
        ), "Cut number must be in the dataframe"

        # create a numpy array of the cut
        # with a final array shape like [no. cuts, len cuts, no. signals]
        cut = self.data[0, cut_no]
        for i, signal_name in enumerate(self.signal_names):
            if i == 0:
                cut_array = cut[signal_name].reshape((9000, 1))
            else:
                cut_array = np.concatenate(
                    (cut_array, cut[signal_name].reshape((9000, 1))), axis=1
                )

        # select the start and end of the cut
        start = self.df_labels[self.df_labels["cut_no"] == cut_no][
            "window_start"
        ].values[0]
        end = self.df_labels[self.df_labels["cut_no"] == cut_no]["window_end"].values[0]
        cut_array = cut_array[start:end, :]

        # instantiate the "temporary" lists to store the sub-cuts and metadata
        sub_cut_list = []
        sub_cut_id_list = []
        sub_cut_label_list = []

        # get the labels for the cut
        label = self.df_labels[self.df_labels["cut_no"] == cut_no]["tool_class"].values[
            0
        ]

        # fit the strided windows into the dummy_array until the length
        # of the window does not equal the proper length (better way to do this???)
        for i in range(cut_array.shape[0]):
            windowed_signal = cut_array[
                i * self.stride : i * self.stride + self.window_len
            ]

            # if the windowed signal is the proper length, add it to the list
            if windowed_signal.shape == (self.window_len, 6):
                sub_cut_list.append(windowed_signal)

                # create sub_cut_id fstring to keep track of the cut_id and the window_id
                sub_cut_id_list.append(f"{cut_no}_{i}")

                # create the sub_cut_label and append it to the list
                sub_cut_label_list.append(int(label))

            else:
                break

        sub_cut_array = np.array(sub_cut_list)

        sub_cut_ids = np.expand_dims(np.array(sub_cut_id_list, dtype=str), axis=1)
        sub_cut_ids = np.repeat(sub_cut_ids, sub_cut_array.shape[1], axis=1)

        sub_cut_labels = np.expand_dims(np.array(sub_cut_label_list, dtype=int), axis=1)
        sub_cut_labels = np.repeat(sub_cut_labels, sub_cut_array.shape[1], axis=1)

        # take the length of the signals in the sub_cut_array
        # and divide it by the frequency (250 Hz) to get the time (seconds) of each sub-cut
        sub_cut_times = np.expand_dims(
            np.arange(0, sub_cut_array.shape[1]) / 250.0, axis=0
        )
        sub_cut_times = np.repeat(
            sub_cut_times,
            sub_cut_array.shape[0],
            axis=0,
        )

        sub_cut_labels_ids_times = np.stack(
            (sub_cut_labels, sub_cut_ids, sub_cut_times), axis=2
        )

        return (
            sub_cut_array,
            sub_cut_labels,
            sub_cut_ids,
            sub_cut_times,
            sub_cut_labels_ids_times,
        )

    def create_xy_arrays(self):
        """Create the x and y arrays used in deep learning.

        Returns
        ===========
        x_array : np.array
            Array of the cut samples. Shape of [no. samples, sample len, features/sample]

        y_array : np.array
            Array of the labels for the cut samples. Shape of [no. samples, sample len, label/ids/times]
            Use y[:,0,:], for example, to get the y in a shape of [no. samples, label/ids/times]
            ( e.g. will be shape (no. samples, 3) )

        """

        # create a list to store the x and y arrays
        x = []  # instantiate X's
        y_labels_ids_times = []  # instantiate y's

        # iterate throught the df_labels
        for i in self.df_labels.itertuples():
            (
                sub_cut_array,
                sub_cut_labels,
                sub_cut_ids,
                sub_cut_times,
                sub_cut_labels_ids_times,
            ) = self.create_data_array(i.cut_no)

            x.append(sub_cut_array)
            y_labels_ids_times.append(sub_cut_labels_ids_times)

        return np.vstack(x), np.vstack(y_labels_ids_times)

    def create_xy_dataframe(self):
        """
        Create a flat dataframe (2D array) of the x and y arrays.

        Amenable for use with TSFresh for feature engineering.

        Returns
        ===========
        df : pd.DataFrame
            Single flat dataframe containing each sample and its labels.

        """

        x, y_labels_ids_times = self.create_xy_arrays()  # create the x and y arrays

        # concatenate the x and y arrays and reshape them to be a flat array (2D)
        x_labels = np.reshape(np.concatenate((x, y_labels_ids_times), axis=2), (-1, 9))

        # define the column names and the data types
        col_names = [s.lower() for s in list(self.signal_names)] + [
            "tool_class",
            "cut_id",
            "time",
        ]

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

        # create a dataframe from the x and y arrays
        df = pd.DataFrame(x_labels, columns=col_names, dtype=str)

        # split the cut_id by "_" and take the first element (cut_no)
        df["cut_no"] = df["cut_id"].str.split("_").str[0]

        # get the case from each cut_no using the df_labels
        df = df.merge(
            self.df_labels[["cut_no", "case"]].astype(dtype=str),
            on="cut_no",
            how="left",
        )

        df = df[col_names_ordered].astype(col_dtype_dict)  # reorder the columns

        return df
