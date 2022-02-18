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
    download_url,
)
import os
from urllib.error import URLError

"""
Contains the data prep class for the Airbus Helicopter Accelerometer Dataset.

Also contains helper functions associated with the dataset.
"""


###############################################################################
# Data Prep Classes
###############################################################################
class AirbusDataLoad(PHMDataset):
    """
    Airbus Helicopter Accelerometer Dataset from .h5 file, and download if necessary.

    Args:
        root (string): Root directory to place all the  data sets.

        dataset_folder_name (string): Name of folder containing raw data.
            This folder will be created in the root directory if not present.

        download (bool): If True, the data will be downloaded from ETH Zurich.

    """

    mirrors = [
        "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/415151/",
    ]

    resources = [
        ("dftrain.h5", None),
        ("dfvalid.h5", None),
        ("dfvalid_groundtruth.csv", None),
    ]

    def __init__(
        self,
        root: Path,
        dataset_folder_name: str = "airbus",
        download: bool = False,
        path_df_labels: Path = None,
    ) -> None:
        super().__init__(root, dataset_folder_name)

        self.dataset_folder_path = self.root / self.dataset_folder_name

        if path_df_labels is not None:
            self.path_df_labels = path_df_labels
        else:
            # path of pyphm source directory using pathlib
            self.path_df_labels = Path(pkg_resources.resource_filename('pyphm', 'datasets/auxilary_metadata/airbus_dfvalid_groundtruth.csv'))
            
        if download:
            self.download()

    def _check_exists(self) -> bool:
        return all(
            check_integrity(self.dataset_folder_path / file_name)
            for file_name, _ in self.resources
        )

    def download(self) -> None:
        """Download the Airbus Helicopter Accelerometer Dataset if it doesn't exist already."""

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

                    download_url(url, self.dataset_folder_path, filename, md5)

                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def load_df(
        self,
        train_or_val: str = "train",
    ) -> None:
        """Load the h5 file as df."""

        if train_or_val == "train":
            df = pd.read_hdf(self.dataset_folder_path / "dftrain.h5", "dftrain")

            # add y column of all zeros (indicating no anomaly)
            df["y"] = 0

        else:  # val dataset
            df = pd.read_hdf(self.dataset_folder_path / "dfvalid.h5", "dfvalid")

            # load the dfvalid_groundtruth.csv as dataframe
            df_labels = pd.read_csv(
                self.path_df_labels,
                dtype={"seqID": int, "anomaly": int},
            )

            # append the anomaly label to the df_val dataframe
            df = df.merge(df_labels, left_index=True, right_on="seqID")

            # drop the seqID column and rename the anomaly column to y
            df = df.drop("seqID", axis=1).rename(columns={"anomaly": "y"})

        return df


class AirbusPrepMethodA(AirbusDataLoad):
    """
    Class used to prepare the Airbus Helicopter Accelerometer Dataset before feature engining or machine learning.
    Method is described in the paper:

    `Temporal signals to images: Monitoring the condition of industrial assets with deep learning image processing algorithms`
    by Garcia et al., 2021 - https://arxiv.org/abs/2005.07031

    Args:
        root (string): Root directory to place all the  data sets. (likely the raw data folder)

        dataset_folder_name (string): Name of folder (within root) containing raw data.
            This folder will be created in the root directory if not present.

        download (bool): If True, the data will be downloaded from the ETH Zurich website.

        path_df_labels (Path, optional): Path to the csv with the labels. If not provided, it
            will default to airbus_dfvalid_groundtruth.csv in the auxilary_metadata folder.

        window_size (int): Size of the window to be used for the sliding window.

        stride (int): Size of the stride to be used for the sliding window.

    """

    def __init__(
        self,
        root: Path,
        dataset_folder_name: str = "airbus",
        download: bool = False,
        path_df_labels: Path = None,
        window_size: int = 64,
        stride: int = 64,
    ) -> None:
        super().__init__(root, dataset_folder_name, download, path_df_labels)

        self.window_size = window_size  # size of the window
        self.stride = stride  # stride between windows

    def create_xy_arrays(self, train_or_val: str = "train"):
        """Create the x and y arrays used in deep learning.

        Parameters
        ----------
        train_or_val : str
            Either 'train' or 'val' to indicate which dataset to use. Default is 'train'.

        Returns
        -------
        x : ndarray
            Array of the signals (samples). Shape: (n_samples, n_windows, window_size)

        y : ndarray
            Array of the labels/meta-data for each signals. Shape: (n_samples, n_windows, window_size, label_columns)
            The label_columns (in order) are:
                time_increments (int) -- the index of each time increment in the window. e.g. (0, 1, 2, ...)
                sample_index (int) -- the index of each sample
                window_index (int) -- the index of each window
                label (int) -- the label of each windowed sample (0 for normal, 1 for anomaly)

        """

        # load the dataframe
        df = self.load_df(train_or_val)

        x = df.drop("y", axis=1).to_numpy()
        y = df["y"].to_numpy()

        # instantiate the "temporary" lists to store the windows and labels
        window_list = []
        y_sample_win_label_list = []

        n_samples = x.shape[0]
        len_sample = x.shape[1]

        # fit the strided windows into the temporary list until the length
        # of the window does not equal the proper length (better way to do this???)
        for window_i in range(len_sample):
            windowed_signal = x[
                :, window_i * self.stride : window_i * self.stride + self.window_size
            ]

            # if the windowed signal is the proper length, add it to the list
            if windowed_signal.shape == (n_samples, self.window_size):
                window_list.append(windowed_signal)

                y_sample_win_label_list.append(
                    [
                        (int(sample_indices), int(window_indices), int(ys))
                        for sample_indices, window_indices, ys in list(
                            zip(list(range(0, n_samples)), [window_i] * n_samples, y)
                        )
                    ]
                )

            else:
                break

        x = np.array(window_list).reshape(n_samples, -1, self.window_size)

        y_sample_win_label_array = np.array(y_sample_win_label_list)[:, :, np.newaxis].repeat(
            self.window_size, axis=2
        )

        time_index = (
            np.arange(0, self.window_size, 1)[np.newaxis, np.newaxis, :]
            .repeat(n_samples, axis=1)
            .repeat(x.shape[1], axis=0)[:, :, :, np.newaxis]
        )

        y_time_sample_win_label_array = np.concatenate(
            (time_index, y_sample_win_label_array), axis=3
        ).reshape(n_samples, -1, self.window_size, 4)
        # window_id_array = np.expand_dims(np.array(window_id_list).reshape(-1), axis=1)
        # window_label_array = np.expand_dims(np.array(window_label_list).reshape(-1), axis=1)

        # x = np.vstack(window_list,)

        # y = np.hstack((window_label_array, window_id_array))
        # return np.vstack(x), np.vstack(y_time_sig_win_label_array)
        return x, y_time_sample_win_label_array

    def create_xy_dataframe(self, train_or_val: str = "train"):
        """
        Create a flat dataframe (2D array) of the x and y arrays.

        Amenable for use with TSFresh for feature engineering.

        Returns
        -------
        df : pd.DataFrame
            Single flat dataframe containing each sample and its labels.
            columns: ['x', 'time_index', 'sample_index', 'window_index', 'y']

        """

        x, y = self.create_xy_arrays(train_or_val)  # create the x and y arrays

        df = pd.DataFrame(np.vstack(x).reshape(-1,1), columns=['x'])

        # add the time_index, sample_index, window_index, and label columns
        # to the dataframe
        df = df.assign(time_index=np.vstack(y[:,:,:,0]).reshape(-1,1))
        df = df.assign(sample_index=np.vstack(y[:,:,:,1]).reshape(-1,1))
        df = df.assign(win_index=np.vstack(y[:,:,:,2]).reshape(-1,1))
        df = df.assign(y=np.vstack(y[:,:,:,3]).reshape(-1,1))

        return df
