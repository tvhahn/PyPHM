from ossaudiodev import control_names
import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
from .pyphm import PHMDataset
import multiprocessing as mp
from typing import Any, Callable, List, Optional, Tuple
from .utils import (
    download_and_extract_archive,
    extract_archive,
    check_integrity,
)
import os
from urllib.error import URLError


class ImsDataLoad(PHMDataset):
    """
    Load the IMS bearing data set from .csv files, and download if necessary.

    Args:
        root (string): Root directory to place all the  data sets.

        dataset_folder_name (string): Name of folder containing raw data.
            This folder will be created in the root directory if not present.

        download (bool): If True, the data will be downloaded from the NASA Prognostics Repository.

    """

    mirrors = [
        "https://ti.arc.nasa.gov/m/project/prognostic-repository/",
    ]

    resources = [
        ("IMS.7z", "d3ca5a418c2ed0887d68bc3f91991f12"),
    ]

    col_1st_names = [
        "b1_ch1",
        "b1_ch2",
        "b2_ch3",
        "b2_ch4",
        "b3_ch5",
        "b3_ch6",
        "b4_ch7",
        "b4_ch8",
    ]
    col_2nd_names = col_3rd_names = ["b1_ch1", "b1_ch2", "b2_ch3", "b2_ch4"]

    def __init__(
        self,
        root: Path,
        dataset_folder_name: str = "ims",
        download: bool = False,
        dataset_path: Path = None,
        data: np.ndarray = None,
        sample_freq: float = 20480.0,
    ) -> None:
        super().__init__(root, dataset_folder_name)

        self.dataset_path = self.root / self.dataset_folder_name
        print(self.dataset_path)

        if download:
            self.download()

            if not self._check_exists():
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )

        # set the paths for the three experiment run folders
        self.path_1st_folder = self.dataset_path / "1st_test"
        self.path_2nd_folder = self.dataset_path / "2nd_test"

        # the third test is labelled as the "4th_test" in the IMS.7z archive
        self.path_3rd_folder = self.dataset_path / "4th_test/txt"

        self.sample_freq = sample_freq

    def _check_exists(self) -> bool:
        return all(
            check_integrity(self.dataset_path / file_name)
            for file_name, _ in self.resources
        )

    def download(self) -> None:
        """Download the UC Berkeley milling data if it doesn't exist already."""

        if self._check_exists():
            print("IMS.7z already exists.")
            return

        # pathlib makdir if not exists
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(
                        url, download_root=self.dataset_path, filename=filename, md5=md5
                    )

                    # sequentially extract the .rar files
                    rar_list = ["1st_test.rar", "2nd_test.rar", "3rd_test.rar"]
                    for rar_file in rar_list:
                        print(f"Extracting {rar_file}")
                        extract_archive(
                            self.dataset_path / rar_file, remove_finished=True
                        )

                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def extract(self) -> None:
        """Extract the data set if it has already been dowloaded."""

        if not self._check_exists():
            print("IMS.7z does not exist. Please download.")
            return

        print("Extracting IMS.7z")

        # start with the .7z file
        extract_archive(self.dataset_path / "IMS.7z", remove_finished=False)

        # sequentially extract the .rar files
        rar_list = ["1st_test.rar", "2nd_test.rar", "3rd_test.rar"]
        for rar_file in rar_list:
            print(f"Extracting {rar_file}")
            extract_archive(self.dataset_path / rar_file, remove_finished=True)

    @staticmethod
    def process_raw_csv(file_info_dict) -> None:
        """Load an individual sample (.csv file) of the IMS data set."""

        path_run_folder = file_info_dict["path_run_folder"]
        file_name = file_info_dict["file_name"]
        sample_freq = file_info_dict["sample_freq"]
        col_names = file_info_dict["col_names"]
        run_no = file_info_dict["run_no"]
        sample_index = file_info_dict["sample_index"]

        # load the .csv file
        signals_array = np.loadtxt(path_run_folder / file_name, delimiter="\t")

        id_list = [f"{run_no}_{sample_index}"] * len(signals_array)
        run_list = [run_no] * len(signals_array)
        file_list = [file_name] * len(signals_array)
        time_step_array = np.linspace(
            0.0, len(signals_array) / sample_freq, len(signals_array)
        )

        df = pd.DataFrame(np.vstack(signals_array), columns=col_names, dtype=np.float32)
        df["id"] = id_list
        df["run"] = run_list
        df["file"] = file_list
        df["time_step"] = np.hstack(time_step_array)

        return df.astype({"id": str, "run": int, "file": str, "time_step": np.float32})

    def load_run_as_df(
        self,
        run_no: int,
        n_jobs: int = None,
    ) -> None:
        """Load the three runs as individual dataframes."""

        if run_no == 1:
            col_names = self.col_1st_names
            path_run_folder = self.path_1st_folder
        elif run_no == 2:
            col_names = self.col_2nd_names
            path_run_folder = self.path_2nd_folder
        else:
            col_names = self.col_3rd_names
            path_run_folder = self.path_3rd_folder

        # get list of every file in the folder and sort by ascending date
        file_list = sorted(os.listdir(path_run_folder))
        print("len file_list:", len(file_list))

        # create a list of dictionaries containing the metadata for each file
        file_info_list = []
        for i, file_name in enumerate(sorted(os.listdir(path_run_folder))):
            file_info_list.append(
                {
                    "path_run_folder": path_run_folder,
                    "file_name": file_name,
                    "sample_freq": self.sample_freq,
                    "col_names": col_names,
                    "run_no": run_no,
                    "sample_index": i,
                }
            )

        # get number of cpu cores
        if n_jobs is None:
            n_jobs = mp.cpu_count() - 2
        if n_jobs < 1:
            n_jobs = 1

        print("n_jobs:", n_jobs)

        with mp.Pool(processes=n_jobs) as pool:

            # from https://stackoverflow.com/a/36590187
            df_run = pool.map(self.process_raw_csv, file_info_list)
            df = pd.concat(df_run, ignore_index=True)

        col_names_ordered = ["id", "run", "file", "time_step"] + col_names

        return df[col_names_ordered]


class ImsPrepMethodA(ImsDataLoad):
    """
    Class used to prepare the IMS bearing dataset before feature engining or machine learning.

    Args:
        root (string): Root directory to place all the  data sets. (likely the raw data folder)

        dataset_folder_name (string): Name of folder containing raw data.
            This folder will be created in the root directory if not present.

        download (bool): If True, the data will be downloaded from the NASA Prognostics Repository.

        path_df_labels (Path, optional): Path to the dataframe with the labels (as a string).
            If not provided, the dataframe must be created.

        window_size (int): Size of the window to be used for the sliding window.

        stride (int): Size of the stride to be used for the sliding window.

        cut_drop_list (list, optional): List of cut numbers to drop. cut_no 17 and 94 are erroneous.
    """

    def __init__(
        self,
        root: Path,
        dataset_folder_name: str = "milling",
        download: bool = False,
        data: np.ndarray = None,
        path_df_labels: Path = None,
        window_size: int = 64,
        stride: int = 64,
        cut_drop_list: list = [17, 94],
    ) -> None:
        super().__init__(root, dataset_folder_name, download, data)

        self.window_size = window_size  # size of the window
        self.stride = stride  # stride between windows
        self.cut_drop_list = cut_drop_list  # list of cut numbers to be dropped

        if path_df_labels is not None:
            self.path_df_labels = path_df_labels
        else:
            # path of pyphm source directory using pathlib
            self.path_df_labels = (
                Path(__file__).parent
                / "auxilary_metadata"
                / "milling_labels_with_tool_class.csv"
            )

        # load the labels dataframe
        self.df_labels = pd.read_csv(self.path_df_labels)

        if self.cut_drop_list is not None:
            self.df_labels.drop(
                self.cut_drop_list, inplace=True
            )  # drop the cuts that are bad

        self.df_labels.reset_index(drop=True, inplace=True)  # reset the index

        self.field_names = self.data.dtype.names

        self.signal_names = self.field_names[7:][::-1]
        print("type field names: ", type(self.field_names))
        print("type signal names: ", type(self.signal_names))

        print(self.field_names)
        print(self.signal_names)
