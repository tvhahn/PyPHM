import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
from .pyphm import PHMDataset
import datetime
import time
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
        "https://drive.google.com/file/d/1iJqTYQpHst_uYSyU5d2THsZkA8Vk6Inx/view?usp=sharing",
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
    def process_raw_csv_to_dict(file_info_dict) -> None:
        """Load an individual sample (.csv file) of the IMS data set."""

        path_run_folder = file_info_dict["path_run_folder"]
        file_name = file_info_dict["file_name"]
        run_no = file_info_dict["run_no"]
        sample_index = file_info_dict["sample_index"]

        # load the .csv file
        signals_array = np.loadtxt(path_run_folder / file_name, delimiter="\t")

        # get the start time (for the first sample) and convert to unix timestamp
        start_time_unix = time.mktime(
            datetime.datetime.strptime(file_name, "%Y.%m.%d.%H.%M.%S").timetuple()
        )

        # create dictionary with the signals_array, id_list, run_list, file_list, time_step_array
        data_dict = {
            "signals_array": signals_array,
            "id": f"{run_no}_{sample_index}",
            "run_no": run_no,
            "file_name": file_name,
            "sample_index": sample_index,
            "start_time_unix": start_time_unix,
        }

        return data_dict

    def load_run_as_dict(
        self,
        run_no: int,
        n_jobs: int = None,
    ) -> None:
        if run_no == 1:
            col_names = self.col_1st_names
            path_run_folder = self.path_1st_folder
        elif run_no == 2:
            col_names = self.col_2nd_names
            path_run_folder = self.path_2nd_folder
        else:
            col_names = self.col_3rd_names
            path_run_folder = self.path_3rd_folder

        # create a list of dictionaries containing the metadata for each file
        file_info_list = []
        for i, file_name in enumerate(sorted(os.listdir(path_run_folder))):
            file_info_list.append(
                {
                    "path_run_folder": path_run_folder,
                    "file_name": file_name,
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
            data_list = pool.map(self.process_raw_csv_to_dict, file_info_list)

        # store the data from data_list as a dictionary, with the key being the file name
        data_dict = {}
        for data_dict_i in data_list:
            data_dict[data_dict_i["file_name"]] = data_dict_i
        return data_dict

    @staticmethod
    def process_raw_csv_to_df(file_info_dict) -> None:
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

        # load the dataframes in parallel
        with mp.Pool(processes=n_jobs) as pool:

            # from https://stackoverflow.com/a/36590187
            df_run = pool.map(self.process_raw_csv_to_df, file_info_list)
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
        dataset_folder_name: str = "ims",
        download: bool = False,
    ) -> None:
        super().__init__(
            root,
            dataset_folder_name,
            download,
        )

    def create_xy_arrays(
        self,
        run_no: int = 1,
        n_jobs: int = None,
    ) -> None:

        # create a list to store the x and y arrays
        x = []  # instantiate X's
        y_ids_runs_files_times_ctimes = []  # instantiate y's

        # create the data dict storing the signals and metadata
        data_dict = self.load_run_as_dict(run_no, n_jobs)

        # get all the file names from the data_dict and sort them
        file_names = sorted(data_dict.keys())

        for i, file_name in enumerate(file_names):

            x.append(data_dict[file_name]["signals_array"])
            y_ids_runs_files_times_ctimes.append(
                [
                    data_dict[file_name]["id"],
                    data_dict[file_name]["run_no"],
                    data_dict[file_name]["file_name"],
                    data_dict[file_name]["sample_index"],
                    data_dict[file_name]["start_time_unix"],
                ]
            )

        x = np.stack(x)
        n_samples = x.shape[0]
        n_signals = x.shape[2]

        return x, np.stack(y_ids_runs_files_times_ctimes).reshape(-1, 5)

    def create_xy_df(
        self,
        run_no: int = 1,
        n_jobs: int = None,
    ) -> None:
        return self.load_run_as_df(run_no, n_jobs)
