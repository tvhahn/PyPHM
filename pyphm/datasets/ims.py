import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
from .pyphm import PHMDataset
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
