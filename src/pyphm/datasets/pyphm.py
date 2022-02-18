import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from .utils import download_and_extract_archive, extract_archive, check_integrity


class PHMDataset:
    """
    Base class for making PyPHM data sets.

    Args:
        root (string): Root directory to place all the  data sets.

        dataset_folder_name (string): Name of folder containing raw data.
            This folder will be created in the root directory if not present.

    """

    def __init__(
        self,
        root: Path,
        dataset_folder_name: str,
    ) -> None:

        self.root = Path(root)
        self.dataset_folder_name = dataset_folder_name
