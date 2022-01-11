import bz2
import os
import os.path
import hashlib
import gzip
import re
import tarfile
from typing import Any, Callable, List, Iterable, Optional, TypeVar, Dict, IO, Tuple, Iterator
from urllib.parse import urlparse
import zipfile
import lzma
import urllib
import urllib.request
import urllib.error
import pathlib
import itertools
from tqdm.auto import tqdm

USER_AGENT = "pytorch/vision"

# From torchvision, copyright BSD-3-Clause, see LICENSE. https://github.com/pytorch/vision
def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)