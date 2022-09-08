![PyPHM Logo](./notebooks/images/logo.png)

# Machinery data, made easy
![example workflow](https://github.com/tvhahn/PyPHM/actions/workflows/main.yml/badge.svg) [![arXiv](https://img.shields.io/badge/arXiv-2205.15489-b31b1b.svg)](https://arxiv.org/abs/2205.15489)


Datasets specific to PHM (prognostics and health management). Use Python to easily download and prepare the data, before feature engineering or model training. 

Current datasets:
- **UC-Berkeley Milling Dataset**: [example notebook](https://github.com/tvhahn/PyPHM/blob/master/notebooks/milling_example.ipynb) ([open in Colab](https://colab.research.google.com/github/tvhahn/PyPHM/blob/master/notebooks/milling_example.ipynb)); [dataset source](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#milling)
- **IMS Bearing Dataset**: [dataset source](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#bearing)
- **Airbus Helicopter Accelerometer Dataset**: [dataset source](https://www.research-collection.ethz.ch/handle/20.500.11850/415151)
- More coming soon!


## Alpha Notice
PyPHM is in active development. Expect considerable changes in the near future.

Our goals are to create:

* A package that implements **common data preprocessing methods** used by others.
* A package with a **coherent and thoughtful API**.
* Thorough **documentation**, with plenty of **examples**.
* A package that is well **tested**, with the use of **type hints**.
* A package built with **continuous integration and continuous deployment**.


## Installation
Install with pip: `pip install pyphm`

Install from github repository: clone with git `clone https://github.com/tvhahn/PyPHM.git`. Then run `python -m pip install -e .` to install the package on your local machine.

Run tests: `python -m unittest discover -s tests`

