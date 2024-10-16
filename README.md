
# FIRECODE - Filtering Refiner and Embedder for Conformationally Dense Ensembles

<div align="center">

[![License: GNU LGPL v3](https://img.shields.io/github/license/ntampellini/firecode)](https://opensource.org/licenses/LGPL-3.0)
![Python Version](https://img.shields.io/badge/Python-3.8.10-blue)
![Size](https://img.shields.io/github/languages/code-size/ntampellini/firecode)
![Lines](https://sloc.xyz/github/ntampellini/firecode/)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/ntampellini/firecode)](https://www.codefactor.io/repository/github/ntampellini/firecode)

[![PyPI](https://img.shields.io/pypi/v/firecode)](https://pypi.org/project/firecode/)
[![Wheel](https://img.shields.io/pypi/wheel/firecode)](https://pypi.org/project/firecode/)
[![Documentation Status](https://readthedocs.org/projects/firecode/badge/?version=latest)](https://firecode.readthedocs.io/en/latest/?badge=latest)
![PyPI - Downloads](https://img.shields.io/pypi/dm/firecode)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fntampellini_&label=%40ntampellini_&link=https%3A%2F%2Ftwitter.com%2Fntampellini_)

</div>

<p align="center">

  <img src="docs/images/logo.png" alt="FIRECODE logo" class="center" width="500"/>

</p>

*FIRECODE is the expanded successor of [TSCoDe](https://github.com/ntampellini/TSCoDe).* 

FIRECODE is a computational chemistry toolbox for the generation, optimization and refinement of conformational ensembles. It features many flexible and highly customizable workflow utilities including conformer generation (via [CREST](https://github.com/crest-lab/crest) or FIRECODE), constrained ensemble optimization through popular calculators like [XTB](https://github.com/grimme-lab/xtb), [ORCA](https://www.orcasoftware.de/tutorials_orca/), [GAUSSIAN](https://gaussian.com/) and Pytorch Neural Network models via [ASE](https://github.com/rosswhitfield/ase) ([AIMNET2](https://github.com/isayevlab/AIMNet2)). It implements a series of conformational pruning routines based on inertia tensors, RMSD, symmetry-corrected RMSD, and more. It can also assemble non-covalent adducts from conformational ensembles (embedding) for fast and automated generation and evaluation of ground and transtition state-like structures. CPU and GPU multithreading is implemented throughout the codebase and linear algebra-intensive modules are compiled at runtime via [Numba](https://github.com/numba/numba).


## Documentation
Documentation on how to install and use the program can be found on [readthedocs](https://firecode.readthedocs.io/en/latest/index.html).