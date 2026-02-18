
# FIRECODE - Filtering Refiner and Embedder for Conformationally Dense Ensembles

<div align="center">

[![License: GNU LGPL v3](https://img.shields.io/github/license/ntampellini/firecode)](https://opensource.org/licenses/LGPL-3.0)
![Python Version](https://img.shields.io/badge/Python-3.12-blue)
![Size](https://img.shields.io/github/languages/code-size/ntampellini/firecode)
![Lines](https://sloc.xyz/github/ntampellini/firecode/)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/ntampellini/firecode)](https://www.codefactor.io/repository/github/ntampellini/firecode)
[![codecov](https://codecov.io/gh/ntampellini/FIRECODE/graph/badge.svg?token=D9TM6S33D8)](https://codecov.io/gh/ntampellini/FIRECODE)

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

FIRECODE is a computational chemistry workflow driver for the generation, optimization and refinement of conformational ensembles, also implementing some transition state ultilities.

It implements flexible and customizable workflows for conformer generation (via [CREST](https://github.com/crest-lab/crest), [RDKit](https://github.com/rdkit/rdkit)), double-ended TS search ([NEB](https://ase-lib.org/ase/neb.html) via [ASE](https://github.com/rosswhitfield/ase), [ML-FSM](https://github.com/thegomeslab/ML-FSM)), and (constrained) ensemble optimization through popular calculators like [XTB](https://github.com/grimme-lab/xtb), [TBLITE](https://github.com/tblite/tblite), [ORCA](https://www.orcasoftware.de/tutorials_orca/), and Pytorch Neural Network models ([AIMNET2](https://github.com/isayevlab/AIMNet2), [UMA](https://huggingface.co/facebook/UMA)) via [ASE](https://github.com/rosswhitfield/ase).

Conformational pruning is performed with the now standalone [PRISM Pruner](https://github.com/ntampellini/prism_pruner).

As a legacy feature from [TSCoDe](https://github.com/ntampellini/TSCoDe), FIRECODE can also assemble non-covalent adducts from conformational ensembles (embedding) programmatically.

## Installation

The package is distributed via `pip`, and the use of [`uv`](https://docs.astral.sh/uv/) is highly recommended. The default installation is minimalistic, and torch/GPU support requires dedicated installs:

```python
uv pip install firecode           # XTB, TBLITE, ORCA
uv pip install firecode[aimnet2]  # + AIMNET2
uv pip install firecode[uma]      # + UMA/OMOL
uv pip install firecode[full]     # + AIMNET2, UMA/OMOL
```

More installation details in the documentation.

## Documentation
Additional documentation on how to install and use the program can be found on [readthedocs](https://firecode.readthedocs.io/en/latest/index.html).