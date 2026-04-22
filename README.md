
# FIRECODE - Filtering Refiner and Embedder for Conformationally Dense Ensembles

<div align="center">

[![License: GNU LGPL v3](https://img.shields.io/github/license/ntampellini/firecode)](https://opensource.org/licenses/LGPL-3.0)
![Python Version](https://img.shields.io/badge/Python-3.12-blue)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
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

FIRECODE is a computational chemistry workflow driver and hub for the generation, optimization and refinement of conformational ensembles, including  transition state and thermochemical utilities.



<!-- It runs flexible workflows for conformer generation (via [CREST](https://github.com/crest-lab/crest), [RDKit](https://github.com/rdkit/rdkit)), double-ended TS search ([NEB](https://ase-lib.org/ase/neb.html) via [ASE](https://github.com/rosswhitfield/ase), [ML-FSM](https://github.com/thegomeslab/ML-FSM)), and (constrained) ensemble optimization through popular calculators like [XTB](https://github.com/grimme-lab/xtb), [TBLITE](https://github.com/tblite/tblite), [ORCA](https://www.orcasoftware.de/tutorials_orca/), and Pytorch Neural Network models ([AIMNET2](https://github.com/isayevlab/AIMNet2), [UMA](https://huggingface.co/facebook/UMA)) via [ASE](https://github.com/rosswhitfield/ase).

Conformational pruning is performed with the now standalone [PRISM Pruner](https://github.com/ntampellini/prism_pruner).

As a legacy feature from [TSCoDe](https://github.com/ntampellini/TSCoDe), FIRECODE can also assemble non-covalent adducts from conformational ensembles (embedding) programmatically. -->

## Calculators

- [xTB](https://github.com/grimme-lab/xtb) *(native)*
- [tblite](https://github.com/tblite/tblite) *(via [ASE](https://github.com/rosswhitfield/ase))*
- [AIMNET2](https://github.com/isayevlab/AIMNet2) *(via [ASE](https://github.com/rosswhitfield/ase))*
- [UMA](https://huggingface.co/facebook/UMA) *(via [ASE](https://github.com/rosswhitfield/ase))*

## Interfaces / utilities

- [CREST](https://github.com/crest-lab/crest) *(conformational search)*
- [GOAT](https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.202500393) *(conformational search)*
- [racerts](https://github.com/digital-chemistry-laboratory/racerts) *(conformational search)*
- [ETKDG](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00025) *(via [rdkit](https://github.com/rdkit/rdkit), conformational search)*
- [TSCoDe](https://github.com/ntampellini/TSCoDe) *(conformational embedding)*
- [prism_pruner](https://github.com/ntampellini/prism_pruner) *(conformational pruning)*
- [ML-FSM](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00025) *(two-ended TS search)*
- [Sella](https://github.com/zadorlab/sella) *(saddle point optimization)*
<!-- - [packmol](https://github.com/m3g/packmol) *(explicit solvation)* -->

...plus frequency calculation, NEB optimization, and more are all implemented in the code in a calculator-agnostic way.

## Installation

The package is distributed via `pip`, and the use of [`uv`](https://docs.astral.sh/uv/) is highly recommended. The default installation is minimalistic, and torch/GPU support requires dedicated installs:

```python
uv pip install firecode           # XTB, TBLITE, ORCA
uv pip install firecode[aimnet2]  # + AIMNET2
uv pip install firecode[uma]      # + UMA/OMOL
uv pip install firecode[full]     # + AIMNET2, UMA/OMOL
```

More installation details in the documentation.

## Usage

Installation exposes the main program working on a plain text file as well as a standalone optimizer.

```
🔥 firecode [-h] [-s] [-t] input.txt [-n NAME] [-p]

    positional arguments:
      inpufile.txt            Input filename, can be any text file.

    optional arguments:
      -h, --help              Show this help message and exit.
      -s, --setup             Guided setup of the calculation settings.
      -n, --name NAME         Specify a custom name for the run.
      -cl,--command_line      Read instructions from the command line instead of from an input file.
      -p, --profile           Profile the run through cProfiler.
```

```
🔥 firecode_opt [-h] [-i] [-t TEMPERATURE] [-c CALCULATOR] [-m METHOD] [-s SOLVENT] [-o] [-f] [--ts] [--irc] [--cfile CFILE] [-n] [--debug]
                    filenames [filenames ...]

positional arguments:
  filenames             Input filename(s), in .xyz format

options:
  -h, --help            show this help message and exit
  -i, --interactive     Set options interactively.
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature, in degrees Celsius.
  -c CALCULATOR, --calculator CALCULATOR
                        Calculator (default UMA).
  -m METHOD, --method METHOD
                        Method (default OMOL for UMA).
  -s SOLVENT, --solvent SOLVENT
                        Solvent (default ch2cl2).
  -o, --opt             Optimize the geometry.
  -f, --freq            Perform vibrational analysis.
  --ts, --saddle        Optimize to a TS.
  --irc                 Run an IRC calculation.
  --cfile CFILE         Uses a constraint file.
  -n, --newfile         Write optimized structure to a new file (*_opt.xyz).
  --debug               Does not delete optimization data.
```

## Documentation
Documentation on how to install and use the program can be found on [readthedocs](https://firecode.readthedocs.io/en/latest/index.html).
