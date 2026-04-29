---
title: 'FIRECODE: A Computational Chemistry Workflow Driver and Modular Hub'
tags:
  - Python
  - chemistry
  - workflow driver
  - conformational search
  - conformational ensemble
  - machine learned interatomic potentials
  - semiempirical methods
  - transition states
  - thermochemistry
authors:
  - name: Nicolò Tampellini
    corresponding: true
    orcid: 0000-0002-6611-556X
    affiliation: 1
affiliations:
  - name: Massachusetts Institute of Technology, USA
    index: 1
date: 23 April 2026
bibliography: paper.bib

---

# Summary

Computational chemistry, the branch of this science dedicated to the modeling of molecular systems for the extraction of properties of interest, is experiencing a recent renaissance thanks to impressive advances in theoretical models enabling ever-faster simulations of complex processes via density functional theory (DFT), semiempirical, and machine-learned interatomic potentials (MLIPs). However, the reduced computational cost will only translate into faster and more robust modeling with the help of automation frameworks, where trivial human intervention does not become the new bottleneck of computational pipelines.

# Statement of need

`FIRECODE` is a modular workflow driver for computational chemistry written in Python. The code interfaces with a series of fast modern calculators (xTB [@xtb], tblite [@tblite], AIMNET2 [@aimnet2], UMA [@uma], usually via ASE [@ase]) and exposes a series of workflows and utilities for geometry optimization, frequency analysis, one- and two-ended transition state search, and conformational search. When possible, these routines are implemented in a calculator-independent way, and almost all of these can be arbitrarily chained to one another to define a complex workflow.

The software was born out of necessity over the last five years of research in asymmetric catalysis: `FIRECODE` was designed to streamline the necessary use of multiple software and routines in the computational modeling of small molecules, in a single centralized hub. The program can be called via command line with a minimal plain text input file defining the workflow, or used as a Python library in jupyter notebooks. The modular nature of the software enables rapid addition of new features, facilitating external contributions, while also retaining a small installation footprint for the core library. The concise input syntax also offers a streamlined definition of workflows and software orchestration with no coding experience required.


# State of the field

While a manifold of computational chemistry software exists, workflow utilities are less common, particularly in the free and open source community. Recently, version 6.0 of `ORCA` [@orca1; @orca2] (free for academic use) started offering compound job functionality, which enables the chained execution of some core ORCA modules. The code is however not open source and functionality is limited to what is included in the ORCA program (*i.e.* no MLIPs, no similarity pruning, no ensemble processing...). While ORCA workflows are more extensively implemented in `WEASEL` [@weasel], this is commercial, closed-source software.

The excellent AQME software [@aqme] features the most similar design philosophy to the one of this work. Key differences with `FIRECODE`, besides code architecture, lie in: 1) the main interaction platform of `AQME` happening via jupyter notebooks, while the software of this work favors the use of plain text input files, keeping the inputs as minimal as possible. 2) The present software having a reactivity-centered set of features, chiefly via fast semiempirical and ML calculators, over `AQME`'s focus on DFT and on the generation of ensemble properties like spectra and molecular descriptors.

Other examples of related software, targeting the exposure of high-level primitives over structured workflows, include `Cuby` [@cuby; @cubygithub] (MD and high-level DFT) written in Ruby, and the popular `ASE` [@ase], which is extensively used as a lower level interface for interacting with calculators in `FIRECODE`.

The development of this software from scratch was motivated by multiple reasons. First, a modern and flexible hub for running computational chemistry workflows, particularly tailored to modeling chemical reactivity and MLIPs, was lacking. While the benefit of developing standalone applications for specific tasks remains, their integration in a single hub permits automation and lowers the barrier of entry for adoption of both individual tools and workflows alike. Second, modern machine-learned interatomic potentials (MLIPs) are more conveniently ran via Python libraries like `torch` [@torch], benefitting greatly from a Python-based workflow manager. Additional features like implicit delta solvation for gas phase-trained MLIPs [^deltasolv] and numerical quasi-RRHO thermochemistry [@qrrho] are additional core library functionalities that were not available in other actively maintained modules at the time of development.

[^deltasolv]: That is, a calculator providing the energy and gradient differences between an atomic structure in an implicit solvent and the same structure in vacuum.

# Software design

The software is developed with the philosophy of maintaining the greatest amount of functionality and module interoperability with the minimal list of dependencies, which should ideally be as modern and lean as possible. All core dependencies are distributed via `pypi`, keeping the installation of the minimal version of the program down to a single command (`uv pip install firecode`).

Non-essential dependencies can be installed based on the desired calculator and interfaces to be used, all via conda/mamba.

The software implements various calculators (xTB [@xtb], tblite [@tblite], AIMNET2 [@aimnet2], UMA [@uma]) via ASE[@ase], which can be used interchangeably in various ensemble routines. Other core libraries leveraged throughout the codebase are `numpy` [@numpy] for algebraic manipulations, `networkx` [@networkx] for graph utilities and `prism_pruner` [@prism_pruner] for the removal of duplicates. `Sella` [@sella] provides a saddle point optimizer, and `rdkit` [@rdkit] is used for `ETKDG` [@etkdg] conformational searches and SMARTS substructure matching. The `ML-FSM` library [@mlfsm; @mlfsm_paper] provides a two-ended TS search utility complementing ASE's implementation of the `NEB` method [@neb].

Other standalone conformational search software is instead interfaced via subprocess calls and file-based I/O, like `CREST` [@crest1; @crest2], ORCA's `GOAT` [@goat] and `racerts` [@racerts; @racerts_paper].

The modular nature of the calculators and interfaces allows the definition of complex workflows in plain text files (input.txt):

```bash
    firecode input.txt -n test_run
```

A second standalone executable exposes the core optimizer routines acting directly on structure files:

```bash
    firecode_opt conformers.xyz --opt --freq --solvent toluene [...]
```

Contributions by way of adding new routines (or interfaces) to the codebase is made straightforward by design: individual "operators" (routines) are simply functions reading a molecular `.xyz` file and returning the filename of the processed ensemble. Run information can be accessed from the Embedder class or via environment variables.

```python
def center_operator(filename: str, embedder: Embedder) -> str:
    """Example operator centering the molecule."""

    # the Embedder class stores global information
    embedder.avail_gpus # 1
    embedder.options.solvent # "ch2cl2"
    embedder.options.T # 298.15

    # get a copy of the molecule of interest
    mol = embedder.mols[filename]

    # center coordinates
    mol.coords[0] -= np.mean(mol.coords[0], axis=0)

    # save any data you might need later
    embedder.options.centered = True
    embedder.options.last_operator = "center"

    # write to global log
    embedder.log(f"--> Center operator: centered molecule {mol.basename}")

    # write outfile and return its name
    outfile = f"{mol.basename}_centered.xyz"
    mol.to_xyz(outfile)

    return outfile
```

# Research impact statement

Since its earliest versions in 2021 (formerly [TSCoDe](https://github.com/ntampellini/TSCoDe)), this software proved essential in orchestrating asymmetric catalysis workflows with ensembles up to thousands of structures [@imidation; @s_oxidation; @rings; @amination; @lineage; @halle_bpy]. Adoption is slowly starting to grow with `pypi` reporting ~150 downloads per month at the present time. Part of the core code was exported to the `prism_pruner` library [@prism_pruner] to facilitate its adoption by the computational chemistry startup Rowan [@rowan] (not affiliated with the author).

# AI usage disclosure

Generative AI tools (Claude, Gemini) were used in some cases to generate first drafts of Python functions that were then manually curated line-by-line and interfaced with the rest of the codebase by hand. No generative AI tools were used in the writing of this manuscript.

# Acknowledgements

We acknowledge years of research support from both Alma Mater Studiorum - University of Bologna (Prof. Giorgio Bencivenni and Prof. Paolo Righi) and Yale University (Prof. Scott J. Miller). These environments provided both the computational resources to run and develop the software as well as the primary problems it was designed to address and was stress-tested against.

# References
