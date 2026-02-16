.. _installation:

Installation
============

This program is written in Python (currently >=3.12). The recommended way to install it is through conda/mamba and uv. 

::

    # create a new conda environment and install firecode with uv
    conda create --name firecode python=3.12 uv
    conda activate firecode
    uv pip install firecode

After installation, run the guided utility to finalize the setup:

::

    firecode --setup


Optional dependencies (WIP)
===========================

CREST
-----

This is free software. See the `GitHub repository <https://github.com/grimme-lab/crest>`__.
Currently only CREST 2.12 is supported, working on supporting the latest version is in progress.

::

    # install mamba if you don't have it already
    conda install -c conda-forge mamba

    mamba install -c conda-forge crest==2.12


AIMNET2 / UMA models
--------------------

The software can be interfaced with AIMNET2 and UMA models. Work is in progress to port AIMNET2 models to the new codebase.
An interface to the UMA models is available provided that the models are present on the system at the location specified in
the settings.py file (change with ``firecode --settings``). The UMA models can be obtained from HuggingFace at https://huggingface.co/facebook/UMA. 



XTB (recommended for Force Field and semiempirical calculations)
----------------------------------------------------------------

This is free software. See the `GitHub
repository <https://github.com/grimme-lab/xtb>`__ and the
`documentation <https://xtb-docs.readthedocs.io/en/latest/contents.html>`__
for how to install it on your machine. The package and its python bindings are available through conda as well.

::

    # install mamba if you don't have it already
    conda install -c conda-forge mamba

    mamba install -c conda-forge xtb xtb-python

TBLITE
------

Light-weight version of the xTB, free for academic use. See the `GitHub repository <https://github.com/tblite/tblite>`__.

::

    # install mamba if you don't have it already
    conda install -c conda-forge mamba

    mamba install -c conda-forge tblite tblite-python


ORCA
----

This software is only free for academic use at an academic institution.
Detailed instructions on how to install and set up ORCA can be found in
`the official
website <https://sites.google.com/site/orcainputlibrary/setting-up-orca>`__.
Make sure to install and set up OpenMPI along with ORCA if you wish to
exploit multiple cores on your machine **(Note: semiempirical methods
cannot yet be parallelized in ORCA!)**
