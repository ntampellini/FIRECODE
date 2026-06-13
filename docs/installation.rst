.. _installation:

Installation
++++++++++++

This program is written in Python (currently >=3.12). The recommended way to install it is through conda/mamba and uv.
Contributors are encouraged to install the program via `pixi <https://pixi.prefix.dev/latest/>`__ for easier development.

::

    # create a new conda environment and install firecode with uv
    conda create --name firecode python=3.12 uv
    conda activate firecode
    uv pip install firecode

After installation, run the guided utility to finalize the setup:

::

    firecode -s


Optional dependencies
=====================

Various optional dependencies can be installed through conda/mamba.

CREST
-----

This is free software. See the `GitHub repository <https://github.com/grimme-lab/crest>`__.
Currently both CREST versions 2.12 and 3.0+ are supported.

::

    conda install -c conda-forge crest


AIMNET2 models
--------------

AIMNET2 models can be accessed if the aimnet library is installed.

::

    uv pip install aimnet[ase]

UMA models
----------

UMA models can be accessed if the fairchem-core library is installed and the path
to a model (.pt file) is specified in ``settings.py``. This can be done by running
``firecode -s``, manually modifying ``settings.py`` or via :ref:`environment variables <env_vars>`.


::

    uv pip install faichem-core


XTB
---

This is free software. See the `GitHub
repository <https://github.com/grimme-lab/xtb>`__ and the
`documentation <https://xtb-docs.readthedocs.io/en/latest/contents.html>`__
for how to install it on your machine. The calculator is available through conda as well.

::

    conda install -c conda-forge xtb

TBLITE (prefer over xTB)
------------------------

Light-weight version of the xTB, free for academic use. See the `GitHub repository <https://github.com/tblite/tblite>`__.

::

    conda install -c conda-forge "tblite>=0.5.0" tblite-python



ORCA
----

ORCA is free for academic use.
Detailed instructions on how to install and set up ORCA can be found on
`the official
website <https://sites.google.com/site/orcainputlibrary/setting-up-orca>`__.
Make sure to install and set up OpenMPI along with ORCA if you wish to
exploit multiple cores on your machine (the command ``mpirun`` should be available).


Running tests
-------------

After installing all the required dependencies for your tasks, you can confirm that the software runs smoothly
by running the tests shipped with the code.

- Installations via `pip`: you can find where the installation folder is by running:

::

    pip show firecode

The input files for the tests are located in ``[your installation directory]/firecode/tests``.

- Installation via pixi: you can run all the CI tests in one command. In the installation root, run:

::

    pixi run -e dev test_cov
