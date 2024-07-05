.. _installation:

Installation
============

This program is written in pure Python and it is intended to use with
Python version 3.8.10. The use of a dedicated conda virtual environment
is highly enocouraged. Installation is possible via pip:

::

    pip install firecode

After installation, run the guided utility to finalize set up:

::

    python -m firecode --setup

Defaults:

-  Force Field optimization is turned ON.
-  Force Field Calculator is XTB

Additional dependencies
=======================

External dependencies: Openbabel (see below) and at least one calculator if geometry optimization is desired.
At the moment, FIRECODE supports:

-  XTB (>=6.3) - recommended
-  AIMNET2 (via ASE) - recommended
-  ORCA (>=4.2)
-  Gaussian (>=9) - (deprecated)
-  MOPAC2016 - (deprecated)

An additional installation of Openbabel is required, as it provides i/o capabilities via cclib.
Read below on how to install these.

Openbabel (required)
--------------------

Openbabel is required, as it provides i/o capabilities via cclib. This is free software and it is available through conda:

::

    conda install -c conda-forge openbabel

Alternatively, you can download it from `the official
website <http://openbabel.org/wiki/Category:Installation>`__. If you
install the software from the website, make sure to install its Python bindings as well.
You can manually compile these by following the `website
guidelines <https://openbabel.org/docs/dev/Installation/install.html#compile-bindings>`__.

XTB (recommended for Force Field and semiempirical calculations)
----------------------------------------------------------------

This is free software. See the `GitHub
repository <https://github.com/grimme-lab/xtb>`__ and the
`documentation <https://xtb-docs.readthedocs.io/en/latest/contents.html>`__
for how to install it on your machine. The package and its python bindings are available through conda as well.

::

    conda install -c conda-forge xtb xtb-python


ORCA (optional, recommended for DFT)
------------------------------------

This software is only free for academic use at an academic institution.
Detailed instructions on how to install and set up ORCA can be found in
`the official
website <https://sites.google.com/site/orcainputlibrary/setting-up-orca>`__.
Make sure to install and set up OpenMPI along with ORCA if you wish to
exploit multiple cores on your machine **(Note: semiempirical methods
cannot yet be parallelized in ORCA!)**

Gaussian (deprecated)
---------------------

This is commercial software available at `the official
website <https://gaussian.com/>`__.


MOPAC2016 (deprecated)
----------------------

This software is closed-source but free for academic use. If you qualify
for this usage, you should `request a licence for
MOPAC2016 <http://openmopac.net/form.php>`__. After installation, be
sure to add the MOPAC folder to your system PATH, to access the program
through command line with the "MOPAC2016.exe" command. To test this, the
command ``MOPAC2016.exe`` should return
`this <https://gist.github.com/ntampellini/82224abb9db1c1880e91ad7e0682e34d>`__
message.
