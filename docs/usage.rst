.. _usg:

Usage
=====

The program can be run from terminal, with the command:

::

    firecode myinput.txt -n [custom_title]

A custom name for the run can be optionally provided with the -n flag, otherwise a time
stamp will be used to name the output files.

A barebones command-line invocation is also possible withthe ``-cl``/``--command_line`` argument:

::

    firecode -cl "csearch> molecule.xyz"


Standalone Optimizer
--------------------

A standalone optimizer interface is also available calling firecode with the ``-o``/``--optimize`` option:

::

   firecode -o cation+.xyz anion-.xyz


Any option can be specified via InquirerPy command line prompts. The charge of the molecule, by default, will be read from
the filename if `-` or `+` are present.
