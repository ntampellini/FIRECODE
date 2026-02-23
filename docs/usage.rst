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


Input formatting
----------------

The input can be any text file, but sticking with ``.txt`` or ``.inp`` is recommended.

-  Any blank line will be ignored
-  Any line starting with ``#`` will be ignored
-  Keywords, if present, need to be on first non-blank, non-comment line

Then, molecule files are specified. A molecule line is made up of these elements, in this order:

-  Zero or more operators (*i.e.* ``csearch>``, ``opt>``, etc.) separated by spaces
-  The molecule file name (required)
-  Optional indices (numbers) and pairings (letters) for the molecule (*i.e.* ``2A 4B 5c``)
-  Optional properties of the molecule (*i.e.* ``charge=1``, ``property=value``)

An example with all four is ``refine> rsearch> butadiene.xyz 6a 8b charge=1``.

FIRECODE works with ``.xyz`` files. **Molecule indices
are zero-based! (counted starting from zero!)**

Operators
+++++++++

The core elements of every run are the operators acting on a given molecule. See the
:ref:`operators <op_kw>` page to see the full set of operators available. This should cater for
most common workflow needs, from conformational search protocols to ensemble optimizations or
double-ended TS-search methods like NEB or FSM.

Standalone Optimizer
--------------------

A standalone optimizer interface is also available calling firecode with the ``-o``/``--optimize`` option:

::

   firecode -o cation+.xyz anion-.xyz


While most options can be specified via InquirerPy command line prompts, the charge of the molecule can be read from
the filename if `-` or `+` are present.
