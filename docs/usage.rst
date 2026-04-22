.. _usg:

Usage
+++++

Command Line Program
====================

The program can be run from terminal, with the command:

::

    firecode myinput.txt -n [custom_title]

A custom name for the run can be optionally provided with the -n flag, otherwise a time
stamp will be used to name the output files.


Standalone Optimizer
====================

A standalone optimizer interface is also available:

::

   firecode_opt cation+.xyz anion-.xyz


::

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

Options can be specified via both command line arguments or InquirerPy command line prompts (``[-i, --interactive]`` flag). The charge of the molecule, by default, will be read from
the filename if `-` or `+` are present.
