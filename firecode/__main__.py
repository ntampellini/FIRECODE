# coding=utf-8
"""FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2026 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

https://github.com/ntampellini/firecode

Nicolo' Tampellini - ntamp@mit.edu

"""

import argparse
import os
import sys
from io import TextIOWrapper

from rich.traceback import install

install(show_locals=True)


def main() -> None:
    # Redirect stdout and stderr to handle encoding errors
    sys.stdout = TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
    )

    sys.stderr = TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
    )

    usage = """\n\n    🔥 python -m firecode [-h] [-s] [-t] input.txt [-n NAME] [-p]
    🔥 python -m firecode -cl "refine> crest_search> mol.xyz"
    🔥 python -m firecode -c
    🔥 python -m firecode -o mol.xyz

        positional arguments:
          inpufile.txt            Input filename, can be any text file.

        optional arguments:
          -h, --help              Show this help message and exit.
          -s, --setup             Guided setup of the calculation settings.
          -n, --name NAME         Specify a custom name for the run.
          -cl,--command_line      Read instructions from the command line instead of from an input file.
          -c, --cite              Print citation links.
          -p, --profile           Profile the run through cProfiler.

          """

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument(
        "-s", "--setup", help="Guided setup of the calculation settings.", action="store_true"
    )
    parser.add_argument(
        "-cl",
        "--command_line",
        help="Read instructions from the command line instead of from an input file.",
        action="store",
    )
    parser.add_argument(
        "inputfile",
        help="Input filename, can be any text file.",
        action="store",
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "-n", "--name", help="Specify a custom name for the run.", action="store", required=False
    )
    parser.add_argument(
        "-c",
        "--cite",
        help="Print the appropriate document links for citation purposes.",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-p",
        "--profile",
        help="Profile the run through cProfiler.",
        action="store_true",
        required=False,
    )

    args = parser.parse_args()

    env_variables_handling()

    if (not (args.setup or args.command_line)) and args.inputfile is None:
        parser.error("One of the following arguments are required: inputfile, -t, -s.\n")

    if args.setup:
        from firecode.modify_settings import run_setup

        run_setup()
        sys.exit(0)

    if args.cite:
        print(
            "No citation link is available for FIRECODE yet. You can link to the code on https://www.github.com/ntampellini/firecode"
        )
        sys.exit(0)

    if args.command_line:
        filename = "input_firecode.txt"
        with open(filename, "w") as f:
            f.write(args.command_line)

        args.inputfile = filename

    filename = os.path.realpath(args.inputfile)

    from firecode.embedder import Embedder

    if args.profile:
        from firecode.profiler import profiled_wrapper

        profiled_wrapper(filename, args.name)
        sys.exit(0)

    embedder = Embedder(filename, stamp=args.name)
    # initialize embedder from input file

    embedder.run()
    # run the program


def env_variables_handling() -> None:
    """Handles global environment variables and associated processes.

    Priority should be given to handling env vars with locally-scoped
    context managers, if possible (see the env_override function).
    """
    from pathlib import Path
    from shutil import rmtree

    # remove compilation cache for jax: we might be running on different
    # hardware from the last firecode run, and that might result in nasty
    # compatibility issues with stale compilation of the jax library.
    jax_comp_cache_dir = Path.home() / ".cache/sella/jax_cache"
    rmtree(str(jax_comp_cache_dir), ignore_errors=True)

    # export "FIRECODE_*" environment variables
    from firecode.settings import ENV_VARS

    for key, value in ENV_VARS.items():
        os.environ.setdefault(key, value)


if __name__ == "__main__":
    main()
