# coding=utf-8
'''

FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2024 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

https://github.com/ntampellini/firecode

Nicolo' Tampellini - nicolo.tampellini@yale.edu

'''
import argparse
import os
import sys
from rich.traceback import install
install(show_locals=True)

__version__ = '1.1.3'

if __name__ == '__main__':


    usage = '''\n\n  🔥 python -m firecode [-h] [-s] [-t] inputfile [-n NAME]
        
        positional arguments:
          inputfile               Input filename, can be any text file.

        optional arguments:
          -h, --help              Show this help message and exit.
          -s, --setup             Guided setup of the calculation settings.
          -t, --test              Perform some tests to check the software setup.
          -n, --name NAME         Specify a custom name for the run.
          -cl,--command_line      Read instructions from the command line instead of from an input file.
          -c, --cite              Print citation links.
          -p, --profile           Profile the run through cProfiler.
          -b, --benchmark FILE    Benchmark the geometry optimization of FILE (.xyz) to get the optimal number of procs per job.
          --procs                 Number of processors to be used by each optimization job.
          '''

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("-s", "--setup", help="Guided setup of the calculation settings.", action="store_true")
    parser.add_argument("-t", "--test", help="Perform some tests to check the software setup.", action="store_true")
    parser.add_argument("-cl", "--command_line", help="Read instructions from the command line instead of from an input file.", action="store")
    parser.add_argument("inputfile", help="Input filename, can be any text file.", action='store', nargs='?', default=None)
    parser.add_argument("-n", "--name", help="Specify a custom name for the run.", action='store', required=False)
    parser.add_argument("-c", "--cite", help="Print the appropriate document links for citation purposes.", action='store_true', required=False)
    parser.add_argument("-p", "--profile", help="Profile the run through cProfiler.", action='store_true', required=False)
    parser.add_argument("-b", "--benchmark", help=("Benchmark the geometry optimization of FILE to get the optimal number " +
                        "of procs per job."), action='store', required=False, default=False)
    # parser.add_argument("-r", "--restart", help="Restarts previous run from an embedder.pickle object.", action='store', required=False, default=False)
    parser.add_argument("--procs", help="Number of processors to be used by each optimization job.", action='store', required=False, default=None)

    args = parser.parse_args()

    if (not (args.test or args.setup or args.command_line or args.benchmark)) and args.inputfile is None:
        parser.error("One of the following arguments are required: inputfile, -t, -s, -b.\n")

    if args.benchmark:
        from firecode.concurrent_test import run_concurrent_test
        run_concurrent_test(args.benchmark)
        sys.exit()

    if args.setup:
        from firecode.modify_settings import run_setup
        run_setup()
        sys.exit()

    if args.cite:
        print('No citation link is available for FIRECODE yet. You can link to the code on https://www.github.com/ntampellini/firecode')
        sys.exit()

    if args.test:
        from firecode.tests import run_tests
        run_tests()
        sys.exit()

    if args.command_line:
        
        filename = 'input_firecode.txt'
        with open(filename, 'w') as f:
            f.write(args.command_line)

        args.inputfile = filename

    filename = os.path.realpath(args.inputfile)

    from firecode.embedder import Embedder

    if args.profile:
        from firecode.profiler import profiled_wrapper
        profiled_wrapper(filename, args.name)
        sys.exit()

    # if args.restart:
    #     import pickle
    #     with open(args.restart, 'rb') as _f:
    #         embedder = pickle.load(_f)
        # initialize embedder from pickle file

        # embedder.run()
        # run the program

    # import faulthandler
    # faulthandler.enable()

    embedder = Embedder(filename, stamp=args.name, procs=args.procs)
    # initialize embedder from input file

    embedder.run()
    # run the program