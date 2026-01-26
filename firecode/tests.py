# coding=utf-8
'''
FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2026 Nicolò Tampellini

SPDX-License-Identifier: LGPL-3.0-or-later

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see
https://www.gnu.org/licenses/lgpl-3.0.en.html#license-text.

'''

import sys


def run_tests():
    
    import os
    import time
    from subprocess import CalledProcessError

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    from firecode.settings import (CALCULATOR, COMMANDS, DEFAULT_FF_LEVELS,
                                   DEFAULT_LEVELS, FF_CALC, FF_OPT_BOOL, PROCS)

    if CALCULATOR not in ('AIMNET2', 'TBLITE', 'ORCA', 'XTB'):
        raise Exception(f'{CALCULATOR} is not a valid calculator. Use AIMNET, TBLITE, ORCA or XTB.')

    import numpy as np
    from ase.atoms import Atoms
    from ase.optimize import LBFGS
    from prism_pruner.utils import time_to_string

    from firecode.optimization_methods import Opt_func_dispatcher
    from firecode.utils import (HiddenPrints, clean_directory, loadbar,
                                read_xyz, run_command, suppress_stdout_stderr)

    os.chdir('tests')

    t_start_run = time.perf_counter()

    data = read_xyz('C2H4.xyz')

    dispatcher = Opt_func_dispatcher(CALCULATOR)
    ase_calc = dispatcher.get_ase_calc(DEFAULT_LEVELS[CALCULATOR], None)

    ##########################################################################

    print('\nRunning tests for FIRECODE. Settings used:')
    print(f'{CALCULATOR=}')

    if CALCULATOR in ('ORCA', 'XTB'):
        print(f'{CALCULATOR} COMMAND = {COMMANDS[CALCULATOR]}')

        print('\nTesting raw (non-ASE) {CALCULATOR} calculator...')

        dispatcher.opt_func(
                            data.atoms,
                            data.coords[0],
                            method=DEFAULT_LEVELS[CALCULATOR],
                            procs=PROCS,
                            read_output=False)

        print(f'{CALCULATOR} raw calculator works.')

    else:
        atoms = Atoms('HH', positions=np.array([[0, 0, 0], [0, 0, 1]]))
        atoms.calc = ase_calc
        
        with suppress_stdout_stderr():
            LBFGS(atoms, logfile=None).run()

        clean_directory()
        print(f'{CALCULATOR} ASE calculator works.')
    
    ##########################################################################

    print(f'\n{FF_OPT_BOOL=}')
    ff = f'on. Calculator is {FF_CALC}. Checking its status.' if FF_OPT_BOOL else 'off.'
    print(f'Force Field optimization is turned {ff}')

    if FF_OPT_BOOL:
        if FF_CALC == 'XTB':
            dispatcher.opt_func(
                                    data.atoms,
                                    data.coords[0],
                                    method=DEFAULT_FF_LEVELS[FF_CALC],
                                    procs=PROCS,
                                    read_output=False)

            print('XTB FF non-ASE calculator works.')

            ##########################################################################
        
            atoms = Atoms('HH', positions=np.array([[0, 0, 0], [0, 0, 1]]))
            atoms.calc = ase_calc
            LBFGS(atoms, logfile=None).run()

            clean_directory()
            print('XTB ASE calculator works.')

        else:
            print('FF optimization only possible via XTB: skipping FF calc check.')

    print('\nNo installation faults detected with the current settings. Running tests.')

    ##########################################################################

    tests = []
    for folder in os.listdir():
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                if file.endswith('.txt'):
                    tests.append((os.path.abspath(folder), os.path.basename(file)))

    # os.chdir(os.path.dirname(os.getcwd()))
    # os.chdir('firecode')
    # # Back to ./firecode

    times = []
    for i, (folder, filename) in enumerate(tests):

        os.chdir(folder)

        name = filename.split('\\')[-1].split('/')[-1][:-4] # trying to make it work for either Win, Linux (and Mac?)
        loadbar(i, len(tests), f'Running FIRECODE tests ({name}): ')
        
        t_start = time.perf_counter()
        try:
            # print(f'python -m firecode {filename} -n {name} [in {os.getcwd()}]')
            with HiddenPrints():
                run_command(f'python -m firecode {filename} -n {name}')

        except CalledProcessError as error:
            print('\n\n--> An error occurred:\n')
            print(error.stderr.decode("utf-8"))
            sys.exit()
                    
        t_end = time.perf_counter()
        times.append(t_end-t_start)

    loadbar(len(tests), len(tests), f'Running FIRECODE tests ({name}): ')  

    print()
    for i, f in enumerate(tests):
        print('    {:25s}{} s'.format(f[1].split('\\')[-1].split('/')[-1][:-4], round(times[i], 3)))

    print(f'FIRECODE tests completed with no errors. ({time_to_string(time.perf_counter() - t_start_run)})\n')