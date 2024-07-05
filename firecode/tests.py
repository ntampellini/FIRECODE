# coding=utf-8
'''
FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2024 Nicolò Tampellini

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

    if CALCULATOR not in ('MOPAC','ORCA','GAUSSIAN','XTB'):
        raise Exception(f'{CALCULATOR} is not a valid calculator. Use MOPAC, ORCA, GAUSSIAN or XTB.')

    import numpy as np
    from ase.atoms import Atoms
    from ase.optimize import LBFGS

    from firecode.ase_manipulations import get_ase_calc
    from firecode.optimization_methods import Opt_func_dispatcher
    from firecode.utils import (HiddenPrints, clean_directory, loadbar, read_xyz,
                              run_command, time_to_string)

    os.chdir('tests')

    t_start_run = time.perf_counter()

    data = read_xyz('C2H4.xyz')

    dispatcher = Opt_func_dispatcher()

    if CALCULATOR == 'AIMNET2':
        dispatcher.load_aimnet2_calc(DEFAULT_LEVELS[CALCULATOR])

    ##########################################################################

    print('\nRunning tests for FIRECODE. Settings used:')
    print(f'{CALCULATOR=}')

    if CALCULATOR != 'XTB':
        print(f'{CALCULATOR} COMMAND = {COMMANDS[CALCULATOR]}')

    print('\nTesting calculator...')

    ##########################################################################

    dispatcher.opt_funcs_dict[CALCULATOR](data.atomcoords[0],
                               data.atomnos,
                               method=DEFAULT_LEVELS[CALCULATOR],
                               procs=PROCS,
                               read_output=False)

    print(f'{CALCULATOR} raw calculator works.')

    ##########################################################################

    atoms = Atoms('HH', positions=np.array([[0, 0, 0], [0, 0, 1]]))
    atoms.calc = get_ase_calc((CALCULATOR, DEFAULT_LEVELS[CALCULATOR], PROCS, None))
    LBFGS(atoms, logfile=None).run()

    clean_directory()
    print(f'{CALCULATOR} ASE calculator works.')
    
    ##########################################################################

    print(f'\n{FF_OPT_BOOL=}')
    ff = f'on. Calculator is {FF_CALC}. Checking its status.' if FF_OPT_BOOL else 'off.'
    print(f'Force Field optimization is turned {ff}')

    if FF_OPT_BOOL:
        if FF_CALC != 'OB': # 'XTB', 'GAUSSIAN'
            dispatcher.opt_funcs_dict[FF_CALC](data.atomcoords[0],
                                    data.atomnos,
                                    method=DEFAULT_FF_LEVELS[FF_CALC],
                                    procs=PROCS,
                                    read_output=False)

        print(f'{FF_CALC} FF raw calculator works.')

        ##########################################################################
        
        if FF_CALC != 'OB':
            atoms.calc = get_ase_calc((FF_CALC, DEFAULT_FF_LEVELS[FF_CALC], PROCS, None))
            LBFGS(atoms, logfile=None).run()

            clean_directory()
            print(f'{FF_CALC} ASE calculator works.')

    print('\nNo installation faults detected with the current settings. Running tests.')

    ##########################################################################

    tests = []
    for f in os.listdir():
        if f.endswith('.txt'):
            tests.append(os.path.realpath(f))

    # os.chdir(os.path.dirname(os.getcwd()))
    # os.chdir('firecode')
    # # Back to ./firecode

    times = []
    for i, f in enumerate(tests):
        name = f.split('\\')[-1].split('/')[-1][:-4] # trying to make it work for either Win, Linux (and Mac?)
        loadbar(i, len(tests), f'Running firecode tests ({name}): ')
        
        t_start = time.perf_counter()
        try:
            with HiddenPrints():
                run_command(f'python -m firecode {f} -n {name}')

        except CalledProcessError as error:
            print('\n\n--> An error occurred:\n')
            print(error.stderr.decode("utf-8"))
            sys.exit()
                    
        t_end = time.perf_counter()
        times.append(t_end-t_start)

    loadbar(len(tests), len(tests), f'Running firecode tests ({name}): ')    

    print()
    for i, f in enumerate(tests):
        print('    {:25s}{} s'.format(f.split('\\')[-1].split('/')[-1][:-4], round(times[i], 3)))

    print(f'\nfirecode tests completed with no errors. ({time_to_string(time.perf_counter() - t_start_run)})\n')