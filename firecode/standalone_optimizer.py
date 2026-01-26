# coding=utf-8
'''
FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2026 Nicolò Tampellini

SPDX-License-Identifier: LGPL-3.0-or-later

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as publishedby
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

import os as op_sys
from subprocess import getoutput
from time import perf_counter

import numpy as np
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator

from firecode.algebra import dihedral, norm_of, point_angle
from firecode.optimization_methods import Opt_func_dispatcher
from firecode.pt import pt
from firecode.settings import CALCULATOR
from firecode.solvents import epsilon_dict
from firecode.units import EH_TO_KCAL
from firecode.utils import Constraint, read_xyz, time_to_string, write_xyz


class Optimizer:

    def __init__(self, calc=None, method=None, solvent=None):

        if any((calc is None, method is None)):

            choices = (
                    Choice(value=('AIMNET2', 'wB97M-D3'), name='AIMNet2/wB97M-D3'),
                    Choice(value=('TBLITE', 'GFN2-xTB'), name='GFN2-xTB (TBLITE)'),
                    # Choice(value=('XTB', 'g-xTB'), name='g-xTB (XTB)'),
                    Choice(value=('XTB', 'GFN-FF'), name='GFN-FF (XTB)'),
                    Choice(value=('UMA', 'OMOL'), name='UMA/OMol25'),
                    Choice(value=('ORCA', 'B97-3c'), name='ORCA/B97-3c'),
                )

            calc, method = inquirer.select(
                message="Which level of theory would you like to use?:",
                choices=choices,
                default=next(c.value for c in choices if c.value[0] == CALCULATOR),
                ).execute()
            
        if solvent is None:

            solvent = inquirer.fuzzy(
                message="Which solvent would you like to use?",
                choices=list(epsilon_dict.keys()) + [Choice(value=None, name="vacuum")],
                default="toluene",
                validate=lambda x: ((x in epsilon_dict) or x is None),
            ).execute()
            
        self.calc = calc
        self.method = method
        self.solvent = solvent

        self.dispatcher = Opt_func_dispatcher(calc)

        # try:
        #     self.dispatcher.get_ase_calc(method, solvent)
        # except NotImplementedError:
        #     pass

    def __repr__(self):
        s = ''
        for attr in ("calc", "method", "solvent"):
            s += f"--> {attr} : {getattr(self, attr)}\n"
        return s

def main(filenames):
    """
    Standalone optimizer entry point.
    args: iterable of strings of structure filenames.
    
    """

    optimizer = Optimizer()

    choices = [
        Choice(value="auto_charge_and_mult", name="Auto charge/mult - Non-singlets will be assumed as doublets.",
               enabled=True),
        Choice(value="constraints",           name="Constraints      - Manually apply constraints to the optimization."),
        Choice(value="constraint_file",       name="Constraint file  - Load a constraint file."),
        Choice(value="sp",                    name="Single point     - Do a single point energy calc, without optimizing."),
        Choice(value="newfile",               name="Newfile          - Write optimized structure to a new file (*_opt.xyz)."),
        Choice(value="free_energy",           name="Free Energy      - Calculate free energy (G)."),
    ]
        
    options_to_set = inquirer.checkbox(
        message="Select options (spacebar to toggle, enter to confirm):",
        choices=choices,
        cycle=False,
        disabled_symbol='⬡',
        enabled_symbol='⬢',
        ).execute()
    
    # set options as booleans, will change idenity later
    for option in choices:
        setattr(optimizer, option.value, option.value in options_to_set)

    optimizer.constraints = []
    smarts_string = None
    optimizer.opt = not optimizer.sp

    if not optimizer.auto_charge_and_mult:

        optimizer.manual_charge = inquirer.text(
            message="Manually specify charge:",
            filter=int,
            validate=lambda x: x.isdigit(),
            invalid_message="Please specify an integer",
        )

    # manually set constraints
    if "constraints" in options_to_set:        

        while True:

            data = input("Constrained indices [+ optional distance or \"ts\", enter to stop]: ").split()

            if not data:
                break

            elif data[-1] == "ts":
                data[-1] = str(get_ts_d_estimate(filenames[0], (int(i) for i in data[0:2])))


            assert len(data) in (2, 3, 4, 5), "Only 2-4 indices as ints + optional target as a float"

            value = None
            if '.' in data[-1]:
                value = float(data.pop(-1))

            constraint = Constraint([int(i) for i in data], value=value)                
            optimizer.constraints.append(constraint)
                
        print(f"Specified {len(optimizer.constraints)} constraints")

    # set constraint_file to textfile filename
    if optimizer.constraint_file:


        optimizer.constraint_file = inquirer.filepath(
                message="Select a constraint file:",
                default="./" if op_sys.name == "posix" else "C:\\",
                validate=PathValidator(is_file=True, message="Input is not a file"),
                only_files=True,
            ).execute() 

        # set constraints from file
        with open(optimizer.constraint_file, 'r') as f:
            lines = f.readlines()

        # see if we are pattern matching
        if lines[0].startswith('SMARTS'):
            smarts_string = lines.pop(0).lstrip('SMARTS ')
            print('--> SMARTS line found: will pattern match and interpret constrained indices relative to the pattern')

        for line in lines:
            data = line.split()
            try:

                assert len(data) in (2, 3, 4, 5), "Only 2-4 indices as ints + optional target as a float"

                value = None
                if '.' in data[-1]:
                    value = float(data.pop(-1))

                constraint = Constraint([int(i) for i in data], value=value)                
                optimizer.constraints.append(constraint)

            except Exception as e:
                print(e)

        print(f'--> Read constraints from {optimizer.constraint_file}')

    if optimizer.free_energy:
        print('--> Requested free energy calculation - performing vibrational analysis')
        from firecode.ase_manipulations import ase_get_free_energy

    if optimizer.sp:
        print("--> Single point calculation requested (no optimization)")

    if optimizer.newfile:
        print("--> Writing optimized structures to new files")

    # if "charge" in [kw.split("=")[0] for kw in sys.argv]:
        # options["charge"] = next((kw.split("=")[-1] for kw in sys.argv if "charge" in kw))
        # sys.argv.remove(f"charge={options['charge']}")

    # if "planar" in sys.argv:
    #     sys.argv.remove(f"planar")
    #     options["constrain_string"] = "dihedral: 7, 8, 9, 15, 180\ndihedral: 9, 8, 15, 7, 180\ndihedral: 15, 8, 7, 9, 180\n force constant=1.0"
    #     print("--> !PLANAR")

    # for option, value in options.items():
    #     print(f"--> {option} = {value}")

    print(optimizer)

    op_sys.chdir(op_sys.getcwd())

    energies, names_confs = [], []

    # load ase_calc only now
    optimizer.dispatcher.get_ase_calc(optimizer.method, optimizer.solvent)

    # start optimizing
    for i, name in enumerate(filenames):
        try:
            data = read_xyz(name)

            print()

            # define outname and clear existing
            outname = name if not optimizer.newfile else name[:-4] + "_opt.xyz"
            if optimizer.newfile and (outname in op_sys.listdir()):
                op_sys.remove(outname)
            write_type = 'a' if optimizer.newfile else 'w'

            # set charge
            if optimizer.auto_charge_and_mult:
                charge = name.count("+") - name.count("-")
            else:
                charge = optimizer.manual_charge

            # set multiplicity
            if multiplicity_check(data.atomnos, int(charge)):
                mult = 1
            elif optimizer.auto_charge_and_mult:
                mult = 2
            else:
                mult = inquirer.text(
                    message=f'It appears {name} is not a singlet. Please specify multiplicity:',
                    validate=lambda inp: inp.isdigit() and int(inp) > 1,
                    default="2",
                    filter=int,
                ).execute()

            for c_n, coords in enumerate(data.coords):

                constrained_indices, constrained_distances, constrained_dihedrals, constrained_dih_angles = [], [], [], []
                constrained_angles_indices, constrained_angles_values = [], []

                for constraint in optimizer.constraints:

                    if smarts_string is not None:
                        # save original indices to revert them later, for the next conformer/molecule
                        constraint.old_indices = constraint.indices[:]

                        # correct indices from relative to the SMARTS
                        # string to absolute for this molecule
                        constraint.convert_constraint_with_smarts(coords, data.atomnos, smarts_string)
                        
                    if constraint.type == 'B':

                        a, b = constraint.indices
                        if constraint.value is None:
                            constraint.value = norm_of(coords[a]-coords[b])

                        constrained_indices.append(constraint.indices)
                        constrained_distances.append(constraint.value)

                        print(f"CONSTRAIN -> d({a}-{b}) = {round(norm_of(coords[a]-coords[b]), 3)} A at start of optimization (target is {round(constraint.value, 3)} A)")

                    elif constraint.type == 'A':

                        a, b, c = constraint.indices
                        if constraint.value is None:
                            constraint.value = point_angle(coords[a],coords[b],coords[c])

                        constrained_angles_indices.append(constraint.indices)
                        constrained_angles_values.append(constraint.value)
                        
                        print(f"CONSTRAIN ANGLE -> Angle({a}-{b}-{c}) = {round(point_angle(coords[a],coords[b],coords[c]), 3)}° at start of optimization, target {round(constraint.value, 3)}°")

                    elif constraint.type == 'D':
                        
                        a, b, c, d = constraint.indices
                        if constraint.value is None:
                            constraint.value = dihedral(np.array([coords[a],coords[b],coords[c], coords[d]]))

                        constrained_dihedrals.append(constraint.indices)
                        constrained_dih_angles.append(constraint.value)

                        print(f"CONSTRAIN DIHEDRAL -> Dih({a}-{b}-{c}-{d}) = {round(dihedral(np.array([coords[a],coords[b],coords[c], coords[d]])), 3)}° at start of optimization, target {round(constraint.value, 3)}°")
                
                action = "Optimizing" if optimizer.opt else "Calculating SP energy on"

                if optimizer.calc in ('AIMNET2', 'UMA') and optimizer.solvent is not None:
                    post = f'+ALPB({optimizer.solvent})'
                else:
                    post = ''

                print(f'{action} {name} - {i+1} of {len(filenames)}, conf {c_n+1} of {len(data.coords)} ({optimizer.method}/{optimizer.calc}{post}) - CHG={charge} MULT={mult}')
                t_start = perf_counter()

                coords, energy, _ = optimizer.dispatcher.opt_func(

                    coords,
                    data.atomnos,
                    method=optimizer.method,

                    constrained_indices=constrained_indices,
                    constrained_distances=constrained_distances,

                    constrained_angles_indices=constrained_angles_indices,
                    constrained_angles_values=constrained_angles_values,

                    constrained_dihedrals_indices=constrained_dihedrals,
                    constrained_dihedrals_values=constrained_dih_angles,

                    ase_calc=optimizer.dispatcher.ase_calc,
                    mult=mult,
                    traj=name[:-4] + "_traj",
                    logfunction=print,
                    charge=charge,
                    maxiter=500 if optimizer.opt else 1,
                    solvent=optimizer.solvent,

                    # title='OPT_temp',
                    # debug=True,
                    )

                elapsed = perf_counter() - t_start

                if energy is None:
                    print(f'--> ERROR: Optimization of {name} crashed. ({time_to_string(elapsed)})')

                elif optimizer.opt:
                    with open(outname, write_type) as f:
                        write_xyz(coords, data.atomnos, f, title=f'Energy = {energy} kcal/mol')
                    print(f"{'Appended' if write_type == 'a' else 'Wrote'} optimized structure at {outname} - {time_to_string(elapsed)}\n")

                if optimizer.free_energy:
                    # sph = (len(constraints) != 0)
                    # print(f'Calculating Free Energy contribution{" (SPH)" if sph else ""} on {name} - {i+1} of {len(names)}, conf {c_n+1} of {len(data.coords)} ({method})')
                    # gcorr = xtb_get_free_energy(coords, data.atomnos, method='GFN-FF', solvent=options["solvent"], charge=options["charge"], sph=sph, grep='Gcorr')
                    # print(f'GCORR: {name}, conf {c_n+1} - {gcorr:.2f} kcal/mol')
                    # energy += gcorr

                    print(f'Performing vibrational analysis on {name} - {i+1} of {len(filenames)}, conf {c_n+1} of {len(data.coords)} ({optimizer.method})')
                    t_start = perf_counter()
                    
                    energy = ase_get_free_energy(
                        coords,
                        data.atomnos,
                        ase_calc=optimizer.dispatcher.ase_calc,
                        energy=energy,
                        charge=charge,
                        mult=mult,
                        title=f"{name[:-4]}",
                        )

                    elapsed = perf_counter() - t_start
                    print(f"Calculated vibrational frequencies in {time_to_string(elapsed)}\n")

                energies.append(energy)
                names_confs.append(name[:-4]+f"_conf{c_n+1}")

        except Exception as e:
            print("--> ", name, " - ", e)
            raise(e)
            continue

        if optimizer.constraints:
            print('Constraints: final values')

            for constraint in optimizer.constraints:
                if constraint.type == 'B':
                    a, b = constraint.indices
                    final_value = norm_of(coords[a]-coords[b])
                    uom = ' Å'
                
                elif constraint.type == 'A':
                    a, b, c = constraint.indices
                    final_value = point_angle(coords[a],coords[b],coords[c])
                    uom = '°'

                elif constraint.type == 'D':
                    a, b, c, d = constraint.indices
                    final_value = dihedral(np.array([coords[a],coords[b],coords[c], coords[d]]))
                    uom = '°'

                indices_string = '-'.join([str(i) for i in constraint.indices])
                print(f"CONSTRAIN -> {constraint.type}({indices_string}) = {round(final_value, 3)}{uom}")

                # revert original indices for the next molecule
                if smarts_string is not None:
                    constraint.indices = constraint.old_indices

            print()

    if None not in energies:

        if len(names_confs) > 1:
            min_e = min(energies)
        else:
            min_e = 0

        ### NICER TABLE PRINTOUT

        from prettytable import PrettyTable

        table = PrettyTable()
        energy_type = 'Free Energy G(Eh)' if optimizer.free_energy else 'Potential Energy E(Eh)'
        letter = 'G' if optimizer.free_energy else 'E'
        table.field_names = ['#', 'Filename', energy_type, f'Rel. {letter} (kcal/mol)']

        print()

        for i, (nc, energy) in enumerate(zip(names_confs, energies)):
            table.add_row([i+1, nc, energy/EH_TO_KCAL, round(energy-min_e, 2)])
            
        print(table.get_string())

def multiplicity_check(atomnos, charge, multiplicity=1) -> bool:
    '''
    Returns True if the multiplicity and the nuber of
    electrons are one odd and one even, and vice versa.

    '''

    electrons = sum(atomnos) - charge
    
    return (multiplicity % 2) != (electrons % 2)

def get_ts_d_estimate(filename, indices, factor=1.35, verbose=True):
    '''
    Returns an estimate for the distance between two
    specific atoms in a transition state, by multipling
    the sum of covalent radii for a constant.
    
    '''
    mol = read_xyz(filename)
    i1, i2 = indices
    a1, a2 = mol.atoms[i1], mol.atoms[i2]
    cr1 = pt.covalent_radius(a1)
    cr2 = pt.covalent_radius(a2)

    est_d = round(factor * (cr1 + cr2), 2)

    if verbose:
        print(f'--> Estimated TS d({a1}-{a2}) = {est_d} Å')
        
    return est_d
