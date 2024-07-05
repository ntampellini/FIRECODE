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

import os
import sys
from subprocess import DEVNULL, STDOUT, check_call

import numpy as np

from firecode.algebra import dihedral, norm, norm_of, vec_angle
from firecode.errors import MopacReadError
from firecode.pt import pt
from firecode.numba_functions import scramble
from firecode.settings import COMMANDS
from firecode.solvents import get_solvent_line


def read_mop_out(filename):
    '''
    Reads a MOPAC output looking for optimized coordinates and energy.
    :params filename: name of MOPAC filename (.out extension)
    :return coords, energy: array of optimized coordinates and absolute energy, in kcal/mol
    '''
    coords = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()

            if 'Too many variables. By definition, at least one force constant is exactly zero' in line:
                success = False
                return None, 1E10, success

            if not line:
                break
            
            if 'SCF FIELD WAS ACHIEVED' in line:
                while True:
                    line = f.readline()
                    if not line:
                        break

                    if 'FINAL HEAT OF FORMATION' in line:
                        energy = float(line.split()[5])
                        # in kcal/mol

                    if 'CARTESIAN COORDINATES' in line:
                        line = f.readline()
                        line = f.readline()
                        while line != '\n':
                            splitted = line.split()
                            # symbols.append(splitted[1])
                            coords.append([float(splitted[2]),
                                            float(splitted[3]),
                                            float(splitted[4])])
                                        
                            line = f.readline()
                            if not line:
                                break
                        break
                break

    coords = np.array(coords)

    if coords.shape[0] != 0:
        success = True
        return coords, energy, success
    
    raise MopacReadError(f'Cannot read file {filename}: maybe a badly specified MOPAC keyword?')

def mopac_opt(coords, atomnos, constrained_indices=None, method='PM7', solvent=None, title='temp', read_output=True, **kwargs):
    '''
    This function writes a MOPAC .mop input, runs it with the subprocess
    module and reads its output. Coordinates used are mixed
    (cartesian and internal) to be able to constrain the reactive atoms
    distances specified in constrained_indices.

    :params coords: array of shape (n,3) with cartesian coordinates for atoms
    :params atomnos: array of atomic numbers for atoms
    :params constrained_indices: array of shape (n,2), with the indices
                                 of atomic pairs to be constrained
    :params method: string, specifiyng the first line of keywords for the MOPAC input file.
    :params title: string, used as a file name and job title for the mopac input file.
    :params read_output: Whether to read the output file and return anything.
    '''

    constrained_indices_list = constrained_indices.ravel() if constrained_indices is not None else []
    constrained_indices = constrained_indices if constrained_indices is not None else []

    if solvent is not None:
        method += ' ' + get_solvent_line(solvent, 'MOPAC', method)

    order = []
    s = [method + '\n' + title + '\n\n']
    for i, num in enumerate(atomnos):
        if i not in constrained_indices:
            order.append(i)
            s.append(' {} {} 1 {} 1 {} 1\n'.format(pt[num].symbol, coords[i][0], coords[i][1], coords[i][2]))

    free_indices = list(set(range(len(atomnos))) - set(constrained_indices_list))
    # print('free indices are', free_indices, '\n')

    if len(constrained_indices_list) == len(set(constrained_indices_list)):
    # block pairs of atoms if no atom is involved in more than one distance constrain

        for a, b in constrained_indices:
            
            order.append(b)
            order.append(a)

            c, d = np.random.choice(free_indices, 2)
            while c == d:
                c, d = np.random.choice(free_indices, 2)
            # indices of reference atoms, from unconstraind atoms set

            dist = norm_of(coords[a] - coords[b]) # in Angstrom
            # print(f'DIST - {dist} - between {a} {b}')

            angle = vec_angle(norm(coords[a] - coords[b]), norm(coords[c] - coords[b]))
            # print(f'ANGLE - {angle} - between {a} {b} {c}')

            d_angle = dihedral([coords[a],
                                coords[b],
                                coords[c],
                                coords[d]])
            d_angle += 360 if d_angle < 0 else 0
            # print(f'D_ANGLE - {d_angle} - between {a} {b} {c} {d}')

            list_len = len(s)
            s.append(' {} {} 1 {} 1 {} 1\n'.format(pt[atomnos[b]].symbol, coords[b][0], coords[b][1], coords[b][2]))
            s.append(' {} {} 0 {} 1 {} 1 {} {} {}\n'.format(pt[atomnos[a]].symbol, dist, angle, d_angle, list_len, free_indices.index(c)+1, free_indices.index(d)+1))
            # print(f'Blocked bond between mopac ids {list_len} {list_len+1}\n')

    elif len(set(constrained_indices_list)) == 3:
    # three atoms, the central bound to the other two
    # OTHERS[0]: cartesian
    # CENTRAL: internal (self, others[0], two random)
    # OTHERS[1]: internal (self, central, two random)
        
        central = max(set(constrained_indices_list), key=lambda x: list(constrained_indices_list).count(x))
        # index of the atom that is constrained to two other

        others = list(set(constrained_indices_list) - {central})

    # OTHERS[0]

        order.append(others[0])
        s.append(' {} {} 1 {} 1 {} 1\n'.format(pt[atomnos[others[0]]].symbol, coords[others[0]][0], coords[others[0]][1], coords[others[0]][2]))
        # first atom is placed in cartesian coordinates, the other two have a distance constraint and are expressed in internal coordinates

    #CENTRAL

        order.append(central)
        c, d = np.random.choice(free_indices, 2)
        while c == d:
            c, d = np.random.choice(free_indices, 2)
        # indices of reference atoms, from unconstraind atoms set

        dist = norm_of(coords[central] - coords[others[0]]) # in Angstrom

        angle = vec_angle(norm(coords[central] - coords[others[0]]), norm(coords[others[0]] - coords[c]))

        d_angle = dihedral([coords[central],
                            coords[others[0]],
                            coords[c],
                            coords[d]])
        d_angle += 360 if d_angle < 0 else 0

        list_len = len(s)
        s.append(' {} {} 0 {} 1 {} 1 {} {} {}\n'.format(pt[atomnos[central]].symbol, dist, angle, d_angle, list_len-1, free_indices.index(c)+1, free_indices.index(d)+1))

    #OTHERS[1]

        order.append(others[1])
        c1, d1 = np.random.choice(free_indices, 2)
        while c1 == d1:
            c1, d1 = np.random.choice(free_indices, 2)
        # indices of reference atoms, from unconstraind atoms set

        dist1 = norm_of(coords[others[1]] - coords[central]) # in Angstrom

        angle1 = np.arccos(norm(coords[others[1]] - coords[central]) @ norm(coords[others[1]] - coords[c1]))*180/np.pi # in degrees

        d_angle1 = dihedral([coords[others[1]],
                             coords[central],
                             coords[c1],
                             coords[d1]])
        d_angle1 += 360 if d_angle < 0 else 0

        list_len = len(s)
        s.append(' {} {} 0 {} 1 {} 1 {} {} {}\n'.format(pt[atomnos[others[1]]].symbol, dist1, angle1, d_angle1, list_len-1, free_indices.index(c1)+1, free_indices.index(d1)+1))

    else:
        raise NotImplementedError('The constraints provided for MOPAC optimization are not yet supported')


    s = ''.join(s)
    with open(f'{title}.mop', 'w') as f:
        f.write(s)
    
    try:
        check_call(f'{COMMANDS["MOPAC"]} {title}.mop'.split(), stdout=DEVNULL, stderr=STDOUT)
    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        sys.exit()

    os.remove(f'{title}.mop')
    # delete input, we do not need it anymore

    if read_output:

        inv_order = [order.index(i) for i, _ in enumerate(order)]
        # undoing the atomic scramble that was needed by the mopac input requirements

        opt_coords, energy, success = read_mop_out(f'{title}.out')
        os.remove(f'{title}.out')

        opt_coords = scramble(opt_coords, inv_order) if opt_coords is not None else coords
        # If opt_coords is None, that is if TS seeking crashed,
        # sets opt_coords to the old coords. If not, unscrambles
        # coordinates read from mopac output.

        return opt_coords, energy, success