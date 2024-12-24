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
from subprocess import DEVNULL, STDOUT, CalledProcessError, check_call

import numpy as np

from firecode.algebra import norm, norm_of
from firecode.calculators.__init__ import NewFolderContext
from firecode.graph_manipulations import get_sum_graph
from firecode.utils import clean_directory, read_xyz, write_xyz


def xtb_opt(
        coords,
        atomnos,

        constrained_indices=None,
        constrained_distances=None,

        constrained_dihedrals_indices=None,
        constrained_dihedrals_values=None,

        constrained_angles_indices=None,
        constrained_angles_values=None,

        method='GFN2-xTB',
        maxiter=500,
        solvent=None,
        charge=0,
        title='temp',
        read_output=True, 
        procs=4,
        opt=True,
        conv_thr="tight",
        assert_convergence=False, 
        constrain_string=None,
        recursive_stepsize=0.3,
        spring_constant=1,

        debug=False,
        **kwargs,
        ):
    '''
    This function writes an XTB .inp file, runs it with the subprocess
    module and reads its output.

    coords: array of shape (n,3) with cartesian coordinates for atoms.

    atomnos: array of atomic numbers for atoms.

    constrained_indices: array of shape (n,2), with the indices
    of atomic pairs to be constrained.

    constrained_distances: optional, target distances for the specified
    distance constraints. 

    constrained_dihedrals: quadruplets of atomic indices to constrain.

    constrained_dih_angles: target dihedral angles for the dihedral constraints.

    method: string, specifying the theory level to be used.

    maxiter: maximum number of geometry optimization steps (maxcycle).

    solvent: solvent to be used in the calculation (ALPB model).

    charge: charge to be used in the calculation.

    title: string, used as a file name and job title for the mopac input file.

    read_output: Whether to read the output file and return anything.

    procs: number of cores to be used for the calculation.

    opt: if false, a single point energy calculation is carried.

    conv_thr: tightness of convergence thresholds. See XTB ReadTheDocs.

    assert_convergence: wheter to raise an error in case convergence is not
    achieved by xtb.

    constrain_string: string to be added to the end of the $geom section of
    the input file.

    recursive_stepsize: magnitude of step in recursive constrained optimizations.
    The smaller, the slower - but potentially safer against scrambling.

    spring_constant: stiffness of harmonic distance constraint (Hartrees/Bohrs^2)

    '''
    # create working folder and cd into it
    with NewFolderContext(title, delete_after=(not debug)):

        if constrained_indices is not None:
            if len(constrained_indices) == 0:
                constrained_indices = None

        if constrained_distances is not None:
            if len(constrained_distances) == 0:
                constrained_distances = None

        # recursive 
        if constrained_distances is not None:

            try:

                for i, (target_d, ci) in enumerate(zip(constrained_distances, constrained_indices)):

                    if target_d is None:
                        continue

                    if len(ci) == 2:
                        a, b = ci
                    else:
                        continue

                    d = norm_of(coords[b] - coords[a])
                    delta = d - target_d

                    if abs(delta) > recursive_stepsize:
                        recursive_c_d = constrained_distances.copy()
                        recursive_c_d[i] = target_d + (recursive_stepsize * np.sign(d-target_d))
                        # print(f"-------->  d is {round(d, 3)}, target d is {round(target_d, 3)}, delta is {round(delta, 3)}, setting new pretarget at {recursive_c_d}")
                        coords, _, _ = xtb_opt(
                                                coords,
                                                atomnos,
                                                constrained_indices,
                                                constrained_distances=recursive_c_d,
                                                method=method,
                                                solvent=solvent,
                                                charge=charge,
                                                maxiter=50,
                                                title=title,
                                                procs=procs,
                                                conv_thr='loose',
                                                constrain_string=constrain_string,
                                                recursive_stepsize=0.3,
                                                spring_constant=0.25,

                                                constrained_dihedrals_indices=constrained_dihedrals_indices,
                                                constrained_dihedrals_values=constrained_dihedrals_values,

                                                constrained_angles_indices=constrained_angles_indices,
                                                constrained_angles_values=constrained_angles_values,

                                            )
                    
                    d = norm_of(coords[b] - coords[a])
                    delta = d - target_d
                    coords[b] -= norm(coords[b] - coords[a]) * delta
                    # print(f"--------> moved atoms from {round(d, 3)} A to {round(norm_of(coords[b] - coords[a]), 3)} A")

            except RecursionError:
                with open(f'{title}_crashed.xyz', 'w') as f:
                    write_xyz(coords, atomnos, f, title=title)
                print("Recursion limit reached in constrained optimization - Crashed.")
                sys.exit()

        with open(f'{title}.xyz', 'w') as f:
            write_xyz(coords, atomnos, f, title=title)

        # outname = f'{title}_xtbopt.xyz' DOES NOT WORK - XTB ISSUE?
        outname = 'xtbopt.xyz'
        trajname = f'{title}_opt_log.xyz'
        maxiter = maxiter if maxiter is not None else 0
        s = f'$opt\n   logfile={trajname}\n   output={outname}\n   maxcycle={maxiter}\n'
            
        if constrained_indices is not None:
            s += f'\n$constrain\n   force constant={spring_constant}\n'

            for (a, b), distance in zip(constrained_indices, constrained_distances):

                distance = distance or 'auto'
                s += f"   distance: {a+1}, {b+1}, {distance}\n"  

        if constrained_angles_indices is not None:

            assert len(constrained_angles_indices) == len(constrained_angles_values)

            if constrained_indices is None:
                s += '\n$constrain\n'

            for (a, b, c), angle in zip(constrained_angles_indices, constrained_angles_values):
                s += f"   angle: {a+1}, {b+1}, {c+1}, {angle}\n"  

        if constrained_dihedrals_indices is not None:

            assert len(constrained_dihedrals_indices) == len(constrained_dihedrals_values)

            if constrained_indices is None:
                s += '\n$constrain\n'

            for (a, b, c, d), angle in zip(constrained_dihedrals_indices, constrained_dihedrals_values):
                s += f"   dihedral: {a+1}, {b+1}, {c+1}, {d+1}, {angle}\n"  

        if constrain_string is not None:
            s += '\n$constrain\n'
            s += constrain_string

        if method.upper() in ('GFN-XTB', 'GFNXTB'):
            s += '\n$gfn\n   method=1\n'

        elif method.upper() in ('GFN2-XTB', 'GFN2XTB'):
            s += '\n$gfn\n   method=2\n'
        
        s += '\n$end'

        s = ''.join(s)
        with open(f'{title}.inp', 'w') as f:
            f.write(s)
        
        flags = '--norestart'
        
        if opt:
            flags += f' --opt {conv_thr}'
            # specify convergence tightness
        
        if method in ('GFN-FF', 'GFNFF'):

            flags += ' --gfnff'
            # declaring the use of FF instead of semiempirical

        if charge != 0:
            flags += f' --chrg {charge}'

        if procs is not None:
            flags += f' -P {procs}'

        if solvent is not None:

            if solvent == 'methanol':
                flags += ' --gbsa methanol'

            else:
                flags += f' --alpb {solvent}'

        elif method.upper() in ('GFN-FF', 'GFNFF'):
            flags += ' --alpb ch2cl2'
            # if using the GFN-FF force field, add CH2Cl2 solvation for increased accuracy

        try:
            with open(f"{title}.out", "w") as f:
                check_call(f'xtb {title}.xyz --input {title}.inp {flags}'.split(), stdout=f, stderr=STDOUT)

        # sometimes the SCC does not converge: only raise the error if specified
        except CalledProcessError:
            if assert_convergence:
                raise CalledProcessError
        
        except KeyboardInterrupt:
            print('KeyboardInterrupt requested by user. Quitting.')
            sys.exit()

        if spring_constant > 0.25:
            print()

        if read_output:
            
            if opt:

                if trajname in os.listdir():
                    coords, energy = read_from_xtbtraj(trajname)

                else:
                    energy = None

                clean_directory((f'{title}.inp', f'{title}.xyz', f"{title}.out", trajname, outname))
            
            else:    
                energy = energy_grepper(f"{title}.out", 'TOTAL ENERGY', 3)
                # clean_directory((f'{title}.inp', f'{title}.xyz', f"{title}.out", trajname, outname))

            for filename in ('gfnff_topo',
                            'charges',
                            'wbo',
                            'xtbrestart',
                            'xtbtopo.mol', 
                            '.xtboptok',
                            'gfnff_adjacency',
                            'gfnff_charges',
                            ):
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass
           
            return coords, energy, True
        
def xtb_pre_opt(
                coords,
                atomnos, 
                graphs,
                constrained_indices=None,
                constrained_distances=None, 
                **kwargs,
                ):
    '''
    Wrapper for xtb_opt that preserves the distance of every bond present in each subgraph provided

    graphs: list of subgraphs that make up coords, in order

    '''
    sum_graph = get_sum_graph(graphs, extra_edges=constrained_indices)

    # we have to check through a list this way, as I have not found
    # an analogous way to check through an array for subarrays in a nice way
    list_of_constr_ids = [[a,b] for a, b in constrained_indices] if constrained_indices is not None else []

    constrain_string = "$constrain\n"
    for constraint in [[a, b] for (a, b) in sum_graph.edges if a!=b]:

        if constrained_distances is None:
            distance = 'auto'

        elif constraint in list_of_constr_ids:
            distance = constrained_distances[list_of_constr_ids.index(constraint)]

        else:
            distance = 'auto'

        indices_string = str([i+1 for i in constraint]).strip("[").strip("]")
        constrain_string += f"  distance: {indices_string}, {distance}\n"
    constrain_string += "\n$end"

    return xtb_opt(
                    coords,
                    atomnos,
                    constrained_indices=constrained_indices,
                    constrained_distances=constrained_distances,
                    constrain_string=constrain_string,
                    **kwargs,
                )

def read_from_xtbtraj(filename):
    '''
    Read coordinates from a .xyz trajfile.

    '''
    with open(filename, 'r') as f:
        lines = f.readlines()

    # look for the last line containing the flag (iterate in reverse order)
    # and extract the line at which coordinates start
    first_coord_line = len(lines) - next(line_num for line_num, line in enumerate(reversed(lines)) if 'energy:' in line)
    xyzblock = lines[first_coord_line:]

    coords = np.array([line.split()[1:] for line in xyzblock], dtype=float)
    energy = float(lines[first_coord_line-1].split()[1]) * 627.5096080305927 # Eh to kcal/mol

    return coords, energy

def energy_grepper(filename, signal_string, position):
    '''
    returns a kcal/mol energy from a Eh energy in a textfile.
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        line = f.readline()
        while True:
            line = f.readline()
            if signal_string in line:
                return float(line.split()[position]) * 627.5096080305927 # Eh to kcal/mol
            if not line:
                raise Exception()

def xtb_get_free_energy(coords, atomnos, method='GFN2-xTB', solvent=None,
                        charge=0, title='temp', sph=False, grep='G', **kwargs):
    '''
    Calculates free energy with XTB,
    without optimizing the provided structure.
    grep: returns either "G" or "Gcorr" in kcal/mol
    sph: whether to run as single point hessian or not
    
    '''
    with NewFolderContext(title):

        with open(f'{title}.xyz', 'w') as f:
            write_xyz(coords, atomnos, f, title=title)

        outname = 'xtbopt.xyz'
        trajname = f'{title}_opt_log.xyz'
        s = f'$opt\n   logfile={trajname}\n   output={outname}\n   maxcycle=1\n'

            
        if method.upper() in ('GFN-XTB', 'GFNXTB'):
            s += '\n$gfn\n   method=1\n'

        elif method.upper() in ('GFN2-XTB', 'GFN2XTB'):
            s += '\n$gfn\n   method=2\n'
        
        s += '\n$end'

        s = ''.join(s)
        with open(f'{title}.inp', 'w') as f:
            f.write(s)
        
        if sph:
            flags = '--bhess'
        else:
            flags = '--ohess'
        
        if method in ('GFN-FF', 'GFNFF'):
            flags += ' --gfnff'
            # declaring the use of FF instead of semiempirical

        if charge != 0:
            flags += f' --chrg {charge}'

        if solvent is not None:

            if solvent == 'methanol':
                flags += ' --gbsa methanol'

            else:
                flags += f' --alpb {solvent}'

        try:
            with open('temp_hess.log', 'w') as outfile:
                check_call(f'xtb --input {title}.inp {title}.xyz {flags}'.split(), stdout=outfile, stderr=STDOUT)
            
        except KeyboardInterrupt:
            print('KeyboardInterrupt requested by user. Quitting.')
            sys.exit()

        try:
            to_grep, index = {
                'G' : ('TOTAL FREE ENERGY', 4),
                'Gcorr' : ('G(RRHO) contrib.', 3),
            }[grep]
            result = energy_grepper('temp_hess.log', to_grep, index)

            clean_directory()
            for filename in ('gfnff_topo', 'charges', 'wbo', 'xtbrestart', 'xtbtopo.mol', '.xtboptok',
                            'hessian', 'g98.out', 'vibspectrum', 'wbo', 'xtbhess.xyz', 'charges', 'temp_hess.log'):
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass

            return result

        except FileNotFoundError:
            # return 1E10
            # print(f'temp_hess.log not present here - we are in', os.getcwd())
            print(os.listdir())
            sys.exit()

def xtb_metadyn_augmentation(coords, atomnos, constrained_indices=None, new_structures:int=5, title=0, debug=False):
    '''
    Runs a metadynamics simulation (MTD) through
    the XTB program to obtain new conformations.
    The GFN-FF force field is used.
    '''
    with open('temp.xyz', 'w') as f:
        write_xyz(coords, atomnos, f, title='temp')

    s = (
        '$md\n'
        '   time=%s\n' % (new_structures) +
        '   step=1\n'
        '   temp=300\n'
        '$end\n'
        '$metadyn\n'
        '   save=%s\n' % (new_structures) +
        '$end'
        )
         
    if constrained_indices is not None:
        s += '\n$constrain\n'
        for a, b in constrained_indices:
            s += '   distance: %s, %s, %s\n' % (a+1, b+1, round(norm_of(coords[a]-coords[b]), 5))

    s = ''.join(s)
    with open('temp.inp', 'w') as f:
        f.write(s)

    try:
        check_call(f'xtb --md --input temp.inp temp.xyz --gfnff > Structure{title}_MTD.log 2>&1'.split(), stdout=DEVNULL, stderr=STDOUT)

    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        sys.exit()

    structures = [coords]
    for n in range(1,new_structures):
        name = 'scoord.'+str(n)
        structures.append(parse_xtb_out(name))
        os.remove(name)

    for filename in ('gfnff_topo', 'xtbmdoc', 'mdrestart'):
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

    # if debug:
    os.rename('xtb.trj', f'Structure{title}_MTD_traj.xyz')

    # else:
    #     os.remove('xtb.traj')  

    structures = np.array(structures)

    return structures

def parse_xtb_out(filename):
    '''
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()

    coords = np.zeros((len(lines)-3,3))

    for _l, line in enumerate(lines[1:-2]):
        coords[_l] = line.split()[:-1]

    return coords * 0.529177249 # Bohrs to Angstroms

def crest_mtd_search(
        coords,
        atomnos,

        constrained_indices=None,
        constrained_distances=None,

        constrained_angles_indices=None,
        constrained_angles_values=None,

        constrained_dihedrals_indices=None,
        constrained_dihedrals_values=None,

        method='GFN2-XTB//GFN-FF',
        solvent='CH2Cl2',
        charge=0,
        kcal=None,
        ncimode=False,
        title='temp',
        procs=4,
        threads=1,
        ):
    '''
    This function runs a crest metadynamic conformational search and 
    returns its output.

    coords: array of shape (n,3) with cartesian coordinates for atoms.

    atomnos: array of atomic numbers for atoms.

    constrained_indices: array of shape (n,2), with the indices
    of atomic pairs to be constrained.

    constrained_distances: optional, target distances for the specified
    distance constraints. 

    constrained_dihedrals: quadruplets of atomic indices to constrain.

    constrained_dih_angles: target dihedral angles for the dihedral constraints.

    method: string, specifying the theory level to be used.

    solvent: solvent to be used in the calculation (ALPB model).

    charge: charge to be used in the calculation.

    title: string, used as a file name and job title for the mopac input file.

    procs: number of cores to be used for the calculation.

    threads: number of parallel threads to be used by the process. 

    '''

    with NewFolderContext(title):

        if constrained_indices is not None:
            if len(constrained_indices) == 0:
                constrained_indices = None

        if constrained_distances is not None:
            if len(constrained_distances) == 0:
                constrained_distances = None

        with open(f'{title}.xyz', 'w') as f:
            write_xyz(coords, atomnos, f, title=title)

        s = '$opt\n   '
            
        if constrained_indices is not None:  
            s += '\n$constrain\n'
            # s += '   atoms: '
            # for i in np.unique(np.array(constrained_indices).flatten()):
            #     s += f"{i+1},"

            for (c1, c2), cd in zip(constrained_indices, constrained_distances):
                cd = "auto" if cd is None else cd
                s += f"    distance: {c1+1}, {c2+1}, {cd}\n"

        if constrained_angles_indices is not None:
            assert len(constrained_angles_indices) == len(constrained_angles_values)
            s += '\n$constrain\n' if constrained_indices is None else ''
            for (a, b, c), angle in zip(constrained_angles_indices, constrained_angles_values):
                s += f"   angle: {a+1}, {b+1}, {c+1}, {angle}\n"

        if constrained_dihedrals_indices is not None:
            assert len(constrained_dihedrals_indices) == len(constrained_dihedrals_values)
            s += '\n$constrain\n' if constrained_indices is None else ''
            for (a, b, c, d), angle in zip(constrained_dihedrals_indices, constrained_dihedrals_values):
                s += f"   dihedral: {a+1}, {b+1}, {c+1}, {d+1}, {angle}\n"  
    
        s += "\n$metadyn\n  atoms: "

        constrained_atoms_cumulative = set()
        if constrained_indices is not None:
            for c1, c2 in constrained_indices:
                constrained_atoms_cumulative.add(c1)
                constrained_atoms_cumulative.add(c2)

        if constrained_angles_indices is not None:
            for c1, c2, c3 in constrained_angles_indices:
                constrained_atoms_cumulative.add(c1)
                constrained_atoms_cumulative.add(c2)
                constrained_atoms_cumulative.add(c3)

        if constrained_dihedrals_indices is not None:
            for c1, c2, c3, c4 in constrained_dihedrals_indices:
                constrained_atoms_cumulative.add(c1)
                constrained_atoms_cumulative.add(c2)
                constrained_atoms_cumulative.add(c3)
                constrained_atoms_cumulative.add(c4)

        # write atoms that need to be moved during metadynamics (all but constrained)
        active_ids = np.array([i+1 for i, _ in enumerate(atomnos) if i not in constrained_atoms_cumulative])

        while len(active_ids) > 2:
            i = next((i for i, _ in enumerate(active_ids[:-2]) if active_ids[i+1]-active_ids[i]>1), len(active_ids)-1)
            if active_ids[0] == active_ids[i]:
                s += f"{active_ids[0]},"
            else:
                s += f"{active_ids[0]}-{active_ids[i]},"
            active_ids = active_ids[i+1:]

        # remove final comma
        s = s[:-1]
        s += '\n$end'

        s = ''.join(s)
        with open(f'{title}.inp', 'w') as f:
            f.write(s)
        
        # avoid restarting the run
        flags = '--norestart'
        
        # add method flag
        if method.upper() in ('GFN-FF', 'GFNFF'):
            flags += ' --gfnff'
            # declaring the use of FF instead of semiempirical

        elif method.upper() in ('GFN2-XTB', 'GFN2'):
            flags += ' --gfn2'

        elif method.upper() in ('GFN2-XTB//GFN-FF', 'GFN2//GFNFF'):
            flags += ' --gfn2//gfnff'

        # adding other options
        if charge != 0:
            flags += f' --chrg {charge}'

        if procs is not None:
            flags += f' -P {procs}'

        if threads is not None:
            flags += f' -T {threads}'

        if solvent is not None:

            if solvent == 'methanol':
                flags += ' --gbsa methanol'

            else:
                flags += f' --alpb {solvent}'

        if kcal is None:
            kcal = 10
        flags += f' --ewin {kcal}'

        if ncimode:
            flags += ' --nci'

        flags += ' --noreftopo'

        try:
            with open(f"{title}.out", "w") as f:
                check_call(f'crest {title}.xyz --cinp {title}.inp {flags}'.split(), stdout=f, stderr=STDOUT)
    
        except KeyboardInterrupt:
            print('KeyboardInterrupt requested by user. Quitting.')
            sys.exit()

        # if CREST crashes, cd into the parent folder before propagating the error
        except CalledProcessError:
            os.chdir(os.path.dirname(os.getcwd()))
            raise CalledProcessError

        new_coords = read_xyz('crest_conformers.xyz').atomcoords

        # clean_directory((f'{title}.inp', f'{title}.xyz', f"{title}.out"))     

        for filename in ('gfnff_topo',
                            'charges',
                            'wbo',
                            'xtbrestart',
                            'xtbtopo.mol', 
                            '.xtboptok',
                            'gfnff_adjacency',
                            'gfnff_charges',
                        ):
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass
        
        return new_coords

def xtb_gsolv(coords, atomnos, model='alpb', charge=0, solvent='ch2cl2', title='temp', assert_convergence=True):
    '''
    Returns the solvation free energy in kcal/mol, as computed by XTB.
    Single-point energy calculation carried out with GFN-FF.

    '''
    
    with NewFolderContext(title):

        with open(f'{title}.xyz', 'w') as f:
            write_xyz(coords, atomnos, f, title=title)

        # outname = f'{title}_xtbopt.xyz' DOES NOT WORK - XTB ISSUE?
        outname = 'xtbopt.xyz'    
        flags = '--norestart'
            
        # declaring the use of FF instead of semiempirical
        flags += ' --gfnff'

        if charge != 0:
            flags += f' --chrg {charge}'

        flags += f' --{model} {solvent}'

        try:
            with open(f"{title}.out", "w") as f:
                check_call(f'xtb {title}.xyz {flags}'.split(), stdout=f, stderr=STDOUT)

        # sometimes the SCC does not converge: only raise the error if specified
        except CalledProcessError:
            if assert_convergence:
                raise CalledProcessError
        
        except KeyboardInterrupt:
            print('KeyboardInterrupt requested by user. Quitting.')
            sys.exit()

                
        else:    
            gsolv = energy_grepper(f"{title}.out", '-> Gsolv', 3)
            clean_directory((f'{title}.inp', f'{title}.xyz', f"{title}.out", outname))

        for filename in ('gfnff_topo',
                            'charges',
                            'wbo',
                            'xtbrestart',
                            'xtbtopo.mol', 
                            '.xtboptok',
                            'gfnff_adjacency',
                            'gfnff_charges',
                        ):
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass
        
        return gsolv