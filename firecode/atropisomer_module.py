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

from time import time

import numpy as np
from ase import Atoms
from ase.constraints import FixInternals
from ase.optimize import LBFGS
from networkx.algorithms.components.connected import connected_components
from networkx.algorithms.shortest_paths.generic import shortest_path

from firecode.algebra import dihedral
from firecode.ase_manipulations import ase_neb, ase_saddle, get_ase_calc
from firecode.errors import ZeroCandidatesError
from firecode.hypermolecule_class import graphize
from firecode.optimization_methods import optimize
from firecode.pruning import prune_by_rmsd
from firecode.utils import (align_structures, clean_directory, loadbar,
                          molecule_check, time_to_string, write_xyz)


def ase_torsion_TSs(embedder,
                    coords,
                    atomnos,
                    indices,
                    threshold_kcal=5,
                    title='temp',
                    optimization=True,
                    logfile=None,
                    bernytraj=None,
                    plot=False):
    '''
    Automated dihedral scan. Runs two preliminary scans
    (clockwise, anticlockwise) in 10 degrees increments,
    then peaks above 'kcal_thresh' are re-scanned accurately
    in 1 degree increments.

    '''
    
    assert len(indices) == 4
    # cyclical = False
    
    ts_structures, energies = [], []

    graph = graphize(coords, atomnos)
    i1, i2, i3, i4 = indices

    if all([len(shortest_path(graph, start, end)) == 2 for start, end in zip(indices[0:-1], indices[1:])]):
        graph.remove_edge(i2, i3)
        subgraphs = connected_components(graph)

        for subgraph in subgraphs:
            if i3 in subgraph:
                indices_to_be_moved = subgraph - {i3}
                break

        if i1 in indices_to_be_moved:

            # cyclical = True
            indices_to_be_moved = [i4]
            # if molecule is cyclical, just move the fourth atom and
            # let the rest of the structure relax

            s = 'The specified dihedral angle is comprised within a cycle. Switching to safe dihedral scan (moving only last index).'
            print(s)
            if logfile is not None:
                logfile.write(s+'\n')

    else:

        if not embedder.options.let:
            raise SystemExit('The specified dihedral angle is made up of non-contiguous atoms. To prevent errors, the\n' +
                             'run has been stopped. Override this behavior with the LET keyword.')

        # if user did not provide four contiguous indices,
        # and did that on purpose, just move the fourth atom and
        # let the rest of the structure relax
        indices_to_be_moved = [i4]
        # cyclical = True

        s = 'The specified dihedral angle is made up of non-contiguous atoms.\nThis might cause some unexpected results.'
        print(s)
        if logfile is not None:
            logfile.write(s+'\n')


    # routine = ((10, 18, '_clockwise'), (-10, 18, '_counterclockwise')) if cyclical else ((10, 36, ''),)
    routine = ((10, 36, '_clockwise'), (-10, 36, '_counterclockwise'))


    for degrees, steps, direction in routine:

        print()
        if logfile is not None:
            logfile.write('\n')

        structures, energies = ase_dih_scan(embedder,
                                        coords,
                                        atomnos,
                                        indices=indices,
                                        degrees=degrees,
                                        steps=steps,
                                        relaxed=optimization,
                                        indices_to_be_moved=indices_to_be_moved,
                                        title='Preliminary scan' + ((' (clockwise)' if direction == '_clockwise' \
                                              else ' (counterclockwise)') if direction != '' else ''),
                                        logfile=logfile)

        min_e = min(energies)
        rel_energies = [e-min_e for e in energies]

        tag = '_relaxed' if optimization else '_rigid'
        
        with open(title + tag + direction + '_scan.xyz', 'w') as outfile:
            for s, structure in enumerate(align_structures(np.array(structures), indices[:-1])):
                write_xyz(structure, atomnos, outfile, title=f'Scan point {s+1}/{len(structures)} - Rel. E = {round(rel_energies[s], 3)} kcal/mol')

        if plot:
            import matplotlib.pyplot as plt

            plt.figure()

            x1 = [dihedral(structure[indices]) for structure in structures]
            y1 = [e-min_e for e in energies]

            for i, (x_, y_) in enumerate(get_plot_segments(x1, y1, max_step=abs(degrees)+1)):

                plt.plot(x_,
                        y_,
                        '-',
                        color='tab:blue',
                        label=('Preliminary SCAN'+direction) if i == 0 else None,
                        linewidth=3,
                        alpha=0.50)

        peaks_indices = atropisomer_peaks(energies, min_thr=min_e+threshold_kcal, max_thr=min_e+75)

        if peaks_indices:

            s = 's' if len(peaks_indices) > 1 else ''
            print(f'Found {len(peaks_indices)} peak{s}. Performing accurate scan{s}.\n')
            if logfile is not None:
                logfile.write(f'Found {len(peaks_indices)} peak{s}. Performing accurate scan{s}.\n\n')


            for p, peak in enumerate(peaks_indices):

                sub_structures, sub_energies = ase_dih_scan(embedder,
                                                        structures[peak-1],
                                                        atomnos,
                                                        indices=indices,
                                                        degrees=degrees/10, #1° or -1°
                                                        steps=20,
                                                        relaxed=optimization,
                                                        ad_libitum=True, # goes on until the hill is crossed
                                                        indices_to_be_moved=indices_to_be_moved,
                                                        title=f'Accurate scan {p+1}/{len(peaks_indices)}',
                                                        logfile=logfile)

                if logfile is not None:
                    logfile.write('\n')

                if plot:
                    x2 = [dihedral(structure[indices]) for structure in sub_structures]
                    y2 = [e-min_e for e in sub_energies]

                    for i, (x_, y_) in enumerate(get_plot_segments(x2, y2, max_step=abs(degrees/10)+1)):

                        plt.plot(x_, 
                                y_,
                                '-o',
                                color='tab:red',
                                label='Accurate SCAN' if (p == 0 and i == 0) else None,
                                markersize=1,
                                linewidth=2,
                                alpha=0.5)

                sub_peaks_indices = atropisomer_peaks(sub_energies, min_thr=threshold_kcal+min_e, max_thr=min_e+75)

                if sub_peaks_indices:

                    s = 's' if len(sub_peaks_indices) > 1 else ''
                    msg = f'Found {len(sub_peaks_indices)} sub-peak{s}.'
                    
                    if embedder.options.saddle or embedder.options.neb:
                        if embedder.options.saddle:
                            tag = 'saddle'
                        else:
                            tag = 'NEB TS'

                        msg += f'Performing {tag} optimization{s}.'

                    print(msg)

                    if logfile is not None:
                        logfile.write(s+'\n')

                    for s, sub_peak in enumerate(sub_peaks_indices):

                        if plot:
                            x = dihedral(sub_structures[sub_peak][indices])
                            y = sub_energies[sub_peak]-min_e
                            plt.plot(x, y, color='gold', marker='o', label='Maxima' if p == 0 else None, markersize=3)

                        if embedder.options.saddle:

                            loadbar_title = f'  > Saddle opt on sub-peak {s+1}/{len(sub_peaks_indices)}'
                            # loadbar(s+1, len(sub_peaks_indices), loadbar_title+' '*(29-len(loadbar_title)))
                            print(loadbar_title)
                        
                            optimized_geom, energy, _ = ase_saddle(embedder,
                                                                    sub_structures[sub_peak],
                                                                    atomnos,
                                                                    title=f'Saddle opt - peak {p+1}, sub-peak {s+1}',
                                                                    logfile=logfile,
                                                                    traj=bernytraj+f'_{p+1}_{s+1}.traj' if bernytraj is not None else None)

                            if molecule_check(coords, optimized_geom, atomnos):
                                ts_structures.append(optimized_geom)
                                energies.append(energy)

                        elif embedder.options.neb:

                            loadbar_title = f'  > NEB TS opt on sub-peak {s+1}/{len(sub_peaks_indices)}, {direction[1:]}'
                            drctn = 'clkws' if direction == '_clockwise' else 'ccws'
                            
                            print(loadbar_title)
                        
                            optimized_geom, energy, success = ase_neb(embedder,
                                                                        sub_structures[sub_peak-2],
                                                                        sub_structures[(sub_peak+1)%len(sub_structures)],
                                                                        atomnos,
                                                                        n_images=5,
                                                                        title=f'{title}_NEB_peak_{p+1}_sub-peak_{s+1}_{drctn}',
                                                                        logfunction=embedder.log)

                            if success and molecule_check(coords, optimized_geom, atomnos):
                                ts_structures.append(optimized_geom)
                                energies.append(energy)

                        else:
                            ts_structures.append(sub_structures[sub_peak])
                            energies.append(sub_energies[sub_peak])

                        print()
            
                else:
                    print('No suitable sub-peaks found.\n')
                    if logfile is not None:
                        logfile.write('No suitable sub-peaks found.\n\n')
        else:
            print('No suitable peaks found.\n')
            if logfile is not None:
                logfile.write('No suitable peaks found.\n\n')

        if plot:
            plt.legend()
            plt.xlabel(f'Dihedral Angle {tuple(indices)}')
            plt.ylabel('Energy (kcal/mol)')
            # with open(f'{title}{direction}_plt.pickle', 'wb') as _f:
            #     pickle.dump(fig, _f)
            plt.savefig(f'{title}{direction}_plt.svg')

    ts_structures = np.array(ts_structures)

    clean_directory()

    return ts_structures, energies

def atropisomer_peaks(data, min_thr, max_thr):
    '''
    data: iterable
    min_thr: peaks must be values greater than min_thr
    max_thr: peaks must be values smaller than max_thr
    return: list of peak indices
    '''
    _l = len(data)
    peaks = [i for i in range(_l-2) if (

        data[i-1] < data[i] >= data[i+1] and
        # peaks have neighbors that are smaller than them

        max_thr > data[i] > min_thr and
        # discard peaks that are too small or too big

        # abs(data[i] - min((data[i-1], data[i+1]))) > 2
        data[i] == max(data[i-2:i+3])
        # discard peaks that are not the highest within close nieghbors
    )]

    return peaks
    
def ase_dih_scan(embedder,
            coords,
            atomnos,
            indices,
            degrees=10,
            steps=36,
            relaxed=True,
            ad_libitum=False,
            indices_to_be_moved=None,
            title='temp scan',
            logfile=None):
    '''
    Performs a dihedral scan via the ASE library
    if ad libitum, steps is the minimum number of performed steps
    '''
    assert len(indices) == 4

    if ad_libitum:
        if not relaxed:
            raise Exception('The ad_libitum keyword is only available for relaxed scans.')

    atoms = Atoms(atomnos, positions=coords)
    structures, energies = [], []

    atoms.calc = get_ase_calc(embedder)

    if indices_to_be_moved is None:
        indices_to_be_moved = range(len(atomnos))

    mask = np.array([i in indices_to_be_moved for i, _ in enumerate(atomnos)], dtype=bool)

    t_start = time()

    if logfile is not None:
        logfile.write(f'  > {title}\n')

    for scan_step in range(1000):

        loadbar_title = f'{title} - step {scan_step+1}'
        if ad_libitum:
            print(loadbar_title, end='\r')
        else:
            loadbar_title += '/'+str(steps)
            loadbar(scan_step+1, steps, loadbar_title+' '*(29-len(loadbar_title)))

        if logfile is not None:
            t_start_step = time()

        if relaxed:
            atoms.set_constraint(FixInternals(dihedrals_deg=[[atoms.get_dihedral(*indices), indices]]))
            
            with LBFGS(atoms, maxstep=0.2, logfile=None, trajectory=None) as opt:
                
                try:
                    opt.run(fmax=0.05, steps=500)
                    exit_str = 'converged'

                except ValueError: # Shake did not converge
                    exit_str = 'crashed'

                iterations = opt.nsteps

            energies.append(atoms.get_total_energy() * 23.06054194532933) # eV to kcal/mol

        if logfile is not None:
            elapsed = time() - t_start_step
            s = '/' + str(steps) if not ad_libitum else ''
            logfile.write(f'        Step {scan_step+1}{s} - {exit_str} - {iterations} iterations ({time_to_string(elapsed)})\n')

        structures.append(atoms.get_positions())

        atoms.rotate_dihedral(*indices, angle=degrees, mask=mask)

        if exit_str == 'crashed':
            break

        elif scan_step+1 >= steps:
            if ad_libitum:
                if any((
                    (max(energies) - energies[-1]) > 1,
                    (max(energies) - energies[-1]) > max(energies)-energies[0],
                    (energies[-1] - min(energies)) > 50
                )):

                    # ad_libitum stops when one of these conditions is met:
                    # - we surpassed and are below the maximum of at least 1 kcal/mol
                    # - we surpassed maximum and are below starting point
                    # - current step energy is more than 50 kcal/mol above starting point

                    print(loadbar_title)
                    break
            else:
                break

    structures = np.array(structures)

    clean_directory()

    if logfile is not None:
        elapsed = time() - t_start
        logfile.write(f'{title} - completed ({time_to_string(elapsed)})\n')

    return align_structures(structures, indices), energies

def get_plot_segments(x, y, max_step=2):
    '''
    Returns a zip object with x, y segments.
    A single segment has x values with separation
    smaller than max_step.
    '''
    x, y = zip(*sorted(zip(x, y), key=lambda t: t[0]))
    
    x_slices, y_slices = [], []
    for i, n in enumerate(x):
        if abs(x[i-1]-n) > max_step:
            x_slices.append([])
            y_slices.append([])

        x_slices[-1].append(n)
        y_slices[-1].append(y[i])

    return zip(x_slices, y_slices)

def dihedral_scan(embedder):
    '''
    Automated dihedral scan. Runs two preliminary scans
    (clockwise, anticlockwise) in 10 degrees increments,
    then peaks above 'kcal_thresh' are re-scanned accurately
    in 1 degree increments.

    '''

    if 'kcal' not in embedder.kw_line.lower():
    # set to 5 if user did not specify a value
        embedder.options.kcal_thresh = 5

    mol = embedder.objects[0]
    embedder.structures, embedder.energies = [], []


    embedder.log(f'\n--> {mol.filename} - performing a scan of dihedral angle with indices {mol.reactive_indices}\n')

    for c, coords in enumerate(mol.atomcoords):

        embedder.log(f'\n--> Pre-optimizing input structure{"s" if len(mol.atomcoords) > 1 else ""} '
                   f'({embedder.options.theory_level} via {embedder.options.calculator})')

        embedder.log(f'--> Performing relaxed scans (conformer {c+1}/{len(mol.atomcoords)})')

        new_coords, ground_energy, success = optimize(
                                                    coords,
                                                    mol.atomnos,
                                                    embedder.options.calculator,
                                                    method=embedder.options.theory_level,
                                                    procs=embedder.procs,
                                                    solvent=embedder.options.solvent
                                                )

        if not success:
            embedder.log(f'Pre-optimization failed - Skipped conformer {c+1}', p=False)
            continue

        structures, energies = ase_torsion_TSs(embedder,
                                                new_coords,
                                                mol.atomnos,
                                                mol.reactive_indices,
                                                threshold_kcal=embedder.options.kcal_thresh,
                                                title=mol.rootname+f'_conf_{c+1}',
                                                optimization=embedder.options.optimization,
                                                logfile=embedder.logfile,
                                                bernytraj=mol.rootname + '_berny' if embedder.options.debug else None,
                                                plot=True)

        for structure, energy in zip(structures, energies):
            embedder.structures.append(structure)
            embedder.energies.append(energy)

    embedder.structures = np.array(embedder.structures)
    embedder.energies = np.array(embedder.energies)
    embedder.atomnos = mol.atomnos

    if len(embedder.structures) == 0:
        s = ('\n--> Dihedral scan did not find any suitable maxima above the set threshold\n'
            f'    ({embedder.options.kcal_thresh} kcal/mol) during the scan procedure. Observe the\n'
                '    generated energy plot and try lowering the threshold value (KCAL keyword).')
        embedder.log(s)
        raise ZeroCandidatesError()

    # remove similar structures (RMSD)
    embedder.structures, mask = prune_by_rmsd(embedder.structures, mol.atomnos, max_rmsd=embedder.options.rmsd, debugfunction=embedder.debuglog)
    embedder.energies = embedder.energies[mask]
    if 0 in mask:
        embedder.log(f'Discarded {int(len([b for b in mask if not b]))} candidates for RMSD similarity ({len([b for b in mask if b])} left)')

    # sort structures based on energy
    embedder.energies, embedder.structures = zip(*sorted(zip(embedder.energies, embedder.structures), key=lambda x: x[0]))
    embedder.structures = np.array(embedder.structures)
    embedder.energies = np.array(embedder.energies)

    # write output and exit
    embedder.write_structures('maxima', indices=mol.reactive_indices, relative=True, extra='(barrier height)', align='moi')
    embedder.normal_termination()