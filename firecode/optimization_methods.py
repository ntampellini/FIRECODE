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

import time
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R

from firecode.algebra import norm, norm_of
from firecode.ase_manipulations import ase_neb, ase_popt
from firecode.calculators._gaussian import gaussian_opt
from firecode.calculators._mopac import mopac_opt
from firecode.calculators._orca import orca_opt
from firecode.calculators._xtb import xtb_opt
from firecode.pruning import prune_by_rmsd
from firecode.settings import DEFAULT_LEVELS
from firecode.utils import (loadbar, molecule_check, pt, scramble_check,
                          time_to_string, write_xyz)


class Opt_func_dispatcher:

    def __init__(self):

        self.opt_funcs_dict = {
            'MOPAC':mopac_opt,
            'ORCA':orca_opt,
            'GAUSSIAN':gaussian_opt,
            'XTB':xtb_opt,
        }

    def load_aimnet2_calc(self, theory_level, logfunction=print):
        
        try:
            from aimnet2_firecode.interface import aimnet2_opt, get_aimnet2_calc

        except ImportError:
            raise Exception(('Cannot import AIMNet2 python bindings for FIRECODE. Install them with:\n'
                            '>>> pip install aimnet2_firecode'))
            
        self.opt_funcs_dict['AIMNET2'] = aimnet2_opt
        self.aimnet2_calc = get_aimnet2_calc(theory_level, logfunction=logfunction)

def optimize(
            coords,
            atomnos,
            calculator,
            method=None,
            maxiter=None,
            conv_thr="tight",

            constrained_indices=None,
            constrained_distances=None,
                  
            constrained_dihedrals_indices=None,
            constrained_dihedrals_values=None,

            constrained_angles_indices=None,
            constrained_angles_values=None,

            mols_graphs=None,
            procs=1,
            solvent=None,
            charge=0,
            max_newbonds=0,
            title='temp',
            check=True, 
            logfunction=None,

            dispatcher=None,
            **kwargs,
            ):
    '''
    Performs a geometry [partial] optimization (OPT/POPT) with MOPAC, ORCA, Gaussian or XTB at $method level, 
    constraining the distance between the specified atom pairs, if any. Moreover, if $check, performs a check on atomic
    pairs distances to ensure that the optimization has preserved molecular identities and no atom scrambling occurred.

    :params calculator: Calculator to be used. ('MOPAC', 'ORCA', 'GAUSSIAN', 'XTB', 'AIMNET2')
    :params coords: list of coordinates for each atom in the TS
    :params atomnos: list of atomic numbers for each atom in the TS
    :params mols_graphs: list of molecule.graph objects, containing connectivity information for each molecule
    :params constrained_indices: indices of constrained atoms in the TS geometry, if this is one
    :params method: Level of theory to be used in geometry optimization. Default if UFF.

    :return opt_coords: optimized structure
    :return energy: absolute energy of structure, in kcal/mol
    :return not_scrambled: bool, indicating if the optimization shifted up some bonds (except the constrained ones)
    '''

    if dispatcher is None:
        dispatcher = Opt_func_dispatcher()

        if calculator == 'AIMNET2':
            dispatcher.load_aimnet2_calc(method)

    if mols_graphs is not None:
        _l = [len(graph.nodes) for graph in mols_graphs]
        assert len(coords) == sum(_l), f'{len(coords)} coordinates are specified but graphs have {_l} = {sum(_l)} nodes'

    if method is None:
        method = DEFAULT_LEVELS[calculator]

    if constrained_distances is not None:
        assert len(constrained_distances) == len(constrained_indices), f'len(cd) = {len(constrained_distances)} != len(ci) = {len(constrained_indices)}'

    constrained_indices = np.array(()) if constrained_indices is None else constrained_indices

    # opt_func = opt_funcs_dict[calculator]
    opt_func = dispatcher.opt_funcs_dict[calculator]

    t_start = time.perf_counter()

    # success checks that calculation had a normal termination
    opt_coords, energy, success = opt_func(coords,
                                            atomnos,

                                            constrained_indices=constrained_indices,
                                            constrained_distances=constrained_distances,
                                            
                                            constrained_dihedrals_indices=constrained_dihedrals_indices,
                                            constrained_dihedrals_values=constrained_dihedrals_values,

                                            constrained_angles_indices=constrained_angles_indices,
                                            constrained_angles_values=constrained_angles_values,

                                            method=method,
                                            procs=procs,
                                            solvent=solvent,
                                            maxiter=maxiter,
                                            conv_thr=conv_thr,
                                            title=title,
                                            charge=charge,

                                            ase_calc = dispatcher.aimnet2_calc if calculator == 'AIMNET2' else None,
                                            **kwargs)
    
    elapsed = time.perf_counter() - t_start

    if success:
        if check:
        # check boolean ensures that no scrambling occurred during the optimization
            if mols_graphs is not None:
                success = scramble_check(opt_coords, atomnos, constrained_indices, mols_graphs, max_newbonds=max_newbonds)
            else:
                success = molecule_check(coords, opt_coords, atomnos, max_newbonds=max_newbonds)

        if logfunction is not None:
            if success:
                logfunction(f'    - {title} - REFINED {time_to_string(elapsed)}')
            else:
                logfunction(f'    - {title} - SCRAMBLED {time_to_string(elapsed)}')             

        return opt_coords, energy, success

    if logfunction is not None:
        logfunction(f'    - {title} - CRASHED')

    return coords, energy, False

def hyperNEB(embedder, coords, atomnos, ids, constrained_indices, title='temp'):
    '''
    Turn a geometry close to TS to a proper TS by getting
    reagents and products and running a climbing image NEB calculation through ASE.
    '''

    reagents = get_reagent(embedder, coords, atomnos, ids, constrained_indices, method=embedder.options.theory_level)
    products = get_product(embedder, coords, atomnos, ids, constrained_indices, method=embedder.options.theory_level)
    # get reagents and products for this reaction

    reagents -= np.mean(reagents, axis=0)
    products -= np.mean(products, axis=0)
    # centering both structures on the centroid of reactive atoms

    aligment_rotation = R.align_vectors(reagents, products)
    # products = np.array([aligment_rotation @ v for v in products])
    products = (aligment_rotation @ products.T).T
    # rotating the two structures to minimize differences

    ts_coords, ts_energy, energies, success = ase_neb(embedder, reagents, products, atomnos, title=title)
    # Use these structures plus the TS guess to run a NEB calculation through ASE

    return ts_coords, ts_energy, energies, success

def get_product(embedder, coords, atomnos, ids, constrained_indices, method='PM7'):
    '''
    Part of the automatic NEB implementation.
    Returns a structure that presumably is the association reaction product
    ([cyclo]additions reactions in mind)
    '''

    opt_func = embedder.dispatcher.opt_funcs_dict[embedder.options.calculator]

    bond_factor = 1.2
    # multiple of sum of covalent radii for two atoms.
    # If two atoms are closer than this times their sum
    # of c_radii, they are considered to converge to
    # products when their geometry is optimized. 

    step_size = 0.1
    # in Angstroms

    if len(ids) == 2:

        mol1_center = np.mean([coords[a] for a, _ in constrained_indices], axis=0)
        mol2_center = np.mean([coords[b] for _, b in constrained_indices], axis=0)
        motion = norm(mol2_center - mol1_center)
        # norm of the motion that, when applied to mol1,
        # superimposes its reactive atoms to the ones of mol2

        threshold_dists = [bond_factor*(pt[atomnos[a]].covalent_radius +
                                        pt[atomnos[b]].covalent_radius) for a, b in constrained_indices]

        reactive_dists = [norm_of(coords[a] - coords[b]) for a, b in constrained_indices]
        # distances between reactive atoms

        while not np.all([reactive_dists[i] < threshold_dists[i] for i, _ in enumerate(constrained_indices)]):
            # print('Reactive distances are', reactive_dists)

            coords[:ids[0]] += motion*step_size

            coords, _, _ = opt_func(coords, atomnos, constrained_indices, method=method)

            reactive_dists = [norm_of(coords[a] - coords[b]) for a, b in constrained_indices]

        newcoords, _, _ = opt_func(coords, atomnos, method=method)
        # finally, when structures are close enough, do a free optimization to get the reaction product

        new_reactive_dists = [norm_of(newcoords[a] - newcoords[b]) for a, b in constrained_indices]

        if np.all([new_reactive_dists[i] < threshold_dists[i] for i, _ in enumerate(constrained_indices)]):
        # return the freely optimized structure only if the reagents did not repel each other
        # during the optimization, otherwise return the last coords, where partners were close
            return newcoords

        return coords

    # trimolecular TSs: the approach is to bring the first pair of reactive
    # atoms closer until optimization bounds the molecules together

    index_to_be_moved = constrained_indices[0,0]
    reference = constrained_indices[0,1]
    moving_molecule_index = next(i for i,n in enumerate(np.cumsum(ids)) if index_to_be_moved < n)
    bounds = [0] + [n+1 for n in np.cumsum(ids)]
    moving_molecule_slice = slice(bounds[moving_molecule_index], bounds[moving_molecule_index+1])
    threshold_dist = bond_factor*(pt[atomnos[constrained_indices[0,0]]].covalent_radius +
                                    pt[atomnos[constrained_indices[0,1]]].covalent_radius)

    motion = (coords[reference] - coords[index_to_be_moved])
    # vector from the atom to be moved to the target reactive atom

    while norm_of(motion) > threshold_dist:
    # check if the reactive atoms are sufficiently close to converge to products

        for i, atom in enumerate(coords[moving_molecule_slice]):
            dist = norm_of(atom - coords[index_to_be_moved])
            # for any atom in the molecule, distance from the reactive atom

            atom_step = step_size*np.exp(-0.5*dist)
            coords[moving_molecule_slice][i] += norm(motion)*atom_step
            # the more they are close, the more they are moved

        # print('Reactive dist -', norm_of(motion))
        coords, _, _ = opt_func(coords, atomnos, constrained_indices, method=method)
        # when all atoms are moved, optimize the geometry with the previous constraints

        motion = (coords[reference] - coords[index_to_be_moved])

    newcoords, _, _ = opt_func(coords, atomnos, method=method)
    # finally, when structures are close enough, do a free optimization to get the reaction product

    new_reactive_dist = norm_of(newcoords[constrained_indices[0,0]] - newcoords[constrained_indices[0,0]])

    if new_reactive_dist < threshold_dist:
    # return the freely optimized structure only if the reagents did not repel each other
    # during the optimization, otherwise return the last coords, where partners were close
        return newcoords

    return coords

def get_reagent(embedder, coords, atomnos, ids, constrained_indices, method='PM7'):
    '''
    Part of the automatic NEB implementation.
    Returns a structure that presumably is the association reaction reagent.
    ([cyclo]additions reactions in mind)
    '''

    opt_func = embedder.dispatcher.opt_funcs_dict[embedder.options.calculator]

    bond_factor = 1.5
    # multiple of sum of covalent radii for two atoms.
    # Putting reactive atoms at this times their bonding
    # distance and performing a constrained optimization
    # is the way to get a good guess for reagents structure. 

    if len(ids) == 2:

        mol1_center = np.mean([coords[a] for a, _ in constrained_indices], axis=0)
        mol2_center = np.mean([coords[b] for _, b in constrained_indices], axis=0)
        motion = norm(mol2_center - mol1_center)
        # norm of the motion that, when applied to mol1,
        # superimposes its reactive centers to the ones of mol2

        threshold_dists = [bond_factor*(pt[atomnos[a]].covalent_radius + pt[atomnos[b]].covalent_radius) for a, b in constrained_indices]

        reactive_dists = [norm_of(coords[a] - coords[b]) for a, b in constrained_indices]
        # distances between reactive atoms

        coords[:ids[0]] -= norm(motion)*(np.mean(threshold_dists) - np.mean(reactive_dists))
        # move reactive atoms away from each other just enough

        coords, _, _ = opt_func(coords, atomnos, constrained_indices=constrained_indices, method=method)
        # optimize the structure but keeping the reactive atoms distanced

        return coords

    # trimolecular TSs: the approach is to bring the first pair of reactive
    # atoms apart just enough to get a good approximation for reagents

    index_to_be_moved = constrained_indices[0,0]
    reference = constrained_indices[0,1]
    moving_molecule_index = next(i for i,n in enumerate(np.cumsum(ids)) if index_to_be_moved < n)
    bounds = [0] + [n+1 for n in np.cumsum(ids)]
    moving_molecule_slice = slice(bounds[moving_molecule_index], bounds[moving_molecule_index+1])
    threshold_dist = bond_factor*(pt[atomnos[constrained_indices[0,0]]].covalent_radius +
                                    pt[atomnos[constrained_indices[0,1]]].covalent_radius)

    motion = (coords[reference] - coords[index_to_be_moved])
    # vector from the atom to be moved to the target reactive atom

    displacement = norm(motion)*(threshold_dist-norm_of(motion))
    # vector to be applied to the reactive atom to push it far just enough

    for i, atom in enumerate(coords[moving_molecule_slice]):
        dist = norm_of(atom - coords[index_to_be_moved])
        # for any atom in the molecule, distance from the reactive atom

        coords[moving_molecule_slice][i] -= displacement*np.exp(-0.5*dist)
        # the closer they are to the reactive atom, the further they are moved

    coords, _, _ = opt_func(coords, atomnos, constrained_indices=np.array([constrained_indices[0]]), method=method)
    # when all atoms are moved, optimize the geometry with only the first of the previous constraints

    newcoords, _, _ = opt_func(coords, atomnos, method=method)
    # finally, when structures are close enough, do a free optimization to get the reaction product

    new_reactive_dist = norm_of(newcoords[constrained_indices[0,0]] - newcoords[constrained_indices[0,0]])

    if new_reactive_dist > threshold_dist:
    # return the freely optimized structure only if the reagents did not approached back each other
    # during the optimization, otherwise return the last coords, where partners were further away
        return newcoords
    
    return coords

def opt_linear_scan(embedder, coords, atomnos, scan_indices, constrained_indices, step_size=0.02, safe=False, title='temp', logfile=None, xyztraj=None):
    '''
    Runs a linear scan along the specified linear coordinate.
    The highest energy structure that passes sanity checks is returned.

    embedder
    coords
    atomnos
    scan_indices
    constrained_indices
    step_size
    safe
    title
    logfile
    xyztraj
    '''
    assert [i in constrained_indices.ravel() for i in scan_indices]

    i1, i2 = scan_indices
    far_thr = 2 * sum([pt[atomnos[i]].covalent_radius for i in scan_indices])
    t_start = time.perf_counter()
    total_iter = 0

    _, energy, _ = optimize(coords,
                            atomnos,
                            embedder.options.calculator,
                            embedder.options.theory_level,
                            constrained_indices=constrained_indices,
                            mols_graphs=embedder.graphs,
                            procs=embedder.procs,
                            max_newbonds=embedder.options.max_newbonds,
                            )

    direction = coords[i1] - coords[i2]
    base_dist = norm_of(direction)
    energies, geometries = [energy], [coords]

    for sign in (1, -1):
    # getting closer for sign == 1, further apart for -1
        active_coords = deepcopy(coords)
        dist = base_dist

        if scan_peak_present(energies):
            break

        for iterations in range(75):
            
            if safe: # use ASE optimization function - more reliable, but locks all interatomic dists

                targets = [norm_of(active_coords[a]-active_coords[b]) - step_size
                           if (a in scan_indices and b in scan_indices)
                           else norm_of(active_coords[a]-active_coords[b])
                           for a, b in constrained_indices]

                active_coords, energy, success = ase_popt(embedder,
                                                            active_coords,
                                                            atomnos,
                                                            constrained_indices,
                                                            targets=targets,
                                                            safe=True,
                                                            )

            else: # use faster raw optimization function, might scramble more often than the ASE one

                active_coords[i2] += sign * norm(direction) * step_size
                active_coords, energy, success = optimize(active_coords,
                                                            atomnos,
                                                            embedder.options.calculator,
                                                            embedder.options.theory_level,
                                                            constrained_indices=constrained_indices,
                                                            mols_graphs=embedder.graphs,
                                                            procs=embedder.procs,
                                                            max_newbonds=embedder.options.max_newbonds,
                                                            )

            if not success:
                if logfile is not None and iterations == 0:
                    logfile.write(f'    - {title} CRASHED at first step\n')

                if embedder.options.debug:
                    with open(title+'_SCRAMBLED.xyz', 'a') as f:
                        write_xyz(active_coords, atomnos, f, title=title+(
                            f' d({i1}-{i2}) = {round(dist, 3)} A, Rel. E = {round(energy-energies[0], 3)} kcal/mol'))

                break
            
            direction = active_coords[i1] - active_coords[i2]
            dist = norm_of(direction)

            total_iter += 1
            geometries.append(active_coords)
            energies.append(energy)

            if xyztraj is not None:
                with open(xyztraj, 'a') as f:
                    write_xyz(active_coords, atomnos, f, title=title+(
                        f' d({i1}-{i2}) = {round(dist, 3)} A, Rel. E = {round(energy-energies[0], 3)} kcal/mol'))

            if (dist < 1.2 and sign == 1) or (
                dist > far_thr and sign == -1) or (
                scan_peak_present(energies)
                ):
                break
            
    distances = [norm_of(g[i1]-g[i2]) for g in geometries]
    best_distance = distances[energies.index(max(energies))]

    distances_delta = [abs(d-best_distance) for d in distances]
    closest_geom = geometries[distances_delta.index(min(distances_delta))]
    closest_dist = distances[distances_delta.index(min(distances_delta))]

    direction = closest_geom[i1] - closest_geom[i2]
    closest_geom[i1] += norm(direction) * (best_distance-closest_dist)

    final_geom, final_energy, _ = optimize(closest_geom,
                                            atomnos,
                                            embedder.options.calculator,
                                            embedder.options.theory_level,
                                            constrained_indices=constrained_indices,
                                            mols_graphs=embedder.graphs,
                                            procs=embedder.procs,
                                            max_newbonds=embedder.options.max_newbonds,
                                            check=False,
                                            )

    if embedder.options.debug:

        if embedder.options.debug:
            with open(xyztraj, 'a') as f:
                write_xyz(active_coords, atomnos, f, title=title+(
                    f' FINAL - d({i1}-{i2}) = {round(norm_of(final_geom[i1]-final_geom[i2]), 3)} A,'
                    f' Rel. E = {round(final_energy-energies[0], 3)} kcal/mol'))

        import matplotlib.pyplot as plt

        plt.figure()

        distances = [norm_of(geom[i1]-geom[i2]) for geom in geometries]
        distances, sorted_energies = zip(*sorted(zip(distances, energies), key=lambda x: x[0]))

        plt.plot(distances,
                [s-energies[0] for s in sorted_energies],
                '-o',
                color='tab:red',
                label=f'Linear SCAN ({i1}-{i2})',
                linewidth=3,
                alpha=0.5)

        plt.plot(norm_of(coords[i1]-coords[i2]),
                0,
                marker='o',
                color='tab:blue',
                label='Starting point (0 kcal/mol)',
                markersize=5,
                )

        plt.plot(best_distance,
                final_energy-energies[0],
                marker='o',
                color='black',
                label='Interpolated best distance, actual energy',
                markersize=5)

        plt.legend()
        plt.title(title)
        plt.xlabel(f'Interatomic distance {tuple(scan_indices)}')
        plt.ylabel('Energy Rel. to starting point (kcal/mol)')
        plt.savefig(f'{title.replace(" ", "_")}_plt.svg')

    if logfile is not None:
        logfile.write(f'    - {title} COMPLETED {total_iter} steps ({time_to_string(time.perf_counter()-t_start)})\n')

    return final_geom, final_energy, True

def scan_peak_present(energies) -> bool:
    '''
    Returns True if the maximum value of the list
    occurs in the middle of it, that is not in first,
    second, second to last or last positions
    '''
    if energies.index(max(energies)) in range(2,len(energies)-1):
        return True
    return False

def fitness_check(coords, constraints, targets, threshold) -> bool:
    '''
    Returns True if the strucure respects
    the imposed pairings specified in constraints.
    targets: target distances for each constraint
    threshold: cumulative threshold to reject a structure (A)

    '''
    error = 0
    for (a, b), target in zip(constraints, targets):
        if target is not None:
            error += (norm_of(coords[a]-coords[b]) - target)
                    
    return error < threshold

def _refine_structures(structures,
                       atomnos,
                       calculator,
                       method,
                       procs,
                       constrained_indices=None,
                       constrained_distances=None,
                       solvent=None,
                       loadstring='',
                       logfunction=None):
    '''
    Refine a set of structures - optimize them and remove similar
    ones and high energy ones (>20 kcal/mol above lowest)
    '''
    energies = []
    for i, conformer in enumerate(deepcopy(structures)):

        loadbar(i, len(structures), f'{loadstring} {i+1}/{len(structures)} ')

        opt_coords, energy, success = optimize(
                                                conformer,
                                                atomnos,
                                                calculator,
                                                constrained_indices=constrained_indices,
                                                constrained_distances=constrained_distances,
                                                method=method,
                                                procs=procs,
                                                solvent=solvent,
                                                title=f'Structure_{i+1}',
                                                logfunction=logfunction,
                                                check=False, # a change in bonding topology is possible and should not be prevented
                                            )

        if success:
            structures[i] = opt_coords
            energies.append(energy)
        else:
            energies.append(1E10)

    loadbar(len(structures), len(structures), f'{loadstring} {len(structures)}/{len(structures)} ')
    energies = np.array(energies)

    # remove similar ones
    structures, mask = prune_by_rmsd(structures, atomnos)
    energies = energies[mask]

    # remove high energy ones
    mask = (energies - np.min(energies)) < 20
    structures, energies = structures[mask], energies[mask]

    return structures, energies