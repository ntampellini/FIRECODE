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

# import pickle
import time
from shutil import which
from subprocess import CalledProcessError

import numpy as np

from firecode.ase_manipulations import ase_saddle
from firecode.atropisomer_module import dihedral_scan
from firecode.automep import automep
from firecode.calculators._xtb import crest_mtd_search
from firecode.errors import FatalError, InputError
from firecode.graph_manipulations import graphize
from firecode.mep_relaxer import ase_mep_relax
from firecode.numba_functions import prune_conformers_tfd
from firecode.optimization_methods import _refine_structures, optimize
from firecode.pka import pka_routine
from firecode.pruning import prune_by_rmsd, prune_by_rmsd_rot_corr
from firecode.settings import (CALCULATOR, DEFAULT_FF_LEVELS, DEFAULT_LEVELS,
                               FF_CALC, FF_OPT_BOOL, PROCS)
from firecode.torsion_module import _get_quadruplets, csearch
from firecode.utils import (align_structures, get_scan_peak_index,
                            molecule_check, read_xyz, time_to_string,
                            write_xyz)


def operate(input_string, embedder):
    '''
    Perform the operations according to the chosen
    operator and return the outname of the (new) .xyz
    file to read instead of the input one.
    '''

    filename = embedder._extract_filename(input_string)
   
    if not hasattr(embedder, "t_start_run"):
        embedder.t_start_run = time.perf_counter()

    if embedder.options.dryrun:
        embedder.log(f'--> Dry run requested: skipping operator \"{input_string}\"')
        return filename

    elif 'csearch>' in input_string:
        outname = csearch_operator(filename, embedder)

    elif 'opt>' in input_string:
        outname = opt_operator(filename,
                                embedder,
                                logfunction=embedder.log)

    elif 'csearch_hb>' in input_string:
        outname = csearch_operator(filename, embedder, keep_hb=True)
        
    elif 'rsearch>' in input_string:
        outname = csearch_operator(filename, embedder, mode=2)

    elif any(string in input_string for string in ('mtd_search>', 'mtd>')):
        outname = mtd_search_operator(filename, embedder)

    elif 'saddle>' in input_string:
        saddle_operator(filename, embedder)
        embedder.normal_termination()
      
    elif 'scan>' in input_string:
        scan_operator(filename, embedder)
        outname = filename

    elif 'automep>' in input_string:
        automep(embedder, n_images=embedder.options.images if hasattr(embedder.options, 'images') else 9)
        # neb_operator(automep_filename, embedder)
        # embedder.normal_termination()

    elif 'neb>' in input_string:
        neb_operator(filename, embedder)
        embedder.normal_termination()

    elif 'refine>' in input_string:
        outname = filename
        # this operator is accounted for in the OptionSetter
        # class of Options, set when the Embedder calls _set_options

    elif 'pka>' in input_string:
        pka_routine(filename, embedder)
        outname = filename

    elif 'mep_relax>' in input_string:

        data = read_xyz(filename)

        # can implement a smart safety feature that is
        # disabled when some bonds have to be let free to break
        no_bonds_breaking = True

        if no_bonds_breaking:

            mep, _, exit_status = ase_mep_relax(
                                                embedder,
                                                data.atomcoords,
                                                data.atomnos,
                                                title=embedder.stamp+"_safe",
                                                n_images=embedder.options.images if hasattr(embedder.options, 'images') else None,
                                                logfunction=embedder.log,
                                                write_plot=True,
                                                verbose_print=True,
                                                safe=True
                                                )
            
        else:
            mep = data.atomcoords
            exit_status = True
        
        if exit_status:

            if no_bonds_breaking:
                print("--> Completed safe optimization, relaxing bond distance constraints.")

            ase_mep_relax(
                embedder,
                mep,
                data.atomnos,
                title=embedder.stamp,
                n_images=embedder.options.images if hasattr(embedder.options, 'images') else None,
                logfunction=embedder.log,
                write_plot=True,
                verbose_print=True,
                safe=True
                )
        
        embedder.normal_termination()

    else:
        op = input_string.split('>')[0]
        raise Exception(f'Operator {op} not recognized.')

    return outname

def csearch_operator(filename, embedder, keep_hb=False, mode=1):
    '''
    '''

    s = f'--> Performing conformational search on {filename}'
    if keep_hb:
        s += ' (preserving current hydrogen bonds)'
    embedder.log(s)

    # t_start = time.perf_counter()

    data = read_xyz(filename)

    if len(data.atomcoords) > 1:
        embedder.log('Requested conformational search on multimolecular file - will do\n' +
                      'an individual search from each conformer (might be time-consuming).')
                                
    # calc, method, procs = _get_lowest_calc(embedder)
    conformers = []

    for i, coords in enumerate(data.atomcoords):

        # opt_coords = optimize(coords, data.atomnos, calculator=calc, method=method, procs=procs)[0] if embedder.options.optimization else coords
        opt_coords = coords
        # optimize starting structure before running csearch

        conf_batch = csearch(
                                opt_coords,
                                data.atomnos,
                                constrained_indices=_get_internal_constraints(filename, embedder),
                                keep_hb=keep_hb,
                                mode=mode,
                                n_out=embedder.options.max_confs//len(data.atomcoords),
                                title=f'{filename}_conf{i}',
                                logfunction=embedder.log,
                                write_torsions=embedder.options.debug
                            )
        # generate the most diverse conformers starting from optimized geometry

        conformers.extend(conf_batch)

    conformers = np.concatenate(conformers)
    # batch_size = conformers.shape[1]

    conformers = conformers.reshape(-1, data.atomnos.shape[0], 3)
    # merging structures from each run in a single array

    # if embedder.embed is not None:
    #     embedder.log(f'\nSelected the most diverse {batch_size} out of {conformers.shape[0]} conformers for {filename} ({time_to_string(time.perf_counter()-t_start)})')
    #     conformers = most_diverse_conformers(batch_size, conformers)

    print(f'Writing conformers to file...{" "*10}', end='\r')

    confname = filename[:-4] + '_confs.xyz'
    with open(confname, 'w') as f:
        for i, conformer in enumerate(conformers):
            write_xyz(conformer, data.atomnos, f, title=f'Generated conformer {i}')

    print(f'{" "*30}', end='\r')

    embedder.log('\n')

    return confname

def opt_operator(filename, embedder, logfunction=None):
    '''
    '''

    mol = next((mol for mol in embedder.objects if mol.filename == filename))
    # load molecule to be optimized from embedder

    if logfunction is not None:
        logfunction(f'--> Performing {embedder.options.calculator} {embedder.options.theory_level}' + (
                    f'{f"/{embedder.options.solvent}" if embedder.options.solvent is not None else ""} optimization on {filename} ({len(mol.atomcoords)} conformers)'))

    constrained_indices = _get_internal_constraints(filename, embedder)
    constrained_distances = [embedder.get_pairing_dists_from_constrained_indices(cp) for cp in constrained_indices]

    energies = []
    lowest_calc = _get_lowest_calc(embedder)

    t_start = time.perf_counter()

    conformers, energies = _refine_structures(mol.atomcoords,
                                              mol.atomnos,
                                              constrained_indices=constrained_indices,
                                              constrained_distances=constrained_distances,
                                              *lowest_calc,
                                              loadstring='Optimizing conformer',
                                              logfunction=lambda s:embedder.log(s, p=False))

    energies, conformers = zip(*sorted(zip(energies, conformers), key=lambda x: x[0]))
    energies = np.array(energies) - np.min(energies)
    conformers = np.array(conformers)
    # sorting structures based on energy

    mask = energies < 20
    # getting the structures to reject (Rel Energy > 20 kcal/mol)

    if logfunction is not None:
        s = 's' if len(conformers) > 1 else ''
        s = f'Completed optimization on {len(conformers)} conformer{s}. ({time_to_string(time.perf_counter()-t_start)}, ~{time_to_string((time.perf_counter()-t_start)/len(conformers))} per structure).\n'

        if max(energies) > 20:
            s += f'Discarded {len(conformers)-np.count_nonzero(mask)}/{len(conformers)} unstable conformers (Rel. E. > 20 kcal/mol)\n'

    conformers, energies = conformers[mask], energies[mask]
    # applying the mask that rejects high energy confs

    optname = filename[:-4] + '_opt.xyz'
    with open(optname, 'w') as f:
        for i, conformer in enumerate(align_structures(conformers)):
            write_xyz(conformer, mol.atomnos, f, title=f'Optimized conformer {i} - Rel. E. = {round(energies[i], 3)} kcal/mol')

    logfunction(s+'\n')
    logfunction(f'Wrote {len(conformers)} optimized structures to {optname}\n')

    return optname

def neb_operator(filename, embedder, attempts=5):
    '''
    '''
    embedder.t_start_run = time.perf_counter()
    data = read_xyz(filename)
    n_str = len(data.atomcoords)
    assert (n_str in (2, 3) or n_str % 2 == 1), 'NEB calculations need a .xyz input file with two, three or an odd number of geometries.'

    if n_str == 2:
        reagents, products = data.atomcoords
        ts_guess = None
        mep_override = None
        embedder.log('--> Two structures as input: using them as start and end points.')

    elif n_str == 3:
        reagents, ts_guess, products = data.atomcoords
        mep_override = None
        embedder.log('--> Three structures as input: using them as start, TS guess and end points.')

    else:
        reagents, *_, products = data.atomcoords
        ts_guess = data.atomcoords[n_str//2]
        mep_override = data.atomcoords
        embedder.log(f'--> {n_str} structures as input: using these as the NEB MEP guess.')

    from firecode.ase_manipulations import ase_neb 

    title = filename[:-4] + '_NEB'

    # if embedder.options.neb.preopt:
    if True:

        embedder.log(f'--> Performing NEB TS optimization. Preoptimizing structures from {filename}\n'
                     f'Theory level is {embedder.options.theory_level}/{embedder.options.solvent or "vacuum"} via {embedder.options.calculator}')

        reagents, reag_energy, _ = optimize(
                                            reagents,
                                            data.atomnos,
                                            embedder.options.calculator,
                                            method=embedder.options.theory_level,
                                            procs=embedder.procs,
                                            solvent=embedder.options.solvent,
                                            title='reagents',
                                            logfunction=embedder.log,
                                            )

        products, prod_energy, _ = optimize(
                                            products,
                                            data.atomnos,
                                            embedder.options.calculator,
                                            method=embedder.options.theory_level,
                                            procs=embedder.procs,
                                            solvent=embedder.options.solvent,
                                            title='products',
                                            logfunction=embedder.log,
                                            )
        
        if mep_override is not None:
            mep_override[0] = reagents
            mep_override[-1] = products

    # else:
    #     embedder.log(f'--> Performing NEB TS optimization. Structures from {filename}\n'
    #                  f'Theory level is {embedder.options.theory_level} via {embedder.options.calculator}')

    #     print('Getting start point energy...', end='\r')
    #     _, reag_energy, _ = ase_popt(embedder, reagents, data.atomnos, steps=0)

    #     print('Getting end point energy...', end='\r')
    #     _, prod_energy, _ = ase_popt(embedder, products, data.atomnos, steps=0)

    for attempt in range(attempts):

        ts_coords, ts_energy, energies, exit_status = ase_neb(
                                                                embedder,
                                                                reagents,
                                                                products,
                                                                data.atomnos,
                                                                # n_images=embedder.options.neb.images,
                                                                n_images=7,
                                                                ts_guess= ts_guess,
                                                                mep_override=mep_override,
                                                                title=title,
                                                                logfunction=embedder.log,
                                                                write_plot=True,
                                                                verbose_print=True
                                                            )

        if exit_status == "CONVERGED":
            break

        elif exit_status == "MAX ITER" and attempt+2 < attempts:
            mep_override = read_xyz(f'{title}_MEP_start_of_CI.xyz').atomcoords
            reagents, *_, products = mep_override
            embedder.log(f'--> Restarting NEB from checkpoint. Attempt {attempt+2}/3.\n')


    e1 = ts_energy - reag_energy
    e2 = ts_energy - prod_energy
    dg1 = ts_energy - min(energies[:3])
    dg2 = ts_energy - min(energies[4:])

    embedder.log(f'NEB completed, relative energy from start/end points (not barrier heights):\n'
               f'  > E(TS)-E(start): {"+" if e1>=0 else "-"}{round(e1, 3)} kcal/mol\n'
               f'  > E(TS)-E(end)  : {"+" if e2>=0 else "-"}{round(e2, 3)} kcal/mol\n')
    
    embedder.log(f'Barrier heights (based on lowest energy point on each side):\n'
               f'  > E(TS)-E(left) : {"+" if dg1>=0 else "-"}{round(dg1, 3)} kcal/mol\n'
               f'  > E(TS)-E(right): {"+" if dg2>=0 else "-"}{round(dg2, 3)} kcal/mol')

    if not (e1 > 0 and e2 > 0):
        embedder.log('\nNEB failed, TS energy is lower than both the start and end points.\n')

    with open(f'{title}_TS.xyz', 'w') as f:
        write_xyz(ts_coords, data.atomnos, f, title='NEB TS - see log for relative energies')

def saddle_operator(filename, embedder):
    '''
    Perform a saddle optimization on the specified structure
    '''

    mol = next((mol for mol in embedder.objects if mol.filename == filename))
    # load molecule to be optimized from embedder

    assert len(mol.atomcoords) == 1, 'saddle> operator works with a single structure as input.'

    logfunction = embedder.log
    
    logfunction(f'--> Performing {embedder.options.calculator} {embedder.options.theory_level}' + (
                    f'{f"/{embedder.options.solvent}" if embedder.options.solvent is not None else ""} saddle optimization on {filename}'))

    new_structure, energy, success = ase_saddle(
                                                embedder,
                                                mol.atomcoords[0],
                                                mol.atomnos,
                                                constrained_indices=None,
                                                mols_graphs=None,
                                                title=mol.rootname,
                                                logfile=mol.rootname+"_saddle_opt_log.txt",
                                                traj=None,
                                                freq=False,
                                                maxiterations=200
                                            )

    with open(mol.rootname+"_saddle.xyz", 'w') as f:
        write_xyz(new_structure, mol.atomnos, f, f"ASE Saddle optimization {'succeded' if success else 'failed'} ({embedder.options.calculator}" +
                f'{embedder.options.theory_level}/{embedder.options.solvent})')
    if success:
        embedder.log(
            f'Saddle optimization completed, relative energy from start/end points (not barrier heights):\n'
            f'  > E(Saddle_point) : {round(energy, 3)} kcal/mol\n')

def mtd_search_operator(filename, embedder):
    '''
    Run a CREST metadynamic conformational search and return the output filename.
    '''

    assert crest_is_installed(), 'CREST 2 does not seem to be installed. Install it with: conda install -c conda-forge crest==2.*'

    mol = next((mol for mol in embedder.objects if mol.filename == filename))
    # load molecule to be optimized from embedder

    if not hasattr(mol, 'charge'):
        mol.charge = 0

    if not embedder.options.let:
        if len(mol.atomcoords) >= 20:
            raise InputError('The mtd_search> operator was given more than 20 input structures. ' +
                             'This would run >20 metadynamic conformational searches. If this was not a mistake, ' +
                             'add the LET keyword an re-run the job.')

    logfunction = embedder.log
    constrained_indices = _get_internal_constraints(filename, embedder)
    constrained_distances = [embedder.get_pairing_dists_from_constrained_indices(cp) for cp in constrained_indices]

    (  
        constrained_angles_indices,
        constrained_angles_values,
        constrained_dihedrals_indices,
        constrained_dihedrals_values,
    ) = embedder._get_angle_dih_constraints()

    logfunction(f'--> {filename}: Geometry optimization pre-mtd_search ({embedder.options.theory_level} via {embedder.options.calculator})')
    return_char = "\n"
    logfunction(f'    {len(constrained_indices)} constraints applied{": "+str(constrained_indices).replace(return_char, " ") if len(constrained_indices) > 0 else ""}')
    
    for c, coords in enumerate(mol.atomcoords.copy()):
        logfunction(f"    Optimizing conformer {c+1}/{len(mol.atomcoords)}")

        opt_coords, _, success = optimize(
                                    coords,
                                    mol.atomnos,
                                    calculator=embedder.options.calculator,
                                    method=embedder.options.theory_level,
                                    solvent=embedder.options.solvent,
                                    charge=embedder.options.charge,
                                    procs=embedder.procs,

                                    constrained_indices=constrained_indices,
                                    constrained_distances=constrained_distances,

                                    constrained_angles_indices=constrained_angles_indices,
                                    constrained_angles_values=constrained_angles_values,

                                    constrained_dihedrals_indices=constrained_dihedrals_indices,
                                    constrained_dihedrals_values=constrained_dihedrals_values,

                                    title=f'{filename.split(".")[0]}_conf{c+1}',
                                ) if embedder.options.optimization else (coords, None, True)
        
        exit_status = "" if success else "CRASHED"
        
        if success:
            success = molecule_check(coords, opt_coords, mol.atomnos)
            exit_status = "" if success else "SCRAMBLED"

        if not success:
            dumpname = filename.split(".")[0] + f"_conf{c+1}_{exit_status}.xyz"
            with open(dumpname, "w") as f:
                write_xyz(opt_coords, mol.atomnos, f, title=f"{filename}, conformer {c+1}/{len(mol.atomcoords)}, {exit_status}")

            logfunction(f"{filename}, conformer {c+1}/{len(mol.atomcoords)} optimization {exit_status}. Inspect geometry at {dumpname}. Aborting run.")

            raise FatalError(filename)
        
        # update embedder structures after optimization
        mol.atomcoords[c] = opt_coords

    logfunction()

    # update mol and embedder graph after optimization 
    mol.graph = graphize(mol.atomcoords[0], mol.atomnos)
    embedder.graphs = [m.graph for m in embedder.objects]

    max_workers = embedder.avail_cpus//2 or 1
    logfunction(f'--> Performing {embedder.options.calculator} GFN2//GFN-FF' + (
                f'{f"/{embedder.options.solvent.upper()}" if embedder.options.solvent is not None else ""} ' +
                f'metadynamic conformational search on {filename} via CREST.\n' +
                f'    (2 cores/thread, {max_workers} threads, {embedder.options.kcal_thresh} kcal/mol thr.)'))

    if embedder.options.crestnci:
        logfunction('--> CRESTNCI: Running crest in NCI mode (wall potential applied)')
    
    if len(mol.atomcoords) > 1:
        embedder.log('--> Requested conformational search on multimolecular file - will do\n' +
                      'an individual search from each conformer (might be time-consuming).')

    t_start = time.perf_counter()
    conformers = []
    for i, coords in enumerate(mol.atomcoords):

        t_start_conf = time.perf_counter()
        try:
            conf_batch = crest_mtd_search(
                                            coords,
                                            mol.atomnos,

                                            constrained_indices=constrained_indices,
                                            constrained_distances=constrained_distances,

                                            constrained_angles_indices=constrained_angles_indices,
                                            constrained_angles_values=constrained_angles_values,

                                            constrained_dihedrals_indices=constrained_dihedrals_indices,
                                            constrained_dihedrals_values=constrained_dihedrals_values,

                                            solvent=embedder.options.solvent,
                                            charge=mol.charge,
                                            method=getattr(embedder.options, "crestlevel", 'GFN2-XTB//GFN-FF'),
                                            kcal=embedder.options.kcal_thresh,
                                            ncimode=embedder.options.crestnci,
                                            title=mol.rootname+"_mtd_csearch",
                                            procs=2,
                                            threads=max_workers,
                                        )
            
        # if the run errors out, we retry with XTB2
        except CalledProcessError:
            logfunction('--> Metadynamics run failed with GFN2-XTB//GFN-FF, retrying with just GFN2-XTB (slower but more stable)')
            conf_batch = crest_mtd_search(
                                            coords,
                                            mol.atomnos,

                                            constrained_indices=constrained_indices,
                                            constrained_distances=constrained_distances,

                                            constrained_angles_indices=constrained_angles_indices,
                                            constrained_angles_values=constrained_angles_values,

                                            constrained_dihedrals_indices=constrained_dihedrals_indices,
                                            constrained_dihedrals_values=constrained_dihedrals_values,
                                            
                                            solvent=embedder.options.solvent,
                                            charge=mol.charge,
                                            method='GFN2-XTB', # try with XTB2
                                            kcal=embedder.options.kcal_thresh,
                                            ncimode=embedder.options.crestnci,
                                            title=mol.rootname+"_mtd_csearch",
                                            procs=2,
                                            threads=max_workers,
                                        )

        conformers.extend(conf_batch)

        elapsed = time.perf_counter() - t_start_conf
        embedder.log(f'  Conformer {i+1:2}/{len(mol.atomcoords):2} - generated {len(conf_batch)} structures in {time_to_string(elapsed)}')

    conformers = np.concatenate(conformers)
    conformers = conformers.reshape(-1, mol.atomnos.shape[0], 3)
    # merging structures from each run in a single array

    embedder.log(f'  MTD conformational search: Generated {len(conformers)} conformers in {time_to_string(time.perf_counter()-t_start)}')
    before = len(conformers)

    ### SIMILARITY PRUNING: TFD
    quadruplets = _get_quadruplets(mol.graph)
    conformers, _ = prune_conformers_tfd(conformers, quadruplets)

    # ### MOI - turned off, as it would get rid of enantiomeric conformations
    # conformers, _ = prune_by_moment_of_inertia(conformers, mol.atomnos)

    ### RMSD
    if len(conformers) < 5E4:
        conformers, _ = prune_by_rmsd(conformers, mol.atomnos, max_rmsd=embedder.options.rmsd, debugfunction=embedder.debuglog)
    if len(conformers) < 1E3:
        conformers, _ = prune_by_rmsd_rot_corr(conformers, mol.atomnos, mol.graph, max_rmsd=embedder.options.rmsd, debugfunction=embedder.debuglog)

    embedder.log(f'  Discarded {before-len(conformers)} RMSD-similar structures ({len(conformers)} left)\n')

    ### PRINTOUT
    with open(f'{mol.rootname}_mtd_confs.xyz', 'w') as f:
        for i, new_s in enumerate(conformers):
            write_xyz(new_s, mol.atomnos, f, title=f'Conformer {i}/{len(conformers)} from CREST MTD')

    
    # check the structures again and warn if some look compenetrated
    embedder.check_objects_compenetration()

    return f'{mol.rootname}_mtd_confs.xyz'

def scan_operator(filename, embedder):
    '''
    Scan operator dispatcher:
    2 indices: distance_scan
    4 indices: dihedral_scan

    '''
    mol = next((mol for mol in embedder.objects if mol.filename == filename))

    assert len(mol.atomcoords) == 1, 'The scan> operator works on a single .xyz geometry.'
    assert len(mol.reactive_indices) in (2,4), 'The scan> operator needs two or four indices' + (
                                              f'({len(mol.reactive_indices)} were provided)')

    if len(mol.reactive_indices) == 2:
        return distance_scan(embedder)
    
    elif len(mol.reactive_indices) == 4:
        return dihedral_scan(embedder)

def distance_scan(embedder):
    '''
    Thought to approach or separate two reactive atoms, looking for the energy maximum.
    Scan direction is inferred by the reactive index distance.
    '''

    import matplotlib.pyplot as plt

    from firecode.algebra import norm_of
    from firecode.pt import pt

    embedder.t_start_run = time.perf_counter()
    mol = embedder.objects[0]
    t_start = time.perf_counter()

    # shorthands for clearer code
    i1, i2 = mol.reactive_indices
    coords = mol.atomcoords[0]

    # getting the start distance between scan indices and start energy
    d = norm_of(coords[i1]-coords[i2])

    # deciding if moving atoms closer or further apart based on distance
    bonds = list(mol.graph.edges)
    step = 0.05 if (i1, i2) in bonds else -0.05

    # logging to file and terminal
    embedder.log(f'--> {mol.rootname} - Performing a distance scan {"approaching" if step < 0 else "separating"} indices {i1} ' +
                 f'and {i2} - step size {round(step, 2)} A\n    Theory level is {embedder.options.theory_level}/{embedder.options.solvent or "vacuum"} ' +
                 f'via {embedder.options.calculator}')

    # creating a dictionary that will hold results
    # and the structure output list
    dists, energies, structures = [], [], []

    # getting atomic symbols
    s1, s2 = mol.atomnos[[i1, i2]]

    # defining the maximum number of iterations
    if step < 0:
        smallest_d = 0.9*(pt[s1].covalent_radius+
                        pt[s2].covalent_radius)
        max_iterations = round((d-smallest_d) / abs(step))
        # so that atoms are never forced closer than
        # a proportionally small distance between those two atoms.

    else:
        max_d = 1.8*(pt[s1].covalent_radius+
                   pt[s2].covalent_radius)
        max_iterations = round((max_d-d) / abs(step))
        # so that atoms are never spaced too far apart

    from firecode.calculators._xtb import xtb_opt
    if embedder.options.calculator == 'AIMNET2':
        from aimnet2_firecode.interface import aimnet2_opt

    for i in range(max_iterations):

        t_start = time.perf_counter()

        if embedder.options.calculator == 'XTB':
            coords, energy, _ = xtb_opt(
                                        coords,
                                        mol.atomnos,
                                        constrained_indices=np.array([mol.reactive_indices]),
                                        constrained_distances=(d,),
                                        method=embedder.options.theory_level,
                                        solvent=embedder.options.solvent,
                                        charge=embedder.options.charge,
                                        title='temp',
                                        procs=embedder.procs,
                                        )
            
        elif embedder.options.calculator == 'AIMNET2':
            coords, energy, _ = aimnet2_opt(
                                        coords,
                                        mol.atomnos,
                                        ase_calc=embedder.dispatcher.aimnet2_calc,
                                        constrained_indices=np.array([mol.reactive_indices]),
                                        constrained_distances=(d,),
                                        solvent=embedder.options.solvent,
                                        charge=embedder.options.charge,
                                        title='temp',
                                        )
            
        else:
            raise NotImplementedError()

        if i == 0:
            e_0 = energy

        energies.append(energy - e_0)
        dists.append(d)
        structures.append(coords)
        # print(f"------> target was {round(d, 3)} A, reached {round(norm_of(coords[mol.reactive_indices[0]]-coords[mol.reactive_indices[1]]), 3)} A")
        # saving the structure, distance and relative energy

        embedder.log(f'Step {i+1:3}/{max_iterations:3} - d={round(d, 2)} A - {round(energy-e_0, 2):4} kcal/mol - {time_to_string(time.perf_counter()-t_start)}')

        with open("temp_scan.xyz", "w") as f:
            for i, (s, d, e) in enumerate(zip(structures, dists, energies)):
                write_xyz(s, mol.atomnos, f, title=f'Scan point {i+1}/{len(structures)} ' +
                        f'- d({i1}-{i2}) = {round(d, 3)} A - Rel. E = {round(e-min(energies), 2)} kcal/mol')

        d += step
        # modify the target distance and reiterate

    ### Start the plotting sequence

    plt.figure()
    plt.plot(
        dists,
        energies,
        color='tab:red',
        label='Scan energy',
        linewidth=3,
    )

    # e_max = max(energies)
    id_max = get_scan_peak_index(energies)
    e_max = energies[id_max]

    # id_max = energies.index(e_max)
    d_opt = dists[id_max]

    plt.plot(
        d_opt,
        e_max,
        color='gold',
        label='Energy maximum (TS guess)',
        marker='o',
        markersize=3,
    )

    title = mol.rootname + ' distance scan'
    plt.legend()
    plt.title(title)
    plt.xlabel(f'indices s{i1}-{i2} distance (A)')

    if step > 0:
        plt.gca().invert_xaxis()
        
    plt.ylabel(f'Rel. E. (kcal/mol) - {embedder.options.theory_level}/{embedder.options.calculator}/{embedder.options.solvent}')
    plt.savefig(f'{title.replace(" ", "_")}_plt.svg')
    # with open(f'{title.replace(" ", "_")}_plt.pickle', 'wb') as _f:
    #     pickle.dump(fig, _f)

    ### Start structure writing 

    # print all scan structures
    with open(f'{mol.filename[:-4]}_scan.xyz', 'w') as f:
        for i, (s, d, e) in enumerate(zip(structures, dists, energies)):
            write_xyz(s, mol.atomnos, f, title=f'Scan point {i+1}/{len(structures)} ' +
                      f'- d({i1}-{i2}) = {round(d, 2)} A - Rel. E = {round(e, 2)} kcal/mol')

    # print the maximum on another file for convienience
    with open(f'{mol.filename[:-4]}_scan_max.xyz', 'w') as f:
        s = structures[id_max]
        d = dists[id_max]
        write_xyz(s, mol.atomnos, f, title=f'Scan point {id_max+1}/{len(structures)} ' +
                    f'- d({i1}-{i2}) = {round(d, 3)} A - Rel. E = {round(e_max, 3)} kcal/mol')

    embedder.log(f'\n--> Written {len(structures)} structures to {mol.filename[:-4]}_scan.xyz ({time_to_string(time.perf_counter() - t_start)})')
    embedder.log(f'\n--> Written energy maximum to {mol.filename[:-4]}_scan_max.xyz\n')

    # Log data to the embedder class
    mol.scan_data = (dists, energies)

def crest_is_installed() -> bool:
    '''
    Returns True if a CREST installation is found.
    For now, only CREST 2 is supported.

    '''
    return (which('crest') is not None)

def _get_lowest_calc(embedder=None):
    '''
    Returns the values for calculator,
    method and processors for the lowest
    theory level available from embedder or settings.
    '''
    if embedder is None:
        if FF_OPT_BOOL:
            return (FF_CALC, DEFAULT_FF_LEVELS[FF_CALC], PROCS)
        return (CALCULATOR, DEFAULT_LEVELS[CALCULATOR], PROCS)

    if embedder.options.ff_opt:
        return (embedder.options.ff_calc, embedder.options.ff_level, embedder.procs)
    return (embedder.options.calculator, embedder.options.theory_level, embedder.procs)

def _get_internal_constraints(filename, embedder):
    '''
    '''
    mol_id = next((i for i, mol in enumerate(embedder.objects) if mol.filename == filename))
    # get embedder,objects index of molecule to get internal constraints of

    out = []
    for _, tgt in embedder.pairings_dict[mol_id].items():
        if isinstance(tgt, tuple):
            out.append(tgt)

    return np.array(out)