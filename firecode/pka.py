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

import numpy as np

from firecode.calculators._xtb import xtb_get_free_energy
from firecode.torsion_module import csearch
from firecode.optimization_methods import _refine_structures, optimize, write_xyz
from firecode.utils import loadbar, graphize
from firecode.graph_manipulations import neighbors
from firecode.algebra import norm


def _get_anions(
                embedder,
                structures, 
                atomnos, 
                index,
                logfunction=print,
            ):
    '''
    structures: array of 3D of coordinates
    atomnos: 1D array of atomic numbers
    index: position of hydrogen atom to be abstracted

    return: anion optimized geomertries, their energies and the new atomnos array
    '''
    assert embedder.options.calculator == 'XTB', 'Charge calculations not yet implemented for Gau, Orca, Mopac, OB'
    # assert atomnos[index] == 1

    atomnos = np.delete(atomnos, index)
    # removing proton from atoms

    solvent = embedder.options.solvent
    if solvent is None:
        logfunction('Solvent for pKa calculation not specified: defaulting to gas phase')

    anions, energies = [], []

    for s, structure in enumerate(structures):

        coords = np.delete(structure, index, axis=0)
        # new coordinates do not include the designated proton

        print(f'Optimizing anion conformer {s+1}/{len(structures)} ...', end='\r')

        opt_coords, energy, success = optimize(
                                                coords,
                                                atomnos,
                                                calculator=embedder.options.calculator,
                                                procs=embedder.procs,
                                                solvent=solvent,
                                                max_newbonds=embedder.options.max_newbonds,
                                                title=f'temp_anion{s}',
                                                check=True,
                                                charge=-1,
                                             )

        if success:
            anions.append(opt_coords)
            energies.append(energy)

    anions, energies = zip(*sorted(zip(anions, energies), key=lambda x: x[1]))

    return anions, energies, atomnos

def _get_cations(
                embedder,
                structures, 
                atomnos, 
                index,
                logfunction=print,
            ):
    '''
    structures: array of 3D of coordinates
    atomnos: 1D array of atomic numbers
    index: position where the new hydrogen atom has to be inserted

    return: cation optimized geomertries, their energies and the new atomnos array
    '''
    assert embedder.options.calculator == 'XTB', 'Charge calculations not yet implemented for Gau, Orca, Mopac, OB'

    cation_atomnos = np.append(atomnos, 1)
    # adding proton to atoms

    solvent = embedder.options.solvent
    if solvent is None:
        logfunction('Solvent for pKa calculation not specified: defaulting to gas phase')

    cations, energies = [], []

    for s, structure in enumerate(structures):

        coords = protonate(structure, atomnos, index)
        # new coordinates which include an additional proton

        print(f'Optimizing cation conformer {s+1}/{len(structures)} ...', end='\r')

        opt_coords, energy, success = optimize(
                                                coords,
                                                cation_atomnos,
                                                calculator=embedder.options.calculator,
                                                procs=embedder.procs,
                                                solvent=solvent,
                                                max_newbonds=embedder.options.max_newbonds,
                                                title=f'temp_cation{s}',
                                                check=True,
                                                charge=+1,
                                             )

        if success:
            cations.append(opt_coords)
            energies.append(energy)

    cations, energies = zip(*sorted(zip(cations, energies), key=lambda x: x[1]))

    return cations, energies, cation_atomnos

def protonate(coords, atomnos, index, length=1):
    '''
    Returns the input structure,
    protonated at the index provided,
    ready to be optimized
    '''

    graph = graphize(coords, atomnos)
    nbs = neighbors(graph, index)
    versor = -norm(np.mean(coords[nbs]-coords[index], axis=0))
    new_proton_coords = coords[index] + length * versor
    coords = np.append(coords, [new_proton_coords], axis=0)

    return coords

def pka_routine(filename, embedder, search=True):
    '''
    Calculates the energy difference between
    the most stable conformer of the provided
    structure and its conjugate base, obtained
    by removing one proton at the specified position.
    '''
    mol_index = [m.filename for m in embedder.objects].index(filename)
    mol = embedder.objects[mol_index]

    assert len(mol.reactive_indices) == 1, 'Please only specify one reactive atom for pKa calculations'

    embedder.log(f'--> pKa computation protocol for {mol.filename}, index {mol.reactive_indices}')

    if search:
        if len(mol.atomcoords) > 1:
            embedder.log(f'Using only the first molecule of {mol.filename} to generate conformers')

        conformers = csearch(
                                mol.atomcoords[0],
                                mol.atomnos,
                                n_out=100,
                                mode=1,
                                logfunction=print,
                                interactive_print=True,
                                write_torsions=False,
                                title=mol.filename
                            )
    else:
        conformers = mol.atomcoords

    conformers, _ =_refine_structures(
                                        conformers,
                                        mol.atomnos, 
                                        calculator=embedder.options.calculator,
                                        method=embedder.options.theory_level,
                                        procs=embedder.procs,
                                        loadstring='Optimizing conformer'
                                    )

    embedder.log()

    free_energies = get_free_energies(embedder, conformers, mol.atomnos, charge=0, title='Starting structure')
    conformers, free_energies = zip(*sorted(zip(conformers, free_energies), key=lambda x: x[1]))

    with open(f'{mol.rootname}_confs_opt.xyz', 'w') as f:

        solvent_string = f', {embedder.options.solvent}' if embedder.options.solvent is not None else ''

        for c, e in zip(conformers, free_energies):
            write_xyz(c, mol.atomnos, f, title=f'G({embedder.options.theory_level}{solvent_string}, charge=0) = {round(e, 3)} kcal/mol')

    if mol.atomnos[mol.reactive_indices[0]] == 1:
    # we have an acid, form and optimize the anions

        anions, _, anions_atomnos = _get_anions(
                                                embedder,
                                                conformers,
                                                mol.atomnos,
                                                mol.reactive_indices[0],
                                                logfunction=embedder.log
                                            )

        anions_free_energies = get_free_energies(embedder, anions, anions_atomnos, charge=-1, title='Anion')
        anions, anions_free_energies = zip(*sorted(zip(anions, anions_free_energies), key=lambda x: x[1]))

        with open(f'{mol.rootname}_anions_opt.xyz', 'w') as f:
            for c, e in zip(anions, anions_free_energies):
                write_xyz(c, anions_atomnos, f, title=f'G({embedder.options.theory_level}{solvent_string}, charge=-1) = {round(e, 3)} kcal/mol')

        e_HA = free_energies[0]
        e_A = anions_free_energies[0]
        embedder.objects[mol_index].pka_data = ('HA -> A-', e_A - e_HA)

        embedder.log()

    else:
    # we have a base, form and optimize the cations

        cations, _, cations_atomnos = _get_cations(
                                                    embedder,
                                                    conformers,
                                                    mol.atomnos,
                                                    mol.reactive_indices[0],
                                                    logfunction=embedder.log
                                                )

        cations_free_energies = get_free_energies(embedder, cations, cations_atomnos, charge=+1, title='Cation')
        cations, cations_free_energies = zip(*sorted(zip(cations, cations_free_energies), key=lambda x: x[1]))

        with open(f'{mol.rootname}_cations_opt.xyz', 'w') as f:
            for c, e in zip(cations, cations_free_energies):
                write_xyz(c, cations_atomnos, f, title=f'G({embedder.options.theory_level}{solvent_string}, charge=+1) = {round(e, 3)} kcal/mol')

        e_B = free_energies[0]
        e_BH = cations_free_energies[0]
        embedder.objects[mol_index].pka_data = ('B -> BH+', e_BH - e_B)

        embedder.log()

def get_free_energies(embedder, structures, atomnos, charge=0, title='Molecule'):
    '''
    '''
    assert embedder.options.calculator == 'XTB', 'Free energy calculations not yet implemented for Gau, Orca, Mopac, OB'

    free_energies = []

    for s, structure in enumerate(structures):

        loadbar(s, len(structures), f'{title} Hessian {s+1}/{len(structures)} ')
        
        free_energies.append(xtb_get_free_energy(
                                                    structure,
                                                    atomnos,
                                                    method=embedder.options.theory_level,
                                                    solvent=embedder.options.solvent,
                                                    charge=charge,
                                                ))

    loadbar(len(structures), len(structures), f'{title} Hessian {len(structures)}/{len(structures)} ')

    return free_energies