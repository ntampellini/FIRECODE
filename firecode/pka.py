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

import numpy as np

from firecode.algebra import normalize
from firecode.calculators._xtb import xtb_get_free_energy
from prism_pruner.graph_manipulations import graphize
from firecode.optimization_methods import (_refine_structures, optimize,
                                           write_xyz)
from firecode.torsion_module import csearch
from firecode.utils import loadbar


def _get_anions(
                embedder,
                atoms, 
                structures, 
                index,
                logfunction=print,
            ):
    '''
    atoms: 1D array of atomic numbers
    structures: array of 3D of coordinates
    index: position of hydrogen atom to be abstracted

    return: anion optimized geomertries, their energies and the new atoms array
    '''
    assert embedder.options.calculator == 'XTB', 'Charge calculations only implemented for XTB'

    # removing proton from atoms
    atoms = np.delete(atoms, index)

    solvent = embedder.options.solvent
    if solvent is None:
        logfunction('Solvent for pKa calculation not specified: defaulting to gas phase')

    anions, energies = [], []

    for s, structure in enumerate(structures):

        coords = np.delete(structure, index, axis=0)
        # new coordinates do not include the designated proton

        print(f'Optimizing anion conformer {s+1}/{len(structures)} ...', end='\r')

        opt_coords, energy, success = optimize(
                                                atoms,
                                                coords,
                                                calculator=embedder.options.calculator,
                                                procs=embedder.procs,
                                                solvent=solvent,
                                                max_newbonds=embedder.options.max_newbonds,
                                                title=f'temp_anion{s}',
                                                check=True,
                                                charge=-1,
                                                dispatcher=embedder.dispatcher,
                                                debug=embedder.options.debug,
                                             )

        if success:
            anions.append(opt_coords)
            energies.append(energy)

    anions, energies = zip(*sorted(zip(anions, energies), key=lambda x: x[1]))

    return anions, energies, atoms

def _get_cations(
                embedder,
                atoms, 
                structures, 
                index,
                logfunction=print,
            ):
    '''
    structures: array of 3D of coordinates
    atoms: 1D array of atomic numbers
    index: position where the new hydrogen atom has to be inserted

    return: cation optimized geomertries, their energies and the new atoms array
    '''
    assert embedder.options.calculator == 'XTB', 'Charge calculations not yet implemented for Gau, Orca, Mopac, OB'

    cation_atoms = np.append(atoms, 1)
    # adding proton to atoms

    solvent = embedder.options.solvent
    if solvent is None:
        logfunction('Solvent for pKa calculation not specified: defaulting to gas phase')

    cations, energies = [], []

    for s, structure in enumerate(structures):

        coords = protonate(structure, atoms, index)
        # new coordinates which include an additional proton

        print(f'Optimizing cation conformer {s+1}/{len(structures)} ...', end='\r')

        opt_coords, energy, success = optimize(
                                                cation_atoms,
                                                coords,
                                                calculator=embedder.options.calculator,
                                                procs=embedder.procs,
                                                solvent=solvent,
                                                max_newbonds=embedder.options.max_newbonds,
                                                title=f'temp_cation{s}',
                                                check=True,
                                                charge=+1,
                                                dispatcher=embedder.dispatcher,
                                                debug=embedder.options.debug,
                                             )

        if success:
            cations.append(opt_coords)
            energies.append(energy)

    cations, energies = zip(*sorted(zip(cations, energies), key=lambda x: x[1]))

    return cations, energies, cation_atoms

def protonate(atoms, coords, index, length=1):
    '''
    Returns the input structure,
    protonated at the index provided,
    ready to be optimized
    '''

    graph = graphize(atoms, coords)
    nbs = graph.neighbors(index)
    versor = -normalize(np.mean(coords[nbs]-coords[index], axis=0))
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
        if len(mol.coords) > 1:
            embedder.log(f'Using only the first molecule of {mol.filename} to generate conformers')

        conformers = csearch(
                                mol.atoms,
                                mol.coords[0],
                                charge=embedder.options.charge,
                                mult=embedder.options.mult,
                                n_out=100,
                                mode=1,
                                logfunction=print,
                                dispatcher=embedder.dispatcher,
                                interactive_print=True,
                                write_torsions=False,
                                title=mol.filename,
                                debug=embedder.options.debug,
                            )
    else:
        conformers = mol.coords

    conformers, _ =_refine_structures(
                                        mol.atoms,
                                        conformers,
                                        charge=embedder.options.charge,
                                        mult=embedder.options.mult,
                                        calculator=embedder.options.calculator,
                                        method=embedder.options.theory_level,
                                        procs=embedder.procs,
                                        loadstring='Optimizing conformer',
                                        dispatcher=embedder.dispatcher,
                                        debug=embedder.options.debug,
                                    )

    embedder.log()

    free_energies = get_free_energies(embedder, mol.atoms, conformers, charge=mol.charge, title='Starting structure')
    conformers, free_energies = zip(*sorted(zip(conformers, free_energies), key=lambda x: x[1]))

    with open(f'{mol.rootname}_confs_opt.xyz', 'w') as f:

        solvent_string = f', {embedder.options.solvent}' if embedder.options.solvent is not None else ''

        for c, e in zip(conformers, free_energies):
            write_xyz(mol.atoms, c, f, title=f'G({embedder.options.theory_level}{solvent_string}, charge={mol.charge}) = {round(e, 3)} kcal/mol')

    if mol.atoms[mol.reactive_indices[0]] == 'H':
    # we have an acid, form and optimize the anions

        anions, _, anions_atoms = _get_anions(
                                                embedder,
                                                mol.atoms,
                                                conformers,
                                                mol.reactive_indices[0],
                                                logfunction=embedder.log
                                            )

        anions_free_energies = get_free_energies(embedder, anions_atoms, anions, charge=-1, title='Anion')
        anions, anions_free_energies = zip(*sorted(zip(anions, anions_free_energies), key=lambda x: x[1]))

        with open(f'{mol.rootname}_anions_opt.xyz', 'w') as f:
            for c, e in zip(anions, anions_free_energies):
                write_xyz(anions_atoms, c, f, title=f'G({embedder.options.theory_level}{solvent_string}, charge=-1) = {round(e, 3)} kcal/mol')

        e_HA = free_energies[0]
        e_A = anions_free_energies[0]
        embedder.objects[mol_index].pka_data = ('HA -> A-', e_A - e_HA)

        embedder.log()

    else:
    # we have a base, form and optimize the cations

        cations, _, cations_atoms = _get_cations(
                                                    embedder,
                                                    mol.atoms,
                                                    conformers,
                                                    mol.reactive_indices[0],
                                                    logfunction=embedder.log
                                                )

        cations_free_energies = get_free_energies(embedder, cations, cations_atoms, charge=+1, title='Cation')
        cations, cations_free_energies = zip(*sorted(zip(cations, cations_free_energies), key=lambda x: x[1]))

        with open(f'{mol.rootname}_cations_opt.xyz', 'w') as f:
            for c, e in zip(cations, cations_free_energies):
                write_xyz(cations_atoms, c, f, title=f'G({embedder.options.theory_level}{solvent_string}, charge=+1) = {round(e, 3)} kcal/mol')

        e_B = free_energies[0]
        e_BH = cations_free_energies[0]
        embedder.objects[mol_index].pka_data = ('B -> BH+', e_BH - e_B)

        embedder.log()

def get_free_energies(embedder, atoms, structures, charge=0, title='Molecule'):
    '''
    '''
    assert embedder.options.calculator == 'XTB', 'Free energy calculations not yet implemented for Gau, Orca, Mopac, OB'

    free_energies = []

    for s, structure in enumerate(structures):

        loadbar(s, len(structures), f'{title} Hessian {s+1}/{len(structures)} ')
        
        free_energies.append(xtb_get_free_energy(
                                                    atoms,
                                                    structure,
                                                    method=embedder.options.theory_level,
                                                    solvent=embedder.options.solvent,
                                                    charge=charge,
                                                ))

    loadbar(len(structures), len(structures), f'{title} Hessian {len(structures)}/{len(structures)} ')

    return free_energies