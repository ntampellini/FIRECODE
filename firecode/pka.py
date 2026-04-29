# coding=utf-8
"""FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
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

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from prism_pruner.algebra import normalize
from prism_pruner.graph_manipulations import graphize

from firecode.optimization_methods import optimize, refine_structures
from firecode.thermochemistry import get_free_energies
from firecode.typing_ import Array1D_float, Array1D_str, Array2D_float, Array3D_float
from firecode.utils import charge_to_str, write_xyz

if TYPE_CHECKING:
    from firecode.embedder import Embedder
    from firecode.hypermolecule_class import Hypermolecule


def _get_anions(
    embedder: Embedder,
    mol: Hypermolecule,
    structures: Array2D_float,
    index: int,
    logfunction: Callable[[str], None] = print,
) -> tuple[Array3D_float, Array1D_float, Array1D_str]:
    """atoms: 1D array of atomic numbers
    structures: array of 3D of coordinates
    index: position of hydrogen atom to be abstracted

    return: anion optimized geomertries, their energies and the new atoms array
    """
    # removing proton from atoms
    atoms = np.delete(mol.atoms, index)

    solvent = embedder.options.solvent
    if solvent is None:
        logfunction("Solvent for pKa calculation not specified: defaulting to gas phase")

    anions, energies = [], []

    for s, structure in enumerate(structures):
        coords = np.delete(structure, index, axis=0)
        # new coordinates do not include the designated proton

        print(f"Optimizing anion conformer {s + 1}/{len(structures)} ...", end="\r")

        opt_coords, energy, success = optimize(
            atoms,
            coords,
            calculator=embedder.options.calculator,
            solvent=solvent,
            max_newbonds=embedder.options.max_newbonds,
            title=f"temp_deprotonated{s}",
            check=True,
            charge=mol.charge - 1,
            mult=mol.mult,
            dispatcher=embedder.dispatcher,
            debug=embedder.options.debug,
        )

        if success:
            anions.append(opt_coords)
            energies.append(energy)

    anions_sorted, energies_sorted = zip(*sorted(zip(anions, energies), key=lambda x: x[1]))

    return np.array(anions_sorted), np.array(energies_sorted), atoms


def _get_cations(
    embedder: Embedder,
    mol: Hypermolecule,
    structures: Array3D_float,
    index: int,
    logfunction: Callable[[str], None] | None = print,
) -> tuple[Array3D_float, Array1D_float, Array1D_str]:
    """structures: array of 3D of coordinates
    atoms: 1D array of atomic numbers
    index: position where the new hydrogen atom has to be inserted

    return: cation optimized geomertries, their energies and the new atoms array
    """
    cation_atoms = np.append(mol.atoms, "H")
    # adding proton to atoms

    solvent = embedder.options.solvent
    if solvent is None and logfunction is not None:
        logfunction("Solvent for pKa calculation not specified: defaulting to gas phase")

    cations, energies = [], []

    for s, structure in enumerate(structures):
        coords = protonate(mol.atoms, structure, index)
        # new coordinates which include an additional proton

        print(f"Optimizing cation conformer {s + 1}/{len(structures)} ...", end="\r")

        opt_coords, energy, success = optimize(
            cation_atoms,
            coords,
            calculator=embedder.options.calculator,
            solvent=solvent,
            max_newbonds=embedder.options.max_newbonds,
            title=f"temp_protonated{s}",
            check=True,
            charge=mol.charge + 1,
            mult=mol.mult,
            dispatcher=embedder.dispatcher,
            debug=embedder.options.debug,
        )

        if success:
            cations.append(opt_coords)
            energies.append(energy)

    cations_sorted, energies_sorted = zip(*sorted(zip(cations, energies), key=lambda x: x[1]))

    return np.array(cations_sorted), np.array(energies_sorted), cation_atoms


def protonate(
    atoms: Array1D_str, coords: Array2D_float, index: int, length: float = 1.0
) -> Array2D_float:
    """Returns the input structure,
    protonated at the index provided,
    ready to be optimized
    """
    graph = graphize(atoms, coords)
    nbs = list(graph.neighbors(index))
    versor = -normalize(np.mean(coords[nbs] - coords[index], axis=0))
    new_proton_coords = coords[index] + length * versor
    coords = np.append(coords, [new_proton_coords], axis=0)

    return coords


def pka_routine(filename: str, embedder: Embedder, search: bool = True) -> None:
    """Calculates the energy difference between
    the most stable conformer of the provided
    structure and its conjugate base, obtained
    by removing one proton at the specified position.
    """
    mol_index = [m.filename for m in embedder.objects].index(filename)
    mol = embedder.objects[mol_index]

    assert len(mol.reactive_indices) == 1, (
        "Please only specify one reactive atom for pKa calculations"
    )

    embedder.log(f"--> pKa computation protocol for {mol.filename}, index {mol.reactive_indices}")

    if search:
        if len(mol.coords) > 1:
            embedder.log(f"Using only the first molecule of {mol.filename} to generate conformers")

        from firecode.torsion_module import csearch

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

    conformers, _ = refine_structures(
        mol.atoms,
        conformers,
        charge=embedder.options.charge,
        mult=embedder.options.mult,
        calculator=embedder.options.calculator,
        method=embedder.options.theory_level,
        loadstring="Optimizing conformer",
        dispatcher=embedder.dispatcher,
        debug=embedder.options.debug,
    )

    embedder.log()

    free_energies = get_free_energies(
        embedder,
        mol.atoms,
        conformers,
        charge=mol.charge,
        mult=mol.mult,
        title=f"{mol.basename}_original",
    )
    conformers, free_energies = zip(*sorted(zip(conformers, free_energies), key=lambda x: x[1]))  # type: ignore[assignment]

    with open(f"{mol.basename}_confs_opt.xyz", "w") as f:
        solvent_string = (
            f", {embedder.options.solvent}" if embedder.options.solvent is not None else ""
        )

        for c, e in zip(conformers, free_energies):
            write_xyz(
                mol.atoms,
                c,
                f,
                title=f"G({embedder.options.theory_level}{solvent_string}, charge={mol.charge}) = {round(e, 3)} kcal/mol",
            )

    if mol.atoms[mol.reactive_indices[0]] == "H":
        # we have an acid, form and optimize the anions
        charge = mol.charge - 1

        anions, _, anions_atoms = _get_anions(
            embedder, mol, conformers, mol.reactive_indices[0], logfunction=embedder.log
        )

        anions_free_energies = get_free_energies(
            embedder,
            anions_atoms,
            anions,
            charge=charge,
            mult=mol.mult,
            title=f"{mol.basename}{charge_to_str(charge)}_deprotonated",
        )
        anions, anions_free_energies = zip(  # type: ignore[assignment]
            *sorted(zip(anions, anions_free_energies), key=lambda x: x[1])
        )

        with open(f"{mol.basename}_anions_opt.xyz", "w") as f:
            for c, e in zip(anions, anions_free_energies):
                write_xyz(
                    anions_atoms,
                    c,
                    f,
                    title=f"G({embedder.options.theory_level}{solvent_string}, charge={charge}) = {round(e, 3)} kcal/mol",
                )

        e_HA = free_energies[0]
        e_A = anions_free_energies[0]
        embedder.objects[mol_index].pka_data = ("HA -> A-", e_A - e_HA)

        embedder.log()

    else:
        # we have a base, form and optimize the cations
        charge = mol.charge + 1
        h = "H" if mol.basename[-1].upper() != "H" else "_H"

        cations, _, cations_atoms = _get_cations(
            embedder, mol, conformers, mol.reactive_indices[0], logfunction=embedder.log
        )

        cations_free_energies = get_free_energies(
            embedder,
            cations_atoms,
            cations,
            charge=charge,
            mult=mol.mult,
            title=f"{mol.basename}{h}{charge_to_str(charge)}_protonated",
        )
        cations, cations_free_energies = zip(  # type: ignore[assignment]
            *sorted(zip(cations, cations_free_energies), key=lambda x: x[1])
        )

        with open(f"{mol.basename}_cations_opt.xyz", "w") as f:
            for c, e in zip(cations, cations_free_energies):
                write_xyz(
                    cations_atoms,
                    c,
                    f,
                    title=f"G({embedder.options.theory_level}{solvent_string}, charge={charge}) = {round(e, 3)} kcal/mol",
                )

        e_B = free_energies[0]
        e_BH = cations_free_energies[0]
        embedder.objects[mol_index].pka_data = ("B -> BH+", e_BH - e_B)

        embedder.log()
