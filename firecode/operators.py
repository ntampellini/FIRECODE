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

# import pickle
import time
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from prism_pruner.graph_manipulations import graphize
from prism_pruner.pruner import prune_by_moment_of_inertia, prune_by_rmsd, prune_by_rmsd_rot_corr
from prism_pruner.utils import align_structures, time_to_string

from firecode.ase_manipulations import fsm_operator
from firecode.atropisomer_module import dihedral_scan
from firecode.errors import FatalError, InputError
from firecode.interfaces.crest import crest_mtd_search, get_crest_version
from firecode.interfaces.goat import goat_operator
from firecode.md.equilibration import equilibrate_operator
from firecode.md.packmol import solvate_molecule
from firecode.optimization_methods import optimize, refine_structures
from firecode.pka import pka_routine
from firecode.pt import pt
from firecode.rdkit_tools import rdkit_search_operator
from firecode.solvents import solvent_data
from firecode.typing_ import Array2D_float, Array3D_float, MaybeNone
from firecode.units import EH_TO_KCAL
from firecode.utils import (
    get_scan_peak_index,
    molecule_check,
    read_xyz,
    read_xyz_energies,
    write_xyz,
)

if TYPE_CHECKING:
    from firecode.embedder import Embedder


def operate(filename: str, operator: str, embedder: Embedder) -> str:
    """Apply the specified operator.

    Perform the operations according to the chosen
    operator and return the outname of the processed
    .xyz file to be read.
    """
    if not hasattr(embedder, "t_start_run"):
        embedder.t_start_run = time.perf_counter()

    if embedder.options.dryrun:
        embedder.log(f'--> Dry run requested: skipping operator "{operator}"')
        return filename

    match operator:
        case "firecode_search":
            outname = csearch_operator(filename, embedder)

        case "opt":
            outname = opt_operator(filename, embedder, logfunction=embedder.log)

        case "firecode_search_hb":
            outname = csearch_operator(filename, embedder, keep_hb=True)

        case "firecode_rsearch":
            outname = csearch_operator(filename, embedder, mode=2)

        case "mtd_search" | "mtd" | "crest" | "crest_search":
            outname = crest_search_operator(filename, embedder)

        case "goat":
            outname = goat_operator(filename, embedder)

        case "rdkit_search" | "rdkit" | "racerts" | "racerts_search":
            outname = rdkit_search_operator(filename, embedder)

        case "scan":
            outname = scan_operator(filename, embedder)

        case "neb":
            outname = neb_operator(filename, embedder)

        case "fsm" | "mlfsm":
            outname = fsm_operator(embedder)

        case "refine":
            outname = filename
            # this operator is accounted for in the Option_setter
            # class of Options, set when the Embedder calls _set_options

        case "pka":
            pka_routine(filename, embedder)
            outname = filename

        case "saddle" | "ts":
            outname = saddle_operator(filename, embedder)

        case "freq" | "thermo":
            outname = freq_operator(filename, embedder)

        case "packmol":
            outname = packmol_operator(filename, embedder)

        case "equilibrate":
            outname = equilibrate_operator(filename, embedder)

        case _:
            raise Exception(f"Operator {operator} not recognized.")

    return outname


def csearch_operator(
    filename: str, embedder: Embedder, keep_hb: bool = False, mode: int = 1
) -> str:
    """ """

    s = f"--> Performing conformational search on {filename}"
    if keep_hb:
        s += " (preserving current hydrogen bonds)"
    embedder.log(s)

    data = embedder.mols[filename]

    if len(data.coords) > 1:
        embedder.log(
            "Requested conformational search on multimolecular file - will do\n"
            + "an individual search from each conformer (might be time-consuming)."
        )

    conformers: list[Array2D_float] = []

    for i, coords in enumerate(data.coords):
        opt_coords = coords

        from firecode.torsion_module import csearch

        conf_batch = csearch(
            data.atoms,
            opt_coords,
            charge=embedder.options.charge,
            mult=embedder.options.mult,
            constrained_indices=embedder._get_internal_constraints(filename),
            keep_hb=keep_hb,
            mode=mode,
            n_out=embedder.options.max_confs // len(data.coords),
            title=f"{filename}_conf{i}",
            logfunction=embedder.log,
            dispatcher=embedder.dispatcher,
            write_torsions=embedder.options.debug,
            debug=embedder.options.debug,
        )
        # generate the most diverse conformers starting from optimized geometry

        conformers.extend(conf_batch)

    conformers_array = np.concatenate(conformers).reshape(-1, data.atomnos.shape[0], 3)
    # merging structures from each run in a single array

    print(f"Writing conformers to file...{' ' * 10}", end="\r")

    confname = data.basename + "_confs.xyz"
    with open(confname, "w") as f:
        for i, conformer in enumerate(conformers_array):
            write_xyz(data.atoms, conformer, f, title=f"Generated conformer {i}")

    print(f"{' ' * 30}", end="\r")

    embedder.log("\n")

    return confname


def opt_operator(
    filename: str, embedder: Embedder, logfunction: Callable[[str], None] | None = None
) -> str:
    """ """

    mol = embedder.mols[filename]

    if logfunction is not None:
        logfunction(
            f"--> Performing {embedder.options.calculator} {embedder.options.theory_level}"
            + (
                f"{f'/{embedder.options.solvent}' if embedder.options.solvent is not None else ''} optimization on {filename} ({len(mol.coords)} conformers)"
            )
        )

    constrained_indices = embedder._get_internal_constraints(filename)
    constrained_distances = [
        embedder.get_pairing_dists_from_constrained_indices(cp) for cp in constrained_indices
    ]

    (
        constrained_angles_indices,
        constrained_angles_values,
        constrained_dihedrals_indices,
        constrained_dihedrals_values,
    ) = embedder._get_angle_dih_constraints(filename)

    constr_str = embedder.get_str_all_constraints(filename)
    n_constraints = (
        len(constrained_indices)
        + len(constrained_angles_indices)
        + len(constrained_dihedrals_indices)
    )
    if logfunction is not None:
        logfunction(
            f"    [{n_constraints} constraints applied{': ' + str(constr_str).replace('\n', ' ') if constr_str != '' else ''}]"
        )

    conformers, energies = refine_structures(
        mol.atoms,
        mol.coords,
        calculator=embedder.options.calculator,
        method=embedder.options.theory_level,
        charge=embedder.options.charge,
        mult=embedder.options.mult,
        constrained_indices=constrained_indices,
        constrained_distances=constrained_distances,
        constrained_angles_indices=list(constrained_angles_indices),
        constrained_angles_values=constrained_angles_values,
        constrained_dihedrals_indices=list(constrained_dihedrals_indices),
        constrained_dihedrals_values=constrained_dihedrals_values,
        loadstring="Optimizing conformer",
        logfunction=embedder.log,
        debug=embedder.options.debug,
        dispatcher=embedder.dispatcher,
    )

    rel_energies = energies - np.min(energies)

    optname = filename[:-4] + "_opt.xyz"
    with open(optname, "w") as f:
        for i, conformer in enumerate(align_structures(conformers)):
            write_xyz(
                mol.atoms,
                conformer,
                f,
                title=f"Optimized conformer {i} - E(kcal/mol) = {energies[i]:.3f} - Rel. E. = {rel_energies[i]:.3f} kcal/mol",
            )

    if logfunction is not None:
        logfunction(f"Wrote {len(conformers)} optimized structures to {optname}\n")

    return optname


def neb_operator(filename: str, embedder: Embedder, attempts: int = 3) -> str:
    """Run a NEB calculation with the ASE module."""
    embedder.t_start_run = time.perf_counter()
    data = embedder.mols[filename]
    n_str = len(data.coords)

    if not hasattr(embedder.options, "neb"):
        embedder.options._init_neb_options()

    if n_str == 2:
        reagents, products = data.coords
        ts_guess = None
        mep_override = None
        embedder.log("--> Two structures as input: using them as start and end points.")

    elif n_str == 3:
        reagents, ts_guess, products = data.coords
        mep_override = None
        embedder.log("--> Three structures as input: using them as start, TS guess and end points.")

    else:
        reagents, *_, products = data.coords
        ts_guess = data.coords[n_str // 2]
        mep_override = data.coords
        embedder.log(
            f"--> {n_str} structures as input: casting {embedder.options.neb.n_images} images from these as the NEB MEP guess."
        )

    from firecode.ase_manipulations import ase_neb

    title = data.basename + "_NEB"
    ci_str = "-CI" if embedder.options.neb.climbing_image else ""

    # preopt unless user specifies not to
    embedder.log(
        f"--> Performing NEB{ci_str} optimization.\n"
        f"Theory level is {embedder.options.theory_level}/{embedder.options.solvent or 'vacuum'} via {embedder.options.calculator}"
    )

    if embedder.options.neb.preopt:
        embedder.log(f"Preoptimizing start/end structures from {filename}")

    else:
        embedder.log(f"Getting energy of start/end structures from {filename}")

    reagents, reag_energy, _ = optimize(
        data.atoms,
        reagents,
        embedder.options.calculator,
        method=embedder.options.theory_level,
        maxiter=750 if embedder.options.neb.preopt else 1,
        charge=embedder.options.charge,
        mult=embedder.options.mult,
        solvent=embedder.options.solvent,
        title="reagents",
        logfunction=embedder.log,
        dispatcher=embedder.dispatcher,
        debug=embedder.options.debug,
    )

    products, prod_energy, _ = optimize(
        data.atoms,
        products,
        embedder.options.calculator,
        method=embedder.options.theory_level,
        maxiter=750 if embedder.options.neb.preopt else 1,
        charge=embedder.options.charge,
        mult=embedder.options.mult,
        solvent=embedder.options.solvent,
        title="products",
        logfunction=embedder.log,
        dispatcher=embedder.dispatcher,
        debug=embedder.options.debug,
    )

    if mep_override is not None:
        mep_override[0] = reagents
        mep_override[-1] = products

    for attempt in range(attempts):
        ts_coords, ts_energy, energies, exit_status = ase_neb(
            embedder,
            data.atoms,
            reagents,
            products,
            n_images=embedder.options.neb.n_images,
            charge=embedder.options.charge,
            mult=embedder.options.mult,
            ts_guess=ts_guess,
            mep_input=mep_override,
            climbing_image=embedder.options.neb.climbing_image,
            title=title,
            logfunction=embedder.log,
            write_plot=True,
            verbose_print=True,
        )

        if exit_status == "CONVERGED":
            break

        elif exit_status == "MAX ITER" and attempt + 2 < attempts:
            mep_override = read_xyz(f"{title}_MEP_start_of_CI.xyz").coords
            reagents, *_, products = mep_override
            embedder.log(f"--> Restarting NEB from checkpoint. Attempt {attempt + 2}/{attempts}.\n")

        elif exit_status == "CRASHED":
            if attempt + 1 < attempts:
                embedder.log(
                    f"--> NEB optimization crashed. Attempting a restart using inner structures as starting points, attempt {attempt + 2}/3."
                )
                crashed_mep = read_xyz(f"{title}_MEP_crashed.xyz").coords

                reag_id = int(len(crashed_mep) // 4) + attempt
                reag_id = min(reag_id, (len(crashed_mep) // 2) - 1)
                reagents = crashed_mep[reag_id]

                prod_id = len(crashed_mep) - reag_id - 1
                products = crashed_mep[prod_id]

                embedder.log(
                    f"    Attempting restart using structure {reag_id + 1}/{len(crashed_mep)} "
                    + f"as reagents and {prod_id + 1}/{len(crashed_mep)} as products.\n"
                )
            else:
                raise FatalError("NEB optimization crashed.")

    e1 = ts_energy - reag_energy
    e2 = ts_energy - prod_energy
    dg1 = ts_energy - min(energies[:3])
    dg2 = ts_energy - min(energies[4:])

    embedder.log(
        f"NEB completed, relative energy from start/end points (not barrier heights):\n"
        f"  > E(TS)-E(start): {'+' if e1 >= 0 else '-'}{e1:.3f} kcal/mol\n"
        f"  > E(TS)-E(end)  : {'+' if e2 >= 0 else '-'}{e2:.3f} kcal/mol\n"
    )

    embedder.log(
        f"Barrier heights (based on lowest energy point on each side):\n"
        f"  > E(TS)-E(left) : {'+' if dg1 >= 0 else '-'}{dg1:.3f} kcal/mol\n"
        f"  > E(TS)-E(right): {'+' if dg2 >= 0 else '-'}{dg2:.3f} kcal/mol"
    )

    if not (e1 > 0 and e2 > 0):
        embedder.log("\nNEB failed, TS energy is lower than both the start and end points.\n")

    with open(f"{title}_TS.xyz", "w") as f:
        write_xyz(data.atoms, ts_coords, f, title="NEB TS - see log for relative energies")

    return f"{title}_TS.xyz"


def crest_search_operator(filename: str, embedder: Embedder) -> str:
    """Run a CREST metadynamic conformational search and return the output filename."""
    crest_version = get_crest_version()
    assert crest_version is not None, (
        "CREST (version 2 or 3) does not seem to be installed. Install it with: mamba install -c conda-forge crest=3"
    )

    # load molecule to be optimized from embedder
    mol = embedder.mols[filename]

    if not embedder.options.let:
        if len(mol.coords) >= 20:
            raise InputError(
                "The crest_search> operator was given more than 20 input structures. "
                + "This would run >20 metadynamic conformational searches. If this was not a mistake, "
                + "add the LET keyword an re-run the job."
            )

    logfunction = embedder.log
    constrained_indices = embedder._get_internal_constraints(filename)
    constrained_distances = [
        embedder.get_pairing_dists_from_constrained_indices(cp) for cp in constrained_indices
    ]

    (
        constrained_angles_indices,
        constrained_angles_values,
        constrained_dihedrals_indices,
        constrained_dihedrals_values,
    ) = embedder._get_angle_dih_constraints(filename)

    logfunction(
        f"--> {filename}: Geometry optimization pre-crest_search ({embedder.options.theory_level} via {embedder.options.calculator})"
    )
    constr_str = embedder.get_str_all_constraints(filename)
    n_constraints = (
        len(constrained_indices)
        + len(constrained_angles_indices)
        + len(constrained_dihedrals_indices)
    )
    logfunction(
        f"    {n_constraints} constraints applied{': ' + str(constr_str).replace('\n', ' ') if constr_str != '' else ''}"
    )

    for c, coords in enumerate(mol.coords.copy()):
        logfunction(f"    Optimizing conformer {c + 1}/{len(mol.coords)}")

        opt_coords, _, success = (
            optimize(
                mol.atoms,
                coords,
                calculator=embedder.options.calculator,
                method=embedder.options.theory_level,
                solvent=embedder.options.solvent,
                charge=embedder.options.charge,
                mult=embedder.options.mult,
                dispatcher=embedder.dispatcher,
                constrained_indices=constrained_indices,
                constrained_distances=constrained_distances,
                constrained_angles_indices=constrained_angles_indices,
                constrained_angles_values=constrained_angles_values,
                constrained_dihedrals_indices=constrained_dihedrals_indices,
                constrained_dihedrals_values=constrained_dihedrals_values,
                title=f"{filename.split('.', maxsplit=1)[0]}_conf{c + 1}",
                debug=embedder.options.debug,
            )
            if embedder.options.optimization
            else (coords, None, True)
        )

        exit_status = "" if success else "CRASHED"

        if success:
            success = molecule_check(mol.atoms, coords, opt_coords)
            exit_status = "" if success else "SCRAMBLED"

        if not success:
            dumpname = filename.split(".", maxsplit=1)[0] + f"_conf{c + 1}_{exit_status}.xyz"
            with open(dumpname, "w") as f:
                write_xyz(
                    mol.atoms,
                    opt_coords,
                    f,
                    title=f"{filename}, conformer {c + 1}/{len(mol.coords)}, {exit_status}",
                )

            logfunction(
                f"{filename}, conformer {c + 1}/{len(mol.coords)} optimization {exit_status}. Inspect geometry at {dumpname}. Aborting run."
            )

            raise FatalError(filename)

        # update embedder structures after optimization
        mol.coords[c] = opt_coords

    logfunction()

    # update mol and embedder graph after optimization
    mol.graph = graphize(mol.atoms, mol.coords[0])
    embedder.graphs = [m.graph for m in embedder.objects]
    crest_method = embedder.options.crestlevel or "GFN2-XTB//GFN-FF"

    crest_threads = embedder.avail_cpus
    logfunction(
        f"--> Performing {crest_method}"
        + (
            f"{f'/{embedder.options.solvent.upper()}' if embedder.options.solvent is not None else ''} "
            + f"metadynamic conformational search on {filename} via CREST {crest_version}.\n"
            + f"    ({crest_threads} threads, {embedder.options.kcal_thresh} kcal/mol thr.)"
        )
    )

    if embedder.options.nci:
        logfunction("--> NCI: Running crest in NCI mode (wall potential applied)")

    if len(mol.coords) > 1:
        embedder.log(
            "--> Requested conformational search on multimolecular file - will do\n"
            + "an individual search from each conformer (might be time-consuming)."
        )

    t_start = time.perf_counter()
    conformers: list[Array2D_float] = []
    for i, coords in enumerate(mol.coords):
        t_start_conf = time.perf_counter()
        try:
            conf_batch = crest_mtd_search(
                mol.atoms,
                coords,
                constrained_indices=constrained_indices,
                constrained_distances=constrained_distances,
                constrained_angles_indices=constrained_angles_indices,
                constrained_angles_values=constrained_angles_values,
                constrained_dihedrals_indices=constrained_dihedrals_indices,
                constrained_dihedrals_values=constrained_dihedrals_values,
                solvent=embedder.options.solvent,
                charge=mol.charge,
                method=crest_method,
                kcal=embedder.options.kcal_thresh,
                ncimode=embedder.options.nci,
                title=mol.basename + "_CREST",
                threads=crest_threads,
            )

        # if the run errors out, we retry with XTB2
        except CalledProcessError:
            logfunction(
                "--> Metadynamics run failed with GFN2-XTB//GFN-FF, retrying with just GFN2-XTB (slower but more stable)"
            )
            conf_batch = crest_mtd_search(
                mol.atoms,
                coords,
                constrained_indices=constrained_indices,
                constrained_distances=constrained_distances,
                constrained_angles_indices=constrained_angles_indices,
                constrained_angles_values=constrained_angles_values,
                constrained_dihedrals_indices=constrained_dihedrals_indices,
                constrained_dihedrals_values=constrained_dihedrals_values,
                solvent=embedder.options.solvent,
                charge=mol.charge,
                method="GFN2-XTB",  # try with XTB2
                kcal=embedder.options.kcal_thresh,
                ncimode=embedder.options.nci,
                title=mol.basename + "_CREST",
                threads=crest_threads,
            )

        conformers.extend(conf_batch)

        elapsed = time.perf_counter() - t_start_conf
        embedder.log(
            f"  Conformer {i + 1:2}/{len(mol.coords):2} - generated {len(conf_batch)} structures in {time_to_string(elapsed)}"
        )

    conformers_array: Array3D_float = np.concatenate(conformers)
    conformers_array = conformers_array.reshape(-1, mol.atomnos.shape[0], 3)
    # merging structures from each run in a single array

    embedder.log(
        f"  CREST conformational search: Generated {len(conformers_array)} conformers in {time_to_string(time.perf_counter() - t_start)}"
    )
    before = len(conformers_array)

    # ### MOI - turned off, as it would get rid of enantiomeric conformations
    conformers_array, _ = prune_by_moment_of_inertia(
        conformers_array, mol.atoms, debugfunction=embedder.debuglog
    )

    ### RMSD
    if len(conformers_array) < 5e4:
        conformers_array, _ = prune_by_rmsd(
            conformers_array,
            mol.atoms,
            max_rmsd=embedder.options.rmsd,
            debugfunction=embedder.debuglog,
        )
    if len(conformers_array) < 1e3:
        conformers_array, _ = prune_by_rmsd_rot_corr(
            conformers_array,
            mol.atoms,
            mol.graph,
            max_rmsd=embedder.options.rmsd,
            debugfunction=embedder.debuglog,
        )

    embedder.log(
        f"  Discarded {before - len(conformers_array)} RMSD-similar structures ({len(conformers_array)} left)\n"
    )

    ### PRINTOUT
    with open(f"{mol.basename}_crest_confs.xyz", "w") as f:
        for i, new_s in enumerate(conformers_array):
            write_xyz(
                mol.atoms, new_s, f, title=f"Conformer {i}/{len(conformers_array)} from CREST MTD"
            )

    # check the structures again and warn if some look compenetrated
    embedder.check_objects_compenetration()

    return f"{mol.basename}_crest_confs.xyz"


def scan_operator(filename: str, embedder: Embedder) -> str | MaybeNone:
    """Scan operator dispatcher:
    2 indices: distance_scan
    4 indices: dihedral_scan

    """
    mol = embedder.mols[filename]

    assert len(mol.coords) == 1, "The scan> operator works on a single .xyz geometry."

    if len(mol.reactive_indices) == 2:
        return distance_scan(filename, embedder)

    elif len(mol.reactive_indices) == 4:
        dihedral_scan(filename, embedder)
        return None

    else:
        raise InputError(
            f"The scan> operator needs two or four indices ({len(mol.reactive_indices)} were provided)"
        )


def distance_scan(filename: str, embedder: Embedder) -> str:
    """Approach or separate two reactive atoms, looking for the energy maximum.

    Scan direction is inferred by the reactive_indices distance.
    """
    t_start = time.perf_counter()

    mol = embedder.mols[filename]

    # shorthands for clearer code
    i1, i2 = mol.reactive_indices
    coords = mol.coords[0]

    # getting the start distance between scan indices and start energy
    d = float(np.linalg.norm(coords[i1] - coords[i2]))

    # deciding if moving atoms closer or further apart based on distance
    bonds = list(mol.graph.edges)
    step = 0.05 if (i1, i2) in bonds else -0.05

    # creating a dictionary that will hold results
    # and the structure output list
    dists, energies, structures = [], [], []

    # getting atomic symbols
    s1, s2 = mol.atoms[[i1, i2]]

    # manually set final distance from input file
    if hasattr(mol, "d"):
        target = float(mol.d)

        # making sure step has the right sign
        step = 0.05 if target > d else -0.05

        max_iterations = round(abs(d - target) / abs(step))
        embedder.log(
            f"--> {mol.basename}: ({i1}-{i2}) final scan distance set to {target:.2f} A ({max_iterations} iterations)"
        )

    # defining the maximum number of iterations
    elif step < 0:
        smallest_d = 0.9 * (pt.covalent_radius(s1) + pt.covalent_radius(s2))
        max_iterations = round((d - smallest_d) / abs(step))
        # so that atoms are never forced closer than
        # a proportionally small distance between those two atoms.

    else:
        max_d = 1.6 * (pt.covalent_radius(s1) + pt.covalent_radius(s2))
        max_iterations = round((max_d - d) / abs(step))
        # so that atoms are never spaced too far apart

    # logging to file and terminal
    embedder.log(
        f"--> {mol.basename} - Performing a distance scan {'approaching' if step < 0 else 'separating'} indices {i1} "
        + f"and {i2} - step size {round(step, 2)} A\n    Theory level is {embedder.options.theory_level}/{embedder.options.solvent or 'vacuum'} "
        + f"via {embedder.options.calculator}"
    )

    for i in range(max_iterations):
        t_start = time.perf_counter()

        coords, energy, _ = optimize(
            mol.atoms,
            coords,
            calculator=embedder.options.calculator,
            constrained_indices=[list(mol.reactive_indices)],
            constrained_distances=(d,),
            solvent=embedder.options.solvent,
            charge=embedder.options.charge,
            mult=embedder.options.mult,
            dispatcher=embedder.dispatcher,
            title="temp",
            debug=embedder.options.debug,
        )

        if energy is None:
            embedder.log(f"--> Optimization crashed at step {i + 1}/{max_iterations}.")
            break

        if i == 0:
            e_0 = energy

        # saving the structure, distance and relative energy
        energies.append(energy - e_0)
        dists.append(d)
        structures.append(coords)
        # print(f"------> target was {round(d, 3)} A, reached {round(np.linalg.norm(coords[mol.reactive_indices[0]]-coords[mol.reactive_indices[1]]), 3)} A")

        embedder.log(
            f"Step {i + 1:3}/{max_iterations:3} - d={d:.2f} Å    {energy - e_0:+.2f} kcal/mol - {time_to_string(time.perf_counter() - t_start)}"
        )

        with open("temp_scan.xyz", "w") as f:
            for i, (s, d, e) in enumerate(zip(structures, dists, energies)):
                write_xyz(
                    mol.atoms,
                    s,
                    f,
                    title=f"Scan point {i + 1}/{len(structures)} "
                    + f"- d({i1}-{i2}) = {round(d, 3)} A - Rel. E = {round(e - min(energies), 2)} kcal/mol",
                )

        # modify the target distance and reiterate
        d += step

    ### Start the plotting sequence

    plt.figure()
    plt.plot(
        dists,
        energies,
        color="tab:red",
        label="Scan energy",
        linewidth=3,
    )

    id_max = get_scan_peak_index(energies)
    e_max = energies[id_max]

    d_opt = dists[id_max]

    plt.plot(
        d_opt,
        e_max,
        color="gold",
        label="Energy maximum (TS guess)",
        marker="o",
        markersize=3,
    )

    title = mol.basename + " distance scan"
    plt.legend()
    plt.title(title)
    plt.xlabel(f"indices s{i1}-{i2} distance (A)")

    if step < 0:
        plt.gca().invert_xaxis()

    plt.ylabel(
        f"Rel. E. (kcal/mol) - {embedder.options.theory_level}/{embedder.options.calculator}/{embedder.options.solvent}"
    )
    plt.savefig(f"{title.replace(' ', '_')}_plt.svg")

    ### Start structure writing

    # print all scan structures
    with open(f"{mol.basename}_scan.xyz", "w") as f:
        for i, (s, d, e) in enumerate(zip(structures, dists, energies)):
            write_xyz(
                mol.atoms,
                s,
                f,
                title=f"Scan point {i + 1}/{len(structures)} "
                + f"- d({i1}-{i2}) = {round(d, 2)} A - Rel. E = {round(e, 2)} kcal/mol",
            )

    # print the maximum on another file for convienience
    with open(f"{mol.basename}_scan_max.xyz", "w") as f:
        s = structures[id_max]
        d = dists[id_max]
        write_xyz(
            mol.atoms,
            s,
            f,
            title=f"Scan point {id_max + 1}/{len(structures)} "
            + f"- d({i1}-{i2}) = {round(d, 3)} A - Rel. E = {round(e_max, 3)} kcal/mol",
        )

    embedder.log(
        f"\n--> Written {len(structures)} structures to {mol.basename}_scan.xyz ({time_to_string(time.perf_counter() - t_start)})"
    )
    embedder.log(f"\n--> Written energy maximum to {mol.basename}_scan_max.xyz\n")

    # Log data to the embedder class
    mol.scan_data = (dists, energies)

    return f"{mol.basename}_scan.xyz"


def saddle_operator(filename: str, embedder: Embedder) -> str:
    """Run a saddle optimization on the input structure(s).

    If more than one are provided, either process all or
    work on the one with highest energy (i.e. coming from a NEB).
    """
    from firecode.ase_manipulations import ase_saddle
    from firecode.thermochemistry import get_free_energies

    constrained_indices = embedder._get_internal_constraints(filename)

    # The "distance" scan> operator
    # (distance_scan function) returns all
    # structures of the scan, but we want to
    # perform a saddle optimization only on
    # the highest energy one
    pick_highest_energy_structure = False
    for i, _ in enumerate(embedder.objects):
        if _.filename == filename:
            ops = embedder.options.operators_dict[i]

            if len(ops) > 1:
                previous_op = ops[ops.index("saddle") - 1]
                if previous_op == "scan":
                    pick_highest_energy_structure = True

            break

    mol = embedder.mols[filename]

    if pick_highest_energy_structure:
        energies = cast("list[float]", read_xyz_energies(filename))
        highest_energy_index = energies.index(max(energies))
        mol.coords = mol.coords[[highest_energy_index]]
        embedder.log(
            f"--> Saddle operator: picked highest energy structure from {filename} (structure {highest_energy_index + 1}/{len(energies)})"
        )

    s = "s" if len(mol.coords) > 1 else ""
    embedder.log(
        f"\n--> Saddle operator: performing {len(mol.coords)} saddle optimization{s} via Sella"
    )

    opt_coords_list: list[Array2D_float] = []

    for c, coords in enumerate(mol.coords):
        title = f"{mol.basename}_conf{c}_saddle"

        # will work in new folders and
        # not delete them since debug=True
        opt_coords, _, success = ase_saddle(
            atoms=mol.atoms,
            coords=coords,
            dispatcher=embedder.dispatcher,
            charge=mol.charge,
            mult=mol.mult,
            calculator=embedder.options.calculator,
            method=embedder.options.theory_level,
            solvent=embedder.options.solvent,
            constrained_indices=constrained_indices,
            assert_convergence=True,
            traj=title + "_traj",
            logfunction=embedder.log,
            title=title,
        )

        if success:
            opt_coords_list.append(opt_coords)

    opt_coords_array = np.array(opt_coords_list)

    free_energies = get_free_energies(
        embedder=embedder,
        atoms=mol.atoms,
        structures=opt_coords_array,
        charge=mol.charge,
        mult=mol.mult,
        title=f"{mol.basename}",
        tighten_opt_before_vib=False,
        logfunction=embedder.log,
    )

    # sorting structures based on energy
    sorted_indices = np.argsort(free_energies)
    free_energies = free_energies[sorted_indices]
    opt_coords_array = opt_coords_array[sorted_indices]

    outname = f"{mol.basename}_saddles.xyz"

    with open(outname, "w") as f:
        for i, (coords, free_energy) in enumerate(
            zip(align_structures(opt_coords_array), free_energies)
        ):
            write_xyz(
                mol.atoms,
                coords,
                f,
                title=f"Saddle-optimized conf. {i} - G = {free_energy / EH_TO_KCAL:.8f} Eh - Rel. G. = {free_energy - free_energies[0]:.2f} kcal/mol",
            )

    return outname


def freq_operator(filename: str, embedder: Embedder) -> str:
    """Run a frequency calculation on the input structure(s)."""
    from firecode.thermochemistry import get_free_energies

    mol = embedder.mols[filename]

    s = "s" if len(mol.coords) > 1 else ""
    embedder.log(
        f"\n--> Freq operator: performing {len(mol.coords)} frequency calculation{s} via ASE"
    )

    free_energies = get_free_energies(
        embedder=embedder,
        atoms=mol.atoms,
        structures=mol.coords,
        charge=mol.charge,
        mult=mol.mult,
        title=f"{mol.basename}",
        tighten_opt_before_vib=False,
        logfunction=embedder.log,
    )

    # sorting structures based on energy
    sorted_indices = np.argsort(free_energies)
    free_energies = free_energies[sorted_indices]
    coords = mol.coords[sorted_indices]

    outname = f"{mol.basename}_freq.xyz"

    with open(outname, "w") as f:
        for i, (coords, free_energy) in enumerate(zip(align_structures(coords), free_energies)):
            write_xyz(
                mol.atoms,
                coords,
                f,
                title=f"Conf. {i} - G = {free_energy / EH_TO_KCAL:.8f} Eh - Rel. G. = {free_energy - free_energies[0]:.2f} kcal/mol",
            )

    return outname


def packmol_operator(filename: str, embedder: Embedder) -> str:
    """Solvate the input molecule."""
    if embedder.options.solvent is None:
        raise Exception("Please specify a solvent for `packmol>`.")

    mol = embedder.mols[filename]

    if len(mol.coords) > 1:
        raise NotImplementedError

    out_dict = solvate_molecule(
        solute_atoms=mol.atoms,
        solute_coords=mol.coords[0],
        solvent_name=embedder.options.solvent,
        solvent_data=solvent_data,
        title=mol.basename,
        logfunction=embedder.log,
    )

    # save meaningful attributes to embedder md_data dict
    embedder.options.md_data = out_dict

    return str(out_dict["output_xyz"])
