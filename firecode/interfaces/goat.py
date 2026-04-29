"""FIRECODE interface to ORCA's GOAT."""

from __future__ import annotations

import os
from subprocess import STDOUT, check_call
from time import perf_counter
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
from prism_pruner.graph_manipulations import graphize
from prism_pruner.pruner import prune
from prism_pruner.utils import time_to_string

from firecode.context_managers import NewFolderContext, orca_env
from firecode.errors import FatalError, InputError
from firecode.optimization_methods import optimize
from firecode.solvents import to_xtb_solvents
from firecode.typing_ import Array1D_str, Array2D_float, Array3D_float
from firecode.utils import (
    clean_directory,
    get_auto_procs_and_mem,
    molecule_check,
    read_xyz,
    write_xyz,
)

if TYPE_CHECKING:
    from firecode.embedder import Embedder


def orca_goat_xtb_search(  # pragma: no cover
    atoms: Array1D_str,
    coords: Array2D_float,
    constrained_indices: Sequence[Sequence[int]] | None = None,
    constrained_distances: Sequence[float | None] | None = None,
    constrained_dihedrals_indices: Sequence[Sequence[int]] | None = None,
    constrained_dihedrals_values: Sequence[float | None] | None = None,
    constrained_angles_indices: Sequence[Sequence[int]] | None = None,
    constrained_angles_values: Sequence[float | None] | None = None,
    method: str = "GFN2-XTB",
    solvent: str | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    kcal: float = 10.0,
    ncimode: bool = False,
    title: str = "temp",
    logfunction: Callable[[str], None] | None = print,
) -> Array3D_float:
    """Runs an ORCA GOAT conformational search with an xTB method and returns
    the conformer ensemble coordinates.

    coords: array of shape (n, 3) with Cartesian coordinates for atoms.

    atoms: array of element symbols.

    constrained_indices: pairs of 0-based atom indices for bond constraints.

    constrained_distances: target bond lengths (Å) for each pair; None means
        constrain to the current value.

    constrained_dihedrals_indices: quadruplets of 0-based atom indices.

    constrained_dihedrals_values: target dihedral angles (degrees); None means
        constrain to the current value.

    constrained_angles_indices: triplets of 0-based atom indices.

    constrained_angles_values: target angles (degrees); None means constrain to
        the current value.

    method: one of 'GFN2-XTB' (default), 'GFN-FF', or 'GFN2-XTB//GFN-FF'
        (dual-level: GFN-FF drives sampling, GFN2-xTB refines).

    solvent: ALPB implicit solvent name (ORCA/xTB convention). None = gas phase.

    charge: total molecular charge.

    multiplicity: spin multiplicity (default 1).

    kcal: conformer energy window in kcal/mol (default 10).

    ncimode: enable NCI-GOAT mode (wall-potential sampling for weakly bound
        complexes).

    title: base name for all generated files and the job folder.

    """
    with NewFolderContext(title, delete_after=False):
        # --- Normalize empty inputs ---
        if constrained_indices is not None and len(constrained_indices) == 0:
            constrained_indices = None
        if constrained_distances is not None and len(constrained_distances) == 0:
            constrained_distances = None

        if constrained_dihedrals_indices is not None and len(constrained_dihedrals_indices) == 0:
            constrained_dihedrals_indices = None
        if constrained_dihedrals_values is not None and len(constrained_dihedrals_values) == 0:
            constrained_dihedrals_values = None

        if constrained_angles_indices is not None and len(constrained_angles_indices) == 0:
            constrained_angles_indices = None
        if constrained_angles_values is not None and len(constrained_angles_values) == 0:
            constrained_angles_values = None

        # ---------------------------------------------------------------
        # Map method strings to ORCA keywords
        # ---------------------------------------------------------------
        def _orca_method_keyword(m: str) -> str:
            m = m.upper()
            if m in ("GFN2-XTB", "GFN2"):
                return "XTB2"
            elif m in ("GFN-FF", "GFNFF"):
                return "GFNFF"
            elif m in ("GFN1-XTB", "GFN1"):
                return "XTB1"
            else:
                return m  # pass through unknown strings as-is

        method_upper = method.upper()
        dual_level = method_upper in ("GFN2-XTB//GFN-FF", "GFN2//GFNFF")
        method_keyword = _orca_method_keyword(method_upper) if not dual_level else "XTB2"

        # ---------------------------------------------------------------
        # Build ORCA .inp file
        # ---------------------------------------------------------------
        lines: list[str] = []

        # --- Simple keywords line ---
        simple_kw = [method_keyword, "GOAT"]

        if solvent is not None:
            orca_solvent = to_xtb_solvents.get(solvent, solvent)
            simple_kw.append(f"ALPB({orca_solvent})")
        lines.append("! " + " ".join(simple_kw) + "\n")

        procs, mem_gb = get_auto_procs_and_mem()

        if procs > 1:
            lines.append(f"%pal\n  nprocs {procs}\nend\n")

        lines.append(f"%maxcore {int(mem_gb * 1024)}\n")

        # %goat block
        lines.append("%goat")
        lines.append(f"  MAXEN {kcal}")
        lines.append("  ALIGN true")
        if dual_level:
            lines.append("  GFNUPHILL GFNFF")
        if ncimode:
            lines.append("  AUTOWALL true")
        lines.append("end\n")

        # --- Constraints via %geom block ---
        # ORCA constraint syntax (0-based indices):
        #   bond:     {B  i j value}  or  {B  i j C}   (C = current value)
        #   angle:    {A  i j k value}  or  {A  i j k C}
        #   dihedral: {D  i j k l value}  or  {D  i j k l C}
        has_constraints = any(
            x is not None
            for x in (
                constrained_indices,
                constrained_angles_indices,
                constrained_dihedrals_indices,
            )
        )
        if has_constraints:
            lines.append("%geom")
            lines.append("  Constraints")

            if constrained_indices is not None:
                resolved_distances = list(
                    constrained_distances or [None] * len(constrained_indices)
                )
                for (c1, c2), cd in zip(constrained_indices, resolved_distances):
                    val_str = f"{round(cd, 4)}" if cd is not None else "C"
                    lines.append(f"    {{B {c1} {c2} {val_str}}}")

            if constrained_angles_indices is not None:
                resolved_angles = list(
                    constrained_angles_values or [None] * len(constrained_angles_indices)
                )
                for (a, b, c), angle in zip(constrained_angles_indices, resolved_angles):
                    val_str = f"{round(angle, 4)}" if angle is not None else "C"
                    lines.append(f"    {{A {a} {b} {c} {val_str}}}")

            if constrained_dihedrals_indices is not None:
                resolved_dihedrals = list(
                    constrained_dihedrals_values or [None] * len(constrained_dihedrals_indices)
                )
                for (a, b, c, d), angle in zip(constrained_dihedrals_indices, resolved_dihedrals):
                    val_str = f"{round(angle, 4)}" if angle is not None else "C"
                    lines.append(f"    {{D {a} {b} {c} {d} {val_str}}}")

            lines.append("  end")
            lines.append("end\n")

        # --- Coordinate block ---
        lines.append(f"* xyz {charge} {multiplicity}")
        for element, (x, y, z) in zip(atoms, coords):
            lines.append(f"  {element:<3s} {x:15.8f} {y:15.8f} {z:15.8f}")
        lines.append("*")
        lines.append("")

        # Write .inp file
        inp_path = f"{title}.inp"
        with open(inp_path, "w") as f:
            f.write("\n".join(lines))

        # ---------------------------------------------------------------
        # Run ORCA
        # ORCA must be called with its full path or be on PATH; stdout is
        # redirected because ORCA writes its log to stdout, not a file.
        # ---------------------------------------------------------------
        try:
            with orca_env():
                with open(f"{title}.out", "w") as f:
                    orca_path = os.environ.get("ORCAEXE") or "orca"
                    check_call([orca_path, inp_path], stdout=f, stderr=STDOUT)

        except KeyboardInterrupt:
            raise KeyboardInterrupt("KeyboardInterrupt requested by user. Quitting.")

        # ORCA GOAT writes the ensemble to <basename>_goat_ensemble.xyz
        ensemble_file = f"{title}.finalensemble.xyz"
        new_coords = read_xyz(ensemble_file).coords

        clean_directory(
            to_remove=(
                f"{title}.gbw",
                f"{title}.densities",
                f"{title}_property.txt",
                f"{title}.xtbrestart",
                "charges",
                "wbo",
            ),
        )

        return new_coords


def goat_operator(filename: str, embedder: Embedder) -> str:  # pragma: no cover
    """Run a GOAT conformational search via ORCA and return the output filename."""
    # load molecule to be optimized from embedder
    mol = next((mol for mol in embedder.objects if mol.filename == filename))

    if not embedder.options.let:
        if len(mol.coords) > 1:
            raise InputError(
                "The `goat>` operator takes one structure as input. "
                f"{len(mol.coords)} were provided."
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
        f"--> {filename}: Geometry optimization pre-goat_search ({embedder.options.theory_level} via {embedder.options.calculator})"
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
    goat_level = "GFN2-XTB"

    logfunction(
        f"--> Performing {goat_level}"
        + (
            f"{f'/{embedder.options.solvent.upper()}' if embedder.options.solvent is not None else ''} "
            + f"GOAT conformational search on {filename} via ORCA.\n"
            + f"    ({embedder.options.kcal_thresh} kcal/mol thr.)"
        )
    )

    if embedder.options.nci:
        logfunction("--> NCI: Running GOAT in NCI mode (wall potential applied)")

    if len(mol.coords) > 1:
        embedder.log(
            "--> Requested conformational search on multimolecular file - will do\n"
            + "an individual search from each conformer (might be time-consuming)."
        )

    t_start = perf_counter()
    conformers: list[Array2D_float] = []
    for i, coords in enumerate(mol.coords):
        t_start_conf = perf_counter()

        conf_batch = orca_goat_xtb_search(
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
            method=goat_level,
            kcal=embedder.options.kcal_thresh,
            ncimode=embedder.options.nci,
            title=mol.basename + "_GOAT",
            logfunction=embedder.log,
        )

        conformers.extend(conf_batch)

        elapsed = perf_counter() - t_start_conf
        embedder.log(
            f"  Conformer {i + 1:2}/{len(mol.coords):2} - generated {len(conf_batch)} structures in {time_to_string(elapsed)}"
        )

    conformers_array: Array3D_float = np.concatenate(conformers)
    conformers_array = conformers_array.reshape(-1, mol.atomnos.shape[0], 3)
    # merging structures from each run in a single array

    logfunction(
        f"  GOAT conformational search: Generated {len(conformers_array)} conformers in {time_to_string(perf_counter() - t_start)}"
    )
    before = len(conformers_array)

    conformers_array, _ = prune(
        conformers_array, mol.atoms, logfunction=logfunction, debugfunction=embedder.debuglog
    )
    after = len(conformers_array)

    if before > after:
        logfunction(f"  Discarded {before - after} similar structures ({after} left)\n")
    else:
        logfunction(f"All {after} structures passed the similarity pruning.")

    ### PRINTOUT
    with open(f"{mol.basename}_goat_confs.xyz", "w") as f:
        for i, new_s in enumerate(conformers_array):
            write_xyz(
                mol.atoms, new_s, f, title=f"Conformer {i}/{len(conformers_array)} from CREST MTD"
            )

    # check the structures again and warn if some look compenetrated
    embedder.check_objects_compenetration()

    return f"{mol.basename}_goat_confs.xyz"
