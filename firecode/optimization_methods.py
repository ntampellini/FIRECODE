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

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import numpy as np
from prism_pruner.utils import align_structures, time_to_string

from firecode.dispatcher import Opt_func_dispatcher
from firecode.ensemble import Ensemble
from firecode.settings import (
    CHECKPOINT_EVERY,
    DEFAULT_LEVELS,
)
from firecode.typing_ import Array1D_float, Array1D_str, Array2D_float, Array3D_float
from firecode.utils import loadbar, molecule_check, scramble_check, write_xyz

if TYPE_CHECKING:
    from networkx import Graph


def optimize(
    atoms: Array1D_str,
    coords: Array2D_float,
    calculator: str,
    method: str | None = None,
    maxiter: int | None = None,
    conv_thr: str = "tight",
    constrained_indices: Sequence[Sequence[int]] | None = None,
    constrained_distances: Sequence[float | None] | None = None,
    constrained_dihedrals_indices: Sequence[Sequence[int]] | None = None,
    constrained_dihedrals_values: Sequence[float | None] | None = None,
    constrained_angles_indices: Sequence[Sequence[int]] | None = None,
    constrained_angles_values: Sequence[float | None] | None = None,
    mols_graphs: Sequence[Graph] | None = None,
    procs: int | None = 1,
    solvent: str | None = None,
    charge: int = 0,
    mult: int = 1,
    max_newbonds: int = 0,
    title: str = "temp",
    check: bool = True,
    logfunction: Callable[[str], None] | None = None,
    debug: bool = False,
    dispatcher: Opt_func_dispatcher | None = None,
    **kwargs: Any,
) -> tuple[Array2D_float, float, bool]:
    """Performs a geometry [partial] optimization (OPT/POPT) with MOPAC, ORCA or XTB at $method level,
    constraining the distance between the specified atom pairs, if any. Moreover, if $check, performs a check on atomic
    pairs distances to ensure that the optimization has preserved molecular identities and no atom scrambling occurred.

    :params calculator: Calculator to be used. ('MOPAC', 'ORCA', 'XTB', 'AIMNET2')
    :params coords: list of coordinates for each atom in the TS
    :params atoms: list of atomic symbols for each atom in the TS
    :params mols_graphs: list of molecule.graph objects, containing connectivity information for each molecule
    :params constrained_indices: indices of constrained atoms in the TS geometry, if this is one
    :params method: Level of theory to be used in geometry optimization. Default if UFF.

    :return opt_coords: optimized structure
    :return energy: absolute energy of structure, in kcal/mol
    :return not_scrambled: bool, indicating if the optimization shifted up some bonds (except the constrained ones)
    """
    dispatcher = dispatcher or Opt_func_dispatcher(calculator)
    ase_calc = dispatcher.get_ase_calc(method, solvent=solvent)

    if mols_graphs is not None:
        _l = [len(graph.nodes) for graph in mols_graphs]
        assert len(coords) == sum(_l), (
            f"{len(coords)} coordinates are specified but graphs have {_l} = {sum(_l)} nodes"
        )

    if method is None:
        method = DEFAULT_LEVELS[calculator]

    if constrained_distances is not None:
        len_ci = len(constrained_indices) if constrained_indices is not None else 0
        assert len(constrained_distances) == len_ci, (
            f"len(cd) = {len(constrained_distances)} != len(ci) = {len_ci}"
        )

    opt_func = dispatcher.opt_func
    t_start = perf_counter()

    constrained_indices = constrained_indices or []
    procs = procs or 1

    # success checks that calculation had a normal termination
    opt_coords, energy, success = opt_func(  # type: ignore[operator]
        atoms,
        coords,
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
        mult=mult,
        debug=debug,
        logfunction=logfunction,
        traj=title + ".traj",
        ase_calc=ase_calc,
        # **kwargs,
    )

    elapsed = perf_counter() - t_start

    if success:
        if check:
            # check boolean ensures that no scrambling occurred during the optimization
            if mols_graphs is not None:
                success = scramble_check(
                    atoms,
                    opt_coords,
                    np.array(constrained_indices),
                    mols_graphs,
                    max_newbonds=max_newbonds,
                )
            else:
                success = molecule_check(atoms, coords, opt_coords, max_newbonds=max_newbonds)

        if logfunction is not None and calculator == "XTB":
            if success:
                logfunction(f"    - {title} - REFINED {time_to_string(elapsed)}")
            else:
                logfunction(f"    - {title} - SCRAMBLED {time_to_string(elapsed)}")

        return opt_coords, energy, success

    if logfunction is not None and calculator == "XTB":
        logfunction(f"    - {title} - CRASHED")

    return coords, energy, False


def fitness_check(
    coords: Array2D_float,
    constraints: Iterable[tuple[int, int]],
    targets: Iterable[float | None],
    threshold: float,
) -> bool:
    """Returns True if the strucure respects
    the imposed pairings specified in constraints.
    targets: target distances for each constraint
    threshold: cumulative threshold to reject a structure (A)

    """
    error: float = 0.0
    for (a, b), target in zip(constraints, targets):
        if target is not None:
            error += float(np.linalg.norm(coords[a] - coords[b]) - target)

    return error < threshold


def refine_structures(
    atoms: Array1D_str,
    structures: Array2D_float,
    calculator: str,
    method: str | None,
    procs: int | None,
    charge: int = 0,
    mult: int = 1,
    constrained_indices: Sequence[Sequence[int]] | None = None,
    constrained_distances: Sequence[float | None] | None = None,
    constrained_dihedrals_indices: Sequence[Sequence[int]] | None = None,
    constrained_dihedrals_values: Sequence[float | None] | None = None,
    constrained_angles_indices: Sequence[Sequence[int]] | None = None,
    constrained_angles_values: Sequence[float | None] | None = None,
    solvent: str | None = None,
    loadstring: str = "",
    logfunction: Callable[[str], None] | None = None,
    dispatcher: Opt_func_dispatcher | None = None,
    debug: bool = False,
) -> tuple[Array3D_float, Array1D_float]:
    """Refine a set of structures - optimize them and remove similar
    and high energy ones (>10 kcal/mol above lowest)
    """
    checkpoint_name = None
    old_checkpoint_name = None

    # make ensemble
    ens = Ensemble(
        atoms,
        structures,
        logfunction=logfunction,
    )

    # remove similar structures
    ens.similarity_pruning()

    # number of structures to be processed
    N = len(ens.coords)

    # start optimizing
    t_start = perf_counter()
    energies_list = []
    opt_structures_list = []

    for i, conformer in enumerate(ens.coords):
        loadbar(i, N, f"{loadstring} {i + 1}/{N} ")

        opt_coords, energy, success = optimize(
            atoms,
            conformer,
            calculator=calculator,
            constrained_indices=constrained_indices,
            constrained_distances=constrained_distances,
            constrained_dihedrals_indices=constrained_dihedrals_indices,
            constrained_dihedrals_values=constrained_dihedrals_values,
            constrained_angles_indices=constrained_angles_indices,
            constrained_angles_values=constrained_angles_values,
            method=method,
            procs=procs,
            solvent=solvent,
            title=f"Structure_{i + 1}",
            logfunction=logfunction,
            charge=charge,
            mult=mult,
            dispatcher=dispatcher,
            check=False,  # a change in bonding topology is possible and should not be prevented
            debug=debug,
        )

        if success:
            opt_structures_list.append(opt_coords)
            energies_list.append(energy)

        # Update checkpoint every 50 optimized structures,
        # and give an estimate of the remaining time
        if i % CHECKPOINT_EVERY == CHECKPOINT_EVERY - 1:
            if checkpoint_name is not None:
                old_checkpoint_name = checkpoint_name[:]

            checkpoint_name = f"firecode_checkpoint_{i + 1}_out_of_{N}.xyz"

            # make new ensemble with optimized structures
            ens = Ensemble(
                atoms,
                np.array(opt_structures_list),
                energies=np.array(energies_list),
                logfunction=logfunction,
            )

            with open(checkpoint_name, "w") as f:
                for j, (coord, energy, rel_e) in enumerate(
                    zip(
                        align_structures(ens.coords),
                        ens.energies,
                        ens.rel_energies,
                    )
                ):
                    write_xyz(
                        atoms,
                        coord,
                        f,
                        title=f"Structure {j + 1} - E = {energy:.2f} kcal/mol, Rel. E. = {rel_e:.2f} kcal/mol ({method} via {calculator})",
                    )

            elapsed = perf_counter() - t_start
            average = (elapsed) / (i + 1)
            time_left = time_to_string((average) * (N - i - 1))

            if logfunction is not None:
                logfunction(
                    f"    - Optimized {i + 1:>4}/{N:>4} structures - saved temporary checkpoint file at {checkpoint_name} (avg. {time_to_string(average)}/struc, est. {time_left} left)",
                )

            # now remove old checkpoint after writing the new one
            if old_checkpoint_name is not None:
                Path(old_checkpoint_name).unlink(missing_ok=True)

    # end of optimization: proces the ensemble and return
    loadbar(N, N, f"{loadstring} {N}/{N} ")

    if logfunction is not None:
        s = "s" if N > 1 else ""
        elapsed = perf_counter() - t_start
        s = (
            f"Completed optimization on {N} conformer{s}. "
            f"({time_to_string(elapsed)}, "
            f"~{time_to_string((elapsed) / N)} per structure).\n"
        )
        logfunction(s)

    # make new ensemble with optimized structures
    ens = Ensemble(
        atoms,
        np.array(opt_structures_list),
        energies=np.array(energies_list),
        logfunction=logfunction,
    )
    ens.sort_by_energy()

    # remove high energy structures (>10 kcal/mol)
    ens.energy_pruning()

    # remove similar structures
    ens.similarity_pruning()

    # remove last checkpoint before returning
    if checkpoint_name is not None:
        Path(checkpoint_name).unlink(missing_ok=True)

    return ens.coords, ens.energies
