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

import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, cast

import numpy as np
from ase.calculators.calculator import Calculator as ASECalculator
from prism_pruner.pruner import prune_by_rmsd
from prism_pruner.utils import time_to_string

from firecode.ase_manipulations import ase_popt, ase_popt_with_alpb
from firecode.calculators._orca import orca_opt
from firecode.calculators._xtb import xtb_opt
from firecode.settings import DEFAULT_LEVELS
from firecode.typing import (Array1D_float, Array1D_str, Array2D_float,
                             Array3D_float)
from firecode.utils import loadbar, molecule_check, scramble_check

if TYPE_CHECKING:
    from networkx import Graph


class Opt_func_dispatcher:
    """Dispatcher for optimization functions."""

    def __init__(self, calculator: str) -> None:
        """Init method."""
        self.opt_func = {
            "ORCA": orca_opt,
            "XTB": xtb_opt,
            "TBLITE": ase_popt,
            "UMA": ase_popt_with_alpb,
            "AIMNET2": ase_popt_with_alpb,
        }[calculator]

        self.calculator = calculator
        self.ase_calc: ASECalculator | None = None

    def get_ase_calc(
        self,
        method: str | None,
        solvent: str | None = None,
        force_reload: bool = False,
        raise_err: bool = True,
    ) -> ASECalculator | None:
        if self.ase_calc is not None and not force_reload:
            pass

        elif self.calculator == "AIMNET2":
            self.ase_calc = self.load_aimnet2_calc(method)

        elif self.calculator == "TBLITE":
            self.ase_calc = self.load_tblite_calc(method, solvent)

        elif self.calculator == "XTB":
            self.ase_calc = self.load_xtb_calc(method, solvent)

        elif self.calculator == "UMA":
            self.ase_calc = self.load_uma_calc(method)

        elif raise_err:
            raise NotImplementedError(
                f"Calculator {self.calculator} not known. Options are AIMNET2, TBLITE, XTB and UMA."
            )

        return self.ase_calc

    def load_aimnet2_calc(
        self, theory_level: str | None, logfunction: Callable[[str], None] | None = print
    ) -> ASECalculator:
        try:
            import torch
            from aimnet.calculators import AIMNet2ASE

        except ImportError:
            raise Exception(
                (
                    "Cannot import AIMNet2 python bindings for FIRECODE. Install them with:\n"
                    "    >>> uv pip install aimnet[ase]\n"
                    'or alternatively, install the "aimnet2" version of firecode:\n'
                    "    >>> uv pip install firecode[aimnet2]\n"
                )
            )

        gpu_bool = torch.cuda.is_available()
        self.aimnet2_calc = cast(ASECalculator, AIMNet2ASE("aimnet2"))

        if logfunction is not None:
            logfunction(f"--> AIMNet2 calculator loaded on {'GPU' if gpu_bool else 'CPU'}.")

        return self.aimnet2_calc

    def load_tblite_calc(self, method: str | None, solvent: str | None) -> ASECalculator:
        try:
            from tblite.ase import TBLite

        except ImportError as e:
            print(e)
            raise Exception(
                (
                    "Cannot import tblite python bindings for FIRECODE. Install them with conda, (or better yet, mamba):\n"
                    ">>> conda install -c conda-forge mamba\n"
                    ">>> mamba install -c conda-forge tblite tblite-python\n"
                )
            )

        solvation = None if solvent is None else ("alpb", solvent)
        method = method or DEFAULT_LEVELS["TBLITE"]

        # tblite is picky with names
        synonyms = {
            "GFN1-XTB": "GFN1-xTB",
            "GFN2-XTB": "GFN2-xTB",
            "G-XTB": "g-xTB",
        }

        method = synonyms.get(method, method)
        self.ase_calc = TBLite(method=method, solvation=solvation)

        return self.ase_calc

    def load_xtb_calc(self, method: str | None, solvent: str | None) -> ASECalculator:
        try:
            from xtb.ase.calculator import XTB
        except ImportError:
            raise Exception(
                (
                    "Cannot import tblite python bindings for FIRECODE. Install them with conda, (or better yet, mamba):\n"
                    ">>> conda install -c conda-forge mamba\n"
                    ">>> mamba install -c conda-forge xtb xtb-python\n"
                )
            )

        synonyms = {
            "GFN1-XTB": "GFN1-xTB",
            "GFN2-XTB": "GFN2-xTB",
            "G-XTB": "g-xTB",
        }
        method = method or DEFAULT_LEVELS["XTB"]
        method = synonyms.get(method.upper(), method)
        self.ase_calc = XTB(method=method, solvation=solvent)

        return self.ase_calc

    def load_uma_calc(
        self, method: str | None, logfunction: Callable[[str], None] | None = print
    ) -> ASECalculator:
        from firecode.calculators._ase_uma import get_uma_calc

        self.uma_calc = get_uma_calc(method, logfunction=logfunction)
        self.ase_calc = self.uma_calc
        return self.uma_calc


def optimize(
    atoms: Array1D_str,
    coords: Array2D_float,
    calculator: ASECalculator,
    method: str | None = None,
    maxiter: int | None = None,
    conv_thr: str = "tight",
    constrained_indices: list[Sequence[int]] | None = None,
    constrained_distances: list[float] | None = None,
    constrained_dihedrals_indices: list[Sequence[int]] | None = None,
    constrained_dihedrals_values: list[float] | None = None,
    constrained_angles_indices: list[Sequence[int]] | None = None,
    constrained_angles_values: list[float] | None = None,
    mols_graphs: Sequence[Graph] | None = None,
    procs: int = 1,
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
    if dispatcher is None:
        dispatcher = Opt_func_dispatcher(calculator)

    ase_calc = dispatcher.get_ase_calc(method, solvent=solvent)

    if mols_graphs is not None:
        _l = [len(graph.nodes) for graph in mols_graphs]
        assert len(coords) == sum(_l), (
            f"{len(coords)} coordinates are specified but graphs have {_l} = {sum(_l)} nodes"
        )

    if method is None:
        method = DEFAULT_LEVELS[calculator]

    if constrained_distances is not None:
        assert len(constrained_distances) == len(constrained_indices), (
            f"len(cd) = {len(constrained_distances)} != len(ci) = {len(constrained_indices)}"
        )

    constrained_indices = np.array(()) if constrained_indices is None else constrained_indices

    opt_func = dispatcher.opt_func
    t_start = time.perf_counter()

    # success checks that calculation had a normal termination
    opt_coords, energy, success = opt_func(
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
        traj=title + ".traj",
        ase_calc=ase_calc,
        # **kwargs,
    )

    elapsed = time.perf_counter() - t_start

    if success:
        if check:
            # check boolean ensures that no scrambling occurred during the optimization
            if mols_graphs is not None:
                success = scramble_check(
                    atoms, opt_coords, constrained_indices, mols_graphs, max_newbonds=max_newbonds
                )
            else:
                success = molecule_check(atoms, coords, opt_coords, max_newbonds=max_newbonds)

        if logfunction is not None:
            if success:
                logfunction(f"    - {title} - REFINED {time_to_string(elapsed)}")
            else:
                logfunction(f"    - {title} - SCRAMBLED {time_to_string(elapsed)}")

        return opt_coords, energy, success

    if logfunction is not None:
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
    calculator: ASECalculator,
    method: str | None,
    procs: int | None,
    charge: int = 0,
    mult: int = 1,
    constrained_indices: list[Sequence[int]] | None = None,
    constrained_distances: list[float] | None = None,
    constrained_dihedrals_indices: list[Sequence[int]] | None = None,
    constrained_dihedrals_values: list[float] | None = None,
    constrained_angles_indices: list[Sequence[int]] | None = None,
    constrained_angles_values: list[float] | None = None,
    solvent: str | None = None,
    loadstring: str = "",
    logfunction: Callable[[str], None] | None = None,
    dispatcher: Opt_func_dispatcher | None = None,
    debug: bool = False,
) -> tuple[Array3D_float, Array1D_float]:
    """Refine a set of structures - optimize them and remove similar
    ones and high energy ones (>20 kcal/mol above lowest)
    """
    energies = []
    for i, conformer in enumerate(deepcopy(structures)):
        loadbar(i, len(structures), f"{loadstring} {i + 1}/{len(structures)} ")

        opt_coords, energy, success = optimize(
            atoms,
            conformer,
            calculator,
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
            structures[i] = opt_coords
            energies.append(energy)
        else:
            energies.append(1e10)

    loadbar(len(structures), len(structures), f"{loadstring} {len(structures)}/{len(structures)} ")
    energies = np.array(energies)

    # remove similar ones
    structures, mask = prune_by_rmsd(structures, atoms)
    energies = energies[mask]

    # remove high energy ones
    mask = (energies - np.min(energies)) < 20
    structures, energies = structures[mask], energies[mask]

    return structures, energies
