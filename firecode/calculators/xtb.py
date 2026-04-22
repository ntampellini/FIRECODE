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

import os
from subprocess import STDOUT, CalledProcessError, check_call
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
from prism_pruner.algebra import normalize

from firecode.context_managers import NewFolderContext
from firecode.graph_manipulations import get_sum_graph
from firecode.solvents import to_xtb_solvents
from firecode.typing_ import Array1D_str, Array2D_float
from firecode.units import EH_TO_KCAL
from firecode.utils import clean_directory, write_xyz

if TYPE_CHECKING:
    from networkx import Graph


def xtb_opt(
    atoms: Array1D_str,
    coords: Array2D_float,
    constrained_indices: Sequence[Sequence[int]] | None = None,
    constrained_distances: Sequence[float | None] | None = None,
    constrained_dihedrals_indices: Sequence[Sequence[int]] | None = None,
    constrained_dihedrals_values: Sequence[float | None] | None = None,
    constrained_angles_indices: Sequence[Sequence[int]] | None = None,
    constrained_angles_values: Sequence[float | None] | None = None,
    method: str = "GFN2-xTB",
    maxiter: int | None = 500,
    solvent: str | None = None,
    charge: int = 0,
    mult: int = 1,
    title: str = "temp",
    read_output: bool = True,
    procs: int = 4,
    opt: bool = True,
    conv_thr: str = "tight",
    assert_convergence: bool = False,
    constrain_string: str | None = None,
    recursive_stepsize: float = 0.3,
    spring_constant: float = 1.0,
    debug: bool = False,
    **kwargs: Any,
) -> tuple[Array2D_float, float | None, Literal[True]] | None:
    """This function writes an XTB .inp file, runs it with the subprocess
    module and reads its output.

    coords: array of shape (n,3) with cartesian coordinates for atoms.

    atoms: array of strings indicating elements.

    constrained_indices: array of shape (n,2), with the indices
    of atomic pairs to be constrained.

    constrained_distances: optional, target distances for the specified
    distance constraints.

    constrained_dihedrals: quadruplets of atomic indices to constrain.

    constrained_dih_angles: target dihedral angles for the dihedral constraints.

    method: string, specifying the theory level to be used.

    maxiter: maximum number of geometry optimization steps (maxcycle).

    solvent: solvent to be used in the calculation (ALPB model).

    charge: charge to be used in the calculation.

    title: string, used as a file name and job title for the mopac input file.

    read_output: Whether to read the output file and return anything.

    procs: number of cores to be used for the calculation.

    opt: if false, a single point energy calculation is carried.

    conv_thr: tightness of convergence thresholds. See XTB ReadTheDocs.

    assert_convergence: wheter to raise an error in case convergence is not
    achieved by xtb.

    constrain_string: string to be added to the end of the $geom section of
    the input file.

    recursive_stepsize: magnitude of step in recursive constrained optimizations.
    The smaller, the slower - but potentially safer against scrambling.

    spring_constant: stiffness of harmonic distance constraint (Hartrees/Bohrs^2)

    """
    # create working folder and cd into it
    with NewFolderContext(title, delete_after=(not debug)):
        if constrained_indices is not None:
            if len(constrained_indices) == 0:
                constrained_indices = None

        if constrained_distances is not None:
            if len(constrained_distances) == 0:
                constrained_distances = None

        # recursive
        if constrained_indices is not None and constrained_distances is not None and read_output:
            try:
                for i, (target_d, ci) in enumerate(zip(constrained_distances, constrained_indices)):
                    if target_d is None:
                        continue

                    if len(ci) == 2:
                        a, b = ci
                    else:
                        continue

                    d = float(np.linalg.norm(coords[b] - coords[a]))
                    delta = d - target_d

                    if abs(delta) > recursive_stepsize:
                        recursive_c_d = list(constrained_distances).copy()
                        recursive_c_d[i] = target_d + (recursive_stepsize * np.sign(d - target_d))
                        # print(f"-------->  d is {round(d, 3)}, target d is {round(target_d, 3)}, delta is {round(delta, 3)}, setting new pretarget at {recursive_c_d}")
                        coords, _, _ = xtb_opt(  # type: ignore
                            atoms,
                            coords,
                            constrained_indices,
                            constrained_distances=recursive_c_d,
                            method=method,
                            solvent=solvent,
                            charge=charge,
                            mult=mult,
                            maxiter=50,
                            title=title,
                            procs=procs,
                            conv_thr="loose",
                            constrain_string=constrain_string,
                            recursive_stepsize=0.3,
                            spring_constant=0.25,
                            constrained_dihedrals_indices=constrained_dihedrals_indices,
                            constrained_dihedrals_values=constrained_dihedrals_values,
                            constrained_angles_indices=constrained_angles_indices,
                            constrained_angles_values=constrained_angles_values,
                            read_output=True,
                        )

                    d = float(np.linalg.norm(coords[b] - coords[a]))
                    delta = d - target_d
                    coords[b] -= normalize(coords[b] - coords[a]) * delta
                    # print(f"--------> moved atoms from {round(d, 3)} A to {round(np.linalg.norm(coords[b] - coords[a]), 3)} A")

            except RecursionError:
                with open(f"{title}_crashed.xyz", "w") as f:
                    write_xyz(atoms, coords, f, title=title)
                raise RecursionError(
                    "Recursion limit reached in constrained optimization - Crashed."
                )

        with open(f"{title}.xyz", "w") as f:
            write_xyz(atoms, coords, f, title=title)

        # outname = f'{title}_xtbopt.xyz' DOES NOT WORK - XTB ISSUE?
        outname = "xtbopt.xyz"
        trajname = f"{title}_opt_log.xyz"
        maxiter = maxiter if maxiter is not None else 0
        s = f"$opt\n   logfile={trajname}\n   output={outname}\n   maxcycle={maxiter}\n"

        if constrained_indices is not None:
            s += f"\n$constrain\n   force constant={spring_constant}\n"

            constrained_distances = constrained_distances or [None for _ in constrained_indices]

            for (a, b), dist in zip(constrained_indices, constrained_distances):
                dist_ = str(dist) if dist is not None else "auto"
                s += f"   distance: {a + 1}, {b + 1}, {dist_}\n"

        if constrained_angles_indices is not None:
            constrained_angles_values = constrained_angles_values or [
                None for _ in constrained_angles_indices
            ]

            if constrained_indices is None:
                s += "\n$constrain\n"

            for (a, b, c), angle in zip(constrained_angles_indices, constrained_angles_values):
                angle_ = str(angle) if angle is not None else "auto"
                s += f"   angle: {a + 1}, {b + 1}, {c + 1}, {angle_}\n"

        if constrained_dihedrals_indices is not None:
            constrained_dihedrals_values = constrained_dihedrals_values or [
                None for _ in constrained_dihedrals_indices
            ]

            if constrained_indices is None:
                s += "\n$constrain\n"

            for (a, b, c, d), angle in zip(
                constrained_dihedrals_indices, constrained_dihedrals_values
            ):
                angle_ = str(angle) if angle is not None else "auto"
                s += f"   dihedral: {a + 1}, {b + 1}, {c + 1}, {d + 1}, {angle_}\n"

        if constrain_string is not None:
            s += "\n$constrain\n"
            s += constrain_string

        if method.upper() in ("GFN-XTB", "GFNXTB"):
            s += "\n$gfn\n   method=1\n"

        elif method.upper() in ("GFN2-XTB", "GFN2XTB"):
            s += "\n$gfn\n   method=2\n"

        s += "\n$end"

        s = "".join(s)
        with open(f"{title}.inp", "w") as f:
            f.write(s)

        flags = "--norestart"

        if opt:
            flags += f" --opt {conv_thr}"
            # specify convergence tightness

        if method in ("GFN-FF", "GFNFF"):
            flags += " --gfnff"
            # declaring the use of FF instead of semiempirical

        if charge != 0:
            flags += f" --chrg {charge}"

        if mult != 1:
            flags += f" --uhf {int(int(mult) - 1)}"

        if procs is not None:
            flags += f" -P {procs}"

        if solvent is not None:
            if solvent == "meoh":
                flags += " --gbsa methanol"

            else:
                flags += f" --alpb {to_xtb_solvents.get(solvent, solvent)}"

        elif method.upper() in ("GFN-FF", "GFNFF"):
            flags += " --alpb ch2cl2"
            # if using the GFN-FF force field, add CH2Cl2 solvation for increased accuracy

        # NOTE: temporary!
        if method == "g-xTB":
            flags += ' --driver "gxtb -grad -c xtbdriver.xyz"'

        try:
            with open(f"{title}.out", "w") as f:
                xtb_path = os.environ.get("FIRECODE_PATH_TO_XTB") or "xtb"
                check_call(
                    f"{xtb_path} {title}.xyz --input {title}.inp {flags}".split(),
                    stdout=f,
                    stderr=STDOUT,
                )

        # sometimes the SCC does not converge: only raise the error if specified
        except CalledProcessError:
            if assert_convergence:
                raise CalledProcessError(
                    1,
                    f"{xtb_path} {title}.xyz --input {title}.inp {flags}",
                    "XTB optimization failed to converge.",
                )

        except KeyboardInterrupt:
            raise KeyboardInterrupt("KeyboardInterrupt requested by user. Quitting.")

        if read_output:
            if opt:
                if trajname in os.listdir():
                    coords, energy = read_from_xtbtraj(trajname)

                else:
                    energy = None

                clean_directory((f"{title}.inp", f"{title}.xyz", f"{title}.out", trajname, outname))

            else:
                energy = energy_grepper(f"{title}.out", "TOTAL ENERGY", 3)
                # clean_directory((f'{title}.inp', f'{title}.xyz', f"{title}.out", trajname, outname))

            clean_directory(
                to_remove=(
                    "gfnff_topo",
                    "charges",
                    "wbo",
                    "xtbrestart",
                    "xtbtopo.mol",
                    ".xtboptok",
                    "gfnff_adjacency",
                    "gfnff_charges",
                ),
            )

            return coords, energy, True

        return None


def xtb_pre_opt(
    atoms: Array1D_str,
    coords: Array2D_float,
    graphs: Sequence[Graph],
    constrained_indices: Sequence[Sequence[int]] | None = None,
    constrained_distances: Sequence[float | None] | None = None,
    **kwargs: Any,
) -> tuple[Array2D_float, float | None, Literal[True]] | None:
    """Wrapper for xtb_opt that preserves the distance of every bond present in each subgraph provided

    graphs: list of subgraphs that make up coords, in order

    """
    sum_graph = get_sum_graph(graphs, extra_edges=constrained_indices)

    # we have to check through a list this way, as I have not found
    # an analogous way to check through an array for subarrays in a nice way
    list_of_constr_ids = (
        [[a, b] for a, b in constrained_indices] if constrained_indices is not None else []
    )

    constrain_string = "$constrain\n"
    for constraint in [[a, b] for (a, b) in sum_graph.edges if a != b]:
        if constrained_distances is None:
            distance = "auto"

        elif constraint in list_of_constr_ids:
            distance = str(constrained_distances[list_of_constr_ids.index(constraint)])

        else:
            distance = "auto"

        indices_string = str([i + 1 for i in constraint]).strip("[").strip("]")
        constrain_string += f"  distance: {indices_string}, {distance}\n"
    constrain_string += "\n$end"

    return xtb_opt(
        atoms,
        coords,
        constrained_indices=constrained_indices,
        constrained_distances=constrained_distances,
        constrain_string=constrain_string,
        **kwargs,
    )


def read_from_xtbtraj(filename: str) -> tuple[Array2D_float, float]:
    """Read coordinates from a .xyz trajfile."""
    with open(filename, "r") as f:
        lines = f.readlines()

    # look for the last line containing the flag (iterate in reverse order)
    # and extract the line at which coordinates start
    first_coord_line = len(lines) - next(
        line_num for line_num, line in enumerate(reversed(lines)) if "energy:" in line
    )
    xyzblock = lines[first_coord_line:]

    coords = np.array([line.split()[1:] for line in xyzblock], dtype=float)
    energy = float(lines[first_coord_line - 1].split()[1]) * EH_TO_KCAL

    return coords, energy


def energy_grepper(filename: str, signal_string: str, position: int) -> float:
    """Returns a kcal/mol energy from a Eh energy in a textfile."""
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while True:
            line = f.readline()
            if signal_string in line:
                return float(line.split()[position]) * EH_TO_KCAL
            if not line:
                raise Exception(f"Could not find '{signal_string}' in file ({filename}).")


def xtb_get_free_energy(
    atoms: Array1D_str,
    coords: Array2D_float,
    method: str = "GFN2-xTB",
    solvent: str | None = None,
    charge: int = 0,
    title: str = "temp",
    sph: bool = False,
    grep: str = "G",
    debug: bool = False,
    **kwargs: Any,
) -> float:
    """Calculates free energy with XTB,
    without optimizing the provided structure.
    grep: returns either "G" or "Gcorr" in kcal/mol
    sph: whether to run as single point hessian or not

    """
    with NewFolderContext(title, delete_after=not debug):
        with open(f"{title}.xyz", "w") as f:
            write_xyz(atoms, coords, f, title=title)

        outname = "xtbopt.xyz"
        trajname = f"{title}_opt_log.xyz"
        s = f"$opt\n   logfile={trajname}\n   output={outname}\n   maxcycle=1\n"

        if method.upper() in ("GFN-XTB", "GFNXTB"):
            s += "\n$gfn\n   method=1\n"

        elif method.upper() in ("GFN2-XTB", "GFN2XTB"):
            s += "\n$gfn\n   method=2\n"

        s += "\n$end"

        s = "".join(s)
        with open(f"{title}.inp", "w") as f:
            f.write(s)

        if sph:
            flags = "--bhess"
        else:
            flags = "--ohess"

        if method in ("GFN-FF", "GFNFF"):
            flags += " --gfnff"
            # declaring the use of FF instead of semiempirical

        if charge != 0:
            flags += f" --chrg {charge}"

        if solvent is not None:
            if solvent == "methanol":
                flags += " --gbsa methanol"

            else:
                flags += f" --alpb {to_xtb_solvents.get(solvent, solvent)}"

        try:
            with open("temp_hess.log", "w") as outfile:
                xtb_path = os.environ.get("FIRECODE_PATH_TO_XTB") or "xtb"
                check_call(
                    f"{xtb_path} --input {title}.inp {title}.xyz {flags}".split(),
                    stdout=outfile,
                    stderr=STDOUT,
                )

        except KeyboardInterrupt:
            raise KeyboardInterrupt("KeyboardInterrupt requested by user. Quitting.")

        # try:
        to_grep, index = {
            "G": ("TOTAL FREE ENERGY", 4),
            "Gcorr": ("G(RRHO) contrib.", 3),
        }[grep]

        try:
            result = energy_grepper("temp_hess.log", to_grep, index)
        except Exception as e:
            os.system(f"cat {outfile}")
            raise e

        clean_directory(
            to_remove=(
                "gfnff_topo",
                "charges",
                "wbo",
                "xtbrestart",
                "xtbtopo.mol",
                ".xtboptok",
                "hessian",
                "g98.out",
                "vibspectrum",
                "wbo",
                "xtbhess.xyz",
                "charges",
                "temp_hess.log",
            ),
        )

        return result


def parse_xtb_out(filename: str) -> Array2D_float:
    """Read an XTB outfile in Bohrs and return the cooodinates in Å."""
    with open(filename, "r") as f:
        lines = f.readlines()

    coords = np.zeros((len(lines) - 3, 3))

    for _l, line in enumerate(lines[1:-2]):
        coords[_l] = line.split()[:-1]

    return coords * 0.529177249  # Bohrs to Angstroms
