# coding=utf-8
"""FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2026 Nicolò Tampellini

SPDX-License-Identifier: LGPL-3.0-or-later

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as publishedby
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

import argparse
import os as op_sys
import sys
from copy import deepcopy
from dataclasses import dataclass
from io import TextIOWrapper
from time import perf_counter
from typing import TYPE_CHECKING, Sequence, cast

import numpy as np
from ase.calculators.calculator import Calculator as ASECalculator
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator
from prism_pruner.algebra import dihedral
from prism_pruner.utils import time_to_string

from firecode.algebra import point_angle
from firecode.ase_manipulations import Constraint, Spring, ase_popt, ase_saddle
from firecode.ensemble import Ensemble
from firecode.optimization_methods import Opt_func_dispatcher
from firecode.rdkit_tools import convert_constraint_with_smarts
from firecode.settings import CALCULATOR, DEFAULT_LEVELS
from firecode.solvents import epsilon_dict, solvent_synonyms
from firecode.typing_ import Array1D_int
from firecode.units import EH_TO_KCAL
from firecode.utils import get_ts_d_estimate, read_xyz, write_xyz

if TYPE_CHECKING:
    from firecode.ase_manipulations import ASEConstraint

_defaults = {
    "calculator": "UMA",
    "method": "OMOL",
    "solvent": "ch2cl2",
}


@dataclass
class OptimizerOptions:
    """Standalone Optimizer Options Class."""

    calc: str
    method: str
    solvent: str | None
    filenames: Sequence[str]

    T_K: float = 298.15
    C_mol_L: float = 0.1
    auto_charge_and_mult: bool = True
    constraint_file: str | None = None
    sp: bool = False
    optimize: bool = True
    newfile: bool = False
    free_energy: bool = False
    saddle: bool = False
    irc: bool = False
    smarts_string: str | None = None

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        self.dispatcher = Opt_func_dispatcher(self.calc)
        self.constraints: dict[str, list[Constraint]] = {f: [] for f in self.filenames}
        self.mols: dict[str, Ensemble] = {f: read_xyz(f) for f in self.filenames}
        self.charge_and_mult_dict = {f: self._get_charge_mult_for_file(f) for f in self.filenames}
        self._set_constraints_from_file()

        if self.irc:
            self.optimize = False

    def _get_charge_mult_for_file(self, filename: str) -> tuple[int, int]:
        """Get charge and multiplicity for a given file."""
        if self.auto_charge_and_mult:
            charge = filename.count("+") - filename.count("-")

            if multiplicity_check(self.mols[filename].atomnos, charge):
                mult = 1
            else:
                mult = 2
                print(
                    f'--> Multiplicity of "{filename}" assumed to be 2 based on filename charge and atom types'
                )

        else:
            charge, mult = inquirer.text(  # type: ignore[attr-defined]
                message=f'Manually specify charge and multiplicity for "{filename}":',
                filter=lambda string: tuple(int(s) for s in string.split()),
                validate=lambda string: (
                    string.replace(" ", "").isdigit() and len(string.split()) == 2
                ),
                invalid_message="Please specify two integers separated by a space",
            ).execute()

        return charge, mult

    def _set_constraints_from_file(self) -> None:
        """Set self.constraints from reading self.constraint_file."""
        if self.constraint_file is not None:
            # set constraints from file
            with open(self.constraint_file, "r") as f:
                lines = f.readlines()

            n_constr: int = 0

            # see if we are pattern matching
            if lines[0].startswith("SMARTS"):
                self.smarts_string = lines.pop(0).lstrip("SMARTS ")
                print(
                    "--> SMARTS line found: will pattern match and translate constrained indices on a per-molecule basis."
                )

            for line in lines:
                data = line.split()
                try:
                    assert len(data) in (2, 3, 4, 5), (
                        "Only 2-4 indices as ints + optional target as a float"
                    )

                    if "." in data[-1]:
                        auto_value = False
                        value = float(data.pop(-1))
                        initial_constr = Constraint([int(i) for i in data], value=value)

                    else:
                        auto_value = True
                        initial_constr = Constraint([int(i) for i in data])

                    # add to dict of constraints, tweaking for each filename if needed
                    for filename in self.filenames:
                        constraint = deepcopy(initial_constr)
                        mol = self.mols[filename]

                        if self.smarts_string is not None:
                            constraint = convert_constraint_with_smarts(
                                constraint, mol.coords[0], mol.atomnos, self.smarts_string
                            )

                        if auto_value:
                            constraint.set_auto_value(mol.coords[0])

                        self.constraints[filename].append(constraint)

                    n_constr += 1

                except Exception as e:
                    print(e)

            print(f"--> Read {n_constr} constraints from {self.constraint_file}")

    @property
    def ase_calc(self) -> ASECalculator:
        """Get the appropriate ASE Calculator."""
        return cast("ASECalculator", self.dispatcher.get_ase_calc(self.method, self.solvent))

    def __repr__(self) -> str:
        """Return string representation of object."""
        s = ""
        for attr in (
            "calc",
            "method",
            "solvent",
            "constraint_file",
            "sp",
            "newfile",
            "free_energy",
            "saddle",
            "irc",
            "T_K",
            "C_mol_L",
        ):
            s += f"--> {attr} : {getattr(self, attr)}\n"
        return s


def main() -> None:
    """Standalone optimizer entry point.
    args: iterable of strings of structure filenames.

    """
    # Redirect stdout and stderr to handle encoding errors
    sys.stdout = TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
    )

    sys.stderr = TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filenames",
        help="Input filename(s), in .xyz format",
        action="store",
        nargs="+",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        help="Set options interactively.",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--temperature",
        help="Temperature, in degrees Celsius.",
        action="store",
        required=False,
        default=25,
    )
    parser.add_argument(
        "-c",
        "--calculator",
        help=f"Calculator (default {_defaults['calculator']}).",
        action="store",
        required=False,
        default=_defaults["calculator"],
    )
    parser.add_argument(
        "-m",
        "--method",
        help=f"Method (default {_defaults['method']} for {_defaults['calculator']}).",
        action="store",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--solvent",
        help=f"Solvent (default {_defaults['solvent']}).",
        action="store",
        required=False,
        default=_defaults["solvent"],
    )
    parser.add_argument(
        "-sp",
        "--singlepoint",
        help="Do a single point energy calc, without optimizing.",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-g",
        "--free_energy",
        help="Calculate free energy (G).",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-ts",
        "--saddle",
        help="Optimize to a TS.",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-irc",
        "--irc",
        help="Run an IRC calculation.",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-cfile",
        "--constraint-file",
        help="Uses a constraint file.",
        action="store",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-n",
        "--newfile",
        help="Write optimized structure to a new file (*_opt.xyz).",
        action="store_true",
        required=False,
    )

    args = parser.parse_args()
    if args.method is None:
        args.method = DEFAULT_LEVELS[args.calculator]

    if args.interactive:
        optimizer = inquire_optimizer_options(args.filenames)

    else:
        optimizer = OptimizerOptions(
            filenames=args.filenames,
            T_K=args.temperature + 273.15,
            calc=args.calculator,
            method=args.method,
            solvent=args.solvent,
            auto_charge_and_mult=True,
            sp=args.singlepoint,
            newfile=args.newfile,
            free_energy=args.free_energy,
            saddle=args.saddle,
            irc=args.irc,
            constraint_file=args.constraint_file,
        )

    return standalone_optimize(optimizer)


def inquire_optimizer_options(filenames: Sequence[str]) -> OptimizerOptions:
    """Inquire optimizer options from user input.
    args: iterable of strings of structure filenames.

    """
    choices: list[Choice] = [
        Choice(value=("AIMNET2", "wB97M-D3"), name="AIMNet2/wB97M-D3"),
        Choice(value=("TBLITE", "GFN2-xTB"), name="GFN2-xTB (TBLITE)"),
        # Choice(value=('TBLITE', 'g-xTB'), name='g-xTB (TBLITE)'),
        Choice(value=("XTB", "GFN-FF"), name="GFN-FF (XTB)"),
        Choice(value=("UMA", "OMOL"), name="UMA/OMol25"),
        Choice(value=("ORCA", "B97-3c"), name="ORCA/B97-3c"),
    ]

    calc, method = inquirer.select(  # type: ignore[attr-defined]
        message="Which level of theory would you like to use?:",
        choices=choices,
        default=next(c.value for c in choices if c.value[0] == CALCULATOR),
    ).execute()

    solvents = (
        list(epsilon_dict.keys())
        + list(solvent_synonyms.keys())
        + [Choice(value=None, name="vacuum")]
    )
    solvent = inquirer.fuzzy(  # type: ignore[attr-defined]
        message="Which solvent would you like to use?",
        choices=solvents,
        default=_defaults["solvent"],
        validate=lambda x: (x in solvents) or x is None,
        filter=lambda solvent: solvent_synonyms.get(solvent, solvent),
    ).execute()

    choices = [
        Choice(
            value="auto_charge_and_mult",
            name="Auto charge/mult - Non-singlets will be assumed as doublets.",
            enabled=True,
        ),
        Choice(
            value="constraints",
            name="Constraints      - Manually apply constraints to the optimization.",
        ),
        Choice(value="constraint_file", name="Constraint file  - Load a constraint file."),
        Choice(
            value="sp", name="Single point     - Do a single point energy calc, without optimizing."
        ),
        Choice(
            value="newfile",
            name="Newfile          - Write optimized structure to a new file (*_opt.xyz).",
        ),
        Choice(value="free_energy", name="Free Energy      - Calculate free energy (G)."),
        Choice(value="saddle", name="Saddle Opt.      - Optimize to a TS."),
        Choice(value="irc", name="IRC              - Run an IRC calculation."),
    ]

    options_to_set = inquirer.checkbox(  # type: ignore[attr-defined]
        message="Select options (spacebar to toggle, enter to confirm):",
        choices=choices,
        cycle=False,
        disabled_symbol="⬡",
        enabled_symbol="⬢",
    ).execute()

    if "constraint_file" in options_to_set:
        constraint_file = inquirer.filepath(  # type: ignore[attr-defined]
            message="Select a constraint file:",
            default="./" if op_sys.name == "posix" else "C:\\",
            validate=PathValidator(is_file=True, message="Input is not a file"),
            only_files=True,
        ).execute()

    else:
        constraint_file = None

    if "free_energy" in options_to_set or "saddle" in options_to_set:
        temp_C = inquirer.number(  # type: ignore[attr-defined]
            message="Specify temperature for free energy calculation (°C):",
            default=25,
            float_allowed=True,
            min_allowed=-273.15,
            filter=float,
        ).execute()

        conc = inquirer.number(  # type: ignore[attr-defined]
            message="Specify concentration for free energy calculation (mol/L):",
            default=0.1,
            float_allowed=True,
            min_allowed=0.0,
            filter=float,
        ).execute()

    else:
        temp_C = 25
        conc = 0.1

    optimizer = OptimizerOptions(
        filenames=filenames,
        T_K=temp_C + 273.15,
        C_mol_L=conc,
        calc=calc,
        method=method,
        solvent=solvent,
        auto_charge_and_mult="auto_charge_and_mult" in options_to_set,
        sp="sp" in options_to_set,
        newfile="newfile" in options_to_set,
        free_energy="free_energy" in options_to_set,
        saddle="saddle" in options_to_set,
        constraint_file=constraint_file,
    )

    if "constraints" in options_to_set:
        while True:
            data = input(
                'Constrained indices [+ optional distance or "ts", enter to stop]: '
            ).split()

            if not data:
                break

            elif data[-1] == "ts":
                mol = read_xyz(filenames[0])
                i1, i2 = (int(i) for i in data[0:2])
                e1, e2 = mol.atoms[i1], mol.atoms[i2]
                data[-1] = str(get_ts_d_estimate(e1, e2))
                print(f"--> Estimated TS d({i1}-{i2}) = {data[-1]} Å")

            assert len(data) in (2, 3, 4, 5), (
                "Only 2-4 indices as ints + optional target as a float"
            )

            # make initial constraint object
            if "." in data[-1]:
                auto_value = False
                value = float(data.pop(-1))
                initial_constr = Constraint([int(i) for i in data], value=value)
            else:
                auto_value = True
                initial_constr = Constraint([int(i) for i in data])

            # add to dict of constraints, tweaking for each filename if needed
            for filename in optimizer.filenames:
                constraint = deepcopy(initial_constr)
                mol = optimizer.mols[filename]

                if auto_value:
                    constraint.set_auto_value(mol.coords[0])

                optimizer.constraints[filename].append(constraint)

        print(f"Specified {len(optimizer.constraints)} global constraints")

    return optimizer


def standalone_optimize(optimizer: OptimizerOptions) -> None:
    """Standalone optimizer main function.
    args: OptimizerOptions object, iterable of strings of structure filenames.

    """
    if optimizer.free_energy:
        print("--> Requested free energy calculation - performing vibrational analysis")
        from firecode.thermochemistry import ase_vib

    if optimizer.saddle:
        print("--> Requested saddle optimization")
        from firecode.thermochemistry import ase_vib

    if optimizer.sp:
        print("--> Single point calculation requested (no optimization)")

    if optimizer.newfile:
        print("--> Writing optimized structures to new files")

    print(optimizer)

    energies, names_confs = [], []

    # start optimizing
    for i, name in enumerate(optimizer.filenames):
        mol = optimizer.mols[name]
        print()

        # set charge
        charge, mult = optimizer.charge_and_mult_dict[name]

        if optimizer.optimize:
            try:
                # define outname and clear existing
                outname = name if not optimizer.newfile else name[:-4] + "_opt.xyz"
                if optimizer.newfile and (outname in op_sys.listdir()):
                    op_sys.remove(outname)
                write_type = "a" if optimizer.newfile else "w"

                for c_n, coords in enumerate(mol.coords):
                    active_ase_constraints: list[ASEConstraint] = []

                    for constraint in optimizer.constraints[name]:
                        if constraint.type_ == "B":
                            a, b = constraint.indices
                            print(
                                f"CONSTRAIN -> d({a}-{b}) = {round(np.linalg.norm(coords[a] - coords[b]), 3)} A at start of optimization (target is {round(constraint.value, 3)} A)"
                            )

                        elif constraint.type_ == "A":
                            a, b, c = constraint.indices
                            print(
                                f"CONSTRAIN ANGLE -> Angle({a}-{b}-{c}) = {round(point_angle(coords[a], coords[b], coords[c]), 3)}° at start of optimization, target {round(constraint.value, 3)}°"
                            )

                        elif constraint.type_ == "D":
                            a, b, c, d = constraint.indices
                            print(
                                f"CONSTRAIN DIHEDRAL -> Dih({a}-{b}-{c}-{d}) = {round(dihedral(np.array([coords[a], coords[b], coords[c], coords[d]])), 3)}° at start of optimization, target {round(constraint.value, 3)}°"
                            )

                        # convert to ASE constraint and add to list of active
                        active_ase_constraints.append(constraint.ase_constraint)

                    if active_ase_constraints or not optimizer.saddle:
                        action = "Calculating SP energy on" if optimizer.sp else "Optimizing"

                        if optimizer.calc in ("AIMNET2", "UMA") and optimizer.solvent is not None:
                            post = f"+ALPB({optimizer.solvent})"
                        else:
                            post = ""

                        print(
                            f"{action} {name} - {i + 1} of {len(optimizer.filenames)}, conf {c_n + 1} of {len(mol.coords)} ({optimizer.method}/{optimizer.calc}{post}) - CHG={charge} MULT={mult}"
                        )
                        t_start = perf_counter()

                        coords, energy, _ = ase_popt(
                            mol.atoms,
                            coords,
                            method=optimizer.method,
                            ase_calc=optimizer.ase_calc,
                            ase_constraints=active_ase_constraints,
                            charge=charge,
                            mult=mult,
                            traj=name[:-4] + "_traj",
                            logfunction=print,
                            maxiter=1 if optimizer.sp else 750,
                            conv_thr="vtight" if optimizer.free_energy else "tight",
                            solvent=optimizer.solvent,
                            # title='OPT_temp',
                            # debug=True,
                        )

                        elapsed = perf_counter() - t_start

                        if energy is None:
                            print(
                                f"--> ERROR: Optimization of {name} crashed. ({time_to_string(elapsed)})"
                            )

                        elif not optimizer.sp:
                            with open(outname, write_type) as f:
                                write_xyz(mol.atoms, coords, f, title=f"Energy = {energy} kcal/mol")  # type: ignore[arg-type]
                            print(
                                f"{'Appended' if write_type == 'a' else 'Wrote'} optimized structure at {outname} - {time_to_string(elapsed)}\n"
                            )

                    if optimizer.saddle:
                        if optimizer.calc in ("AIMNET2", "UMA") and optimizer.solvent is not None:
                            post = f"+ALPB({optimizer.solvent})"
                        else:
                            post = ""

                        print(
                            f"Optimizing TS for {name} - {i + 1} of {len(optimizer.filenames)}, conf {c_n + 1} of {len(mol.coords)} ({optimizer.method}/{optimizer.calc}{post}) - CHG={charge} MULT={mult}"
                        )
                        t_start = perf_counter()

                        constrained_indices_saddle = [
                            (c.i1, c.i2) for c in active_ase_constraints if type(c) == Spring
                        ]

                        if constrained_indices_saddle:
                            s = "s" if len(constrained_indices_saddle) > 1 else ""
                            i_str = ""
                            for i1, i2 in constrained_indices_saddle:
                                i_str += f"B({i1}-{i2}) "

                            print(f" Biasing v0 with bond vibration{s} [{i_str[:-1]}]")

                        coords, energy, success = ase_saddle(
                            mol.atoms,
                            coords,
                            method=optimizer.method,
                            ase_calc=optimizer.ase_calc,
                            constrained_indices=constrained_indices_saddle,
                            charge=charge,
                            mult=mult,
                            traj=name[:-4] + "_traj",
                            title=f"{name[:-4]}_saddle",
                            logfunction=print,
                            maxiter=750,
                            solvent=optimizer.solvent,
                        )

                        elapsed = perf_counter() - t_start

                        if not success:
                            print(
                                f"--> ERROR: Optimization of {name} crashed. ({time_to_string(elapsed)})"
                            )

                        elif not optimizer.sp:
                            with open(outname, write_type) as f:
                                write_xyz(mol.atoms, coords, f, title=f"Energy = {energy} kcal/mol")  # type: ignore[arg-type]
                            print(
                                f"{'Appended' if write_type == 'a' else 'Wrote'} saddle structure at {outname} - {time_to_string(elapsed)}\n"
                            )

                    if optimizer.free_energy or optimizer.saddle:
                        # sph = (len(constraints) != 0)
                        # print(f'Calculating Free Energy contribution{" (SPH)" if sph else ""} on {name} - {i+1} of {len(names)}, conf {c_n+1} of {len(data.coords)} ({method})')
                        # gcorr = xtb_get_free_energy(coords, data.atomnos, method='GFN-FF', solvent=options["solvent"], charge=options["charge"], sph=sph, grep='Gcorr')
                        # print(f'GCORR: {name}, conf {c_n+1} - {gcorr:.2f} kcal/mol')
                        # energy += gcorr

                        print(
                            f"Performing vibrational analysis on {name} - {i + 1} of {len(optimizer.filenames)}, conf {c_n + 1} of {len(mol.coords)} ({optimizer.method})"
                        )
                        t_start = perf_counter()

                        freqs, gcorr = ase_vib(
                            mol.atoms,
                            coords,
                            ase_calc=optimizer.ase_calc,
                            charge=charge,
                            mult=mult,
                            T_K=optimizer.T_K,
                            solvent=optimizer.solvent,
                            add_alpb_solvation=optimizer.calc in ("AIMNET2", "UMA"),
                            C_mol_L=optimizer.C_mol_L,
                            title=f"{name[:-4]}",
                            tighten_opt_before_vib=(not optimizer.saddle),
                        )

                        energy += gcorr
                        num_neg = np.count_nonzero(freqs < 0.0)
                        elapsed = perf_counter() - t_start
                        print(
                            f"Calculated vibrational frequencies ({num_neg} negative) in {time_to_string(elapsed)}\n"
                        )

                    energies.append(energy)
                    names_confs.append(name[:-4] + f"_conf{c_n + 1}")

            except Exception as e:
                print("--> ", name, " - ", e)
                raise (e)

            if optimizer.constraints[name]:
                print("Constraints: final values")

                for constraint in optimizer.constraints[name]:
                    if constraint.type_ == "B":
                        a, b = constraint.indices
                        final_value = float(np.linalg.norm(coords[a] - coords[b]))
                        uom = " Å"

                    elif constraint.type_ == "A":
                        a, b, c = constraint.indices
                        final_value = point_angle(coords[a], coords[b], coords[c])
                        uom = "°"

                    elif constraint.type_ == "D":
                        a, b, c, d = constraint.indices
                        final_value = dihedral(
                            np.array([coords[a], coords[b], coords[c], coords[d]])
                        )
                        uom = "°"

                    indices_string = "-".join([str(i) for i in constraint.indices])
                    print(
                        f"CONSTRAIN -> {constraint.type_}({indices_string}) = {round(final_value, 3)}{uom}"
                    )

                    # revert original indices for the next molecule
                    if optimizer.smarts_string is not None:
                        constraint.indices = constraint.old_indices  # type: ignore[attr-defined]

                print()

        if optimizer.irc:
            from firecode.ase_manipulations import ase_irc

            for c_n, coords in enumerate(mol.coords):
                _, _ = ase_irc(
                    mol.atoms,
                    coords,
                    method=optimizer.method,
                    ase_calc=optimizer.ase_calc,
                    charge=charge,
                    mult=mult,
                    traj=name[:-4] + "_traj",
                    title=f"{name[:-4]}_conf{c_n}",
                    logfunction=print,
                    solvent=optimizer.solvent,
                )

    if None not in energies:
        if len(names_confs) > 1:
            min_e = min(energies)
        else:
            min_e = 0

        ### NICER TABLE PRINTOUT

        from prettytable import PrettyTable

        table = PrettyTable()
        energy_type = "Free Energy G(Eh)" if optimizer.free_energy else "Potential Energy E(Eh)"
        letter = "G" if optimizer.free_energy else "E"
        table.field_names = ["#", "Filename", energy_type, f"Rel. {letter} (kcal/mol)"]

        print()

        for i, (nc, energy) in enumerate(zip(names_confs, energies)):
            table.add_row([i + 1, nc, energy / EH_TO_KCAL, round(energy - min_e, 2)])

        print(table.get_string())


def multiplicity_check(atomnos: Array1D_int, charge: int, multiplicity: int = 1) -> bool:
    """Returns True if the multiplicity and the nuber of
    electrons are one odd and one even, and vice versa.

    """
    electrons = sum(atomnos) - charge

    return (multiplicity % 2) != (electrons % 2)
