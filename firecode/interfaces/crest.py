"""FIRECODE interface to CREST 2 and 3"""

from __future__ import annotations

from shutil import which
from subprocess import STDOUT, check_call, getoutput
from typing import Any, Sequence

import numpy as np

from firecode.context_managers import NewFolderContext, env_override
from firecode.solvents import to_xtb_solvents
from firecode.typing_ import Array1D_str, Array2D_float, Array3D_float
from firecode.utils import clean_directory, read_xyz, write_xyz


def crest2_mtd_search(
    atoms: Array1D_str,
    coords: Array2D_float,
    constrained_indices: Sequence[Sequence[int]] | None = None,
    constrained_distances: Sequence[float | None] | None = None,
    constrained_dihedrals_indices: Sequence[Sequence[int]] | None = None,
    constrained_dihedrals_values: Sequence[float | None] | None = None,
    constrained_angles_indices: Sequence[Sequence[int]] | None = None,
    constrained_angles_values: Sequence[float | None] | None = None,
    method: str = "GFN2-XTB//GFN-FF",
    solvent: str | None = "CH2Cl2",
    charge: int = 0,
    kcal: float | None = None,
    ncimode: bool = False,
    title: str = "temp",
    procs: int = 4,
    threads: int = 1,
) -> Array3D_float:
    """This function runs a crest metadynamic conformational search and
    returns its output.

    coords: array of shape (n,3) with cartesian coordinates for atoms.

    atoms: array of strings for elements.

    constrained_indices: array of shape (n,2), with the indices
    of atomic pairs to be constrained.

    constrained_distances: optional, target distances for the specified
    distance constraints.

    constrained_dihedrals: quadruplets of atomic indices to constrain.

    constrained_dih_angles: target dihedral angles for the dihedral constraints.

    method: string, specifying the theory level to be used.

    solvent: solvent to be used in the calculation (ALPB model).

    charge: charge to be used in the calculation.

    title: string, used as a file name and job title for the mopac input file.

    procs: number of cores to be used for the calculation.

    threads: number of parallel threads to be used by the process.

    """
    with NewFolderContext(title, delete_after=False):
        if constrained_indices is not None:
            if len(constrained_indices) == 0:
                constrained_indices = None

        if constrained_distances is not None:
            if len(constrained_distances) == 0:
                constrained_distances = None

        with open(f"{title}.xyz", "w") as f:
            write_xyz(atoms, coords, f, title=title)

        s = "$opt\n   "

        if constrained_indices is not None and constrained_distances is not None:
            s += "\n$constrain\n"
            # s += '   atoms: '
            # for i in np.unique(np.array(constrained_indices).flatten()):
            #     s += f"{i+1},"

            constrained_distances = constrained_distances or [None for _ in constrained_indices]

            for (c1, c2), cd in zip(constrained_indices, constrained_distances):
                cd_ = "auto" if cd is None else str(round(cd, 3))
                s += f"    distance: {c1 + 1}, {c2 + 1}, {cd_}\n"

        if constrained_angles_indices is not None:
            constrained_angles_values = constrained_angles_values or [
                None for _ in constrained_angles_indices
            ]
            s += "\n$constrain\n" if constrained_indices is None else ""
            for (a, b, c), angle in zip(constrained_angles_indices, constrained_angles_values):
                angle_ = "auto" if angle is None else str(round(angle, 3))
                s += f"   angle: {a + 1}, {b + 1}, {c + 1}, {angle_}\n"

        if constrained_dihedrals_indices is not None:
            constrained_dihedrals_values = constrained_dihedrals_values or [
                None for _ in constrained_dihedrals_indices
            ]
            s += "\n$constrain\n" if constrained_indices is None else ""
            for (a, b, c, d), angle in zip(
                constrained_dihedrals_indices, constrained_dihedrals_values
            ):
                angle_ = "auto" if angle is None else str(round(angle, 3))
                s += f"   dihedral: {a + 1}, {b + 1}, {c + 1}, {d + 1}, {angle_}\n"

        s += "\n$metadyn\n  atoms: "

        constrained_atoms_cumulative = set()
        if constrained_indices is not None:
            for c1, c2 in constrained_indices:
                constrained_atoms_cumulative.add(c1)
                constrained_atoms_cumulative.add(c2)

        if constrained_angles_indices is not None:
            for c1, c2, c3 in constrained_angles_indices:
                constrained_atoms_cumulative.add(c1)
                constrained_atoms_cumulative.add(c2)
                constrained_atoms_cumulative.add(c3)

        if constrained_dihedrals_indices is not None:
            for c1, c2, c3, c4 in constrained_dihedrals_indices:
                constrained_atoms_cumulative.add(c1)
                constrained_atoms_cumulative.add(c2)
                constrained_atoms_cumulative.add(c3)
                constrained_atoms_cumulative.add(c4)

        # write atoms that need to be moved during metadynamics (all but constrained)
        active_ids = np.array(
            [i + 1 for i, _ in enumerate(atoms) if i not in constrained_atoms_cumulative]
        )

        while len(active_ids) > 2:
            i = next(
                (i for i, _ in enumerate(active_ids[:-2]) if active_ids[i + 1] - active_ids[i] > 1),
                len(active_ids) - 1,
            )
            if active_ids[0] == active_ids[i]:
                s += f"{active_ids[0]},"
            else:
                s += f"{active_ids[0]}-{active_ids[i]},"
            active_ids = active_ids[i + 1 :]

        # remove final comma
        s = s[:-1]
        s += "\n$end"

        s = "".join(s)
        with open(f"{title}.inp", "w") as f:
            f.write(s)

        # avoid restarting the run
        flags = "--norestart"

        # add method flag
        if method.upper() in ("GFN-FF", "GFNFF"):
            flags += " --gfnff"
            # declaring the use of FF instead of semiempirical

        elif method.upper() in ("GFN2-XTB", "GFN2"):
            flags += " --gfn2"

        elif method.upper() in ("GFN2-XTB//GFN-FF", "GFN2//GFNFF"):
            flags += " --gfn2//gfnff"

        # adding other options
        if charge != 0:
            flags += f" --chrg {charge}"

        if procs is not None:
            flags += f" -P {procs}"

        if threads is not None:
            flags += f" -T {threads}"

        if solvent is not None:
            if solvent == "methanol":
                flags += " --gbsa methanol"

            else:
                flags += f" --alpb {to_xtb_solvents.get(solvent, solvent)}"

        if kcal is None:
            kcal = 10
        flags += f" --ewin {kcal}"

        if ncimode:
            flags += " --nci"

        flags += " --noreftopo"

        try:
            with open(f"{title}.out", "w") as f:
                check_call(
                    f"crest {title}.xyz --cinp {title}.inp {flags}".split(), stdout=f, stderr=STDOUT
                )

        except KeyboardInterrupt:
            raise KeyboardInterrupt("KeyboardInterrupt requested by user. Quitting.")

        new_coords = read_xyz("crest_conformers.xyz").coords

        # clean_directory((f'{title}.inp', f'{title}.xyz', f"{title}.out"))

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

        return new_coords


def crest3_mtd_search(
    atoms: Array1D_str,
    coords: Array2D_float,
    constrained_indices: Sequence[Sequence[int]] | None = None,
    constrained_distances: Sequence[float | None] | None = None,
    constrained_dihedrals_indices: Sequence[Sequence[int]] | None = None,
    constrained_dihedrals_values: Sequence[float | None] | None = None,
    constrained_angles_indices: Sequence[Sequence[int]] | None = None,
    constrained_angles_values: Sequence[float | None] | None = None,
    method: str = "GFN2-XTB//GFN-FF",
    solvent: str | None = "CH2Cl2",
    charge: int = 0,
    kcal: float | None = None,
    ncimode: bool = False,
    title: str = "temp",
    threads: int = 1,
) -> Array3D_float:
    """Runs a CREST 3 metadynamic conformational search and returns its output.
    Generates a TOML input file (CREST >= 3.0) instead of the legacy .inp format.

    coords: array of shape (n,3) with cartesian coordinates for atoms.

    atoms: array of strings for elements.

    constrained_indices: array of shape (n,2), with the indices
        of atomic pairs to be constrained.

    constrained_distances: optional, target distances for the specified
        distance constraints (None means "auto").

    constrained_dihedrals_indices: quadruplets of atomic indices to constrain.

    constrained_dihedrals_values: target dihedral angles for dihedral constraints.

    constrained_angles_indices: triplets of atomic indices to constrain.

    constrained_angles_values: target angles for angle constraints.

    method: string, specifying the theory level to be used.

    solvent: solvent to be used in the calculation (ALPB/GBSA model).

    charge: charge to be used in the calculation.

    kcal: energy window for conformer ensemble (default 10 kcal/mol).

    ncimode: use NCI-MTD runtype (wall potential sampling).

    title: string, used as a file name and job title.

    threads: CREST 3 'threads'
    """
    with NewFolderContext(title, delete_after=False):
        # --- Normalize empty inputs ---
        if constrained_indices is not None and len(constrained_indices) == 0:
            constrained_indices = None
        if constrained_distances is not None and len(constrained_distances) == 0:
            constrained_distances = None

        # --- Write coordinate file ---
        with open(f"{title}.xyz", "w") as f:
            write_xyz(atoms, coords, f, title=title)

        # ---------------------------------------------------------------
        # Build CREST 3 TOML input
        # ---------------------------------------------------------------
        lines: list[str] = [f"# CREST 3 input file - {title}"]

        # Top-level keys
        lines.append(f"input = '{title}.xyz'")
        lines.append(f"runtype = '{'nci-mtd' if ncimode else 'imtd-gc'}'")
        lines.append("topo = false")  # replaces --noreftopo
        lines.append(f"threads = {threads}")
        lines.append("")

        # --- Resolve method(s) ---
        method_upper = method.upper()
        dual_level = method_upper in ("GFN2-XTB//GFN-FF", "GFN2//GFNFF")

        def _method_key(m: str) -> str:
            """Map old-style method names to CREST 3 method strings."""
            m = m.upper()
            if m in ("GFN-FF", "GFNFF"):
                return "gfnff"
            elif m in ("GFN2-XTB", "GFN2"):
                return "gfn2"
            elif m in ("GFN1-XTB", "GFN1"):
                return "gfn1"
            else:
                return m.lower()

        def _level_block(method_key: str) -> list[str]:
            """Emit a [[calculation.level]] block."""
            blk = ["[[calculation.level]]", f"method = '{method_key}'"]
            if charge != 0:
                blk.append(f"chrg = {charge}")
            if solvent is not None:
                if solvent.lower() == "methanol":
                    blk.append("gbsa = 'methanol'")  # ALPB unavailable for MeOH
                else:
                    xtb_solvent = to_xtb_solvents.get(solvent, solvent)
                    blk.append(f"alpb = '{xtb_solvent}'")
            return blk

        if dual_level:
            # GFN-FF drives the metadynamics; GFN2 is used for optimizations.
            # Equivalent to old --gfn2//gfnff flag.
            lines += _level_block("gfnff")
            lines.append("")
            lines += _level_block("gfn2")
            lines.append("")
            lines.append("[dynamics]")
            lines.append("active = [1]")  # level 1 (GFN-FF) active for dynamics
            lines.append("")
        else:
            lines += _level_block(_method_key(method_upper))
            lines.append("")

        # --- Constraints ---
        # CREST 3 uses [[calculation.constraint]] blocks; each has:
        #   type  = 'bond' | 'angle' | 'dihedral'
        #   atoms = [1-based indices]
        #   val   = target value (optional; omit for "auto")

        if constrained_indices is not None and constrained_distances is not None:
            constrained_distances = list(constrained_distances) or [None] * len(constrained_indices)
            for (c1, c2), cd in zip(constrained_indices, constrained_distances):
                lines += [
                    "[[calculation.constraint]]",
                    "type = 'bond'",
                    f"atoms = [{c1 + 1}, {c2 + 1}]",
                ]
                if cd is not None:
                    lines.append(f"val = {round(cd, 3)}")
                lines.append("")

        if constrained_angles_indices is not None:
            constrained_angles_values = list(
                constrained_angles_values or [None] * len(constrained_angles_indices)
            )
            for (a, b, c), angle in zip(constrained_angles_indices, constrained_angles_values):
                lines += [
                    "[[calculation.constraint]]",
                    "type = 'angle'",
                    f"atoms = [{a + 1}, {b + 1}, {c + 1}]",
                ]
                if angle is not None:
                    lines.append(f"val = {round(angle, 3)}")
                lines.append("")

        if constrained_dihedrals_indices is not None:
            constrained_dihedrals_values = list(
                constrained_dihedrals_values or [None] * len(constrained_dihedrals_indices)
            )
            for (a, b, c, d), angle in zip(
                constrained_dihedrals_indices, constrained_dihedrals_values
            ):
                lines += [
                    "[[calculation.constraint]]",
                    "type = 'dihedral'",
                    f"atoms = [{a + 1}, {b + 1}, {c + 1}, {d + 1}]",
                ]
                if angle is not None:
                    lines.append(f"val = {round(angle, 3)}")
                lines.append("")

        # --- Energy window via [cregen] ---
        if kcal is None:
            kcal = 10
        lines += ["[cregen]", f"ewin = {kcal}", ""]

        # Write TOML file
        with open(f"{title}.toml", "w") as f:
            f.write("\n".join(lines))

        # ---------------------------------------------------------------
        # Run CREST 3
        # CREST 3 accepts the .toml file directly as the only argument.
        # ---------------------------------------------------------------
        try:
            with env_override(
                OMP_NUM_THREADS=threads,
                MKL_NUM_THREADS=threads,
                # needed to suppress nasty printouts
                OPENBLAS_NUM_THREADS=1,
            ):
                with open(f"{title}.out", "w") as f:
                    check_call(["crest", f"{title}.toml", "--noreftopo"], stdout=f, stderr=STDOUT)
        except KeyboardInterrupt:
            raise KeyboardInterrupt("KeyboardInterrupt requested by user. Quitting.")

        new_coords = read_xyz("crest_conformers.xyz").coords

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

        return new_coords


def get_crest_version() -> int | None:
    """Returns an integer (2 or 3) representing the version of CREST that is installed."""
    if which("crest") is None:
        return None

    crest_version = int(getoutput("crest --version | grep Version").split()[1].split(".")[0])

    return crest_version


def crest_mtd_search(*args: Any, **kwargs: Any) -> Any:
    """Return 2 or 3"""
    crest_version = get_crest_version()

    match crest_version:
        case 2:
            return crest2_mtd_search(*args, **kwargs)
        case 3:
            return crest3_mtd_search(*args, **kwargs)
        case _:
            raise AssertionError(
                "CREST (version 2 or 3) does not seem to be installed. "
                "Install it with: mamba install -c conda-forge crest=3"
            )
