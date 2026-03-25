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
import re
import shutil
import sys
from pathlib import Path
from shutil import rmtree
from subprocess import getoutput
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from prism_pruner.algebra import rot_mat_from_pointer
from prism_pruner.graph_manipulations import d_min_bond, graphize
from prism_pruner.rmsd import rmsd_and_max
from scipy.spatial.distance import cdist

from firecode.algebra import count_clashes
from firecode.ensemble import Ensemble
from firecode.errors import TriangleError
from firecode.typing_ import Array1D_float, Array1D_int, Array1D_str, Array2D_float, Array3D_float
from firecode.units import EH_TO_KCAL

if TYPE_CHECKING:
    from io import TextIOWrapper

    from networkx import Graph

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class suppress_stdout_stderr(object):
    """A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self) -> None:
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self) -> None:
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_: object) -> None:
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class HiddenPrints:
    def __enter__(self) -> None:
        self._original_stdout: Any = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]
    ) -> None:
        sys.stdout.close()
        sys.stdout = self._original_stdout


def clean_directory(
    to_remove: Iterable[str] | None = None,
    to_remove_startswith: Iterable[str] | None = None,
    to_remove_endswith: Iterable[str] | None = None,
    to_remove_contains: Iterable[str] | None = None,
) -> None:
    """Cleans the current directory from temporary files created during a run."""
    if to_remove is not None:
        for name in to_remove:
            try:
                os.remove(name)
            except IsADirectoryError:
                rmtree(os.path.join(os.getcwd(), name))
            except FileNotFoundError:
                pass

    to_remove_startswith = to_remove_startswith or []
    to_remove_endswith = to_remove_endswith or []
    to_remove_contains = to_remove_contains or []
    for f in os.listdir():
        if (
            f.startswith(("temp", *to_remove_startswith))
            or f.endswith(("temp", *to_remove_endswith))
            or any([s in f for s in to_remove_contains])
        ):
            try:
                os.remove(f)
            except IsADirectoryError:
                rmtree(os.path.join(os.getcwd(), f))
            except FileNotFoundError:
                pass


def write_xyz(
    atoms: Array1D_str, coords: Array2D_float, output: TextIOWrapper, title: str = "temp"
) -> None:
    """Output is of _io.TextIOWrapper type"""
    assert atoms.shape[0] == coords.shape[0]
    assert coords.shape[1] == 3
    string = ""
    string += str(len(coords))
    string += f"\n{title}\n"
    for atom, coord in zip(atoms, coords):
        string += "%s     % .6f % .6f % .6f\n" % (atom, coord[0], coord[1], coord[2])
    output.write(string)


def read_xyz(filename: str) -> Ensemble:
    """FIRECODE's xyz reader."""
    return Ensemble.from_xyz(filename)


def read_xyz_energies(
    filename: str,
    verbose: bool = True,
    logfunction: Callable[[str], None] | None = print,
) -> list[float] | None:
    """Read energies from a .xyz file. Returns None or a list of floats (in Hartrees)."""
    energies = None

    # get lines right after the number of atom, which should contain the energy
    comment_lines = getoutput(
        f'grep -A1 "^[[:space:]]*[0-9]\\+$" {filename} | grep -v "^[[:space:]]*[0-9]\\+$" | grep -v "^--$"'
    ).split("\n")

    if len(comment_lines[0].split()) == 1:
        if set(comment_lines[0].split()[0]).issubset("0123456789.-"):
            # only one energy found with no UOM, assume it's in Eh
            energies = [float(e.split()[0].strip()) for e in comment_lines]

            if verbose and logfunction is not None:
                print(
                    f"--> Read {len(energies)} energies from {filename} (single number, no UOM: assuming Eh units)."
                )

        elif verbose and logfunction is not None:
            logfunction(f"--> Could not parse energies for {filename} - skipping.")

    else:
        # multiple energies found, parse units
        hartree_matches = re.findall(r"-*\d+.\d+\sEH", comment_lines[0].upper())
        kcal_matches = re.findall(r"-*\d+.\d+\sKCAL/MOL", comment_lines[0].upper())
        number_matches = re.findall(r"-*\d+.\d+", comment_lines[0])

        if hartree_matches:
            energies = [
                float(re.findall(r"-*\d+.\d+\sEH", e.upper())[0].split()[0].strip())
                for e in comment_lines
            ]
            if verbose and logfunction is not None:
                logfunction(
                    f"--> Read {len(comment_lines)} energies from {filename} (first number followed by Eh units)."
                )

        elif kcal_matches:
            energies = [
                float(re.findall(r"-*\d+.\d+\sKCAL/MOL", e.upper())[0].split()[0].strip())
                / EH_TO_KCAL
                for e in comment_lines
            ]
            if verbose and logfunction is not None:
                logfunction(
                    f"--> Read {len(comment_lines)} energies from {filename} (first number followed by kcal/mol units)."
                )

        # last resort, parse the first thing that looks like an energy and assume it's in Eh
        elif number_matches:
            energies = [float(re.findall(r"-*\d+.\d+", e)[0].strip()) for e in comment_lines]
            if verbose and logfunction is not None:
                logfunction(
                    f"--> Read {len(comment_lines)} energies from {filename} (first number, no UOM: assuming Eh units)."
                )

        elif verbose and logfunction is not None:
            logfunction(f"--> Could not parse energies for {filename} - skipping.")

    return energies


def pretty_num(n: float) -> str:
    if n < 1e3:
        return str(n)
    if n < 1e6:
        return str(round(n / 1e3, 2)) + " k"
    return str(round(n / 1e6, 2)) + " M"


def loadbar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 50,
    fill: str = "#",
) -> None:
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    if iteration == total:
        print()


def cartesian_product(*arrays: Iterable[Any]) -> NDArray[Any]:
    arrays_converted = [np.asarray(arr) for arr in arrays]
    return np.stack(np.meshgrid(*arrays_converted), -1).reshape(-1, len(arrays))


def rotation_matrix_from_vectors(vec1: Array1D_float, vec2: Array1D_float) -> Array2D_float:
    """Find the rotation matrix that aligns vec1 to vec2.
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.

    """
    assert vec1.shape == (3,)
    assert vec2.shape == (3,)

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if np.linalg.norm(v) != 0:
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix: Array2D_float = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
        return rotation_matrix

    # if the cross product is zero, then vecs must be parallel or perpendicular
    if np.linalg.norm(a + b) == 0:
        pointer = np.array([0, 0, 1])
        rot_mat: Array2D_float = rot_mat_from_pointer(pointer, 180)
        return rot_mat

    return np.eye(3)


def polygonize(lengths: Array1D_float) -> NDArray[Any]:
    """Returns coordinates for the polygon vertices used in cyclical TS construction,
    as a list of vector couples specifying starting and ending point of each pivot
    vector. For bimolecular TSs, returns vertices for the centered superposition of
    two segments. For trimolecular TSs, returns triangle vertices.

    :params vertices: list of floats, used as polygon side lenghts.
    :return vertices_out: list of vectors couples (start, end)
    """
    assert len(lengths) in (2, 3)

    arr = np.zeros((len(lengths), 2, 3))

    if len(lengths) == 2:
        arr[0, 0] = np.array([-lengths[0] / 2, 0, 0])
        arr[0, 1] = np.array([+lengths[0] / 2, 0, 0])
        arr[1, 0] = np.array([-lengths[1] / 2, 0, 0])
        arr[1, 1] = np.array([+lengths[1] / 2, 0, 0])

        vertices_out = np.vstack(([arr], [arr]))
        vertices_out[1, 1] *= -1

    else:
        if not all([lengths[i] < lengths[i - 1] + lengths[i - 2] for i in (0, 1, 2)]):
            raise TriangleError(f"Impossible to build a triangle with sides {lengths}")
            # check that we can build a triangle with the specified vectors

        arr[0, 1] = np.array([lengths[0], 0, 0])
        arr[1, 0] = np.array([lengths[0], 0, 0])

        a = np.power(lengths[0], 2)
        b = np.power(lengths[1], 2)
        c = np.power(lengths[2], 2)
        x = (a - b + c) / (2 * a**0.5)
        y = (c - x**2) ** 0.5

        arr[1, 1] = np.array([x, y, 0])
        arr[2, 0] = np.array([x, y, 0])

        vertices_out = np.vstack(([arr], [arr], [arr], [arr], [arr], [arr], [arr], [arr]))

        swaps = [
            (1, 2),
            (2, 1),
            (3, 1),
            (3, 2),
            (4, 0),
            (5, 0),
            (5, 1),
            (6, 0),
            (6, 2),
            (7, 0),
            (7, 1),
            (7, 2),
        ]

        for t, v in swaps:
            # triangle, vector couples to be swapped
            vertices_out[t, v][[0, 1]] = vertices_out[t, v][[1, 0]]

    return vertices_out


def get_scan_peak_index(energies: list[float], max_thr: float = 50.0, min_thr: float = 0.1) -> int:
    """Returns the index of the energies iterable corresponding to the most prominent peak."""
    _l = len(energies)
    peaks = [
        i
        for i in range(_l)
        if (
            energies[i - 1] < energies[i] >= energies[(i + 1) % _l]
            and max_thr > energies[i] > min_thr
            # discard peaks that are too small or too big
        )
    ]

    if not peaks:
        return energies.index(max(energies))
    # if no peaks are present, return the highest

    if len(peaks) == 1:
        return peaks[0]
    # if one is present, return that

    peaks_nrg = [energies[i] for i in peaks]
    return energies.index(max(peaks_nrg))
    # if more than one, return the highest


def molecule_check(
    atoms: Array1D_str, old_coords: Array2D_float, new_coords: Array2D_float, max_newbonds: int = 0
) -> bool:
    """Checks if two molecules have the same bonds between the same atomic indices."""
    old_bonds = {(a, b) for a, b in list(graphize(atoms, old_coords).edges) if a != b}
    new_bonds = {(a, b) for a, b in list(graphize(atoms, new_coords).edges) if a != b}

    delta_bonds = (old_bonds | new_bonds) - (old_bonds & new_bonds)

    if len(delta_bonds) > max_newbonds:
        return False

    return True


def scramble_check(
    embedded_atoms: Array1D_str,
    embedded_structure: Array2D_float,
    excluded_atoms: Iterable[int],
    mols_graphs: Iterable[Graph],
    max_newbonds: int = 0,
    logfunction: Callable[[str], None] | None = None,
    title: str | None = None,
) -> bool:
    """Check if a multimolecular arrangement has scrambled during some optimization
    steps. If more than a given number of bonds changed (formed or broke) the
    structure is considered scrambled, and the method returns False.
    """
    assert len(embedded_structure) == sum([len(graph.nodes) for graph in mols_graphs])

    bonds = set()
    for i, graph in enumerate(mols_graphs):
        pos = sum([len(other_graph.nodes) for j, other_graph in enumerate(mols_graphs) if j < i])

        for bond in [tuple(sorted((a + pos, b + pos))) for a, b in list(graph.edges) if a != b]:
            bonds.add(bond)
    # creating bond set containing all bonds present in the desired molecular assembly

    new_bonds = {
        tuple(sorted((a, b)))
        for a, b in list(graphize(embedded_atoms, embedded_structure).edges)
        if a != b
    }
    delta_bonds = (bonds | new_bonds) - (bonds & new_bonds)
    # delta_bonds -= {tuple(sorted(pair)) for pair in constrained_indices}

    for bond in delta_bonds.copy():
        for a in excluded_atoms:
            if a in bond:
                delta_bonds -= {bond}
    # removing bonds involving constrained atoms: they are not counted as scrambled bonds

    if len(delta_bonds) > max_newbonds:
        if logfunction is not None:
            logfunction(
                f"{title}, scramble_check - found {len(delta_bonds)} extra bonds: {delta_bonds}"
            )
        return False

    return True


def auto_newline(string: str, max_line_len: int = 50, padding: int = 2) -> str:
    """Inserts newline chars into string to limit its length."""
    string = str(string)

    out = [" " * padding]
    line_len = 0
    for word in string.split():
        out.append(word)
        line_len += len(word) + 1

        if line_len >= max_line_len:
            out.append("\n" + " " * padding)
            line_len = 0

    return " ".join(out)


def str_to_var(
    string: str, enforced_type: Callable[[str], bool | float | int] | None = None
) -> bool | float | int | str:
    """Cast a string into the most appropriate type."""
    if enforced_type is not None:
        return enforced_type(string)

    string_lower = string.lower()
    if string_lower in ("true", "yes", "on", "1"):
        return True
    if string_lower in ("false", "no", "off", "0"):
        return False

    # Check for int, including negative numbers - max 1 neg. sign
    if string.replace("-", "", 1).isdigit():
        return int(string)

    # Check for float, including negative numbers - max 1 dot/neg. sign
    if string.count(".") <= 1 and (string.replace(".", "", 1).replace("-", "", 1).isdigit()):
        return float(string)

    return string


def charge_to_str(charge: int) -> str:
    """Return a string representation of molecular charge."""
    if charge == 0:
        return ""
    elif charge > 0:
        return "+" * charge
    else:
        return "-" * abs(charge)


def timing_wrapper(function: Callable[P, R], *args: Any, **kwargs: Any) -> tuple[R, float]:
    """Generic function wrapper that appends the
    execution time in seconds at the end of return.
    """
    start_time = perf_counter()
    func_return = function(*args, **kwargs)
    elapsed = perf_counter() - start_time

    return func_return, elapsed


def timing_wrapper_with_payload(
    function: Callable[P, R], *args: Any, payload: T, **kwargs: Any
) -> tuple[R, T, float]:
    """Generic function wrapper that appends a specific payload and the
    execution time in seconds at the end of return.
    """
    start_time = perf_counter()
    func_return = function(*args, **kwargs)
    elapsed = perf_counter() - start_time
    return func_return, payload, elapsed


def timing_decorator() -> Callable[[Callable[P, R]], Callable[P, tuple[R, float]]]:
    """Generic function wrapper that appends the
    execution time in seconds at the end of return.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, tuple[R, float]]:
        def wrapper(*args: Any, **kwargs: Any) -> tuple[R, float]:
            return timing_wrapper(func, *args, **kwargs)

        return wrapper

    return decorator


def timing_decorator_with_payload() -> Callable[
    [Callable[P, R]], Callable[..., tuple[R, T, float]]
]:
    """Generic function decorator that appends a specific payload and the
    execution time in seconds at the end of return.
    """

    def decorator(func: Callable[P, R]) -> Callable[..., tuple[R, T, float]]:
        def wrapper(*args: Any, payload: T, **kwargs: Any) -> tuple[R, T, float]:
            return timing_wrapper_with_payload(func, *args, payload=payload, **kwargs)

        return wrapper

    return decorator


def saturation_check(atoms: Array1D_str, charge: int = 0) -> bool:
    """Checks that the molecule saturation looks reasonable given the assigned charge."""
    transition_metals = [
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
    ]

    # if we have any transition metal, it's hard to tell
    # if the structure looks ok: in this case we assume it is.
    organometallic = any([el in transition_metals for el in atoms])

    if organometallic:
        return True

    odd_valent = [  # 1 valent
        "H",
        "Li",
        "Na",
        "K",
        "Rb",
        "Cs",
        "F",
        "Cl",
        "Br",
        "I",
        "At",
        # 3/5 valent
        "N",
        "P",
        "As",
        "Sb",
        "Bi",
        "B",
        "Al",
        "Ga",
        "In",
        "Tl",
    ]

    n_odd_valent = sum([1 for a in atoms if a in odd_valent])
    looks_ok = ((n_odd_valent + charge) / 2) % 1 < 0.001

    return looks_ok


def rmsd_similarity(ref: Array2D_float, structures: Array3D_float, rmsd_thr: float = 0.5) -> bool:
    """Simple, RMSD similarity eval function."""
    # iterate over target structures
    for structure in structures:
        # compute RMSD and max deviation
        rmsd_value, maxdev_value = rmsd_and_max(ref, structure)

        if rmsd_value < rmsd_thr and maxdev_value < 2 * rmsd_thr:
            return True

    return False


def compenetration_check(
    coords: Array2D_float,
    ids: Sequence[int] | Array1D_int | None = None,
    thresh: float = 1.5,
    max_clashes: int = 0,
) -> bool:
    """coords: 3D molecule coordinates
    ids: 1D array with the number of atoms for each
    molecule (contiguous fragments in array)
    thresh: threshold value for when two atoms are considered clashing
    max_clashes: maximum number of clashes to pass a structure
    returns True if the molecule shows less than max_clashes

    """
    if ids is None:
        return count_clashes(coords) <= max_clashes

    if len(ids) == 2:
        # Bimolecular

        m1 = coords[0 : ids[0]]
        m2 = coords[ids[0] :]
        # fragment identification by length (contiguous)

        return int(np.count_nonzero(cdist(m2, m1) < thresh)) <= max_clashes

    # if len(ids) == 3:

    clashes = 0
    # max_clashes clashes is good, max_clashes + 1 is not

    m1 = coords[0 : ids[0]]
    m2 = coords[ids[0] : ids[0] + ids[1]]
    m3 = coords[ids[0] + ids[1] :]
    # fragment identification by length (contiguous)

    clashes += int(np.count_nonzero(cdist(m2, m1) <= thresh))
    if clashes > max_clashes:
        return False

    clashes += int(np.count_nonzero(cdist(m3, m2) <= thresh))
    if clashes > max_clashes:
        return False

    clashes += int(np.count_nonzero(cdist(m1, m3) <= thresh))
    if clashes > max_clashes:
        return False

    return True


def get_ts_d_estimate(
    e1: str,
    e2: str,
    factor: float = 1.35,
) -> float:
    """Returns an estimate for the distance between two
    specific elements in a transition state, by multipling
    the sum of covalent radii for a constant.

    """
    return cast("float", d_min_bond(e1, e2, factor=factor))


class NewFolderContext:
    """Context manager: creates a new directory and moves into it on entry.

    On exit, moves out of the directory and deletes it if instructed to do so.

    """

    def __init__(
        self, new_folder_name: str, delete_after: bool = True, overwrite_if_exists: bool = True
    ) -> None:

        self.new_folder_name = os.path.join(os.getcwd(), new_folder_name)
        self.delete_after = delete_after
        self.overwrite_if_exists = overwrite_if_exists

    def __enter__(self) -> None:
        if self.overwrite_if_exists:
            shutil.rmtree(self.new_folder_name, ignore_errors=True)

        if not os.path.isdir(self.new_folder_name):
            # create working folder and cd into it
            new_dir = Path(self.new_folder_name)
            new_dir.mkdir()

        os.chdir(self.new_folder_name)

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        # get out of working folder
        os.chdir(os.path.dirname(os.getcwd()))

        # only delete if instructed to
        # and no unhandled exception occurred
        if self.delete_after and exc_type is None:
            shutil.rmtree(self.new_folder_name, ignore_errors=True)


class FolderContext:
    """Context manager: works in the specified directory and moves back after."""

    def __init__(self, target_folder: str) -> None:
        self.target_folder = os.path.join(os.getcwd(), target_folder)
        self.initial_folder = os.getcwd()

    def __enter__(self) -> None:
        """Move into folder on entry."""
        if os.path.isdir(self.target_folder):
            os.chdir(self.target_folder)

        else:
            raise NotADirectoryError(self.target_folder)

    def __exit__(self, *args: object) -> None:
        """Get out of working folder on exit."""
        os.chdir(self.initial_folder)
