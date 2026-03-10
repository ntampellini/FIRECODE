# coding=utf-8
"""FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
from numpy.linalg import LinAlgError
from prism_pruner.algebra import get_inertia_moments
from prism_pruner.graph_manipulations import graphize
from prism_pruner.rmsd import get_alignment_matrix

from firecode.errors import NoOrbitalError
from firecode.graph_manipulations import is_sigmatropic, is_vicinal
from firecode.pt import pt
from firecode.reactive_atoms_classes import get_atom_type
from firecode.typing_ import Array1D_float, Array1D_int, Array1D_str, Array2D_float, Array3D_float
from firecode.utils import read_xyz

if TYPE_CHECKING:
    from networkx import Graph

    from firecode.ase_manipulations import Constraint
    from firecode.reactive_atoms_classes import RAtom

warnings.simplefilter("ignore", UserWarning)


def align_by_moi(atoms: Array1D_str, structures: Array2D_float) -> Array3D_float:
    """Aligns molecules of a structure array (shape is (n_structures, n_atoms, 3))
    to the first one, based on the the moments of inertia vectors.
    Returns the aligned array.

    """
    reference, *targets = structures

    masses = np.array([pt.mass(el) for el in atoms])

    # center all the structures at the origin
    reference -= np.mean(reference, axis=0)
    for t, target in enumerate(targets):
        targets[t] -= np.mean(target, axis=0)

    # initialize output array
    output = np.zeros(structures.shape)
    output[0] = reference

    # reference vectors
    ref_moi_vecs = np.eye(3)
    (ref_moi_vecs[0, 0], ref_moi_vecs[1, 1], ref_moi_vecs[2, 2]) = get_inertia_moments(
        reference, masses
    )

    for t, target in enumerate(targets):
        tgt_moi_vecs = np.eye(3)
        (tgt_moi_vecs[0, 0], tgt_moi_vecs[1, 1], tgt_moi_vecs[2, 2]) = get_inertia_moments(
            target, masses
        )

        try:
            matrix = get_alignment_matrix(ref_moi_vecs, tgt_moi_vecs)

        except LinAlgError:
            # it is actually possible for the kabsch alg not to converge
            matrix = np.eye(3)

        # output[t+1] = np.array([matrix @ vector for vector in target])
        output[t + 1] = (matrix @ target.T).T

    return output


class Hypermolecule:
    """Molecule class to be used within firecode."""

    def __repr__(self) -> str:
        """String representation."""
        r = self.rootname
        if hasattr(self, "reactive_atoms_classes_dict"):
            r += f" {[str(atom) for atom in self.reactive_atoms_classes_dict[0].values()]}"
        return r

    def __init__(
        self,
        filename: str,
        reactive_indices: Sequence[int] | None = None,
        charge: int | None = None,
        mult: int = 1,
        T: float = 298.15,
        debug_logfunction: Callable[[str], None] | None = None,
    ) -> None:
        """Initializing class properties."""
        if not os.path.isfile(filename):
            if "." in filename:
                raise SyntaxError(
                    (f"Molecule {filename} cannot be read. Please check your syntax.")
                )

        self.rootname = filename.split(".", maxsplit=1)[0]
        self.filename = filename
        self.charge = charge or self.filename.count("+") - self.filename.count("-")
        self.mult = mult
        self.T = T
        self.debug_logfunction = debug_logfunction

        # properties that can be set/overwritten later
        self.constraints: list[Constraint] = []
        self.pivots: list[list[Pivot]]
        self.pka_data: tuple[str, float] | None = None
        self.sp3_sigmastar: bool = False
        self.scan_data: tuple[list[float], list[float]] | None = None

        if reactive_indices is None:
            self.reactive_indices = np.array([])
        else:
            self.reactive_indices = np.array(reactive_indices)

        conf_ensemble_object = read_xyz(filename)

        coordinates = conf_ensemble_object.coords

        self.atomnos: Array1D_int = conf_ensemble_object.atomnos
        self.atoms: Array1D_str = conf_ensemble_object.atoms
        self.position: Array1D_float = np.array([0, 0, 0], dtype=float)  # used in Embedder class
        self.rotation: Array1D_float = np.identity(3)  # used in Embedder class - rotation matrix

        assert all(
            [len(coordinates[i]) == len(coordinates[0]) for i in range(1, len(coordinates))]
        ), "Ensembles must have constant atom number."
        # Checking that ensemble has constant length
        if self.debug_logfunction is not None:
            self.debug_logfunction(
                f"DEBUG: Hypermolecule Class __init__ ({filename}) - Initializing object {filename}, read {len(coordinates)} structures with {len(coordinates[0])} atoms"
            )

        self.centroid: Array1D_float = np.sum(np.sum(coordinates, axis=0), axis=0) / (
            len(coordinates) * len(coordinates[0])
        )

        self.coords: Array3D_float = coordinates - self.centroid
        self.graph: Graph = graphize(self.atoms, self.coords[0])

        self.all_atoms_coords = np.array(
            [coord for structure in self.coords for coord in structure]
        )  # single list with all atomic positions

    def compute_orbitals(self, override: str | None = None) -> None:
        """Computes orbital positions for atoms in self.reactive_atoms"""
        if len(self.reactive_indices) == 0:
            return

        self._inspect_reactive_atoms(override=override)
        # sets reactive atoms properties

        # self.coords = align_structures(self.coords, self.get_alignment_indices())
        self.sigmatropic: list[bool] = [is_sigmatropic(self, c) for c, _ in enumerate(self.coords)]
        self.sp3_sigmastar = is_vicinal(self)

        for c, _ in enumerate(self.coords):
            for index, reactive_atom in self.reactive_atoms_classes_dict[c].items():
                reactive_atom.init(self, index, update=True, conf=c)
                # update properties into reactive_atom class.
                # Since now we have mol.sigmatropic and mol.sigmastar,
                # We can update, that is set the reactive_atom.center attribute

    def get_alignment_indices(self) -> list[tuple[int]] | None:
        """Return the indices to align the molecule to, given a list of
        atoms that should be reacting. List is composed by reactive atoms
        plus adjacent atoms.
        :param coords: coordinates of a single molecule
        :param reactive atoms: int or list of ints
        :return: list of indices
        """
        if not self.reactive_indices:
            return None

        indices = set()
        for atom in self.reactive_indices:
            indices |= set(list([(a, b) for a, b in self.graph.adjacency()][atom][1].keys()))

        if self.debug_logfunction is not None:
            self.debug_logfunction(
                f"DEBUG: Hypermolecule.get_aligment_indices {self.filename} - Alignment indices are {list(indices)})"
            )

        return list(indices)

    def _inspect_reactive_atoms(self, override: str | None = None) -> None:
        """Control the type of reactive atoms and sets the class attribute self.reactive_atoms_classes_dict."""
        self.reactive_atoms_classes_dict: dict[int, dict[int, RAtom]] = {
            c: dict() for c, _ in enumerate(self.coords)
        }

        for c, _ in enumerate(self.coords):
            for index in self.reactive_indices:
                symbol = self.atoms[index]

                try:
                    atom_type = get_atom_type(self.graph, index, override=override)()

                    # setting the reactive_atom class type
                    atom_type.init(self, index, conf=c)

                    # understanding the type of reactive atom in order to align the ensemble correctly and build the correct pseudo-orbitals
                    self.reactive_atoms_classes_dict[c][index] = atom_type

                    if self.debug_logfunction is not None:
                        self.debug_logfunction(
                            f"DEBUG: Hypermolecule._inspect_reactive_atoms {self.filename} - Reactive atom {index} is a {symbol} atom of {atom_type} type. It is bonded to {len(list(self.graph.neighbors(index)))} neighbor(s): {atom_type.neighbors_symbols}"
                        )

                except KeyError as err:
                    raise KeyError(err)

    def _scale_orbs(self, value: float) -> None:
        """Scale each orbital dimension according to value."""
        for c, _ in enumerate(self.coords):
            for index, atom in self.reactive_atoms_classes_dict[c].items():
                orb_dim = np.linalg.norm(atom.center[0] - atom.coord)
                atom.init(self, index, update=True, orb_dim=orb_dim * value, conf=c)

    def get_r_atoms(self, c: int) -> list[RAtom]:
        """c: conformer number"""
        return list(self.reactive_atoms_classes_dict[c].values())

    def get_centers(self, c: int) -> Array3D_float:
        """c: conformer number"""
        return np.array([[v for v in atom.center] for atom in self.get_r_atoms(c)])

    def write_hypermolecule(self) -> None:
        """ """

        hyp_name = self.rootname + "_hypermolecule.xyz"
        with open(hyp_name, "w") as f:
            for c, _ in enumerate(self.coords):
                f.write(
                    str(
                        sum(
                            [
                                len(atom.center)
                                for atom in self.reactive_atoms_classes_dict[c].values()
                            ]
                        )
                        + len(self.coords[0])
                    )
                )
                f.write(
                    f"\nFIRECODE Hypermolecule {c} for {self.rootname} - reactive indices {self.reactive_indices}\n"
                )
                orbs_stacked = np.vstack(
                    [atom_type.center for atom_type in self.reactive_atoms_classes_dict[c].values()]
                ).ravel()
                orbs = orbs_stacked.reshape((int(len(orbs_stacked) / 3), 3))
                for i, atom in enumerate(self.coords[c]):
                    f.write(
                        "%-5s %-8s %-8s %-8s\n"
                        % (self.atoms[i], round(atom[0], 6), round(atom[1], 6), round(atom[2], 6))
                    )
                for orb in orbs:
                    f.write(
                        "%-5s %-8s %-8s %-8s\n"
                        % ("X", round(orb[0], 6), round(orb[1], 6), round(orb[2], 6))
                    )

    def get_orbital_length(self, index: int) -> float:
        """index: reactive atom index"""
        if index not in self.reactive_indices:
            raise NoOrbitalError(
                f"Index provided must be a molecule reactive index ({index}, {self.filename})"
            )

        r_atom = self.reactive_atoms_classes_dict[0][index]
        return float(np.linalg.norm(r_atom.center[0] - r_atom.coord))


from dataclasses import dataclass


@dataclass
class Pivot:
    """(Cyclical embed)
    Pivot object: vector connecting two lobes of a
    molecule, starting from v1 (first reactive atom in
    mol.reacitve_atoms_classes_dict) and ending on v2.

    For molecules involved in chelotropic reactions,
    that is molecules that undergo a cyclical embed
    while having only one reactive atom, pivots are
    built on that single atom.
    """

    # def __init__(self, c1, c2, a1, a2, index1, index2):
    """c: centers (orbital centers)
    v: vectors (orbital vectors, non-normalized)
    i: indices (of coordinates, in mol.center)
    """
    start: Array1D_float
    end: Array1D_float

    start_atom: RAtom
    end_atom: RAtom

    # the pivot starts from the index1-th
    # center of the first reactive atom
    # and to the index2-th center of the second
    index1: int
    index2: int

    def __post_init__(self) -> None:
        """Compute some extra attributes."""
        self.pivot = self.start - self.end
        self.meanpoint = np.mean((self.start, self.end), axis=0)
        self.index_ = (self.index1, self.index2)

    def __repr__(self) -> str:
        return f"Pivot object - index {self.index_}, norm {round(np.linalg.norm(self.pivot), 3)}, meanpoint {self.meanpoint}"
