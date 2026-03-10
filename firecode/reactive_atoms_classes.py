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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
from prism_pruner.algebra import normalize, rot_mat_from_pointer, vec_angle

from firecode.parameters import orb_dim_dict
from firecode.typing_ import Array1D_float, Array2D_float, MaybeNone

if TYPE_CHECKING:
    from networkx import Graph

    from firecode.hypermolecule_class import Hypermolecule


@dataclass
class RAtom:
    """Reactive atom stub class."""

    index: int = field(init=False)
    cumnum: int = field(init=False)
    symbol: str = field(init=False)
    init: Callable[..., None] = field(init=False)
    center: Array1D_float = field(init=False)
    coord: Array1D_float = field(init=False)
    orb_vecs: Array2D_float = field(init=False)
    neighbors_symbols: list[str] = field(init=False)


class Single(RAtom):
    def __repr__(self) -> str:
        return "Single Bond"

    def init(
        self,
        mol: Hypermolecule,
        i: int,
        update: bool = False,
        orb_dim: float | None = None,
        conf: int = 0,
    ) -> None:
        """ """
        self.index = i
        self.symbol = mol.atoms[i]
        neighbors_indices = list(mol.graph.neighbors(i))

        self.neighbors_symbols = [mol.atoms[i] for i in neighbors_indices]
        self.coord = mol.coords[conf][i]
        self.other = mol.coords[conf][neighbors_indices][0]

        if not hasattr(self, "cumnum"):
            self.cumnum: int | MaybeNone = None

        if not mol.sp3_sigmastar:
            self.orb_vecs = np.array([normalize(self.coord - self.other)])

        else:
            other_reactive_indices = list(mol.reactive_indices)
            other_reactive_indices.remove(i)
            for index in other_reactive_indices:
                if index in neighbors_indices:
                    parnter_index = index
                    break
            # obtain the reference partner index

            partner = mol.coords[conf][parnter_index]
            pivot = normalize(partner - self.coord)

            neighbors_of_partner = list(mol.graph.neighbors(parnter_index))
            neighbors_of_partner.remove(i)
            orb_vec = normalize(mol.coords[conf][neighbors_of_partner[0]] - partner)
            orb_vec = orb_vec - orb_vec @ pivot * pivot

            steps = 3  # number of total orbitals
            self.orb_vecs = np.array(
                [
                    rot_mat_from_pointer(pivot, angle + 60) @ orb_vec
                    for angle in range(0, 360, int(360 / steps))
                ]
            )
            # orbitals are staggered in relation to sp3 substituents

            self.orb_vers = normalize(self.orb_vecs[0])

        if update:
            if orb_dim is None:
                key = self.symbol + " " + str(self).split(" (")[0]
                orb_dim = orb_dim_dict.get(key)

                if orb_dim is None:
                    orb_dim = float(np.linalg.norm(self.coord - self.other))
                    # print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using the bonding distance ({round(orb_dim, 3)} A).')

            self.center = orb_dim * self.orb_vecs + self.coord


class Sp2(RAtom):
    def __repr__(self) -> str:
        return "sp2"

    def init(
        self,
        mol: Hypermolecule,
        i: int,
        update: bool = False,
        orb_dim: float | None = None,
        conf: int = 0,
    ) -> None:
        """ """
        self.index = i

        if not hasattr(self, "cumnum"):
            self.cumnum: int | MaybeNone = None

        self.symbol = mol.atoms[i]
        neighbors_indices = list(mol.graph.neighbors(i))

        self.neighbors_symbols = [mol.atoms[i] for i in neighbors_indices]
        self.coord = mol.coords[conf][i]
        self.others = mol.coords[conf][neighbors_indices]

        self.vectors = self.others - self.coord  # vectors connecting reactive atom with neighbors
        self.orb_vec = normalize(
            np.mean(
                np.array(
                    [
                        np.cross(normalize(self.vectors[0]), normalize(self.vectors[1])),
                        np.cross(normalize(self.vectors[1]), normalize(self.vectors[2])),
                        np.cross(normalize(self.vectors[2]), normalize(self.vectors[0])),
                    ]
                ),
                axis=0,
            )
        )

        self.orb_vecs = np.vstack((self.orb_vec, -self.orb_vec))

        if update:
            if orb_dim is None:
                key = self.symbol + " " + str(self).split(" (")[0]
                orb_dim = orb_dim_dict.get(key)

                if orb_dim is None:
                    orb_dim = orb_dim_dict["Fallback"]
                    # print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            self.center = self.orb_vecs * orb_dim

            self.center += self.coord


class Sp3(RAtom):
    def __repr__(self) -> str:
        return "sp3"

    def init(
        self,
        mol: Hypermolecule,
        i: int,
        update: bool = False,
        orb_dim: float | None = None,
        conf: int = 0,
    ) -> None:
        self.index = i

        if not hasattr(self, "cumnum"):
            self.cumnum: int | MaybeNone = None

        self.symbol = mol.atoms[i]
        neighbors_indices = list(mol.graph.neighbors(i))
        self.neighbors_symbols = [mol.atoms[i] for i in neighbors_indices]
        self.coord = mol.coords[conf][i]
        self.others = mol.coords[conf][neighbors_indices]

        if not mol.sp3_sigmastar:
            if not hasattr(self, "leaving_group_index"):
                self.leaving_group_index: int | None = None

            if (
                len(
                    [atom for atom in self.neighbors_symbols if atom in ["O", "N", "Cl", "Br", "I"]]
                )
                == 1
            ):  # if we can tell where is the leaving group
                self.leaving_group_coords = self.others[
                    self.neighbors_symbols.index(
                        [atom for atom in self.neighbors_symbols if atom in ["O", "Cl", "Br", "I"]][
                            0
                        ]
                    )
                ]

            elif (
                len([atom for atom in self.neighbors_symbols if atom not in ["H"]]) == 1
            ):  # if no clear leaving group but we only have one atom != H
                self.leaving_group_coords = self.others[
                    self.neighbors_symbols.index(
                        [atom for atom in self.neighbors_symbols if atom not in ["H"]][0]
                    )
                ]

            else:
                # probably a bad embedding, but we still need to go through this for refine> runs, so let's pick one
                self.leaving_group_coords = self.others[0]

            self.orb_vecs = np.array([self.coord - self.leaving_group_coords])
            self.orb_vers = normalize(self.orb_vecs[0])

        else:  # Sigma bond type
            other_reactive_indices = list(mol.reactive_indices)
            other_reactive_indices.remove(i)
            for index in other_reactive_indices:
                if index in neighbors_indices:
                    parnter_index = index
                    break
            # obtain the reference partner index

            pivot = normalize(mol.coords[conf][parnter_index] - self.coord)

            other_neighbors = deepcopy(neighbors_indices)
            other_neighbors.remove(parnter_index)
            orb_vec = normalize(mol.coords[conf][other_neighbors[0]] - self.coord)
            orb_vec = orb_vec - orb_vec @ pivot * pivot

            steps = 3  # number of total orbitals
            self.orb_vecs = np.array(
                [
                    rot_mat_from_pointer(pivot, angle + 60) @ orb_vec
                    for angle in range(0, 360, int(360 / steps))
                ]
            )
            # orbitals are staggered in relation to sp3 substituents

            self.orb_vers = normalize(self.orb_vecs[0])

        if update:
            if orb_dim is None:
                key = self.symbol + " " + str(self).split(" (")[0]
                orb_dim = orb_dim_dict.get(key)

                if orb_dim is None:
                    orb_dim = orb_dim_dict["Fallback"]
                    # print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            self.center = np.array([orb_dim * normalize(vec) + self.coord for vec in self.orb_vecs])


class Ether(RAtom):
    def __repr__(self) -> str:
        return "Ether"

    def init(
        self,
        mol: Hypermolecule,
        i: int,
        update: bool = False,
        orb_dim: float | None = None,
        conf: int = 0,
    ) -> None:
        """ """
        self.index = i

        if not hasattr(self, "cumnum"):
            self.cumnum: int | MaybeNone = None

        self.symbol = mol.atoms[i]
        neighbors_indices = list(mol.graph.neighbors(i))

        self.neighbors_symbols = [mol.atoms[i] for i in neighbors_indices]
        self.coord = mol.coords[conf][i]
        self.others = mol.coords[conf][neighbors_indices]

        self.orb_vecs = (
            self.others - self.coord
        )  # vectors connecting center to each of the two substituents

        if update:
            if orb_dim is None:
                key = self.symbol + " " + str(self).split(" (")[0]
                orb_dim = orb_dim_dict.get(key)

                if orb_dim is None:
                    orb_dim = orb_dim_dict["Fallback"]
                    # print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            self.orb_vecs = orb_dim * np.array(
                [normalize(v) for v in self.orb_vecs]
            )  # making both vectors a fixed, defined length

            orb_mat = rot_mat_from_pointer(
                np.mean(self.orb_vecs, axis=0), 90
            ) @ rot_mat_from_pointer(np.cross(self.orb_vecs[0], self.orb_vecs[1]), 180)

            # self.orb_vecs = np.array([orb_mat @ v for v in self.orb_vecs])
            self.orb_vecs = (orb_mat @ self.orb_vecs.T).T

            self.center = self.orb_vecs + self.coord
            # two vectors defining the position of the two orbital lobes centers


class Ketone(RAtom):
    def __repr__(self) -> str:
        return f"Ketone ({self.subtype})"

    def init(
        self,
        mol: Hypermolecule,
        i: int,
        update: bool = False,
        orb_dim: float | None = None,
        conf: int = 0,
    ) -> None:
        """ """
        self.index = i

        if not hasattr(self, "cumnum"):
            self.cumnum: int | MaybeNone = None

        self.symbol = mol.atoms[i]
        neighbors_indices = list(mol.graph.neighbors(i))
        self.subtype = "pre-init"

        self.neighbors_symbols = [mol.atoms[i] for i in neighbors_indices]
        self.coord = mol.coords[conf][i]
        self.other = mol.coords[conf][neighbors_indices][0]

        self.vector = self.other - self.coord  # vector connecting center to substituent

        if update:
            if orb_dim is None:
                key = self.symbol + " " + str(self).split(" (")[0]
                orb_dim = orb_dim_dict.get(key)

                if orb_dim is None:
                    orb_dim = orb_dim_dict["Fallback"]
                    # print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            neighbors_of_neighbor_indices = list(mol.graph.neighbors(neighbors_indices[0]))
            neighbors_of_neighbor_indices.remove(i)

            self.vector = normalize(self.vector) * orb_dim

            if len(neighbors_of_neighbor_indices) == 1:
                # ketene

                ketene_sub_indices = list(mol.graph.neighbors(neighbors_of_neighbor_indices[0]))
                ketene_sub_indices.remove(neighbors_indices[0])

                ketene_sub_coords = mol.coords[conf][ketene_sub_indices[0]]
                n_o_n_coords = mol.coords[conf][neighbors_of_neighbor_indices[0]]

                # vector connecting ketene R with C (O=C=C(R)R)
                v = ketene_sub_coords - n_o_n_coords

                # this vector is orthogonal to the ketene O=C=C and coplanar with the ketene
                pointer = v - ((v @ normalize(self.vector)) * self.vector)
                pointer = normalize(pointer) * orb_dim

                self.center = np.array(
                    [rot_mat_from_pointer(self.vector, 90 * step) @ pointer for step in range(4)]
                )

                self.subtype = "p+p"

            elif len(neighbors_of_neighbor_indices) == 2:
                # if it is a normal ketone (or an enolate), n orbital lobes must be coplanar with
                # atoms connecting to ketone C atom, or p lobes must be placed accordingly

                a1 = mol.coords[conf][neighbors_of_neighbor_indices[0]]
                a2 = mol.coords[conf][neighbors_of_neighbor_indices[1]]
                pivot = normalize(np.cross(a1 - self.coord, a2 - self.coord))

                if mol.sigmatropic[conf]:
                    # two p lobes
                    self.center = np.concatenate(([pivot * orb_dim], [-pivot * orb_dim]))
                    self.subtype = "p"

                else:
                    # two n lobes
                    self.center = np.array(
                        [rot_mat_from_pointer(pivot, angle) @ self.vector for angle in (120, 240)]
                    )
                    self.subtype = "sp2"

            elif len(neighbors_of_neighbor_indices) == 3:
                # alkoxide, sulfonamide

                v1, v2, v3 = mol.coords[conf][neighbors_of_neighbor_indices] - self.coord
                v1, v2, v3 = normalize(v1), normalize(v2), normalize(v3)
                v1, v2, v3 = v1 * orb_dim, v2 * orb_dim, v3 * orb_dim
                pivot = normalize(np.cross(self.vector, v1))

                self.center = np.array([rot_mat_from_pointer(pivot, 180) @ v for v in (v1, v2, v3)])
                self.subtype = "trilobe"

            self.orb_vecs = np.array([normalize(center) for center in self.center])
            # unit vectors connecting reactive atom coord with orbital centers

            self.center += self.coord
            # two vectors defining the position of the two orbital lobes centers


class Imine(RAtom):
    def __repr__(self) -> str:
        return "Imine"

    def init(
        self,
        mol: Hypermolecule,
        i: int,
        update: bool = False,
        orb_dim: float | None = None,
        conf: int = 0,
    ) -> None:
        """ """
        self.index = i

        if not hasattr(self, "cumnum"):
            self.cumnum: int | MaybeNone = None

        self.symbol = mol.atoms[i]
        neighbors_indices = list(mol.graph.neighbors(i))

        self.neighbors_symbols = [mol.atoms[i] for i in neighbors_indices]
        self.coord = mol.coords[conf][i]
        self.others = mol.coords[conf][neighbors_indices]

        self.vectors = self.others - self.coord  # vector connecting center to substituent

        if update:
            if orb_dim is None:
                key = self.symbol + " " + str(self).split(" (")[0]
                orb_dim = orb_dim_dict.get(key)

                if orb_dim is None:
                    orb_dim = orb_dim_dict["Fallback"]
                    # print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            if mol.sigmatropic[conf]:
                # two p lobes
                p_lobe = normalize(np.cross(self.vectors[0], self.vectors[1])) * orb_dim
                self.orb_vecs = np.concatenate(([p_lobe], [-p_lobe]))

            else:
                # lone pair lobe
                self.orb_vecs = np.array(
                    [-normalize(np.mean([normalize(v) for v in self.vectors], axis=0)) * orb_dim]
                )

            self.center = self.orb_vecs + self.coord
            # two vectors defining the position of the two orbital lobes centers


class Sp_or_carbene(RAtom):
    def __repr__(self) -> str:
        return self.type

    def init(
        self,
        mol: Hypermolecule,
        i: int,
        update: bool = False,
        orb_dim: float | None = None,
        conf: int = 0,
    ) -> None:
        self.index = i

        if not hasattr(self, "cumnum"):
            self.cumnum: int | MaybeNone = None

        self.symbol = mol.atoms[i]
        neighbors_indices = list(mol.graph.neighbors(i))

        self.neighbors_symbols = [mol.atoms[i] for i in neighbors_indices]

        self.coord = mol.coords[conf][i]
        self.others = mol.coords[conf][neighbors_indices]

        self.vectors = self.others - self.coord  # vector connecting center to substituent

        angle = vec_angle(
            normalize(self.others[0] - self.coord), normalize(self.others[1] - self.coord)
        )

        if np.abs(angle - 180) < 5:
            self.type = "sp"

        else:
            self.type = "bent carbene"

        self.allene = False
        self.ketene = False
        if self.type == "sp" and all([s == "C" for s in self.neighbors_symbols]):
            neighbors_of_neighbors_indices = (
                list(mol.graph.neighbors(neighbors_indices[0])),
                list(mol.graph.neighbors(neighbors_indices[1])),
            )

            neighbors_of_neighbors_indices[0].remove(i)
            neighbors_of_neighbors_indices[1].remove(i)

            if (len(side1) == len(side2) == 2 for side1, side2 in neighbors_of_neighbors_indices):
                self.allene = True

        elif self.type == "sp" and sorted(self.neighbors_symbols) in (["C", "O"], ["C", "S"]):
            self.ketene = True

            neighbors_of_neighbors_indices = (
                list(mol.graph.neighbors(neighbors_indices[0])),
                list(mol.graph.neighbors(neighbors_indices[1])),
            )

            neighbors_of_neighbors_indices[0].remove(i)
            neighbors_of_neighbors_indices[1].remove(i)

            if len(neighbors_of_neighbors_indices[0]) == 2:
                substituent = mol.coords[conf][neighbors_of_neighbors_indices[0][0]]
                ketene_atom = mol.coords[conf][neighbors_indices[0]]
                self.ketene_ref = substituent - ketene_atom

            elif len(neighbors_of_neighbors_indices[1]) == 2:
                substituent = mol.coords[conf][neighbors_of_neighbors_indices[1][0]]
                ketene_atom = mol.coords[conf][neighbors_indices[1]]
                self.ketene_ref = substituent - ketene_atom

            else:
                self.ketene = False

        if update:
            if orb_dim is None:
                key = self.symbol + " " + self.type
                orb_dim = orb_dim_dict.get(key)

                if orb_dim is None:
                    orb_dim = orb_dim_dict["Fallback"]
                    # print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            if self.type == "sp":
                v = np.random.rand(3)
                pivot1 = v - ((v @ normalize(self.vectors[0])) * self.vectors[0])

                if self.allene or self.ketene:
                    # if we have an allene or ketene, pivot1 is aligned to
                    # one substituent so that the resulting positions
                    # for the four orbital centers make chemical sense.

                    axis = normalize(self.others[0] - self.others[1])
                    # versor connecting reactive atom neighbors

                    if self.allene:
                        ref = (
                            mol.coords[conf][neighbors_of_neighbors_indices[0][0]]
                            - mol.coords[conf][neighbors_indices[0]]
                        )
                    else:
                        ref = self.ketene_ref

                    pivot1 = ref - ref @ axis * axis
                    # projection of ref orthogonal to axis (vector rejection)

                pivot2 = normalize(np.cross(pivot1, self.vectors[0]))

                self.orb_vecs = (
                    np.array(
                        [
                            rot_mat_from_pointer(pivot2, 90)
                            @ rot_mat_from_pointer(pivot1, angle)
                            @ normalize(self.vectors[0])
                            for angle in (0, 90, 180, 270)
                        ]
                    )
                    * orb_dim
                )

                self.center = self.orb_vecs + self.coord
                # four vectors defining the position of the four orbital lobes centers

            else:  # bent carbene case: three centers, sp2+p
                self.orb_vecs = np.array(
                    [-normalize(np.mean([normalize(v) for v in self.vectors], axis=0)) * orb_dim]
                )
                # one sp2 center first

                p_vec = np.cross(normalize(self.vectors[0]), normalize(self.vectors[1]))
                p_vecs = np.array([normalize(p_vec) * orb_dim, -normalize(p_vec) * orb_dim])
                self.orb_vecs = np.concatenate((self.orb_vecs, p_vecs))
                # adding two p centers

                self.center = self.orb_vecs + self.coord
                # three vectors defining the position of the two p lobes and main sp2 lobe centers


class Metal(RAtom):
    def __repr__(self) -> str:
        return "Metal"

    def init(
        self,
        mol: Hypermolecule,
        i: int,
        update: bool = False,
        orb_dim: float | None = None,
        conf: int = 0,
    ) -> None:
        self.index = i

        if not hasattr(self, "cumnum"):
            self.cumnum: int | MaybeNone = None

        self.symbol = mol.atoms[i]
        neighbors_indices = list(mol.graph.neighbors(i))

        self.neighbors_symbols = [mol.atoms[i] for i in neighbors_indices]
        self.coord = mol.coords[conf][i]
        self.others = mol.coords[conf][neighbors_indices]

        self.vectors = self.others - self.coord  # vectors connecting reactive atom with neighbors

        v1 = self.vectors[0]
        # v1 connects first bonded atom to the metal itself

        neighbor_of_neighbor_index = list(mol.graph.neighbors(neighbors_indices[0]))[0]
        v2 = mol.coords[conf][neighbor_of_neighbor_index] - self.coord
        # v2 connects first neighbor of the first neighbor to the metal itself

        self.orb_vec = normalize(rot_mat_from_pointer(np.cross(v1, v2), 120) @ v1)
        # setting the pointer (orb_vec) so that orbitals are oriented correctly
        # (Lithium enolate in mind)

        steps = 4  # number of total orbitals
        self.orb_vecs = np.array(
            [
                rot_mat_from_pointer(v1, angle) @ self.orb_vec
                for angle in range(0, 360, int(360 / steps))
            ]
        )

        if update:
            if orb_dim is None:
                orb_dim = orb_dim_dict[str(self)]

            self.center = (self.orb_vecs * orb_dim) + self.coord


class SingleAtom(RAtom):
    def __repr__(self) -> str:
        return "SingleAtom"

    def init(
        self,
        mol: Hypermolecule,
        i: int,
        update: bool = False,
        orb_dim: float | None = None,
        conf: int = 0,
    ) -> None:
        """ """
        self.index = i
        self.cumnum: int | MaybeNone = None
        self.symbol = mol.atoms[i]

        self.neighbors_symbols: list[str] = []
        self.coord = mol.coords[conf][i]
        self.other = mol.coords[conf][i] + np.array([0, 0, 1], dtype=float)
        self.orb_vecs = np.array([normalize(self.coord - self.other)])
        self.orb_vers = normalize(self.orb_vecs[0])

        if update:
            if orb_dim is None:
                key = self.symbol + " " + str(self).split(" (")[0]
                orb_dim = orb_dim_dict.get(key)

                if orb_dim is None:
                    orb_dim = float(np.linalg.norm(self.coord - self.other))
                    # print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using the bonding distance ({round(orb_dim, 3)} A).')

            self.center = orb_dim * self.orb_vecs + self.coord


# Keys are made of atom symbol and number of bonds that it makes
atom_type_dict = {
    "H1": Single,
    "B3": Sp2,
    "B4": Sp3,
    "C1": Single,  # deprotonated terminal alkyne. What if it is a carbylidene? Very rare by the way...
    "C2": Sp_or_carbene,  # sp if straight, carbene if bent
    "C3": Sp2,  # double lobe
    "C4": Sp3,  # one lobe, on the back of the leaving group. If we can't tell which one it is, we ask user
    "N1": Single,
    "N2": Imine,  # one lobe on free side
    "N3": Sp2,  # double lobe
    "N4": Sp3,  # leaving group
    "O1": Ketone,  # two lobes 120° apart. Also for alkoxides, good enough
    "O2": Ether,  # or alcohol, two lobes about 109,5° apart
    "P2": Imine,  # one lobe on free side
    "P3": Sp2,  # double lobe
    "P4": Sp3,  # leaving group
    "S1": Ketone,
    "S2": Ether,
    "S3": Sp2,  # basically treating it as a bent carbonyl
    #  'S3' : Sulfoxide, # Should we consider this? Or just ok with Sp2()?
    "S4": Sp3,
    "F1": Single,
    "Cl1": Single,
    "Br1": Single,
    "I1": Single,
    ############### Name associations
    "Single": Single,
    "Sp2": Sp2,
    "Sp3": Sp3,
    "Ether": Ether,
    "Ketone": Ketone,
    "Imine": Imine,
    "Sp_or_carbene": Sp_or_carbene,
    "Metal": Metal,
    "X0": SingleAtom,
}

metals = (
    "Li",
    "Na",
    "Mg",
    "K",
    "Ca",
    "Ti",
    "Rb",
    "Sr",
    "Cs",
    "Ba",
    "Zn",
)

for metal in metals:
    for bonds in range(1, 9):
        atom_type_dict[metal + str(bonds)] = Metal


def get_atom_type(graph: Graph, index: int, override: str | None = None) -> type[RAtom]:
    """Returns the appropriate class to represent
    the atom with the given index on the graph.
    If override is not None, returns the class
    with that name.
    """
    if override is not None:
        return atom_type_dict[override]

    nb = list(graph.neighbors(index))

    if not nb:
        return atom_type_dict["X0"]

    code = graph.nodes[index]["atoms"] + str(len(nb))
    try:
        return atom_type_dict[code]

    except KeyError:
        raise KeyError(f"Orbital type {code} not known (index {index})")
