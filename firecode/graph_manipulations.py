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
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
from networkx import (
    all_simple_paths,
    connected_components,
    get_node_attributes,
    set_node_attributes,
)
from prism_pruner.graph_manipulations import get_sp_n

if TYPE_CHECKING:
    from networkx import Graph

    from firecode.hypermolecule_class import Hypermolecule


def is_sigmatropic(mol: Hypermolecule, conf: int) -> bool:
    """mol: Hypermolecule object
    conf: conformer index

    A hypermolecule is considered sigmatropic when:
    - has 2 reactive atoms
    - they are of sp2 or analogous types
    - they are connected, or at least one path connecting them
    is made up of atoms that do not make more than three bonds each
    - they are less than 3 A apart (cisoid propenal makes it, transoid does not)

    Used to set the mol.sigmatropic attribute, that affects orbital
    building (p or n lobes) for Ketone and Imine reactive atoms classes.
    """
    sp2_types = ("Ketone", "Imine", "sp2", "sp", "bent carbene")
    if len(mol.reactive_indices) == 2:
        i1, i2 = mol.reactive_indices
        if np.linalg.norm(mol.coords[conf][i1] - mol.coords[conf][i2]) < 3:
            if all(
                [
                    str(r_atom) in sp2_types
                    for r_atom in mol.reactive_atoms_classes_dict[conf].values()
                ]
            ):
                paths = all_simple_paths(mol.graph, i1, i2)

                for path in paths:
                    path = path[1:-1]

                    full_sp2 = True
                    for index in path:
                        if len(list(mol.graph.neighbors(index))) - 2 > 1:
                            full_sp2 = False
                            break

                    if full_sp2:
                        return True
    return False


def is_vicinal(mol: Hypermolecule) -> bool:
    """A hypermolecule is considered vicinal when:
    - has 2 reactive atoms
    - they are of sp3 or Single Bond type
    - they are bonded

    Used to set the mol.sp3_sigmastar attribute, that affects orbital
    building (BH4 or agostic-like behavior) for Sp3 and Single Bond reactive atoms classes.
    """
    vicinal_types = (
        "sp3",
        "Single Bond",
    )

    if len(mol.reactive_indices) == 2:
        i1, i2 = mol.reactive_indices

        if all(
            [str(r_atom) in vicinal_types for r_atom in mol.reactive_atoms_classes_dict[0].values()]
        ):
            if i1 in mol.graph.neighbors(i2):
                return True

    return False


def is_sp_n(index: int, graph: Graph, n: int) -> bool:
    """Returns True if the sp_n value matches the input"""
    sp_n = get_sp_n(index, graph)
    if sp_n == n:
        return True
    return False


def get_sum_graph(
    graphs: Sequence[Graph], extra_edges: Iterable[Iterable[int]] | None = None
) -> Graph:
    """Creates a graph containing all graphs, added in
    sequence, and then adds the specified extra edges
    (with cumulative numbering).
    """
    graph, *extra = graphs
    out = deepcopy(graph)
    cum_atoms = list(get_node_attributes(graphs[0], "atoms").values())

    for g in extra:
        n = len(out.nodes())
        for e1, e2 in g.edges():
            out.add_edge(e1 + n, e2 + n)

        cum_atoms += list(get_node_attributes(g, "atoms").values())

    out.is_single_molecule = len(list(connected_components(out))) == 1

    if extra_edges is not None:
        for e1, e2 in extra_edges:
            out.add_edge(e1, e2)

    set_node_attributes(out, dict(enumerate(cum_atoms)), "atoms")

    return out
