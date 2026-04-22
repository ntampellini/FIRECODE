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
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Iterable, Sequence, cast

import numpy as np
from networkx import (
    Graph,
    connected_components,
    has_path,
    is_isomorphic,
    minimum_spanning_tree,
    shortest_path,
    subgraph,
)
from prism_pruner.algebra import dihedral, normalize, vec_angle
from prism_pruner.graph_manipulations import (
    get_phenyl_ids,
    get_sp_n,
    graphize,
    is_amide_n,
    is_ester_o,
)
from prism_pruner.utils import flatten, get_double_bonds_indices, rotate_dihedral, time_to_string
from scipy.spatial.distance import cdist

from firecode.errors import SegmentedGraphError
from firecode.graph_manipulations import is_sp_n
from firecode.typing_ import (
    Array1D_bool,
    Array1D_float,
    Array1D_str,
    Array2D_float,
    Array2D_int,
    Array3D_float,
    MaybeNone,
)
from firecode.utils import cartesian_product, write_xyz

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator as ASECalculator

    from firecode.dispatcher import Opt_func_dispatcher


class Torsion:
    def __repr__(self) -> str:
        if hasattr(self, "n_fold"):
            return f"Torsion({self.i1}, {self.i2}, {self.i3}, {self.i4}; {self.n_fold}-fold)"
        return f"Torsion({self.i1}, {self.i2}, {self.i3}, {self.i4})"

    def __init__(self, i1: int, i2: int, i3: int, i4: int, mode: str | None = None) -> None:
        self.i1 = i1
        self.i2 = i2
        self.i3 = i3
        self.i4 = i4
        self.torsion = (i1, i2, i3, i4)
        self.mode = mode

    def in_cycle(self, graph: Graph) -> bool:
        """Returns True if the torsion is part of a cycle."""
        graph.remove_edge(self.i2, self.i3)
        cyclical = bool(has_path(graph, self.i1, self.i4))
        graph.add_edge(self.i2, self.i3)
        return cyclical

    def is_rotable(
        self, graph: Graph, hydrogen_bonds: Sequence[tuple[int, int]], keepdummy: bool = False
    ) -> bool:
        """Hydrogen bonds: iterable with pairs of sorted atomic indices"""
        if tuple(sorted((self.i2, self.i3))) in hydrogen_bonds:
            # self.n_fold = 6
            # # This has to be an intermolecular HB: rotate it
            # return True
            return False

        if _is_free(self.i2, graph) or (_is_free(self.i3, graph)):
            if keepdummy or (
                _is_nondummy(self.i2, self.i3, graph) and (_is_nondummy(self.i3, self.i2, graph))
            ):
                self.n_fold = self.get_n_fold(graph)
                return True

        return False

    def get_n_fold(self, graph: Graph) -> int:
        symbols = (graph.nodes[self.i2]["atoms"], graph.nodes[self.i3]["atoms"])

        if "H" in symbols:
            return 6  # H-N, H-O hydrogen bonds

        if is_amide_n(self.i2, graph, mode=2) or (is_amide_n(self.i3, graph, mode=2)):
            # tertiary amides rotations are 2-fold
            return 2

        if ("C" in symbols) or ("N" in symbols) or ("S" in symbols):
            sp_n_i2 = get_sp_n(self.i2, graph)
            sp_n_i3 = get_sp_n(self.i3, graph)

            if 3 == sp_n_i2 == sp_n_i3:
                return 3

            if 3 in (sp_n_i2, sp_n_i3):  # Csp3-X, Nsp3-X, Ssulfone-X
                if self.mode == "csearch":
                    return 3

                elif self.mode == "symmetry":
                    return sp_n_i3 or 2

            if 2 in (sp_n_i2, sp_n_i3):
                return 2

        return 4  # O-O, S-S, Ar-Ar, Ar-CO, and everything else

    def get_angles(self) -> tuple[int, ...] | MaybeNone:
        return {
            2: (0, 180),
            3: (0, 120, 240),
            4: (0, 90, 180, 270),
            6: (0, 60, 120, 180, 240, 300),
        }.get(self.n_fold)

    def sort_torsion(
        self, graph: Graph, constrained_indices: Sequence[tuple[int, int]] | None
    ) -> None:
        """Acts on the self.torsion tuple leaving it as it is or
        reversing it, so that the first index of it (from which
        rotation will act) is external to the molecule constrained
        indices. That is we make sure to rotate external groups
        and not the whole structure.
        """
        if constrained_indices is None:
            return

        graph.remove_edge(self.i2, self.i3)
        for d in flatten(constrained_indices):
            if has_path(graph, self.i2, d):
                self.torsion = cast("tuple[int, int, int, int]", tuple(reversed(self.torsion)))
        graph.add_edge(self.i2, self.i3)


def _is_free(index: int, graph: Graph) -> bool:
    """Return True if the index specified
    satisfies all of the following:
    - Is not a sp2 carbonyl carbon atom
    - Is not the oxygen atom of an ester
    - Is not the nitrogen atom of a secondary amide (CONHR)

    """
    if all(
        (
            graph.nodes[index]["atoms"] == "C",
            is_sp_n(index, graph, 2),
            "O" in (graph.nodes[n]["atoms"] for n in graph.neighbors(index)),
        )
    ):
        return False

    if is_amide_n(index, graph, mode=1):
        return False

    if is_ester_o(index, graph):
        return False

    return True


def _is_nondummy(i: int, root: int, graph: Graph) -> bool:
    """Checks that a molecular rotation along the dihedral
    angle (*, root, i, *) is non-dummy, that is the atom
    at index i, in the direction opposite to the one leading
    to root, has different substituents. i.e. methyl, CF3 and tBu
    rotations should return False.
    """
    if graph.nodes[i]["atoms"] not in ("C", "N"):
        return True
    # for now, we only discard rotations around carbon
    # and nitrogen atoms, like methyl/tert-butyl/triphenyl
    # and flat symmetrical rings like phenyl, N-pyrrolyl...

    G = deepcopy(graph)
    nb = list(G.neighbors(i))
    # nb.remove(root)

    if len(nb) == 1:
        if len(G.neighbors(nb[0])) == 2:
            return False
    # if node i has two bonds only (one with root and one with a)
    # and the other atom (a) has two bonds only (one with i)
    # the rotation is considered dummy: some other rotation
    # will account for its freedom (i.e. alkynes, hydrogen bonds)

    # check if it is a phenyl-like rotation
    if len(nb) == 2:
        # get the 6 indices of the aromatic atoms (i1-i6)
        phenyl_indices = get_phenyl_ids(i, G)

        # compare the two halves of the 6-membered ring (indices i2-i3 region with i5-i6 region)
        if phenyl_indices is not None:
            i1, i2, i3, i4, i5, i6 = phenyl_indices
            G.remove_edge(i3, i4)
            G.remove_edge(i4, i5)
            G.remove_edge(i1, i2)
            G.remove_edge(i1, i6)

            subgraphs = [
                subgraph(G, _set) for _set in connected_components(G) if i2 in _set or i6 in _set
            ]

            if len(subgraphs) == 2:
                return not is_isomorphic(
                    subgraphs[0], subgraphs[1], node_match=lambda n1, n2: n1["atoms"] == n2["atoms"]
                )

            # We should not end up here, but if we do, rotation should not be dummy
            return True

    # if not, compare immediate neighbors of i
    for n in nb:
        G.remove_edge(i, n)

    # make a set of each fragment around the chopped n-i bonds,
    # but only for fragments that are not root nor contain other random,
    # disconnected parts of the graph
    subgraphs_nodes = [
        _set for _set in connected_components(G) if root not in _set and any(n in _set for n in nb)
    ]

    if len(subgraphs_nodes) == 1:
        return True
        # if not, the torsion is likely to be rotable
        # (tetramethylguanidyl alanine C(β)-N bond)

    subgraphs = [subgraph(G, s) for s in subgraphs_nodes]
    for sub in subgraphs[1:]:
        if not is_isomorphic(
            subgraphs[0], sub, node_match=lambda n1, n2: n1["atoms"] == n2["atoms"]
        ):
            return True
    # Care should be taken because chiral centers are not taken into account: a rotation
    # involving an index where substituents only differ by stereochemistry, and where a
    # rotation is not an element of symmetry of the subsystem, the rotation is considered
    # dummy even if it would be more correct not to. For rotaionally corrected RMSD this
    # should only cause small inefficiencies and not lead to discarding any good conformer.

    return False


def _get_hydrogen_bonds(
    atoms: Array1D_str,
    coords: Array2D_float,
    graph: Graph,
    d_min: float = 2.5,
    d_max: float = 3.3,
    max_angle: int = 45,
    elements: Sequence[Sequence[int]] | None = None,
    fragments: Sequence[Sequence[int]] | None = None,
) -> list[tuple[int, int]]:
    """Returns a list of tuples with the indices
    of hydrogen bonding partners.

    An HB is a pair of atoms:
    - with one H and one X (N or O) atom
    - with an Y-X distance between d_min and d_max (i.e. N-O, Angstroms)
    - with an Y-H-X angle below max_angle (i.e. N-H-O, degrees)

    elements: iterable of donors and acceptors atomic numbers. default: ((7, 8), (7, 8))

    If fragments is specified (iterable of iterable of indices for each fragment)
    the function only returns inter-fragment hydrogen bonds.
    """
    hbs: list[tuple[int, int]] = []
    # initializing output list

    if elements is None:
        elements = ((7, 8), (7, 8, 9))

    het_idx_from = np.array([i for i, a in enumerate(atoms) if a in elements[0]], dtype=int)
    het_idx_to = np.array([i for i, a in enumerate(atoms) if a in elements[1]], dtype=int)
    # indices where N or O (or user-specified elements) atoms are present.

    for i1 in het_idx_from:
        for i2 in het_idx_to:
            # if inter-fragment HBs are requested, skip intra-HBs
            if fragments is not None:
                if any(((i1 in f and i2 in f) for f in fragments)):
                    continue

            # keep close pairs
            if d_min < np.linalg.norm(coords[i1] - coords[i2]) < d_max:
                # getting the indices of all H atoms attached to them
                Hs = [i for i in (graph.neighbors(i1)) if graph.nodes[i]["atoms"] == "H"]

                # versor connectring the two Heteroatoms
                versor = normalize(coords[i2] - coords[i1])

                for iH in Hs:
                    # vectors connecting heteroatoms to H
                    v1 = coords[iH] - coords[i1]
                    v2 = coords[iH] - coords[i2]

                    # lengths of these vectors
                    d1 = np.linalg.norm(v1)
                    d2 = np.linalg.norm(v2)

                    # scalar projection in the heteroatom direction
                    l1 = v1 @ versor
                    l2 = v2 @ -versor

                    # largest planar angle between Het-H and Het-Het, in degrees (0 to 90°)
                    alfa = vec_angle(v1, versor) if l1 < l2 else vec_angle(v2, -versor)

                    # if the three atoms are not too far from being in line
                    if alfa < max_angle:
                        # adding the correct pair of atoms to results
                        if d1 < d2:
                            hbs.append(tuple(sorted((iH, i2))))
                        else:
                            hbs.append(tuple(sorted((iH, i1))))

                        break

    return hbs


def _get_rotation_mask(graph: Graph, torsion: tuple[int, int, int, int]) -> Array1D_bool:
    """Get mask for the atoms that will rotate in a torsion:
    all the ones in the graph reachable from the last index
    of the torsion but not going through the central two
    atoms in the torsion quadruplet.

    """
    _, i2, i3, i4 = torsion

    graph.remove_edge(i2, i3)
    reachable_indices = shortest_path(graph, i4).keys()
    # get all indices reachable from i4 not going through i2-i3

    graph.add_edge(i2, i3)
    # restore modified graph

    mask = np.array([i in reachable_indices for i in graph.nodes], dtype=bool)
    # generate boolean mask

    # if np.count_nonzero(mask) > int(len(mask)/2):
    #     mask = ~mask
    # if we want to rotate more than half of the indices,
    # invert the selection so that we do less math

    mask[i3] = False
    # do not rotate i3: it would not move,
    # since it lies on the rotation axis

    return mask


def get_quadruplets(graph: Graph) -> Array2D_int:
    """Returns list of quadruplets that indicate potential torsions"""
    # Step 1: Find spanning tree
    spanning_tree = minimum_spanning_tree(graph)

    # Step 2: Add dihedrals for spanning tree
    dihedrals = []

    # For each edge in the spanning tree, we can potentially define a dihedral
    # We need edges that have at least 2 neighbors each to form a 4-point dihedral
    for edge in spanning_tree.edges():
        i, j = edge

        # Find neighbors of i and j in the original graph
        i_neighbors = [n for n in graph.neighbors(i) if n not in (i, j)]
        j_neighbors = [n for n in graph.neighbors(j) if n not in (i, j)]

        if len(i_neighbors) > 0 and len(j_neighbors) > 0:
            # Form dihedral: neighbor_of_i - i - j - neighbor_of_j
            k = i_neighbors[0]  # Choose first available neighbor
            _l = j_neighbors[0]  # Choose first available neighbor
            dihedrals.append((k, i, j, _l))

    return np.array(dihedrals)


def _get_torsions(
    graph: Graph,
    hydrogen_bonds: Sequence[tuple[int, int]],
    double_bonds: Sequence[tuple[int, int]],
    keepdummy: bool = False,
    mode: str = "csearch",
) -> list[Torsion]:
    """Returns list of Torsion objects"""
    torsions = []
    for path in get_quadruplets(graph):
        _, i2, i3, _ = path
        bt: tuple[int, int] = tuple(sorted((i2, i3)))

        if bt not in double_bonds:
            t = Torsion(*path, mode=mode)  # type: ignore[misc]

            if (not t.in_cycle(graph)) and t.is_rotable(graph, hydrogen_bonds, keepdummy=keepdummy):
                torsions.append(t)
    # Create non-redundant torsion objects
    # Rejects (4,3,2,1) if (1,2,3,4) is present
    # Rejects torsions that do not represent a rotable bond

    return torsions


def random_csearch(
    atoms: Array1D_str,
    coords: Array2D_float,
    torsions: Sequence[Torsion],
    graph: Graph,
    constrained_indices: Sequence[Sequence[int]] | None = None,
    n_out: int = 100,
    max_tries: int = 10000,
    rotations: int | None = None,
    title: str = "test",
    logfunction: Callable[[str], None] | None = print,
    interactive_print: bool = True,
    write_torsions: bool = False,
) -> Array3D_float:
    """Random dihedral rotations - quickly generate n_out conformers

    n_out: number of output structures
    max_tries: if n_out conformers are not generated after these number of tries, stop trying
    rotations: number of dihedrals to rotate per conformer. If none, all will be rotated
    """
    t_start_run = time.perf_counter()

    ############################################## LOG TORSIONS

    if logfunction is not None:
        logfunction("\n> Torsion list: (indices: n-fold)")
        for t, torsion in enumerate(torsions):
            logfunction(
                " {:2s} - {:21s} : {}{}{}{} : {}-fold".format(
                    str(t),
                    str(torsion.torsion),
                    atoms[torsion.torsion[0]],
                    atoms[torsion.torsion[1]],
                    atoms[torsion.torsion[2]],
                    atoms[torsion.torsion[3]],
                    torsion.n_fold,
                )
            )

    central_ids = set(flatten([t.torsion[1:3] for t in torsions], int))
    if logfunction is not None:
        logfunction(f"\n> Rotable bonds ids: {' '.join([str(i) for i in sorted(central_ids)])}")

    if write_torsions:
        _write_torsion_vmd(atoms, coords, [torsions], constrained_indices, title=title)
        # logging torsions to file

        torsions_indices = [t.torsion for t in torsions]
        torsions_centers = np.array(
            [np.mean((coords[i2], coords[i3]), axis=0) for _, i2, i3, _ in torsions_indices]
        )

        with open(f"{title}_torsion_centers.xyz", "w") as f:
            write_xyz(torsions_centers, np.array([3 for _ in torsions_centers]), f)

    ############################################## END LOG TORSIONS

    if logfunction is not None:
        logfunction(
            f"\n--> Random dihedral CSearch on {title}\n    mode 2 (random) - {len(torsions)} torsions"
        )

    angles = cartesian_product(*[t.get_angles() for t in torsions])
    # calculating the angles for rotation based on step values

    if rotations is not None:
        mask = np.count_nonzero(angles, axis=1) == rotations
        angles = angles[mask]

    np.random.shuffle(angles)
    # shuffle them so we don't bias conformational sampling

    new_structures: list[Array2D_float] = []

    for a, angle_set in enumerate(angles):
        if interactive_print:
            print(
                f"Generating conformers... ({round(len(new_structures) / n_out * 100)} %) {' ' * 10}",
                end="\r",
            )

        # get a copy of the molecule position as a starting point
        new_coords = np.copy(coords)

        # initialize the number of bonds that actually rotate
        rotated_bonds = 0

        for t, torsion in enumerate(torsions):
            angle = angle_set[t]

            # for every angle we have to rotate, calculate the new coordinates
            if angle != 0:
                mask = _get_rotation_mask(graph, torsion.torsion)
                temp_coords = rotate_dihedral(new_coords, torsion.torsion, angle, mask=mask)

                # if these coordinates are bad and compenetration is present
                if not torsion_comp_check(
                    temp_coords, torsion=torsion.torsion, mask=mask, thresh=1.5
                ):
                    # back off five degrees
                    for _ in range(angle // 5):
                        temp_coords = rotate_dihedral(temp_coords, torsion.torsion, -5, mask=mask)

                        # and reiterate until we have no more compenetrations,
                        # or until we have undone the previous rotation
                        if torsion_comp_check(
                            temp_coords, torsion=torsion.torsion, mask=mask, thresh=1.5
                        ):
                            # print(f'------> DEBUG - backed off {_*5}/{angle} degrees')
                            rotated_bonds += 1
                            break

                else:
                    rotated_bonds += 1

                # update the active coordinates with the temp ones
                new_coords = temp_coords

        # add the rotated molecule to the output list
        if rotated_bonds != 0:
            new_structures.append(new_coords)

            # after adding a molecule to the output, check if we
            # have reached the number of desired output structures
            if len(new_structures) == n_out or a == max_tries:
                break

    # Get a descriptor for how exhaustive the sampling has been
    exhaustiveness = len(new_structures) / np.prod([t.n_fold for t in torsions])

    if logfunction is not None:
        logfunction(
            f"  Generated {len(new_structures)} conformers, (est. {round(100 * exhaustiveness, 2)} % of the total conformational space) - CSearch time {time_to_string(time.perf_counter() - t_start_run)}"
        )

    return np.array(new_structures)


def most_diverse_conformers(
    n: int, structures: Array3D_float | list[Array2D_float]
) -> list[Array2D_float]:
    """TEMP: JUST RETURNS THE TOP n STRUCTURES.

    Previous algo required scikit-learn which was dropped.
    """
    if len(structures) <= n:
        return list(np.array(structures))
    # if we already pruned enough structures to meet the requirement, return them

    indices = np.sort(np.random.choice(len(structures), size=n))
    return list(np.array(structures)[indices])


def csearch(
    atoms: Array1D_str,
    coords: Array2D_float,
    charge: int = 0,
    mult: int = 1,
    constrained_indices: Sequence[tuple[int, int]] | None = None,
    keep_hb: bool = False,
    n: int = 100,
    n_out: int = 100,
    mode: int = 1,
    calc: ASECalculator | None = None,
    method: str | None = None,
    title: str = "test",
    logfunction: Callable[[str], None] | None = print,
    dispatcher: Opt_func_dispatcher | None = None,
    debug: bool = False,
    interactive_print: bool = True,
    write_torsions: bool = False,
) -> Array3D_float:
    """n: number of structures to keep from each torsion cluster
    mode: 0 - torsion clustered - keep the n lowest energy conformers
    1 - torsion clustered - keep the n most diverse conformers
    2 - random dihedral rotations - quickly generate n_out conformers

    n_out: maximum number of output structures

    keep_hb: whether to preserve the presence of current hydrogen bonds or not
    """
    if logfunction is not None:
        if constrained_indices is not None and len(constrained_indices) > 0:
            logfunction(
                f"Constraining {len(constrained_indices)} distance{'s' if len(constrained_indices) > 1 else ''} - {constrained_indices}"
            )
        else:
            logfunction("Free conformational search: no constraints provided.")
            constrained_indices = []

    graph = graphize(atoms, coords)

    if constrained_indices is not None:
        for i1, i2 in constrained_indices:
            graph.add_edge(i1, i2)
    # build a molecular graph of the TS
    # that includes constrained indices pairs

    # ... and hydrogen bonding, if requested
    if keep_hb:
        hydrogen_bonds = _get_hydrogen_bonds(atoms, coords, graph)
        for hb in hydrogen_bonds:
            graph.add_edge(*hb)

        if logfunction is not None:
            if hydrogen_bonds:
                logfunction(f"Preserving {len(hydrogen_bonds)} hydrogen bonds - {hydrogen_bonds}")
            else:
                logfunction("No hydrogen bonds found.")

    else:
        hydrogen_bonds = []
    # get informations on the intra/intermolecular hydrogen
    # bonds that we should avoid disrupting

    if len(fragments := list(connected_components(graph))) > 1:
        # if the molecule graph is not made up of a single connected component

        s = (
            f"{title} has a segmented connectivity graph: double check the input geometry.\n"
            + "if this is supposed to be a complex, FIRECODE was not able to find hydrogen bonds\n"
            + "connecting the molecules, and the algorithm is not designed to reliably perform\n"
            + "conformational searches on loosely bound multimolecular arrangements."
        )

        if keep_hb:
            raise SegmentedGraphError(s)
        # if we already looked for HBs, raise the error

        hydrogen_bonds.extend(_get_hydrogen_bonds(atoms, coords, graph, fragments=fragments))
        # otherwise, look for INTERFRAGMENT HBs only

        if not hydrogen_bonds:
            raise SegmentedGraphError(s)
        # if they are not present, raise error

        for hb in hydrogen_bonds:
            graph.add_edge(*hb)

        if len(list(connected_components(graph))) > 1:
            raise SegmentedGraphError(s)
        # otherwise, add the new HBs linking the pieces
        # and make sure that now we only have one connected component

    double_bonds = get_double_bonds_indices(coords, atoms)
    # get all double bonds - do not rotate these

    torsions = _get_torsions(graph, hydrogen_bonds, double_bonds)
    # get all torsions that we should explore

    for t in torsions:
        t.sort_torsion(graph, constrained_indices)
    # sort torsion indices so that first index of each torsion
    # is the half that will move and is external to the structure

    if not torsions:
        if logfunction is not None:
            logfunction(f"No rotable bonds found for {title}.")
        return np.array([coords])

    if mode in (0, 1):
        return clustered_csearch(
            atoms,
            coords,
            torsions,
            graph,
            constrained_indices=constrained_indices,
            n=n,
            n_out=n_out,
            title=title,
            logfunction=logfunction,
            interactive_print=interactive_print,
            write_torsions=write_torsions,
            debug=debug,
        )

    return random_csearch(
        atoms,
        coords,
        torsions,
        graph,
        constrained_indices=constrained_indices,
        n_out=n_out,
        title=title,
        logfunction=logfunction,
        interactive_print=interactive_print,
        write_torsions=write_torsions,
    )


def clustered_csearch(
    atoms: Array1D_str,
    coords: Array2D_float,
    torsions: Sequence[Torsion],
    graph: Graph,
    charge: int = 0,
    mult: int = 1,
    constrained_indices: Sequence[Sequence[int]] | None = None,
    n: int = 100,
    n_out: int = 100,
    title: str = "test",
    logfunction: Callable[[str], None] | None = print,
    interactive_print: bool = True,
    write_torsions: bool = False,
    debug: bool = False,
) -> Array3D_float:
    """"""

    t_start_run = time.perf_counter()

    tag = "diverse"
    # criteria to choose the best structure of each torsional cluster

    grouped_torsions = [torsions]

    ############################################## LOG TORSIONS

    if logfunction is not None:
        logfunction("\n> Torsion list: (indices: n-fold)")
        for t, torsion in enumerate(torsions):
            logfunction(" {} - {:21s} : {}-fold".format(t, str(torsion.torsion), torsion.n_fold))

        central_ids = set(flatten([t.torsion[1:3] for t in torsions], int))
        logfunction(f"\n> Rotable bonds ids: {' '.join([str(i) for i in sorted(central_ids)])}")

    if write_torsions:
        _write_torsion_vmd(atoms, coords, grouped_torsions, constrained_indices, title=title)
        # logging torsions to file

        torsions_indices = [t.torsion for t in torsions]
        torsions_centers = np.array(
            [np.mean((coords[i2], coords[i3]), axis=0) for _, i2, i3, _ in torsions_indices]
        )

        with open(f"{title}_torsion_centers.xyz", "w") as f:
            write_xyz(torsions_centers, np.array([3 for _ in torsions_centers]), f)

    ############################################## END LOG TORSIONS

    if logfunction is not None:
        logfunction(
            f"\n--> Clustered CSearch on {title}\n"
            + f"    - {len(torsions)} torsions in {len(grouped_torsions)} group{'s' if len(grouped_torsions) != 1 else ''} - "
            + f"{[len(t) for t in grouped_torsions]}"
        )

    output_structures = []
    starting_points = [coords]
    for tg, torsions_group in enumerate(grouped_torsions):
        angles = cartesian_product(*[t.get_angles() for t in torsions_group])
        candidates = len(angles) * len(starting_points)
        # calculating the angles for rotation based on step values

        if logfunction is not None:
            logfunction(
                f"\n> Group {tg + 1}/{len(grouped_torsions)} - {len(torsions_group)} bonds, "
                + f"{[t.n_fold for t in torsions_group]} n-folds, {len(starting_points)} "
                + f"starting point{'s' if len(starting_points) > 1 else ''} = {candidates} conformers"
            )

        new_structures: list[Array2D_float] = []

        for s, sp in enumerate(starting_points):
            if interactive_print:
                print(
                    f"Generating conformers... ({round(s / len(starting_points) * 100)} %) {' ' * 10}",
                    end="\r",
                )

            new_structures.append(sp)

            for angle_set in angles:
                new_coords = np.copy(sp)
                # get a copy of the molecule position as a starting point

                rotated_bonds = 0
                # initialize the number of bonds that actually rotate

                for t, torsion in enumerate(torsions_group):
                    angle = angle_set[t]

                    if angle != 0:
                        mask = _get_rotation_mask(graph, torsion.torsion)
                        temp_coords = rotate_dihedral(new_coords, torsion.torsion, angle, mask=mask)
                        # for every angle we have to rotate, calculate the new coordinates

                        if not torsion_comp_check(
                            temp_coords, torsion=torsion.torsion, mask=mask, thresh=1.5
                        ):
                            # if these coordinates are bad and compenetration is present

                            for _ in range(angle // 5):
                                temp_coords = rotate_dihedral(
                                    temp_coords, torsion.torsion, -5, mask=mask
                                )
                                # back off five degrees

                                if torsion_comp_check(
                                    temp_coords, torsion=torsion.torsion, mask=mask, thresh=1.5
                                ):
                                    # print(f'------> DEBUG - backed off {_*5}/{angle} degrees')
                                    rotated_bonds += 1
                                    break
                                # and reiterate until we have no more compenetrations,
                                # or until we have undone the previous rotation

                        else:
                            rotated_bonds += 1

                        new_coords = temp_coords
                        # update the active coordinates with the temp ones

                if rotated_bonds != 0:
                    new_structures.append(new_coords)
                    # add the rotated molecule to the output list

        torsion_array = np.array([t.torsion for t in torsions])

        if tg + 1 != len(grouped_torsions):
            if n is not None and len(new_structures) > n:
                new_structures = most_diverse_conformers(
                    n,
                    new_structures,
                )

            if logfunction is not None:
                logfunction(
                    f"  Kept the most {tag} {len(new_structures)} starting points for next rotation cluster"
                )

        output_structures.extend(new_structures)
        starting_points = new_structures

    output_structures = list(prune_conformers_tfd(np.array(output_structures), torsion_array)[0])

    if len(new_structures) > n_out:
        output_structures = most_diverse_conformers(
            n_out,
            output_structures,
        )

    exhaustiveness = len(output_structures) / np.prod([t.n_fold for t in torsions])

    if logfunction is not None:
        logfunction(
            f"  Selected the most diverse {len(output_structures)} conformers, corresponding\n"
            + f"  to about {round(100 * exhaustiveness, 2)} % of the total conformational space - CSearch time {time_to_string(time.perf_counter() - t_start_run)}"
        )

    return np.array(output_structures)


def torsion_comp_check(
    coords: Array2D_float,
    torsion: tuple[int, int, int, int],
    mask: Array1D_bool,
    thresh: float = 1.5,
    max_clashes: int = 0,
) -> bool:
    """coords: 3D molecule coordinates
    mask: 1D boolean array with the mask torsion
    thresh: threshold value for when two atoms are considered clashing
    max_clashes: maximum number of clashes to pass a structure
    returns True if the molecule shows less than max_clashes
    """
    _, i2, i3, _ = torsion

    antimask = ~mask
    antimask[i2] = False
    antimask[i3] = False
    # making sure the i2-i3 bond is not included in the clashes

    m1 = coords[mask]
    m2 = coords[antimask]
    # fragment identification by boolean masking

    return int(np.count_nonzero(cdist(m2, m1) < thresh)) <= max_clashes


def _write_torsion_vmd(
    atoms: Array1D_str,
    coords: Array2D_float,
    grouped_torsions: Iterable[Iterable[Torsion]],
    constrained_indices: Sequence[Sequence[int]] | None = None,
    title: str = "test",
) -> None:
    with open(f"{title}.xyz", "w") as f:
        write_xyz(atoms, coords, f)

    path = os.path.join(os.getcwd(), f"{title}_torsional_clusters.vmd")
    with open(path, "w") as f:
        s = (
            "display resetview\n"
            + "mol new {%s}\n" % (os.path.join(os.getcwd(), f"{title}.xyz"))
            + "mol representation Lines 2\n"
            + "mol color ColorID 16\n"
        )

        for group, color in zip(grouped_torsions, (7, 9, 10, 11, 29, 16)):
            for torsion in group:
                s += (
                    "mol selection index %s\n" % (" ".join([str(i) for i in torsion.torsion[1:-1]]))
                    + "mol representation CPK 0.7 0.5 50 50\n"
                    + f"mol color ColorID {color}\n"
                    + "mol material Transparent\n"
                    + "mol addrep top\n"
                )

        if constrained_indices is not None:
            for a, b in constrained_indices:
                s += f"label add Bonds 0/{a} 0/{b}\n"

        f.write(s)


def prune_conformers_tfd(
    structures: Array3D_float, quadruplets: Array2D_int, thresh: int = 10, verbose: bool = False
) -> tuple[Array3D_float, Array1D_bool]:
    """Removes similar structures by repeatedly grouping them into k
    subgroups and removing similar ones. A cache is present to avoid
    repeating TFD computations.

    Similarity occurs for structures with a total angle difference
    greater than thresh degrees
    """
    # Get torsion fingerprints for structures
    tf_mat = _get_tf_mat(structures, quadruplets)

    cache_set = set()
    final_mask = np.ones(structures.shape[0], dtype=bool)

    for k in (5e5, 2e5, 1e5, 5e4, 2e4, 1e4, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1):
        num_active_str = np.count_nonzero(final_mask)

        if k == 1 or 5 * k < num_active_str:
            # proceed only of there are at least five structures per group

            if verbose:
                print(
                    f"Working on subgroups with k={k} ({num_active_str} candidates left) {' ' * 10}",
                    end="\r",
                )

            d = int(len(structures) // k)

            for step in range(int(k)):
                # operating on each of the k subdivisions of the array
                if step == k - 1:
                    _l = len(range(d * step, num_active_str))
                else:
                    _l = len(range(d * step, int(d * (step + 1))))

                # similarity_mat = np.zeros((_l, _l))
                matches = set()

                for i_rel in range(_l):
                    for j_rel in range(i_rel + 1, _l):
                        i_abs = i_rel + (d * step)
                        j_abs = j_rel + (d * step)

                        if (i_abs, j_abs) not in cache_set:
                            # if we have already performed the comparison,
                            # structures were not similar and we can skip them

                            if tfd_similarity(tf_mat[i_abs], tf_mat[j_abs], thresh=thresh):
                                # similarity_mat[i_rel,j_rel] = 1
                                matches.add((i_rel, j_rel))
                                break
                            else:
                                i_abs = i_rel + (d * step)
                                j_abs = j_rel + (d * step)
                                cache_set.add((i_abs, j_abs))

                # for i_rel, j_rel in zip(*np.where(similarity_mat == False)):
                #     i_abs = i_rel+(d*step)
                #     j_abs = j_rel+(d*step)
                #     cache_set.add((i_abs, j_abs))
                # adding indices of structures that are considered equal,
                # so as not to repeat computing their TFD
                # Their index accounts for their position in the initial
                # array (absolute index)

                # matches = [(i,j) for i,j in zip(*np.where(similarity_mat))]
                g = Graph(matches)

                subgraphs = [g.subgraph(c) for c in connected_components(g)]
                groups = [tuple(graph.nodes) for graph in subgraphs]

                best_of_cluster = [group[0] for group in groups]
                # of each cluster, keep the first structure

                rejects_sets = [set(a) - {b} for a, b in zip(groups, best_of_cluster)]
                rejects = []
                for s in rejects_sets:
                    for i in s:
                        rejects.append(i)

                for i in rejects:
                    abs_index = i + d * step
                    final_mask[abs_index] = 0

    return structures[final_mask], final_mask


def _get_tf_mat(structures: Array3D_float, quadruplets: Array2D_int) -> Array2D_float:
    """Get the torsional fingerprint matrix."""
    tf_mat = np.empty(shape=(len(structures), len(quadruplets)), dtype=float)

    for i in range(len(structures)):
        tf_mat[i] = get_torsion_fingerprint(structures[i], quadruplets)

    return tf_mat


def tfd_similarity(tfp1: Array1D_float, tfp2: Array1D_float, thresh: int = 10) -> bool:
    """Return True if the two structure are similar under the torsion fingeprint criteria."""
    # Compute their absolute difference
    deltas = np.abs(tfp1 - tfp2)

    # Correct for rotations over 180 deg
    deltas = np.abs(deltas - (deltas > 180) * 360)

    if np.sum(deltas) < thresh:
        return True

    return False


def get_torsion_fingerprint(coords: Array1D_float, quadruplets: Array2D_int) -> Array1D_float:
    """Get vector with dihedral angle values."""
    out = np.zeros(quadruplets.shape[0], dtype=float)
    for i, q in enumerate(quadruplets):
        i1, i2, i3, i4 = q
        out[i] = dihedral([coords[i1], coords[i2], coords[i3], coords[i4]])
    return out
