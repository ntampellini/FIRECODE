# coding=utf-8
'''
FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2024 Nicolò Tampellini

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

'''
from copy import deepcopy
from itertools import combinations

import numpy as np
from networkx import (all_simple_paths, connected_components,
                      from_numpy_matrix, get_node_attributes,
                      set_node_attributes)

from firecode.algebra import all_dists, dihedral, norm_of
from firecode.pt import pt


def d_min_bond(e1, e2, factor=1.2):
    return factor * (pt[e1].covalent_radius + pt[e2].covalent_radius)
    # return 0.2 + (pt[e1].covalent_radius + pt[e2].covalent_radius)
# if this is somewhat prone to bugs, this might help https://cccbdb.nist.gov/calcbondcomp1x.asp

def graphize(coords, atomnos, mask=None):
    '''
    :params coords: atomic coordinates as 3D vectors
    :params atomnos: atomic numbers as a list
    :params mask: bool array, with False for atoms
    to be excluded in the bond evaluation
    :return connectivity graph
    
    '''

    mask = np.array([True for _ in atomnos], dtype=bool) if mask is None else mask

    matrix = np.zeros((len(coords),len(coords)))
    for i, _ in enumerate(coords):
        for j in range(i,len(coords)):
            if mask[i] and mask[j]:
                if norm_of(coords[i]-coords[j]) < d_min_bond(atomnos[i], atomnos[j]):
                    matrix[i][j] = 1

    graph = from_numpy_matrix(matrix)
    set_node_attributes(graph, dict(enumerate(atomnos)), 'atomnos')

    return graph

def neighbors(graph, index):
    # neighbors = list([(a, b) for a, b in graph.adjacency()][index][1].keys())
    neighbors = list(graph.neighbors(index))
    if index in neighbors:
        neighbors.remove(index)
    return neighbors

def is_sp_n(index, graph, n):
    '''
    Returns True if the sp_n value matches the input
    '''
    sp_n = get_sp_n(index, graph)
    if sp_n == n:
        return True
    return False

def get_sp_n(index, graph):
    '''
    Returns n, that is the apex of sp^n hybridization for CONPS atoms.
    This is just an assimilation to the carbon geometry in relation to sp^n:
    - sp(1) is linear
    - sp2 is planar
    - sp3 is tetraedral
    This is mainly used to understand if a torsion is to be rotated or not.
    '''
    element = graph.nodes[index]['atomnos']

    if element not in (6,7,8,15,16):
        return None

    d = {
        6:{2:1, 3:2, 4:3}, # C - 2 neighbors means sp, 3 nb means sp2, 4 nb sp3
        7:{2:2, 3:None, 4:3},      # N - 2 neighbors means sp2, 3 nb could mean sp3 or sp2, 4 nb sp3
        8:{1:2, 2:3, 3:3, 4:3},    # O
        15:{2:2, 3:3, 4:3},        # P - like N
        16:{2:2, 3:3, 4:3},        # S
    }
    return d[element].get(len(neighbors(graph, index)))

def is_amide_n(index, graph, mode=-1):
    '''
    Returns true if the nitrogen atom at the given
    index is a nitrogen and is part of an amide.
    Carbamates and ureas are considered amides.

    mode:
    -1 - any amide
    0 - primary amide (CONH2)
    1 - secondary amide (CONHR)
    2 - tertiary amide (CONR2)
    '''
    if graph.nodes[index]['atomnos'] == 7:
        # index must be a nitrogen atom

        nb = neighbors(graph, index)
        nb_atomnos = [graph.nodes[j]['atomnos'] for j in nb]

        if mode != -1:
            if nb_atomnos.count(1) != (2,1,0)[mode]:
                # primary amides need to have 1H, secondary amides none
                return False

        for n in nb:
            if graph.nodes[n]['atomnos'] == 6:
            # there must be at least one carbon atom next to N

                nb_nb = neighbors(graph, n)
                if len(nb_nb) == 3:
                # bonded to three atoms

                    nb_nb_sym = [graph.nodes[i]['atomnos'] for i in nb_nb]
                    if 8 in nb_nb_sym:
                        return True
                        # and at least one of them has to be an oxygen
    return False

def is_ester_o(index, graph):
    '''
    Returns true if the atom at the given
    index is an oxygen and is part of an ester.
    Carbamates and carbonates return True,
    Carboxylic acids return False.
    '''
    if graph.nodes[index]['atomnos'] == 8:
        nb = neighbors(graph, index)
        if 1 not in nb:
            for n in nb:
                if graph.nodes[n]['atomnos'] == 6:
                    nb_nb = neighbors(graph, n)
                    if len(nb_nb) == 3:
                        nb_nb_sym = [graph.nodes[i]['atomnos'] for i in nb_nb]
                        if nb_nb_sym.count(8) > 1:
                            return True
    return False

def is_phenyl(coords):
    '''
    :params coords: six coordinates of C/N atoms
    :return tuple: bool indicating if the six atoms look like part of a
                   phenyl/naphtyl/pyridine system, coordinates for the center of that ring

    NOTE: quinones would show as aromatic: it is okay, since they can do π-stacking as well.
    '''

    if np.max(all_dists(coords, coords)) > 3:
        return False
    # if any atomic couple is more than 3 A away from each other, this is not a Ph

    threshold_delta = 1 - np.cos(10 * np.pi/180)
    flat_delta = 1 - np.abs(np.cos(dihedral(coords[[0,1,2,3]]) * np.pi/180))

    if flat_delta < threshold_delta:
        flat_delta = 1 - np.abs(np.cos(dihedral(coords[[0,1,2,3]]) * np.pi/180))
        if flat_delta < threshold_delta:
            # print('phenyl center at', np.mean(coords, axis=0))
            return True
    
    return False

def get_phenyls(coords, atomnos):
    '''
    returns a (n, 6, 3) array where the first
    dimension is the aromatic rings detected
    '''
    if len(atomnos) < 6:
        return np.array([])

    output = []

    c_n_indices = np.fromiter((i for i, a in enumerate(atomnos) if a in (6,7)), dtype=atomnos.dtype)
    comb = combinations(c_n_indices, 6)

    for c in comb:
        mask = np.fromiter((i in c for i in range(len(atomnos))), dtype=bool)
        coords_ = coords[mask]
        if is_phenyl(coords_):
            output.append(coords_)

    return np.array(output)

def _get_phenyl_ids(i, G):
    '''
    If i is part of a phenyl group, return the six
    heavy atom indices associated with the ring
    '''
    for n in neighbors(G, i):
        paths = all_simple_paths(G, source=i, target=n, cutoff=6)
        for path in paths:
            if len(path) == 6:
                if all(G.nodes[n]['atomnos'] != 1 for n in path):
                    if all(len(neighbors(G, i)) == 3 for i in path):
                        return path
    
    return None

def findPaths(G, u, n, excludeSet = None):
    '''
    Recursively find all paths of a NetworkX
    graph G with length = n, starting from node u
    '''
    if excludeSet is None:
        excludeSet = set([u])

    else:
        excludeSet.add(u)

    if n == 0:
        return [[u]]

    paths = [[u]+path for neighbor in G.neighbors(u) if neighbor not in excludeSet for path in findPaths(G,neighbor,n-1,excludeSet)]
    excludeSet.remove(u)

    return paths

def is_sigmatropic(mol, conf):
    '''
    mol: Hypermolecule object
    conf: conformer index

    A hypermolecule is considered sigmatropic when:
    - has 2 reactive atoms
    - they are of sp2 or analogous types
    - they are connected, or at least one path connecting them
    is made up of atoms that do not make more than three bonds each
    - they are less than 3 A apart (cisoid propenal makes it, transoid does not)

    Used to set the mol.sigmatropic attribute, that affects orbital
    building (p or n lobes) for Ketone and Imine reactive atoms classes.
    '''
    sp2_types = (
                'Ketone',
                'Imine',
                'sp2',
                'sp',
                'bent carbene'
                )
    if len(mol.reactive_indices) == 2:

        i1, i2 = mol.reactive_indices
        if norm_of(mol.atomcoords[conf][i1] - mol.atomcoords[conf][i2]) < 3:

            if all([str(r_atom) in sp2_types for r_atom in mol.reactive_atoms_classes_dict[conf].values()]):

                paths = all_simple_paths(mol.graph, i1, i2)

                for path in paths:
                    path = path[1:-1]

                    full_sp2 = True
                    for index in path:
                        if len(neighbors(mol.graph, index))-2 > 1:
                            full_sp2 = False
                            break

                    if full_sp2:
                        return True
    return False

def is_vicinal(mol):
    '''
    A hypermolecule is considered vicinal when:
    - has 2 reactive atoms
    - they are of sp3 or Single Bond type
    - they are bonded

    Used to set the mol.sp3_sigmastar attribute, that affects orbital
    building (BH4 or agostic-like behavior) for Sp3 and Single Bond reactive atoms classes.
    '''
    vicinal_types = (
                'sp3',
                'Single Bond',
                )

    if len(mol.reactive_indices) == 2:

        i1, i2 = mol.reactive_indices

        if all([str(r_atom) in vicinal_types for r_atom in mol.reactive_atoms_classes_dict[0].values()]):
            if i1 in neighbors(mol.graph, i2):
                return True

    return False

def get_sum_graph(graphs, extra_edges=None):
    '''
    Creates a graph containing all graphs, added in 
    sequence, and then adds the specified extra edges
    (with cumulative numbering).
    '''

    graph, *extra = graphs
    out = deepcopy(graph)
    cum_atomnos = list(get_node_attributes(graphs[0], "atomnos").values())

    for g in extra:
        n = len(out.nodes())
        for e1, e2 in g.edges():
            out.add_edge(e1+n, e2+n)

        cum_atomnos += list(get_node_attributes(g, "atomnos").values())

    out.is_single_molecule = (len(list(connected_components(out))) == 1)

    if extra_edges is not None:
        for e1, e2 in extra_edges:
            out.add_edge(e1, e2)

    set_node_attributes(out, dict(enumerate(cum_atomnos)), 'atomnos')

    return out
