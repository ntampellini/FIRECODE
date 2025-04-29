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
import os
import time
from copy import deepcopy

import numpy as np
from networkx import (connected_components, has_path, is_isomorphic,
                      shortest_path, subgraph)
from sklearn.cluster import KMeans, dbscan

from firecode.algebra import norm, norm_of, vec_angle
from firecode.errors import SegmentedGraphError
from firecode.graph_manipulations import (_get_phenyl_ids, findPaths, get_sp_n,
                                        is_amide_n, is_ester_o, is_sp_n,
                                        neighbors)
from firecode.hypermolecule_class import graphize
from firecode.numba_functions import prune_conformers_tfd, torsion_comp_check
from firecode.pt import pt
from firecode.rmsd import rmsd_and_max_numba
from firecode.settings import DEFAULT_FF_LEVELS, FF_CALC
from firecode.utils import (align_structures, cartesian_product, flatten,
                          get_double_bonds_indices, rotate_dihedral,
                          time_to_string, write_xyz)


class Torsion:
    def __repr__(self):
        if hasattr(self, 'n_fold'):
            return f'Torsion({self.i1}, {self.i2}, {self.i3}, {self.i4}; {self.n_fold}-fold)'
        return f'Torsion({self.i1}, {self.i2}, {self.i3}, {self.i4})'

    def __init__(self, i1, i2, i3, i4):
        self.i1 = i1
        self.i2 = i2
        self.i3 = i3
        self.i4 = i4
        self.torsion = (i1, i2, i3 ,i4)

    def in_cycle(self, graph):
        '''
        Returns True if the torsion is part of a cycle
        '''
        graph.remove_edge(self.i2, self.i3)
        cyclical = has_path(graph, self.i1, self.i4)
        graph.add_edge(self.i2, self.i3)
        return cyclical

    def is_rotable(self, graph, hydrogen_bonds, keepdummy=False) -> bool:
        '''
        hydrogen bonds: iterable with pairs of sorted atomic indices
        '''

        if sorted((self.i2, self.i3)) in hydrogen_bonds:
            # self.n_fold = 6
            # # This has to be an intermolecular HB: rotate it
            # return True
            return False

        if _is_free(self.i2, graph) or (
           _is_free(self.i3, graph)):

            if keepdummy or (
               _is_nondummy(self.i2, self.i3, graph) and (
               _is_nondummy(self.i3, self.i2, graph))):

                self.n_fold = self.get_n_fold(graph)
                return True

        return False

    def get_n_fold(self, graph) -> int:

        nums = (graph.nodes[self.i2]['atomnos'],
                graph.nodes[self.i3]['atomnos'])

        if 1 in nums:
            return 6 # H-N, H-O hydrogen bonds
        
        if is_amide_n(self.i2, graph, mode=2) or (
           is_amide_n(self.i3, graph, mode=2)):
           # tertiary amides rotations are 2-fold
           return 2

        if (6 in nums) or (7 in nums) or (16 in nums): # if C, N or S atoms

            sp_n_i2 = get_sp_n(self.i2, graph)
            sp_n_i3 = get_sp_n(self.i3, graph)

            if 3 == sp_n_i2 == sp_n_i3:
                return 3

            if 3 in (sp_n_i2, sp_n_i3): # Csp3-X, Nsp3-X, Ssulfone-X

                if self.mode == 'csearch':
                    return 3
                
                elif self.mode == 'symmetry':
                    return sp_n_i3 or 2
            
            if 2 in (sp_n_i2, sp_n_i3):
                return 2

        return 4 #O-O, S-S, Ar-Ar, Ar-CO, and everything else

    def get_angles(self):
        return {
                2:(0, 180),
                3:(0, 120, 240),
                4:(0, 90, 180, 270),
                6:(0, 60, 120, 180, 240, 300),
                }.get(self.n_fold)

    def sort_torsion(self, graph, constrained_indices) -> None:
        '''
        Acts on the self.torsion tuple leaving it as it is or
        reversing it, so that the first index of it (from which
        rotation will act) is external to the molecule constrained
        indices. That is we make sure to rotate external groups
        and not the whole structure.
        '''
        graph.remove_edge(self.i2, self.i3)
        for d in constrained_indices.flatten():
            if has_path(graph, self.i2, d):
                self.torsion = tuple(reversed(self.torsion))
        graph.add_edge(self.i2, self.i3)

def _is_free(index, graph):
    '''
    Return True if the index specified
    satisfies all of the following:
    - Is not a sp2 carbonyl carbon atom
    - Is not the oxygen atom of an ester
    - Is not the nitrogen atom of a secondary amide (CONHR)

    '''
    if all((
            graph.nodes[index]['atomnos'] == 6,
            is_sp_n(index, graph, 2),
            8 in (graph.nodes[n]['atomnos'] for n in neighbors(graph, index))
          )):
        return False

    if is_amide_n(index, graph, mode=1):
        return False

    if is_ester_o(index, graph):
        return False

    return True

def _is_nondummy(i, root, graph) -> bool:
    '''
    Checks that a molecular rotation along the dihedral
    angle (*, root, i, *) is non-dummy, that is the atom
    at index i, in the direction opposite to the one leading
    to root, has different substituents. i.e. methyl, CF3 and tBu
    rotations should return False.
    '''

    if graph.nodes[i]['atomnos'] not in (6,7):
        return True
    # for now, we only discard rotations around carbon
    # and nitrogen atoms, like methyl/tert-butyl/triphenyl
    # and flat symmetrical rings like phenyl, N-pyrrolyl...

    G = deepcopy(graph)
    nb = neighbors(G, i)
    nb.remove(root)

    if len(nb) == 1:
        if len(neighbors(G, nb[0])) == 2:
            return False
    # if node i has two bonds only (one with root and one with a)
    # and the other atom (a) has two bonds only (one with i)
    # the rotation is considered dummy: some other rotation
    # will account for its freedom (i.e. alkynes, hydrogen bonds)

    # check if it is a phenyl-like rotation
    if len(nb) == 2:

        # get the 6 indices of the aromatic atoms (i1-i6)
        phenyl_indices = _get_phenyl_ids(i, G)

        # compare the two halves of the 6-membered ring (indices i2-i3 region with i5-i6 region)
        if phenyl_indices is not None:
            i1, i2, i3, i4, i5, i6 = phenyl_indices
            G.remove_edge(i3, i4)
            G.remove_edge(i4, i5)
            G.remove_edge(i1, i2)
            G.remove_edge(i1, i6)

            subgraphs = [subgraph(G, _set) for _set in connected_components(G)
                    if i2 in _set or i6 in _set]

            if len(subgraphs) == 2:
                return not is_isomorphic(subgraphs[0], subgraphs[1],
                                            node_match=lambda n1, n2: n1['atomnos'] == n2['atomnos'])
            
            # We should not end up here, but if we do, rotation should not be dummy
            return True

    # if not, compare immediate neighbors of i
    for n in nb:
        G.remove_edge(i, n)

    # make a set of each fragment around the chopped n-i bonds,
    # but only for fragments that are not root nor contain other random,
    # disconnected parts of the graph
    subgraphs_nodes = [_set for _set in connected_components(G)
                       if root not in _set and any(
                           n in _set for n in nb
                       )]

    if len(subgraphs_nodes) == 1:
        return True
        # if not, the torsion is likely to be rotable
        # (tetramethylguanidyl alanine C(β)-N bond)

    subgraphs = [subgraph(G, s) for s in subgraphs_nodes]
    for sub in subgraphs[1:]:
        if not is_isomorphic(subgraphs[0], sub,
                                node_match=lambda n1, n2: n1['atomnos'] == n2['atomnos']):
            return True
    # Care should be taken because chiral centers are not taken into account: a rotation 
    # involving an index where substituents only differ by stereochemistry, and where a 
    # rotation is not an element of symmetry of the subsystem, the rotation is considered
    # dummy even if it would be more correct not to. For rotaionally corrected RMSD this 
    # should only cause small inefficiencies and not lead to discarding any good conformer.

    return False

def _get_hydrogen_bonds(coords, atomnos, graph, d_min=2.5, d_max=3.3, max_angle=45, elements=None, fragments=None):
    '''
    Returns a list of tuples with the indices
    of hydrogen bonding partners.

    An HB is a pair of atoms:
    - with one H and one X (N or O) atom
    - with an Y-X distance between d_min and d_max (i.e. N-O, Angstroms)
    - with an Y-H-X angle below max_angle (i.e. N-H-O, degrees)

    elements: iterable of donors and acceptors atomic numbers. default: ((7, 8), (7, 8))

    If fragments is specified (iterable of iterable of indices for each fragment)
    the function only returns inter-fragment hydrogen bonds.
    '''

    hbs = []
    # initializing output list

    if elements is None:
        elements = ((7, 8), (7, 8, 9))

    het_idx_from = np.array([i for i, a in enumerate(atomnos) if a in elements[0]], dtype=int)
    het_idx_to = np.array([i for i, a in enumerate(atomnos) if a in elements[1]], dtype=int)
    # indices where N or O (or user-specified elements) atoms are present.

    for i1 in het_idx_from:
        for i2 in het_idx_to:

            # if inter-fragment HBs are requested, skip intra-HBs
            if fragments is not None:
                if any(((i1 in f and i2 in f) for f in fragments)):
                    continue

            # keep close pairs
            if d_min < norm_of(coords[i1]-coords[i2]) < d_max:

                # getting the indices of all H atoms attached to them
                Hs = [i for i in (neighbors(graph, i1)) if graph.nodes[i]['atomnos'] == 1]

                # versor connectring the two Heteroatoms
                versor = norm(coords[i2]-coords[i1])

                for iH in Hs:

                    # vectors connecting heteroatoms to H
                    v1 = coords[iH]-coords[i1]
                    v2 = coords[iH]-coords[i2]

                    # lengths of these vectors
                    d1 = norm_of(v1)
                    d2 = norm_of(v2)

                    # scalar projection in the heteroatom direction
                    l1 = v1 @ versor
                    l2 = v2 @ -versor

                    # largest planar angle between Het-H and Het-Het, in degrees (0 to 90°)
                    alfa = vec_angle(v1, versor) if l1 < l2 else vec_angle(v2, -versor)

                    # if the three atoms are not too far from being in line
                    if alfa < max_angle:

                        # adding the correct pair of atoms to results
                        if d1 < d2:
                            hbs.append(sorted((iH,i2)))
                        else:
                            hbs.append(sorted((iH,i1)))

                        break

    return hbs

def _get_rotation_mask(graph, torsion):
    '''
    Get mask for the atoms that will rotate in a torsion:
    all the ones in the graph reachable from the last index
    of the torsion but not going through the central two
    atoms in the torsion quadruplet.
    
    '''
    _, i2, i3, i4 = torsion

    graph.remove_edge(i2, i3)
    reachable_indices = shortest_path(graph, i4).keys()
    # get all indices reachable from i4 not going through i2-i3

    graph.add_edge(i2, i3)
    # restore modified graph

    mask = np.array([i in reachable_indices for i in graph.nodes], dtype=bool)
    # generate boolean mask

    if np.count_nonzero(mask) > int(len(mask)/2):
        mask = ~mask
    # if we want to rotate more than half of the indices,
    # invert the selection so that we do less math

    mask[i3] = False
    # do not rotate i3: would not move,
    # since it lies on rotation axis
    
    return mask

def _get_quadruplets(graph):
    '''
    Returns list of quadruplets that indicate potential torsions
    '''

    allpaths = []
    for node in graph:
        allpaths.extend(findPaths(graph, node, 3))
    # get all possible continuous indices quadruplets

    quadruplets, q_ids = [], []
    for path in allpaths:
        _, i2, i3, _ = path
        q_id = tuple(sorted((i2, i3)))

        if (q_id not in q_ids):

            quadruplets.append(path)
            q_ids.append(q_id)

    # Yields non-redundant quadruplets
    # Rejects (4,3,2,1) if (1,2,3,4) is present

    return np.array(quadruplets)

def _get_torsions(graph, hydrogen_bonds, double_bonds, keepdummy=False, mode="csearch"):
    '''
    Returns list of Torsion objects
    '''

    torsions = []
    for path in _get_quadruplets(graph):
        _, i2, i3, _ = path
        bt = tuple(sorted((i2, i3)))

        if bt not in double_bonds:
            t = Torsion(*path)
            t.mode = mode

            if (not t.in_cycle(graph)) and t.is_rotable(graph, hydrogen_bonds, keepdummy=keepdummy):
                torsions.append(t)
    # Create non-redundant torsion objects
    # Rejects (4,3,2,1) if (1,2,3,4) is present
    # Rejects torsions that do not represent a rotable bond

    return torsions

def _group_torsions_dbscan(coords, torsions, max_size=5):
    '''
    '''
    torsions_indices = [t.torsion for t in torsions]
    # get torsion indices

    torsions_centers = np.array([np.mean((coords[i2], coords[i3]), axis=0) for _, i2, i3, _ in torsions_indices])
    # compute spatial distance

    for eps in np.arange(10, 1.5, -0.5):
        labels = dbscan(torsions_centers, eps=eps, min_samples=1)[1]
        n_clusters = max(labels) + 1
        biggest_cluster_size = max([np.count_nonzero(labels==i) for i in set(labels)])

        if biggest_cluster_size <= max_size:
            break

    output = [[] for _ in range(n_clusters)]
    for torsion, cluster in zip(torsions, labels):
        output[cluster].append(torsion)

    output = sorted(output, key=len)
    # largest groups last
    
    return output

def random_csearch(
                    coords,
                    atomnos,
                    torsions,
                    graph,
                    constrained_indices=None,
                    n_out=100,
                    max_tries=10000,
                    rotations=None,
                    title='test',
                    logfunction=print,
                    interactive_print=True,
                    write_torsions=False
                ):
    '''
    Random dihedral rotations - quickly generate n_out conformers

    n_out: number of output structures
    max_tries: if n_out conformers are not generated after these number of tries, stop trying
    rotations: number of dihedrals to rotate per conformer. If none, all will be rotated
    '''

    t_start_run = time.perf_counter()

    ############################################## LOG TORSIONS

    logfunction('\n> Torsion list: (indices: n-fold)')
    for i, t in enumerate(torsions):
        logfunction(' {:2s} - {:21s} : {}{}{}{} : {}-fold'.format(
                                                               str(i),
                                                               str(t.torsion),
                                                               pt[atomnos[t.torsion[0]]].symbol,
                                                               pt[atomnos[t.torsion[1]]].symbol,
                                                               pt[atomnos[t.torsion[2]]].symbol,
                                                               pt[atomnos[t.torsion[3]]].symbol,
                                                               t.n_fold))

    central_ids = set(flatten([t.torsion[1:3] for t in torsions], int))
    logfunction(f'\n> Rotable bonds ids: {" ".join([str(i) for i in sorted(central_ids)])}')

    if write_torsions:
        _write_torsion_vmd(coords, atomnos, constrained_indices, [torsions], title=title)
        # logging torsions to file

        torsions_indices = [t.torsion for t in torsions]
        torsions_centers = np.array([np.mean((coords[i2], coords[i3]), axis=0) for _, i2, i3, _ in torsions_indices])

        with open(f'{title}_torsion_centers.xyz', 'w') as f:
            write_xyz(torsions_centers, np.array([3 for _ in torsions_centers]), f)

    ############################################## END LOG TORSIONS

    logfunction(f'\n--> Random dihedral CSearch on {title}\n    mode 2 (random) - {len(torsions)} torsions')
    
    angles = cartesian_product(*[t.get_angles() for t in torsions])
    # calculating the angles for rotation based on step values

    if rotations is not None:
        mask = (np.count_nonzero(angles, axis=1) == rotations)
        angles = angles[mask]

    np.random.shuffle(angles)
    # shuffle them so we don't bias conformational sampling

    new_structures = []

    for a ,angle_set in enumerate(angles):

        if interactive_print:
            print(f'Generating conformers... ({round(len(new_structures)/n_out*100)} %) {" "*10}', end='\r')

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
                if not torsion_comp_check(temp_coords, torsion=torsion.torsion, mask=mask, thresh=1.5):

                    # back off five degrees
                    for _ in range(angle//5):
                        temp_coords = rotate_dihedral(temp_coords, torsion.torsion, -5, mask=mask)
                        
                        # and reiterate until we have no more compenetrations,
                        # or until we have undone the previous rotation
                        if torsion_comp_check(temp_coords, torsion=torsion.torsion, mask=mask, thresh=1.5):
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

    # make an array out of them
    new_structures = np.array(new_structures)

    # Get a descriptor for how exhaustive the sampling has been
    exhaustiveness = len(new_structures) / np.prod([t.n_fold for t in torsions])

    logfunction(f'  Generated {len(new_structures)} conformers, (est. {round(100*exhaustiveness, 2)} % of the total conformational space) - CSearch time {time_to_string(time.perf_counter()-t_start_run)}')

    return new_structures

def csearch(
            coords,
            atomnos,
            constrained_indices=None,
            keep_hb=False,
            ff_opt=False,
            n=100,
            n_out=100,
            mode=1,
            calc=None,
            method=None,
            title='test',
            logfunction=print,
            interactive_print=True,
            write_torsions=False):
    '''
    n: number of structures to keep from each torsion cluster
    mode: 0 - torsion clustered - keep the n lowest energy conformers
    1 - torsion clustered - keep the n most diverse conformers
    2 - random dihedral rotations - quickly generate n_out conformers

    n_out: maximum number of output structures

    keep_hb: whether to preserve the presence of current hydrogen bonds or not
    '''

    calc = FF_CALC if calc is None else calc
    method = DEFAULT_FF_LEVELS[calc] if method is None else method
    # Set default calculator attributes if user did not specify them

    if constrained_indices is not None and len(constrained_indices) > 0:
        logfunction(f'Constraining {len(constrained_indices)} distance{"s" if len(constrained_indices) > 1 else ""} - {constrained_indices}')
    else:
        logfunction('Free conformational search: no constraints provided.')
        constrained_indices = np.array([])

    graph = graphize(coords, atomnos)
    for i1, i2 in constrained_indices:
        graph.add_edge(i1, i2)
    # build a molecular graph of the TS
    # that includes constrained indices pairs
    
    # ... and hydrogen bonding, if requested
    if keep_hb:
        hydrogen_bonds = _get_hydrogen_bonds(coords, atomnos, graph)
        for hb in hydrogen_bonds:
            graph.add_edge(*hb)

        if hydrogen_bonds:
            logfunction(f'Preserving {len(hydrogen_bonds)} hydrogen bonds - {hydrogen_bonds}')
        else:
            logfunction('No hydrogen bonds found.')

    else:
        hydrogen_bonds = []
    # get informations on the intra/intermolecular hydrogen
    # bonds that we should avoid disrupting

    if len(fragments := list(connected_components(graph))) > 1:
    # if the molecule graph is not made up of a single connected component

        s = (f'{title} has a segmented connectivity graph: double check the input geometry.\n' +
              'if this is supposed to be a complex, FIRECODE was not able to find hydrogen bonds\n' +
              'connecting the molecules, and the algorithm is not designed to reliably perform\n'+ 
              'conformational searches on loosely bound multimolecular arrangements.')

        if keep_hb:
            raise SegmentedGraphError(s)
        # if we already looked for HBs, raise the error

        hydrogen_bonds.extend(_get_hydrogen_bonds(coords, atomnos, graph, fragments=fragments))
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

    double_bonds = get_double_bonds_indices(coords, atomnos)
    # get all double bonds - do not rotate these
    
    torsions = _get_torsions(graph, hydrogen_bonds, double_bonds)
    # get all torsions that we should explore

    for t in torsions:
        t.sort_torsion(graph, constrained_indices)
    # sort torsion indices so that first index of each torsion
    # is the half that will move and is external to the structure

    if not torsions:
        logfunction(f'No rotable bonds found for {title}.')
        return np.array([coords])

    if mode in (0,1):
        return clustered_csearch(
                                    coords,
                                    atomnos,
                                    torsions,
                                    graph,
                                    constrained_indices=constrained_indices,
                                    ff_opt=ff_opt,
                                    n=n,
                                    n_out=n_out,
                                    mode=mode,
                                    calc=calc,
                                    method=method,
                                    title=title,
                                    logfunction=logfunction,
                                    interactive_print=interactive_print,
                                    write_torsions=write_torsions
                                )

    return random_csearch(
                                    coords,
                                    atomnos,
                                    torsions,
                                    graph,
                                    constrained_indices=constrained_indices,
                                    n_out=n_out,
                                    title=title,
                                    logfunction=logfunction,
                                    interactive_print=interactive_print,
                                    write_torsions=write_torsions
                                )

def clustered_csearch(
                        coords,
                        atomnos,
                        torsions,
                        graph,
                        constrained_indices=None,
                        ff_opt=False,
                        n=100,
                        n_out=100,
                        mode=1,
                        calc=None,
                        method=None,
                        title='test',
                        logfunction=print,
                        interactive_print=True,
                        write_torsions=False):
    '''
    n: number of structures to keep from each torsion cluster
    mode: 0 - torsion clustered - keep the n lowest energy conformers
    1 - torsion clustered - keep the n most diverse conformers

    n_out: maximum number of output structures

    keep_hb: whether to preserve the presence of current hydrogen bonds or not
    '''

    assert mode != 0 or ff_opt, 'Either leave mode=1 or turn on force field optimization'
    assert mode in (0,1), 'The mode keyword can only be 0 or 1'

    t_start_run = time.perf_counter()

    tag = ('stable', 'diverse')[mode]
    # criteria to choose the best structure of each torsional cluster

    if len(torsions) < 9:
        grouped_torsions = [torsions]

    else:
        grouped_torsions = _group_torsions_dbscan(coords,
                                              torsions,
                                              max_size=3 if ff_opt else 5)

    ############################################## LOG TORSIONS

    logfunction('\n> Torsion list: (indices: n-fold)')
    for i, t in enumerate(torsions):
        logfunction(' {} - {:21s} : {}-fold'.format(i, str(t.torsion), t.n_fold))

    central_ids = set(flatten([t.torsion[1:3] for t in torsions], int))
    logfunction(f'\n> Rotable bonds ids: {" ".join([str(i) for i in sorted(central_ids)])}')

    if write_torsions:
        _write_torsion_vmd(coords, atomnos, constrained_indices, grouped_torsions, title=title)
        # logging torsions to file

        torsions_indices = [t.torsion for t in torsions]
        torsions_centers = np.array([np.mean((coords[i2], coords[i3]), axis=0) for _, i2, i3, _ in torsions_indices])

        with open(f'{title}_torsion_centers.xyz', 'w') as f:
            write_xyz(torsions_centers, np.array([3 for _ in torsions_centers]), f)

    ############################################## END LOG TORSIONS

    logfunction(f'\n--> Clustered CSearch on {title}\n    mode {mode} ({"stability" if mode == 0 else "diversity"}) - ' +
                f'{len(torsions)} torsions in {len(grouped_torsions)} group{"s" if len(grouped_torsions) != 1 else ""} - ' +
                f'{[len(t) for t in grouped_torsions]}')
    
    output_structures = []
    starting_points = [coords]
    for tg, torsions_group in enumerate(grouped_torsions):

        angles = cartesian_product(*[t.get_angles() for t in torsions_group])
        candidates = len(angles)*len(starting_points)
        # calculating the angles for rotation based on step values

        logfunction(f'\n> Group {tg+1}/{len(grouped_torsions)} - {len(torsions_group)} bonds, ' +
                      f'{[t.n_fold for t in torsions_group]} n-folds, {len(starting_points)} ' + 
                      f'starting point{"s" if len(starting_points) > 1 else ""} = {candidates} conformers')

        new_structures = []

        for s, sp in enumerate(starting_points):

            if interactive_print:
                print(f'Generating conformers... ({round(s/len(starting_points)*100)} %) {" "*10}', end='\r')

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
                       
                        if not torsion_comp_check(temp_coords, torsion=torsion.torsion, mask=mask, thresh=1.5):
                        # if these coordinates are bad and compenetration is present

                            for _ in range(angle//5):
                                temp_coords = rotate_dihedral(temp_coords, torsion.torsion, -5, mask=mask)
                                # back off five degrees
                                
                                if torsion_comp_check(temp_coords, torsion=torsion.torsion, mask=mask, thresh=1.5):
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

        new_structures = np.array(new_structures)
        torsion_array = np.array([t.torsion for t in torsions])

        energies = None
        if ff_opt:

            t_start = time.perf_counter()

            energies = np.zeros(new_structures.shape[0])
            for c, new_coords in enumerate(np.copy(new_structures)):

                from firecode.optimization_methods import optimize
                opt_coords, energy, success = optimize(new_coords,
                                                        atomnos,
                                                        calc,
                                                        method=method,
                                                        constrained_indices=constrained_indices)

                if success:
                    new_structures[c] = opt_coords
                    energies[c] = energy

                else:
                    energies[c] = 1E10

            logfunction(f'Optimized {len(new_structures)} structures at {method} level ({time_to_string(time.perf_counter()-t_start)})')

        if tg+1 != len(grouped_torsions):
            if n is not None and len(new_structures) > n:

                if mode == 0:
                    new_structures, energies = zip(*sorted(zip(new_structures, energies), key=lambda x: x[1]))
                    new_structures = new_structures[0:n]

                if mode == 1:
                    new_structures = most_diverse_conformers(n, new_structures, torsion_array,
                                                                energies=energies,
                                                                interactive_print=interactive_print)

            logfunction(f'  Kept the most {tag} {len(new_structures)} starting points for next rotation cluster')

        output_structures.extend(new_structures)
        starting_points = new_structures

    output_structures = np.array(output_structures)
    output_structures, _ = prune_conformers_tfd(output_structures, torsion_array)

    if len(new_structures) > n_out:

        if mode == 0:
            output_structures, energies = zip(*sorted(zip(output_structures, energies), key=lambda x: x[1]))
            output_structures = output_structures[0:n_out]
            output_structures = np.array(output_structures)

        if mode == 1:
            output_structures = most_diverse_conformers(n_out, output_structures,
                                                        torsion_array=torsion_array,
                                                        energies=energies,
                                                        interactive_print=interactive_print)

    exhaustiveness = len(output_structures) / np.prod([t.n_fold for t in torsions])

    logfunction(f'  Selected the {"best" if mode == 0 else "most diverse"} {len(output_structures)} conformers, corresponding\n' +
                f'  to about {round(100*exhaustiveness, 2)} % of the total conformational space - CSearch time {time_to_string(time.perf_counter()-t_start_run)}')

    return output_structures

def most_diverse_conformers(n, structures, torsion_array, energies=None, interactive_print=False):
    '''
    Return the n most diverse structures from the set.
    First removes similar structures based on torsional fingerprints, then divides them in n subsets and:
    - If the enrgy list is given, chooses the
    one with the lowest energy from each.
    - If it is not, picks the most diverse structures.
    
    '''
        
    if len(structures) <= n:
        return structures
    # if we already pruned enough structures to meet the requirement, return them

    if n > 300:
        indices = np.sort(np.random.choice(len(structures), size=n))
        return structures[indices]
    # For now, the algorithm scales badly with number of clusters.
    # If there are too many to compute, just choose randomly

    if interactive_print:
        print(f'Removing similar structures...{" "*10}', end='\r')

    structures, _ = prune_conformers_tfd(structures, torsion_array)
    # remove structrures with too similar TFPs

    if len(structures) <= n:
        return structures
    # if we already pruned enough structures to meet the requirement, return them

    if interactive_print:
        print(f'Aligning structures...{" "*10}', end='\r')

    structures = align_structures(structures)
    features = structures.reshape((structures.shape[0], structures.shape[1]*structures.shape[2]))
    # reduce the dimensionality of the rest of the structure array to cluster them with KMeans

    if interactive_print:
        print(f'Performing KMeans clustering...{" "*10}', end='\r')

    kmeans = KMeans(n_clusters=n)
    kmeans.fit(features)
    # Generate and train the model

    # if energies are given, pick the lowest energy structure from each cluster
    if energies is not None:
        clusters = [[] for _ in range(n)]
        for coords, energy, c in zip(structures, energies, kmeans.labels_):
            clusters[c].append((coords, energy))

        output = []
        for group in clusters:
            sorted_s, _ = zip(*sorted(group, key=lambda x: x[1]))
            output.append(sorted_s[0])

    # if not, from each non-empty cluster yield the structure that is more distant from the other clusters
    else:
        centers = kmeans.cluster_centers_.reshape((n, *structures.shape[1:3]))

        clusters = [[] for _ in range(n)]
        for coords, c in zip(structures, kmeans.labels_):
            clusters[c].append(coords)

        r = np.arange(len(clusters))
        output = []

        # take one from each non-empty cluster
        for cluster in clusters:

            if cluster:
                cumdists = [np.sum(np.linalg.norm(centers[r!=c]-ref, axis=2)) for c, ref in enumerate(cluster)]

                furthest = cluster[cumdists.index(max(cumdists))]
                output.append(furthest)

    return np.array(output)

def _write_torsion_vmd(coords, atomnos, constrained_indices, grouped_torsions, title='test'):

    with open(f'{title}.xyz', 'w') as f:
        write_xyz(coords, atomnos, f)

    path = os.path.join(os.getcwd(), f'{title}_torsional_clusters.vmd')
    with open(path, 'w') as f:
        s = ('display resetview\n' +
            'mol new {%s}\n' % (os.path.join(os.getcwd() + f'\{title}.xyz')) +
            'mol representation Lines 2\n' +
            'mol color ColorID 16\n'
            )

        for group, color in zip(grouped_torsions, (7,9,10,11,29,16)):
            for torsion in group:
                s += ('mol selection index %s\n' % (' '.join([str(i) for i in torsion.torsion[1:-1]])) +
                    'mol representation CPK 0.7 0.5 50 50\n' +
                f'mol color ColorID {color}\n' +
                    'mol material Transparent\n' +
                    'mol addrep top\n')

        for a, b in constrained_indices:
            s += f'label add Bonds 0/{a} 0/{b}\n'


        f.write(s)

def rotationally_corrected_rmsd_and_max(ref, coord, atomnos, torsions, graph, angles, debugfunction=None):

    torsion_corrections = [0 for _ in torsions]

    # Now rotate every dummy torsion by the appropriate increment until we minimize local RMSD 
    for i, torsion in enumerate(torsions):
       
        best_rmsd = 1E10

        # for angle_set in combinations
        # Look for the rotational angle set that minimizes the torsion RMSD and save it for later
        for angle in angles[i]:

            coord = rotate_dihedral(coord,
                                    torsion, 
                                    angle,
                                    indices_to_be_moved=[torsion[3]])
            
            locally_corrected_rmsd, _ = rmsd_and_max_numba(ref[torsion], coord[torsion])

            if locally_corrected_rmsd < best_rmsd:
                best_rmsd = locally_corrected_rmsd
                torsion_corrections[i] = angle

            # it is faster to undo the rotation rather than working with a copy of coords
            coord = rotate_dihedral(coord,
                                    torsion, 
                                    -angle,
                                    indices_to_be_moved=[torsion[3]])
            
        # now rotate that angle to the desired orientation before going to the next angle
        if torsion_corrections[i] != 0:
            coord = rotate_dihedral(coord,
                                    torsion, 
                                    torsion_corrections[i],
                                    mask=_get_rotation_mask(graph, torsion))

        if debugfunction is not None:
            debugfunction(f"Torsion {i+1} - {torsion}: best corr = {torsion_corrections[i]}°, 4-atom RMSD: " +
                          f"{best_rmsd:.3f} A, global RMSD: {rmsd_and_max_numba(ref[(atomnos != 1)], coord[(atomnos != 1)])[0]:.3f}")

    # we should have the optimal orientation on all torsions now:
    # calculate the RMSD (only on heavy atoms)
    rmsd, maxdev = rmsd_and_max_numba(ref[(atomnos != 1)], coord[(atomnos != 1)])

    # since we could have segmented graphs, and therefore potentially only rotate
    # subsets of the graph where the torsion last two indices are,
    # we have to undo the final rotation too (would not be needed for connected graphs)
    for torsion, optimal_angle in zip(reversed(torsions), reversed(torsion_corrections)):
        coord = rotate_dihedral(coord,
                                torsion, 
                                -optimal_angle,
                                mask=_get_rotation_mask(graph, torsion))
        

    return rmsd, maxdev
