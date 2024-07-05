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

from networkx import Graph, connected_components 
import numpy as np
from numba import njit, float32, prange

from firecode.algebra import all_dists, dihedral, norm_of

@njit
def torsion_comp_check(coords, torsion, mask, thresh=1.5, max_clashes=0) -> bool:
    '''
    coords: 3D molecule coordinates
    mask: 1D boolean array with the mask torsion
    thresh: threshold value for when two atoms are considered clashing
    max_clashes: maximum number of clashes to pass a structure
    returns True if the molecule shows less than max_clashes
    '''
    _, i2, i3, _ = torsion


    antimask = ~mask
    antimask[i2] = False
    antimask[i3] = False
    # making sure the i2-i3 bond is not included in the clashes

    m1 = coords[mask]
    m2 = coords[antimask]
    # fragment identification by boolean masking

    return 0 if np.count_nonzero(all_dists(m2,m1) < thresh) > max_clashes else 1
 
@njit
def count_clashes(coords):
    '''
    '''
    return np.count_nonzero(
                            (all_dists(coords,coords) < 0.5) & (
                             all_dists(coords,coords) > 0)
                            )


@njit
def compenetration_check(coords, ids=None, thresh=1.5, max_clashes=0) -> bool:
    '''
    coords: 3D molecule coordinates
    ids: 1D array with the number of atoms for each 
    molecule (contiguous fragments in array)
    thresh: threshold value for when two atoms are considered clashing
    max_clashes: maximum number of clashes to pass a structure
    returns True if the molecule shows less than max_clashes
    
    '''

    if ids is None:
        return 0 if count_clashes(coords) > max_clashes else 1

    if len(ids) == 2:
    # Bimolecular

        m1 = coords[0:ids[0]]
        m2 = coords[ids[0]:]
        # fragment identification by length (contiguous)

        return 0 if np.count_nonzero(all_dists(m2,m1) < thresh) > max_clashes else 1

    # if len(ids) == 3:

    clashes = 0
    # max_clashes clashes is good, max_clashes + 1 is not

    m1 = coords[0:ids[0]]
    m2 = coords[ids[0]:ids[0]+ids[1]]
    m3 = coords[ids[0]+ids[1]:]
    # fragment identification by length (contiguous)

    clashes += np.count_nonzero(all_dists(m2,m1) < thresh)
    if clashes > max_clashes:
        return 0

    clashes += np.count_nonzero(all_dists(m3,m2) < thresh)
    if clashes > max_clashes:
        return 0

    clashes += np.count_nonzero(all_dists(m1,m3) < thresh)
    if clashes > max_clashes:
        return 0

    return 1

def scramble(array, sequence):
    return np.array([array[s] for s in sequence])

def prune_conformers_tfd(structures, quadruplets, thresh=10, verbose=False):
    '''
    Removes similar structures by repeatedly grouping them into k
    subgroups and removing similar ones. A cache is present to avoid
    repeating TFD computations.
    
    Similarity occurs for structures with a total angle difference
    greater than thresh degrees
    '''

    # Get torsion fingerprints for structures
    tf_mat = _get_tf_mat(structures, quadruplets)

    cache_set = set()
    final_mask = np.ones(structures.shape[0], dtype=bool)
    
    for k in (5e5, 2e5, 1e5, 5e4, 2e4, 1e4,
              5000, 2000, 1000, 500, 200, 100,
              50, 20, 10, 5, 2, 1):

        num_active_str = np.count_nonzero(final_mask)
        
        if k == 1 or 5*k < num_active_str:
        # proceed only of there are at least five structures per group

            if verbose:      
                print(f'Working on subgroups with k={k} ({num_active_str} candidates left) {" "*10}', end='\r')

            d = int(len(structures) // k)

            for step in range(int(k)):
            # operating on each of the k subdivisions of the array
                if step == k-1:
                    _l = len(range(d*step, num_active_str))
                else:
                    _l = len(range(d*step, int(d*(step+1))))

                # similarity_mat = np.zeros((_l, _l))
                matches = set()

                for i_rel in range(_l):
                    for j_rel in range(i_rel+1,_l):

                        i_abs = i_rel+(d*step)
                        j_abs = j_rel+(d*step)

                        if (i_abs, j_abs) not in cache_set:
                        # if we have already performed the comparison,
                        # structures were not similar and we can skip them

                            if tfd_similarity(tf_mat[i_abs],
                                              tf_mat[j_abs],
                                              thresh=thresh):

                                # similarity_mat[i_rel,j_rel] = 1
                                matches.add((i_rel,j_rel))
                                break
                            else:
                                i_abs = i_rel+(d*step)
                                j_abs = j_rel+(d*step)
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
                    abs_index = i + d*step
                    final_mask[abs_index] = 0

    return structures[final_mask], final_mask

@njit(parallel=True)
def _get_tf_mat(structures, quadruplets):
    '''
    '''
    tf_mat = np.empty(shape=(len(structures), len(quadruplets)), dtype=float32)

    for i in prange(len(structures)):
        tf_mat[i] = get_torsion_fingerprint(structures[i], quadruplets)

    return tf_mat

@njit
def tfd_similarity(tfp1, tfp2, thresh=10) -> bool:
    '''
    Return True if the two structure are similar under the torsion fingeprint criteria.
    '''

    # Compute their absolute difference
    deltas = np.abs(tfp1 - tfp2)

    # Correct for rotations over 180 deg
    deltas = np.abs(deltas - (deltas > 180) * 360)

    if np.sum(deltas) < thresh:
        return True

    return False

@njit
def get_torsion_fingerprint(coords, quadruplets):
    out = np.zeros(quadruplets.shape[0], dtype=float32)
    for i, q in enumerate(quadruplets):
        i1, i2, i3, i4 = q
        out[i] = dihedral([coords[i1],
                           coords[i2],
                           coords[i3],
                           coords[i4]])
    return out

@njit(parallel=True)
def _score_embed_poses(structures, constrained_indices, constrained_distances):
    '''
    Returns array of scores for embedded structures.
    The score is calculated as the sum of deltas from
    the desired embed distances.
    '''
    _l = len(structures)
    scores = np.zeros(shape=_l, dtype=float32)

    for j in prange(_l):
        for i, (i1, i2) in enumerate(constrained_indices[j]):
            dist = norm_of(structures[j][i1] - structures[j][i2])
            scores[j] += np.abs(dist - constrained_distances[j][i])

    return scores