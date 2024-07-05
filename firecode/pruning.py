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
import numpy as np
from networkx import connected_components
from numba import njit

from firecode.algebra import all_dists, get_inertia_moments
from firecode.pt import pt
from firecode.rmsd import rmsd_and_max_numba
from firecode.torsion_module import (_get_hydrogen_bonds, _get_torsions,
                                     _is_nondummy,
                                     rotationally_corrected_rmsd_and_max)
from firecode.utils import get_double_bonds_indices


class Pruner:
    def __init__(self, structures, atomnos, debugfunction=None):
        self.structures = structures
        self.atomnos = atomnos
        self.calls = 0
        self.cache_calls = 0
        self.debugfunction = debugfunction

        self.defaults_dict = {

            "rmsd_rot_corr" : (
                rotationally_corrected_rmsd_and_max,

                [
                    "atomnos",
                    "torsions",
                    "graph",
                    "angles",
                ], # args

                dict(), # kwargs

                ["max_rmsd", "max_dev"], # thresholds
            ),

            "rmsd" : (
                rmsd_and_max_numba,

                [], # args

                dict(), # kwargs

                ["max_rmsd", "max_dev"], # thresholds
            ),

            "moi" : (
                get_moi_deviation_vec,

                ["masses"], # args

                dict(), # kwargs

                ["max_dev", "max_dev", "max_dev"], # thresholds
            ),

        }

    # set the operating mode 
    def set_mode(self, mode):
        if mode not in self.defaults_dict.keys():
            raise NameError(f"pruning mode \"{mode}\" not recognized.")
        self.mode = mode

        self.eval_func, args_names, kwargs_names, thresholds_names = self.defaults_dict[self.mode]
        self.args = [getattr(self, name) for name in args_names]
        self.kwargs = {name:getattr(self, value) for name, value in kwargs_names.items()}
        self.thresholds = [getattr(self, name) for name in thresholds_names]

        for name, value in zip(thresholds_names, self.thresholds):
            if value is None:
                raise UnboundLocalError(f'Class Pruner({self.mode}) does not have a \"{name}\" attriubute.')
    
    def _main_eval_similarity(self, coords1, coords2):
        results = self.eval_func(coords1, coords2, *self.args, **self.kwargs)
        for r, t in zip(results, self.thresholds):
            if r > t:
                return 0
        return 1

    def _main_compute_subrow(self, ref, structures, in_mask, first_abs_index):
        '''
        Returns True (as the int 1) if ref is similar to any
        structure in structures, returning at the first instance of a match.
        Ignores structures that are False (0) in in_mask and saves pairs
        that evaluate to False (0) by returning them in computed_pairs.

        '''

        # iterate over target structures
        for i, structure in enumerate(structures):

            # only compare active structures
            if in_mask[i]:
               
                # if first_abs_index == 12: print(f'Comparing 12 with {first_abs_index+1+i}')
                # check if we have performed this computation already,
                # and in that case we know the structures were not similar,
                # since the in_mask attribute is not False for ref nor i
                hash_value = (first_abs_index, first_abs_index+1+i)
                self.calls += 1
                if hash_value in self.cache:
                    self.cache_calls += 1
                
                # if we have not computed the value before, do it
                # function will return True (1) if the structures are similar
                elif self._main_eval_similarity(ref, structure):
                    return 1
                
                # if structures are not similar, add the result to the
                # cache, because they will potentially return here,
                # while similar structures are discarded and won't come back
                else:
                    self.cache.add(hash_value)
                           
        return 0

    def _main_compute_row(self, structures, in_mask, first_abs_index):
        '''
        For a given set of structures, check if each is similar
        to any other after itself. Returns a boolean mask to slice
        the array, only retaining the structures that are dissimilar.
        The inner subrow function caches computed non-similar pairs.

        '''
        #initialize the result container
        out_mask = np.ones(shape=in_mask.shape, dtype=np.bool_)

        # loop over the structures
        for i, ref in enumerate(structures):

            # only check for similarity if the structure is active
            if in_mask[i]:

                # reject structure i if it is similar to any other after itself
                similar = self._main_compute_subrow(
                                                    ref,
                                                    structures[i+1:],
                                                    in_mask[i+1:],
                                                    first_abs_index=first_abs_index+i,
                                                )
                out_mask[i] = not similar

            else:
                out_mask[i] = 0

        return out_mask

    def _main_compute_group(self, structures, in_mask, k):
        '''
        Acts on chunks of the structures array,
        returning the updated mask and the non-similar pairs computed.

        '''
        # initialize final result container
        out_mask = np.ones(shape=structures.shape[0], dtype=np.bool_)

        # calculate the size of each chunk
        chunksize = int(len(structures) // k)

        # iterate over chunks (multithreading here?)
        for chunk in range(int(k)):
            first = chunk*chunksize
            if chunk == k-1:
                last = len(structures)
            else:
                last = chunksize*(chunk+1)

            # get the structure chunk
            structures_chunk = structures[first:last]

            # compare structures within that chunk and save results to the out_mask
            out_mask[first:last] = self._main_compute_row(
                                                            structures_chunk,
                                                            in_mask[first:last],
                                                            first_abs_index=first,
                                                        )
        return out_mask

    def prune(self):
        '''
        Removes similar structures by repeatedly grouping them into k
        subgroups and removing similar ones. A cache is present to avoid
        repeating RMSD computations.
        
        Similarity occurs for structures with both rmsd < rmsd_thr and
        maximum absolute atomic deviation < 2 * rmsd_thr.

        Returns the pruned structures and the corresponding boolean mask.

        '''

        if self.mode in ("rmsd_rot_corr"):
            # all atoms passed, but still only the 
            # heavy ones are used for the RMSD calc
            structures = self.structures
            
        else:
            # only feed non-hydrogen atoms to eval funcs 
            heavy_atoms = (self.atomnos != 1)
            structures = np.array([structure[heavy_atoms] for structure in self.structures])

        # initialize the output mask
        out_mask = np.ones(shape=self.structures.shape[0], dtype=np.bool_)
        self.cache = set()

        # split the structure array in subgroups and prune them internally
        for k in (5e5, 2e5, 1e5, 5e4, 2e4, 1e4,
                5000, 2000, 1000, 500, 200, 100,
                50, 20, 10, 5, 2, 1):
            
            # choose only k values such that every subgroup
            # has on average at least twenty active structures in it
            if k == 1 or 20*k < np.count_nonzero(out_mask):

                before = np.count_nonzero(out_mask)

                # compute similarities and get back the out_mask
                # and the pairings to be added to cache
                out_mask = self._main_compute_group(
                                                    structures,
                                                    out_mask,
                                                    k=k,
                                                )
                
                after = np.count_nonzero(out_mask)
                newly_discarded = before - after

                if self.debugfunction is not None:
                    self.debugfunction(f'DEBUG: Pruner({self.mode}) - k={k}, rejected {newly_discarded} (keeping {after}/{len(out_mask)})')

        del self.cache
        self.mask = out_mask
        self.structures = self.structures[self.mask]

def prune_by_rmsd(structures, atomnos, max_rmsd=0.25, max_dev=None, debugfunction=None):
    '''
    Remove duplicate (enantiomeric or rotameric) structures based on the
    moments of inertia on principal axes. If all three MOI
    are within max_deviation percent from another structure,
    they are classified as rotamers or enantiomers and therefore only one
    of them is kept.
    '''

    # set default max_dev if not provided
    max_dev = max_dev or 2*max_rmsd

    pruner = Pruner(structures, atomnos, debugfunction=debugfunction)
    pruner.max_rmsd = max_rmsd
    pruner.max_dev = max_dev
    pruner.set_mode('rmsd')
    pruner.prune()
    final_mask = pruner.mask

    if debugfunction is not None:
        fraction = 0 if pruner.calls == 0 else pruner.cache_calls/pruner.calls
        debugfunction(f"DEBUG: prune_by_rmsd - Used cached data {pruner.cache_calls}/{pruner.calls} times, {100*fraction:.2f}% of total calls")
    
    return structures[final_mask], final_mask

def prune_by_rmsd_rot_corr(structures, atomnos, graph, max_rmsd=0.25, max_dev=None, logfunction=None, debugfunction=None):
    '''
    Removes similar structures by repeatedly grouping them into k
    subgroups and removing similar ones. A cache is present to avoid
    repeating RMSD computations.
    
    Similarity occurs for structures with both RMSD < max_rmsd and
    maximum deviation < max_dev.
    '''

    # center structures
    structures = np.array([s - s.mean(axis=0) for s in structures])
    ref = structures[0]

    # get the number of molecular fragments
    subgraphs = list(connected_components(graph))

    # if they are more than two, give up on pruning by rot corr rmsd
    if len(subgraphs) > 2:
        return structures, np.ones(structures.shape[0], dtype=bool)
    
    # if they are two, we can add a fictitious bond between the closest
    # atoms on the two molecular fragment in the provided graph, and
    # then removing it before returning
    if len(subgraphs) == 2:
        subgraphs =  [list(set) for set in connected_components(graph)]
        all_dists_array = all_dists(ref[list(subgraphs[0])], ref[list(subgraphs[1])])
        min_d = np.min(all_dists_array)
        s1, s2 = np.where(all_dists_array == min_d)
        i1, i2 = subgraphs[0][s1[0]], subgraphs[1][s2[0]]
        graph.add_edge(i1, i2)

        if debugfunction is not None:
            debugfunction(f"DEBUG: prune_by_rmsd_rot_corr - temporarily added edge {i1}-{i2} to the graph (will be removed before returning)")

    # set default max_dev if not provided
    max_dev = max_dev or 2*max_rmsd

    # add hydrogen bonds to molecular graph 
    hydrogen_bonds = _get_hydrogen_bonds(ref, atomnos, graph)
    for hb in hydrogen_bonds:
        graph.add_edge(*hb)

    # get all rotable bonds in the molecule, including dummy rotations
    torsions = _get_torsions(graph,
                            hydrogen_bonds=_get_hydrogen_bonds(ref, atomnos, graph),
                            double_bonds=get_double_bonds_indices(ref, atomnos),
                            keepdummy=True,
                            mode='symmetry')

    # only keep dummy rotations (checking both directions)
    torsions = [t for t in torsions if not (
                                _is_nondummy(t.i2, t.i3, graph) and (
                                _is_nondummy(t.i3, t.i2, graph)))]

    # since we only compute RMSD based on heavy atoms, discard quadruplets that involve hydrogen atoms
    torsions = [t for t in torsions if 1 not in [atomnos[i] for i in t.torsion]]

    # get torsions angles
    angles = [t.get_angles() for t in torsions]

    # Used specific directionality of torsions so that we always rotate the dummy portion (the one attached to the last index)
    torsions = [list(t.torsion) if _is_nondummy(t.i2, t.i3, graph) else list(reversed(t.torsion)) for t in torsions]

    # Set up final mask and cache
    final_mask = np.ones(structures.shape[0], dtype=bool)
   
    # Halt the run if there are too many structures or no subsymmetrical bonds
    if len(torsions) == 0:
        if debugfunction is not None:
            debugfunction('DEBUG: prune_by_rmsd_rot_corr - No subsymmetrical torsions found: skipping symmetry-corrected RMSD pruning')

        return structures[final_mask], final_mask

    # Print out torsion information
    if logfunction is not None:
        logfunction('\n >> Dihedrals considered for subsymmetry corrections:')
        for i, (torsion, angle) in enumerate(zip(torsions, angles)):
            logfunction(' {:2s} - {:21s} : {}{}{}{} : {}-fold'.format(
                                                                str(i+1),
                                                                str(torsion),
                                                                pt[atomnos[torsion[0]]].symbol,
                                                                pt[atomnos[torsion[1]]].symbol,
                                                                pt[atomnos[torsion[2]]].symbol,
                                                                pt[atomnos[torsion[3]]].symbol,
                                                                len(angle)))
        logfunction("\n")

    pruner = Pruner(structures, atomnos, debugfunction=debugfunction)
    pruner.graph = graph
    pruner.torsions = torsions
    pruner.angles = angles
    pruner.max_rmsd = max_rmsd
    pruner.max_dev = max_dev
    pruner.set_mode('rmsd_rot_corr')
    pruner.prune()
    final_mask = pruner.mask

    # remove the extra bond in the molecular graph
    if len(subgraphs) == 2:
        graph.remove_edge(i1, i2)

    if debugfunction is not None:
        fraction = 0 if pruner.calls == 0 else pruner.cache_calls/pruner.calls
        debugfunction(f"DEBUG: prune_by_rmsd_rot_corr - Used cached data {pruner.cache_calls}/{pruner.calls} times, {100*fraction:.2f}% of total calls")
    
    return structures[final_mask], final_mask

def prune_by_moment_of_inertia(structures, atomnos, max_deviation=1e-2, debugfunction=None):
    '''
    Remove duplicate (enantiomeric or rotameric) structures based on the
    moments of inertia on principal axes. If all three MOI
    are within max_deviation percent from another structure,
    they are classified as rotamers or enantiomers and therefore only one
    of them is kept.
    '''

    pruner = Pruner(structures, atomnos, debugfunction=debugfunction)
    pruner.max_dev = max_deviation
    pruner.masses = np.array([pt[a].mass for a in atomnos])
    pruner.set_mode('moi')
    pruner.prune()
    mask = pruner.mask

    if debugfunction is not None:
        fraction = 0 if pruner.calls == 0 else pruner.cache_calls/pruner.calls
        debugfunction(f"DEBUG: prune_by_moment_of_inertia - Used cached data {pruner.cache_calls}/{pruner.calls} times, {100*fraction:.2f}% of total calls")

    return structures[mask], mask

@njit
def get_moi_deviation_vec(coords1, coords2, masses):
    
    im_1 = get_inertia_moments(coords1, masses)
    im_2 = get_inertia_moments(coords2, masses)

    return np.abs(im_1 - im_2) / im_1