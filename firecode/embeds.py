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

import numpy as np

from firecode.algebra import (align_vec_pair, norm, rot_mat_from_pointer,
                              vec_angle)
from firecode.ase_manipulations import ase_bend
from firecode.errors import TriangleError, ZeroCandidatesError
from firecode.graph_manipulations import get_sum_graph
from firecode.numba_functions import (compenetration_check,
                                      get_torsion_fingerprint, tfd_similarity)
from firecode.rmsd import _rmsd_similarity
from firecode.torsion_module import _get_quadruplets
from firecode.utils import (cartesian_product, loadbar, polygonize, pretty_num,
                            rotation_matrix_from_vectors)


def string_embed(embedder):
    '''
    return poses: return embedded structures, ready to be refined
    Algorithm used is the "string" algorithm (see docs).
    '''
    assert len(embedder.objects) == 2

    def is_new_structure(coords, quadruplets, lru_cache, cache_size=5):
        '''
        Checks if the torsion fingerprint of a structure is
        similar to the ones present in lru_cache. If the
        structure is new, updates the cache.
        '''

        # get the structure torsion fingerprint
        tfp = get_torsion_fingerprint(coords, quadruplets)

        # compare it to the ones in lru_cache
        for ref_tfp in lru_cache:
            if tfd_similarity(tfp, ref_tfp):
                return False

        # if it is different than all of them, add it to cache
        lru_cache.append(tfp)

        # update cache if that is too big
        if len(lru_cache) == cache_size + 1:
            lru_cache = lru_cache[1:]

        return True

    embedder.log(f'\n--> Performing string embed ({pretty_num(embedder.candidates)} candidates)')

    conf_number = [len(mol.atomcoords) for mol in embedder.objects]
    conf_indices = cartesian_product(*[np.array(range(i)) for i in conf_number])
    # (n,2) vectors where the every element is the conformer index for that molecule

    r_atoms_centers_indices = cartesian_product(*[np.array(range(len(mol.get_centers(0)[0]))) for mol in embedder.objects])
    # for two mols with 3 and 2 centers: [[0 0][0 1][1 0][1 1][2 0][2 1]]

    # explicit individual molecules
    mol1, mol2 = embedder.objects

    # get quadruplets needed for the tfd similarity check
    constrained_indices = [[int(embedder.objects[0].reactive_indices[0]),
                            int(embedder.objects[1].reactive_indices[0] + embedder.ids[0])]]

    quadruplets = _get_quadruplets(get_sum_graph((mol1.graph, mol2.graph), constrained_indices))

    lru_cache = []
    poses = []
    for i, (c1, c2) in enumerate(conf_indices):

        loadbar(i, len(conf_indices),
                prefix='Embedding structures ',
                suffix=f'({len(poses)} found)')

        for ai1, ai2 in r_atoms_centers_indices:
            for angle in embedder.systematic_angles:

                ra1 = mol1.get_r_atoms(c1)[0]
                ra2 = mol2.get_r_atoms(c2)[0]

                p1 = ra1.center[ai1]
                p2 = ra2.center[ai2]
                ref_vec = ra1.orb_vecs[ai1]
                mol_vec = ra2.orb_vecs[ai2]

                mol2.rotation = rotation_matrix_from_vectors(mol_vec, -ref_vec)

                if angle != 0:                  
                    delta_rot = rot_mat_from_pointer(ref_vec, angle)
                    mol2.rotation = delta_rot @ mol2.rotation

                mol2.position = p1 - mol2.rotation @ p2

                embedded_structure = get_embed((mol1, mol2), (c1, c2))

                if compenetration_check(embedded_structure, ids=embedder.ids, thresh=embedder.options.clash_thresh):
                    if is_new_structure(embedded_structure, quadruplets, lru_cache):
                        poses.append(embedded_structure)

    loadbar(1, 1, prefix='Embedding structures ')
    
    if not poses:
        s = ('\n--> Cyclical embed did not find any suitable disposition of molecules.\n' +
               '    This is probably because the two molecules cannot find a correct interlocking pose.\n' +
               '    Try expanding the conformational space with the csearch> operator or see the SHRINK keyword.')
        embedder.log(s, p=False)
        raise ZeroCandidatesError(s)

    embedder.constrained_indices = _get_string_constrained_indices(embedder, len(poses))

    return np.array(poses)

def _get_string_constrained_indices(embedder, n):
    '''
    Get constrained indices referring to the transition states, repeated n times.
    :params n: int
    :return: list of lists consisting in atomic pairs to be constrained.
    '''
    # Two molecules, string algorithm, one constraint for all, repeated n times
    return np.array([[[int(embedder.objects[0].reactive_indices[0]),
                       int(embedder.objects[1].reactive_indices[0] + embedder.ids[0])]] for _ in range(n)])

def cyclical_embed(embedder, max_norm_delta=5):
    '''
    return threads: return embedded structures, with position and rotation attributes set, ready to be pumped
                    into embedder.structures. Algorithm used is the "cyclical" algorithm (see docs).
    '''
    
    if len(embedder.objects) == 2 and embedder.options.rigid:
        return _fast_bimol_rigid_cyclical_embed(embedder, max_norm_delta=max_norm_delta)
        # shortened, simplified version that is somewhat faster

    def _get_directions(norms):
        '''
        Returns two or three vectors specifying the direction in which each molecule should be aligned
        in the cyclical TS, pointing towards the center of the polygon.
        '''
        assert len(norms) in (2,3)

        if len(norms) == 2:
            return np.array([[0, 1,0],
                             [0,-1,0]])

        vertices = np.zeros((3,2))

        vertices[1] = np.array([norms[0],0])

        a = np.power(norms[0], 2)
        b = np.power(norms[1], 2)
        c = np.power(norms[2], 2)
        x = (a-b+c)/(2*a**0.5)
        y = (c-x**2)**0.5

        vertices[2] = np.array([x,y])
        # similar to the code from polygonize, to get the active triangle
        # but without the orientation specified in the polygonize function
        
        a = vertices[1,0] # first point, x
        b = vertices[2,0] # second point, x
        c = vertices[2,1] # second point, y

        x = a/2
        y = (b**2 + c**2 - a*b)/(2*c)
        cc = np.array([x,y])
        # 2D coordinates of the triangle circocenter

        v0, v1, v2 = vertices

        meanpoint1 = np.mean((v0,v1), axis=0)
        meanpoint2 = np.mean((v1,v2), axis=0)
        meanpoint3 = np.mean((v2,v0), axis=0)

        dir1 = cc - meanpoint1
        dir2 = cc - meanpoint2
        dir3 = cc - meanpoint3
        # 2D direction versors connecting center of side with circumcenter.
        # Now we need to understand if we want these or their negative

        if np.any([np.all(d == 0) for d in (dir1, dir2, dir3)]):
        # We have a right triangle. To aviod numerical
        # errors, a small perturbation is made.
        # This should not happen, but just in case...
            norms[0] += 1e-5
            dir1, dir2, dir3 = [t[:-1] for t in _get_directions(norms)]

        angle0_obtuse = (vec_angle(v1-v0, v2-v0) > 90)
        angle1_obtuse = (vec_angle(v0-v1, v2-v1) > 90)
        angle2_obtuse = (vec_angle(v0-v2, v1-v2) > 90)

        dir1 = -dir1 if angle2_obtuse else dir1
        dir2 = -dir2 if angle0_obtuse else dir2
        dir3 = -dir3 if angle1_obtuse else dir3
        # invert the versors sign of circumcenter if
        # one angle is obtuse, because then
        # circumcenter is outside the triangle
        
        dir1 = norm(np.concatenate((dir1, [0])))
        dir2 = norm(np.concatenate((dir2, [0])))
        dir3 = norm(np.concatenate((dir3, [0])))

        return np.vstack((dir1, dir2, dir3))

    def _adjust_directions(embedder, directions, constrained_indices, triangle_vectors, pivots, conf_ids):
        '''
        For trimolecular TSs, correct molecules pre-alignment. That is, after the initial estimate
        based on pivot triangle circocentrum, systematically rotate each molecule around its pivot
        by fixed increments and look for the arrangement with the smallest deviation from orbital
        parallel interaction. This optimizes the obtainment of poses with the correct inter-reactive
        atoms distances.

        '''
        assert directions.shape[0] == 3

        mols = deepcopy(embedder.objects)
        p0, p1, p2 = [end - start for start, end in triangle_vectors]
        p0_mean, p1_mean, p2_mean = [np.mean((end, start), axis=0) for start, end in triangle_vectors]

        ############### get triangle vertices

        vertices = np.zeros((3,2))
        vertices[1] = np.array([norms[0],0])

        a = np.power(norms[0], 2)
        b = np.power(norms[1], 2)
        c = np.power(norms[2], 2)
        x = (a-b+c)/(2*a**0.5)
        y = (c-x**2)**0.5

        vertices[2] = np.array([x,y])
        # similar to the code from polygonize, to get the active triangle
        # but without the orientation specified in the polygonize function
        
        a = vertices[1,0] # first point, x
        b = vertices[2,0] # second point, x
        c = vertices[2,1] # second point, y

        x = a/2
        y = (b**2 + c**2 - a*b)/(2*c)
        # cc = np.array([x,y])
        # 2D coordinates of the triangle circocenter

        v0, v1, v2 = vertices

        v0 = np.concatenate((v0, [0]))
        v1 = np.concatenate((v1, [0]))
        v2 = np.concatenate((v2, [0]))

        ############### set up mols -> pos + rot

        for i in (0,1,2):

            start, end = triangle_vectors[i]

            mol_direction = pivots[i].meanpoint - np.mean(embedder.objects[i].atomcoords[conf_ids[i]][embedder.objects[i].reactive_indices], axis=0)
            if np.all(mol_direction == 0.):
                mol_direction = pivots[i].meanpoint

            mols[i].rotation = align_vec_pair(np.array([end-start, directions[i]]),
                                              np.array([pivots[i].pivot, mol_direction]))
            mols[i].position = np.mean(triangle_vectors[i], axis=0) - mols[i].rotation @ pivots[i].meanpoint

        ############### set up pairings between reactive atoms

        pairings = [[None, None] for _ in constrained_indices]
        for i, c in enumerate(constrained_indices):
            for m, mol in enumerate(embedder.objects):
                for index, r_atom in mol.reactive_atoms_classes_dict[0].items():
                    if r_atom.cumnum == c[0]:
                        pairings[i][0] = (m, index)
                    if r_atom.cumnum == c[1]:
                        pairings[i][1] = (m, index)

        r = np.zeros((3,3), dtype=int)

        for first, second in pairings:
            mol_index = first[0]
            partner_index = second[0]
            reactive_index = first[1]
            r[mol_index, partner_index] = reactive_index

            mol_index = second[0]
            partner_index = first[0]
            reactive_index = second[1]
            r[mol_index, partner_index] = reactive_index

        # r[0,1] is the reactive_index of molecule 0 that faces molecule 1 and so on
        # diagonal of r (r[0,0], r[1,1], r[2,2]) is just unused

        ############### calculate reactive atoms positions

        mol0, mol1, mol2 = mols

        a01 = mol0.rotation @ mol0.atomcoords[0][r[0,1]] + mol0.position
        a02 = mol0.rotation @ mol0.atomcoords[0][r[0,2]] + mol0.position

        a10 = mol1.rotation @ mol1.atomcoords[0][r[1,0]] + mol1.position
        a12 = mol1.rotation @ mol1.atomcoords[0][r[1,2]] + mol1.position

        a20 = mol2.rotation @ mol2.atomcoords[0][r[2,0]] + mol2.position
        a21 = mol2.rotation @ mol2.atomcoords[0][r[2,1]] + mol2.position

        ############### explore all angles combinations

        steps = 6
        angle_range = 30
        step_angle = 2*angle_range/steps
        angles_list = cartesian_product(*[range(steps+1) for _ in range(3)]) * step_angle - angle_range
        # Molecules are rotated around the +angle_range/-angle_range range in the given number of steps.
        # Therefore, the angular resolution between candidates is step_angle (10 degrees)

        candidates = []
        for angles in angles_list:

            rot0 = rot_mat_from_pointer(p0, angles[0])
            new_a01 = rot0 @ a01
            new_a02 = rot0 @ a02
            d0 = p0_mean - np.mean((new_a01, new_a02), axis=0)

            rot1 = rot_mat_from_pointer(p1, angles[1])
            new_a10 = rot1 @ a10
            new_a12 = rot1 @ a12
            d1 = p1_mean - np.mean((new_a10, new_a12), axis=0)

            rot2 = rot_mat_from_pointer(p2, angles[2])
            new_a20 = rot2 @ a20
            new_a21 = rot2 @ a21
            d2 = p2_mean - np.mean((new_a20, new_a21), axis=0)

            cost = 0
            cost += vec_angle(v0 - new_a02, new_a20 - v0)
            cost += vec_angle(v1 - new_a01, new_a10 - v1)
            cost += vec_angle(v2 - new_a21, new_a12 - v2)
                    
            candidates.append((cost, angles, (d0, d1, d2)))

        ############### choose the one with the best alignment, that is minor cost

        cost, angles, directions = sorted(candidates, key=lambda x: x[0])[0]
        
        return np.array(directions)

    s = f'\n--> Performing {embedder.embed} embed ({pretty_num(embedder.candidates)} candidates)'

    embedder.log(s)

    if not embedder.options.rigid:
        embedder.ase_bent_mols_dict = {}
        # used as molecular cache for ase_bend
        # keys are tuples with: ((identifier, pivot.index, target_pivot_length), obtained with:
        # (np.sum(original_mol.atomcoords[0]), tuple(sorted(pivot.index)), round(threshold,3))

    # if not embedder.options.let:
    #     for mol in embedder.objects:
    #         if len(mol.atomcoords) > 10:
    #             mol.atomcoords = most_diverse_conformers(10, mol.atomcoords)
    #             embedder.log(f'Using only the most diverse 10 conformers of molecule {mol.filename} (override with LET keyword)')
        # Do not keep more than 10 conformations, unless LET keyword is provided

    conf_number = [len(mol.atomcoords) for mol in embedder.objects]
    conf_indices = cartesian_product(*[np.array(range(i)) for i in conf_number])

    poses = []
    constrained_indices = []
    for ci, conf_ids in enumerate(conf_indices):

        pivots_indices = cartesian_product(*[range(len(mol.pivots[conf_ids[i]])) for i, mol in enumerate(embedder.objects)])
        # indices of pivots in each molecule self.pivots[conf] list. For three mols with 2 pivots each: [[0,0,0], [0,0,1], [0,1,0], ...]
        
        for p, pi in enumerate(pivots_indices):

            loadbar(p+ci*(len(pivots_indices)), len(pivots_indices)*len(conf_indices), prefix='Embedding structures ')
            
            pivots = [embedder.objects[m].pivots[conf_ids[m]][pi[m]] for m, _ in enumerate(embedder.objects)]
            # getting the active pivot for each molecule for this run
            
            norms = np.linalg.norm(np.array([p.pivot for p in pivots]), axis=1)
            # getting the pivots norms to feed into the polygonize function

            if len(norms) == 2:

                if abs(norms[0] - norms[1]) < max_norm_delta:
                    norms_type = 'digon'

                else:
                    norms_type = 'impossible_digon'

            else:
                if all([norms[i] < norms[i-1] + norms[i-2] for i in (0,1,2)]):
                    norms_type = 'triangle'

                else:
                    norms_type = 'impossible_triangle'

            if norms_type in ('triangle', 'digon'):

                polygon_vectors = polygonize(norms)

            elif norms_type == 'impossible_triangle':
                # Accessed if we cannot build a triangle with the given norms.
                # Try to bend the structure if it was close or just skip this triangle and go on.

                deltas = [norms[i] - (norms[i-1] + norms[i-2]) for i in range(3)]

                rel_delta = max([deltas[i]/norms[i] for i in range(3)])
                # s = 'Rejected triangle, delta was %s, %s of side length' % (round(delta, 3), str(round(100*rel_delta, 3)) + ' %')
                # embedder.log(s, p=False)

                if rel_delta < 0.2 and not embedder.options.rigid:
                # correct the molecule structure with the longest
                # side if the distances are at most 20% off.

                    index = deltas.index(max(deltas))
                    mol = embedder.objects[index]

                    if tuple(sorted(mol.reactive_indices)) not in list(mol.graph.edges):
                        # do not try to bend molecules where the two reactive indices are bonded

                        pivot = pivots[index]

                        # ase_view(mol)
                        maxval = norms[index-1] + norms[index-2]

                        traj = f'bend_{mol.filename}_p{p}_tgt_{round(0.9*maxval, 3)}' if embedder.options.debug else None

                        bent_mol = ase_bend(embedder,
                                            mol,
                                            conf_ids[index],
                                            pivot,
                                            0.9*maxval,
                                            title=f'{mol.rootname} - pivot {p}',
                                            traj=traj
                                            )

                        embedder.objects[index] = bent_mol

                        try:
                            pivots = [embedder.objects[m].pivots[conf_ids[m]][pi[m]] for m, _ in enumerate(embedder.objects)]
                            # updating the active pivot for each molecule for this run
                        except IndexError:
                            raise Exception((f'The number of pivots for molecule {index} ({bent_mol.filename}) most likely decreased during ' +
                                            'its bending, causing this error. Adding the RIGID (and maybe also SHRINK) keyword to the ' +
                                            'input file should solve the issue. I do not think this should ever happen under common ' +
                                            'circumstances, but if it does, it may be reasonable to print a statement on the log, ' +
                                            'discard the bent molecule, and then proceed with the embed. If you see this error, ' +
                                            'please report your input and structures on a GitHub issue. Thank you.'))
                        
                        norms = np.linalg.norm(np.array([p.pivot for p in pivots]), axis=1)
                        # updating the pivots norms to feed into the polygonize function

                        try:
                            polygon_vectors = polygonize(norms)
                            # repeating the failed polygon creation. If it fails again, skip these pivots

                        except TriangleError:
                            continue

                    else:
                        continue

                else:
                    continue
            
            else: # norms type == 'impossible_digon', that is sides are too different in length

                if not embedder.options.rigid:

                    if embedder.embed == 'chelotropic':
                        target_length = min(norms)

                    else:
                        maxgap = 3 # in Angstrom
                        gap = abs(norms[0]-norms[1])
                        r = 0.3 + 0.5*(gap/maxgap)
                        r = np.clip(5, 0.5, 0.8)

                        # r is the ratio for calculating target_length based
                        # on the gap that deformations will need to cover.
                        # It ranges from 0.5 to 0.8 and is shifted more toward
                        # the shorter norm as the gap rises. For gaps of more
                        # than maxgap Angstroms, the target length is very close
                        # to the shortest molecule, and only the molecule 
                        # with the longest pivot is bent.

                        target_length = min(norms)*r + max(norms)*(1-r)
                
                    for i, mol in enumerate(deepcopy(embedder.objects)):

                        if len(mol.reactive_indices) > 1:
                        # do not try to bend molecules that react with a single atom

                            if tuple(sorted(mol.reactive_indices)) not in list(mol.graph.edges):
                            # do not try to bend molecules where the two reactive indices are bonded

                                traj = f'bend_{mol.filename}_p{p}_tgt_{round(target_length, 3)}' if embedder.options.debug else None

                                bent_mol = ase_bend(embedder,
                                                    mol,
                                                    conf_ids[i],
                                                    pivots[i],
                                                    target_length,
                                                    title=f'{mol.rootname} - pivot {p}',
                                                    traj=traj
                                                    )

                                # ase_view(bent_mol)
                                embedder.objects[i] = bent_mol

                    # Repeating the previous polygonization steps with the bent molecules

                    pivots = [embedder.objects[m].pivots[conf_ids[m]][pi[m]] for m, _ in enumerate(embedder.objects)]
                    # updating the active pivot for each molecule for this run
                    
                    norms = np.linalg.norm(np.array([p.pivot for p in pivots]), axis=1)
                    # updating the pivots norms to feed into the polygonize function

                    polygon_vectors = polygonize(norms)
                    # repeating the failed polygon creation

                else:
                    continue # do not embed digons with too different lengths if RIGID

            directions = _get_directions(norms)
            # directions to orient the molecules toward, orthogonal to each vec_pair

            for v, vecs in enumerate(polygon_vectors):
            # getting vertices to embed molecules with and iterating over start/end points

                ids = _get_cyclical_reactive_indices(embedder, pivots, v)
                # get indices of atoms that face each other

                if not embedder.pairings_table or all((pair in ids) or (pair in embedder.internal_constraints) for pair in embedder.pairings_table.values()):
                # ensure that the active arrangement has all the pairings that the user specified

                    angular_poses = []
                    # initialize a container for the poses generated for this combination of conformations,
                    # pairing and polygon_vectors orientation. These will be used not to generate poses
                    # that are too similar to each other.

                    if len(embedder.objects) == 3:

                        directions = _adjust_directions(embedder, directions, ids, vecs, pivots, conf_ids)
                        # For trimolecular TSs, the alignment direction previously get is 
                        # just a general first approximation that needs to be corrected
                        # for the specific case through another algorithm.
                        
                    for angles in embedder.systematic_angles:

                        for i, vec_pair in enumerate(vecs):
                        # setting molecular positions and rotations (embedding)
                        # i is the molecule index, vecs is a tuple of start and end positions
                        # for the pivot vector

                            start, end = vec_pair
                            angle = angles[i]

                            reactive_coords = embedder.objects[i].atomcoords[conf_ids[i]][embedder.objects[i].reactive_indices]
                            # coordinates for the reactive atoms in this run

                            atomic_pivot_mean = np.mean(reactive_coords, axis=0)
                            # mean position of the atoms active in this run 

                            mol_direction = pivots[i].meanpoint-atomic_pivot_mean
                            if np.all(mol_direction == 0.):
                                mol_direction = pivots[i].meanpoint
                                # log.write(f'mol {i} - improper pivot? Thread {len(threads)-1}\n')

                            # Direction in which the molecule should be oriented, based on the mean of reactive
                            # atom positions and the mean point of the active pivot for the run.
                            # If this vector is too small and gets rounded to zero (as it can happen for
                            # "antrafacial" vectors), we fallback to the vector starting from the molecule
                            # center (mean of atomic positions) and ending in pivot_means[i], so to avoid
                            # numeric errors in the next function.
                                
                            alignment_rotation = align_vec_pair(np.array([end-start, directions[i]]),
                                                                np.array([pivots[i].pivot, mol_direction]))
                            # this rotation superimposes the molecular orbitals active in this run (pivots[i].pivot
                            # goes to end-start) and also aligns the molecules so that they face each other
                            # (mol_direction goes to directions[i])
                            
                            if len(reactive_coords) == 2:
                                axis_of_step_rotation = alignment_rotation @ (reactive_coords[0]-reactive_coords[1])
                            else:
                                axis_of_step_rotation = alignment_rotation @ pivots[i].pivot
                            # molecules with two reactive atoms are step-rotated around the line connecting
                            # the reactive atoms, while single reactive atom mols around their active pivot

                            step_rotation = rot_mat_from_pointer(axis_of_step_rotation, angle)
                            # this rotation cycles through all different rotation angles for each molecule

                            center_of_rotation = alignment_rotation @ atomic_pivot_mean
                            # center_of_rotation is the mean point between the reactive atoms so
                            # as to keep the reactive distances constant

                            embedder.objects[i].rotation = step_rotation @ alignment_rotation
                            # overall rotation for the molecule is given by the matrices product

                            pos = np.mean(vec_pair, axis=0) - alignment_rotation @ pivots[i].meanpoint
                            embedder.objects[i].position = center_of_rotation - step_rotation @ center_of_rotation + pos
                            # overall position is given by superimposing mean of active pivot (connecting orbitals)
                            # to mean of vec_pair (defining the target position - the side of a triangle for three molecules)

                        embedded_structure = get_embed(embedder.objects, conf_ids)
                        if compenetration_check(embedded_structure, ids=embedder.ids, thresh=embedder.options.clash_thresh):
                            if not _rmsd_similarity(embedded_structure, angular_poses, rmsd_thr=1):
                                poses.append(embedded_structure)
                                angular_poses.append(embedded_structure)
                                constrained_indices.append(ids)
                                # Save indices to be constrained later in the optimization step

    loadbar(1, 1, prefix='Embedding structures ')

    embedder.constrained_indices = np.array(constrained_indices)

    if not poses:
        s = ('\n--> Cyclical embed did not find any suitable disposition of molecules.\n' +
                '    This is probably because one molecule has two reactive centers at a great distance,\n' +
                '    preventing the other two molecules from forming a closed, cyclical structure.')
        embedder.log(s, p=False)
        raise ZeroCandidatesError(s)

    return np.array(poses)

def _fast_bimol_rigid_cyclical_embed(embedder, max_norm_delta=10):
    '''
    return threads: return embedded structures, with position and rotation attributes set, ready to be pumped
                    into embedder.structures. Algorithm used is the "cyclical" algorithm (see docs).
    '''
    
    embedder.log(f'\n--> Performing {embedder.embed} embed ({embedder.candidates} candidates)')

    conf_number = [len(mol.atomcoords) for mol in embedder.objects]
    conf_indices = cartesian_product(*[np.array(range(i)) for i in conf_number])

    poses = []
    constrained_indices = []
    for ci, conf_ids in enumerate(conf_indices):

        pivots_indices = cartesian_product(*[range(len(mol.pivots[conf_ids[i]])) for i, mol in enumerate(embedder.objects)])
        # indices of pivots in each molecule self.pivots[conf] list. For three mols with 2 pivots each: [[0,0,0], [0,0,1], [0,1,0], ...]

        for p, pi in enumerate(pivots_indices):

            loadbar(p+ci*(len(pivots_indices)), len(pivots_indices)*len(conf_indices), prefix='Embedding structures ')
            
            pivots = [embedder.objects[m].pivots[conf_ids[m]][pi[m]] for m, _ in enumerate(embedder.objects)]
            # getting the active pivot for each molecule for this run
            
            norms = np.linalg.norm(np.array([p.pivot for p in pivots]), axis=1)
            # getting the pivots norms to feed into the polygonize function

            if abs(norms[0] - norms[1]) > max_norm_delta:
                continue
            # skip if norms are too different

            polygon_vectors = polygonize(norms)

            directions = np.array([[0, 1,0], [0,-1,0]])
            # directions to orient the molecules toward, orthogonal to each vec_pair          

            for v, vecs in enumerate(polygon_vectors):
            # getting vertices to embed molecules with and iterating over start/end points

                ids = _get_cyclical_reactive_indices(embedder, pivots, v)
                # get indices of atoms that face each other

                if not embedder.pairings_table or all((pair in ids) or (pair in embedder.internal_constraints) for pair in embedder.pairings_table.values()):
                # ensure that the active arrangement has all the pairings that the user specified
                        
                    angular_poses = []
                    # initialize a container for the poses generated for this combination of conformations,
                    # pairing and polygon_vectors orientation. These will be used not to generate poses
                    # that are too similar to each other.

                    for angles in embedder.systematic_angles:

                        for i, vec_pair in enumerate(vecs):
                        # setting molecular positions and rotations (embedding)
                        # i is the molecule index, vecs is a tuple of start and end positions
                        # for the pivot vector

                            start, end = vec_pair
                            angle = angles[i]

                            reactive_coords = embedder.objects[i].atomcoords[conf_ids[i]][embedder.objects[i].reactive_indices]
                            # coordinates for the reactive atoms in this run

                            atomic_pivot_mean = np.mean(reactive_coords, axis=0)
                            # mean position of the atoms active in this run 

                            mol_direction = pivots[i].meanpoint-atomic_pivot_mean
                            if np.all(mol_direction == 0.):
                                mol_direction = pivots[i].meanpoint
                                # log.write(f'mol {i} - improper pivot? Thread {len(threads)-1}\n')

                            # Direction in which the molecule should be oriented, based on the mean of reactive
                            # atom positions and the mean point of the active pivot for the run.
                            # If this vector is too small and gets rounded to zero (as it can happen for
                            # "antrafacial" vectors), we fallback to the vector starting from the molecule
                            # center (mean of atomic positions) and ending in pivot_means[i], so to avoid
                            # numeric errors in the next function.
                                
                            alignment_rotation = align_vec_pair(np.array([end-start, directions[i]]),
                                                                np.array([pivots[i].pivot, mol_direction]))
                            # this rotation superimposes the molecular orbitals active in this run (pivots[i].pivot
                            # goes to end-start) and also aligns the molecules so that they face each other
                            # (mol_direction goes to directions[i])
                            
                            if len(reactive_coords) == 2:
                                axis_of_step_rotation = alignment_rotation @ (reactive_coords[0]-reactive_coords[1])
                            else:
                                axis_of_step_rotation = alignment_rotation @ pivots[i].pivot
                            # molecules with two reactive atoms are step-rotated around the line connecting
                            # the reactive atoms, while single reactive atom mols around their active pivot

                            step_rotation = rot_mat_from_pointer(axis_of_step_rotation, angle)
                            # this rotation cycles through all different rotation angles for each molecule

                            center_of_rotation = alignment_rotation @ atomic_pivot_mean
                            # center_of_rotation is the mean point between the reactive atoms so
                            # as to keep the reactive distances constant

                            embedder.objects[i].rotation = step_rotation @ alignment_rotation
                            # overall rotation for the molecule is given by the matrices product

                            pos = np.mean(vec_pair, axis=0) - alignment_rotation @ pivots[i].meanpoint
                            embedder.objects[i].position = center_of_rotation - step_rotation @ center_of_rotation + pos
                            # overall position is given by superimposing mean of active pivot (connecting orbitals)
                            # to mean of vec_pair (defining the target position - the side of a triangle for three molecules)

                        embedded_structure = get_embed(embedder.objects, conf_ids)
                        if compenetration_check(embedded_structure, ids=embedder.ids, thresh=embedder.options.clash_thresh):
                            if not _rmsd_similarity(embedded_structure, angular_poses, rmsd_thr=1):
                                poses.append(embedded_structure)
                                angular_poses.append(embedded_structure)
                                constrained_indices.append(ids)
                                # Save indices to be constrained later in the optimization step

    loadbar(1, 1, prefix='Embedding structures ')

    embedder.constrained_indices = np.array(constrained_indices)

    if not poses:
        s = ('\n--> Cyclical embed did not find any suitable disposition of molecules.\n' +
                '    This is probably because one molecule has two reactive centers at a great distance,\n' +
                '    preventing the other two molecules from forming a closed, cyclical structure.')
        embedder.log(s, p=False)
        raise ZeroCandidatesError(s)

    return np.array(poses)

def _get_cyclical_reactive_indices(embedder, pivots, n):
    '''
    :params n: index of the n-th disposition of vectors yielded by the polygonize function.
    :return: list of index couples, to be constrained during the partial optimization.
    '''

    cumulative_pivots_ids = [[p.start_atom.cumnum, p.end_atom.cumnum] for p in pivots]

    def orient(i,ids,n):
        if swaps[n][i]:
            return list(reversed(ids))
        return ids

    if len(embedder.objects) == 2:

        swaps = [(0,0),
                    (0,1)]

        oriented = [orient(i,ids,n) for i, ids in enumerate(cumulative_pivots_ids)]
        couples = [[oriented[0][0], oriented[1][0]], [oriented[0][1], oriented[1][1]]]

        return couples

    swaps = [(0,0,0),
                (0,0,1),
                (0,1,0),
                (0,1,1),
                (1,0,0),
                (1,1,0),
                (1,0,1),
                (1,1,1)]

    oriented = [orient(i,ids,n) for i, ids in enumerate(cumulative_pivots_ids)]
    couples = [[oriented[0][1], oriented[1][0]], [oriented[1][1], oriented[2][0]], [oriented[2][1], oriented[0][0]]]
    couples = [sorted(c) for c in couples]

    return couples

def monomolecular_embed(embedder):
    '''
    return threads: embeds structures by bending molecules, storing them
    in embedder.structures. Algorithm used is the "monomolecular" algorithm (see docs).
    '''

    assert len(embedder.objects) == 1

    embedder.log(f'\n--> Performing monomolecular embed ({embedder.candidates} candidates)')

    mol = embedder.objects[0]
    
    embedder.structures = []

    for c, _ in enumerate(mol.atomcoords):
        for p, pivot in enumerate(mol.pivots[c]):

            loadbar(p, len(mol.pivots[c]), prefix='Bending structures ')

            traj = f'bend_{p}_monomol' if embedder.options.debug else None

            bent_mol = ase_bend(embedder,
                                mol,
                                c,
                                pivot,
                                1, # bend until we are within 1 A to
                                # the target distance between atoms
                                title=f'{mol.rootname} - pivot {p}',
                                traj=traj,
                                check=False, # avoid returning the non-bent molecule,
                                            # even if this means having it scrambled
                                )

            for conformer in bent_mol.atomcoords:
                embedder.structures.append(conformer)

    loadbar(1, 1, prefix='Bending structures ')

    embedder.structures = np.array(embedder.structures)

    embedder.atomnos = mol.atomnos
    embedder.energies = np.zeros(len(embedder.structures))
    embedder.exit_status = np.zeros(len(embedder.structures), dtype=bool)
    embedder.graphs = [mol.graph]

    embedder.constrained_indices = _get_monomolecular_reactive_indices(embedder)

    return embedder.structures

def _get_monomolecular_reactive_indices(embedder):
    '''
    '''
    if embedder.pairings_table:
        return np.array([list(embedder.pairings_table.values())
                        for _ in embedder.structures])
    # This option gives the possibility to specify pairings in
    # refine>/REFINE runs, so as to make constrained optimizations
    # accessible.

    return np.array([[] for _ in embedder.structures])

def get_embed(mols, conf_ids):
    '''
    mols: iterable of Hypermolecule objects
    conf_ids: iterable of conformer indices for each mol

    Returns an np.array with the coordinates
    of every molecule as a concatenated array.
    '''
    return np.concatenate([(mol.rotation @ mol.atomcoords[c].T).T + mol.position for mol, c in zip(mols, conf_ids)])