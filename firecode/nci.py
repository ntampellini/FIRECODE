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
from itertools import combinations

import numpy as np

from firecode.algebra import norm_of
from firecode.parameters import nci_dict
from firecode.pt import pt
from firecode.graph_manipulations import is_phenyl


def get_nci(coords, atomnos, constrained_indices, ids):
    '''
    Returns a list of guesses for intermolecular non-covalent
    interactions between molecular fragments/atoms. Used to get
    a hint of the most prominent NCIs that drive stereo/regio selectivity.
    '''
    nci = []
    print_list = []
    
    symbols = [pt[i].symbol for i in atomnos]
    constrained_indices = constrained_indices.ravel()

    print_list, nci = _get_nci_atomic_pairs(coords, symbols, constrained_indices, ids)
    # Initialize with atomic pairs NCIs

    # Start checking group contributions
    aromatic_centers = _get_aromatic_centers(coords, symbols, ids)
    # print(f'structure has {len(aromatic_centers)} phenyl rings')

    # checking phenyl-atom pairs and phenyl-phenyl pairs
    pl, nc = _get_nci_aromatic_rings(coords, symbols, ids, aromatic_centers)
    print_list += pl
    nci += nc

    return nci, print_list

def _get_nci_atomic_pairs(coords, symbols, constrained_indices, ids):
    '''
    '''
    print_list = []
    nci = []

    cum_ids = np.cumsum(ids)

    for i1, _ in enumerate(coords):
    # check atomic pairs (O-H, N-H, ...)

        start_of_next_mol = cum_ids[next(i for i,n in enumerate(cum_ids) if i1 < n)]
        # ensures that we are only taking into account intermolecular NCIs

        for i2, _ in enumerate(coords[start_of_next_mol:]):
            i2 += start_of_next_mol

            if (i1 not in constrained_indices) and (i2 not in constrained_indices):
                # ignore atoms involved in constraints

                    s = ''.join(sorted([symbols[i1], symbols[i2]]))
                    # print(f'Checking pair {i1}/{i2}')

                    if s in nci_dict:
                        threshold, nci_type = nci_dict[s]
                        dist = norm_of(coords[i1]-coords[i2])

                        if dist < threshold:

                            print_list.append(nci_type + f' ({round(dist, 2)} A, indices {i1}/{i2})')
                            # string to be printed in log

                            nci.append((nci_type, i1, i2))
                            # tuple to be used in identifying the NCI

    return print_list, nci

def _get_nci_aromatic_rings(coords, symbols, ids, aromatic_centers):
    '''
    '''
    cum_ids = np.cumsum(ids)
    print_list, nci = [], []

    for owner, center in aromatic_centers:
        for i, atom in enumerate(coords):

            if i < cum_ids[0]:
                atom_owner = 0
            else:
                atom_owner = next(i for i,n in enumerate(np.cumsum(ids)) if i < n)

            if atom_owner != owner:
            # if this atom belongs to a molecule different than the one that owns the phenyl

                s = ''.join(sorted(['Ph', symbols[i]]))
                if s in nci_dict:

                    threshold, nci_type = nci_dict[s]
                    dist = norm_of(center - atom)

                    if dist < threshold:

                        print_list.append(nci_type + f' ({round(dist, 2)} A, atom {i}/ring)')
                        # string to be printed in log

                        nci.append((nci_type, i, 'ring'))
                        # tuple to be used in identifying the NCI

    # checking phenyl-phenyl pairs
    for i, owner_center in enumerate(aromatic_centers):
        owner1, center1 = owner_center
        for owner2, center2 in aromatic_centers[i+1:]:
            if owner1 != owner2:
            # if this atom belongs to a molecule different than owner

                    threshold, nci_type = nci_dict['PhPh']
                    dist = norm_of(center1 - center2)

                    if dist < threshold:

                        print_list.append(nci_type + f' ({round(dist, 2)} A, ring/ring)')
                        # string to be printed in log

                        nci.append((nci_type, 'ring', 'ring'))
                        # tuple to be used in identifying the NCI
    return print_list, nci

def _get_aromatic_centers(coords, symbols, ids):
    '''
    '''
    cum_ids = np.cumsum(ids)
    masks = []

    for mol, _ in enumerate(ids):

        if mol == 0:
            mol_mask = slice(0, cum_ids[0])
            filler = 0
        else:
            mol_mask = slice(cum_ids[mol-1], cum_ids[mol])
            filler = cum_ids[mol-1]

        aromatics_indices = np.array([i+filler for i, s in enumerate(symbols[mol_mask]) if s in ('C','N')])

        if len(aromatics_indices) > 5:
        # only check for phenyls in molecules with more than 5 C/N atoms

            masks.append(list(combinations(aromatics_indices, 6)))
            # all possible combinations of picking 6 C/N atoms from this molecule

    aromatic_centers = []

    if masks:

        masks = np.concatenate(masks)

        for mask in masks:

            if is_phenyl(coords[mask]):

                center = np.mean(coords[mask], axis=0)

                owner = next(i for i,n in enumerate(np.cumsum(ids)) if np.all(mask < n))
                # index of the molecule that owns that phenyl ring

                aromatic_centers.append((owner, center))

    return aromatic_centers