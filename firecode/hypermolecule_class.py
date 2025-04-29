# coding=utf-8
'''

FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''

import os
import warnings
from copy import deepcopy

import numpy as np
from numpy.linalg import LinAlgError

from firecode.algebra import get_inertia_moments, norm_of
from firecode.errors import CCReadError, NoOrbitalError
from firecode.graph_manipulations import (graphize, is_sigmatropic, is_vicinal,
                                        neighbors)
from firecode.pt import pt
from firecode.reactive_atoms_classes import get_atom_type
from firecode.rmsd import get_alignment_matrix
from firecode.utils import flatten, read_xyz, smi_to_3d

warnings.simplefilter("ignore", UserWarning)

def align_by_moi(structures, atomnos, **kwargs):
    '''
    Aligns molecules of a structure array (shape is (n_structures, n_atoms, 3))
    to the first one, based on the the moments of inertia vectors.
    Returns the aligned array.

    '''

    reference, *targets = structures
  
    masses = np.array([pt[a].mass for a in atomnos])

    # center all the structures at the origin
    reference -= np.mean(reference, axis=0)
    for t, target in enumerate(targets):
        targets[t] -= np.mean(target, axis=0)

    # initialize output array
    output = np.zeros(structures.shape)
    output[0] = reference

    # reference vectors   
    ref_moi_vecs = np.eye(3)
    (ref_moi_vecs[0,0],
     ref_moi_vecs[1,1],
     ref_moi_vecs[2,2]) = get_inertia_moments(reference, masses)

    for t, target in enumerate(targets):

        tgt_moi_vecs = np.eye(3)
        (tgt_moi_vecs[0,0],
         tgt_moi_vecs[1,1],
         tgt_moi_vecs[2,2]) = get_inertia_moments(target, masses)

        try:
            matrix = get_alignment_matrix(ref_moi_vecs, tgt_moi_vecs)

        except LinAlgError:
        # it is actually possible for the kabsch alg not to converge
            matrix = np.eye(3)
        
        # output[t+1] = np.array([matrix @ vector for vector in target])
        output[t+1] = (matrix @ target.T).T

    return output

class Hypermolecule:
    '''
    Molecule class to be used within firecode.
    '''

    def __repr__(self):
        r = self.rootname
        if hasattr(self, 'reactive_atoms_classes_dict'):
            r += f' {[str(atom) for atom in self.reactive_atoms_classes_dict[0].values()]}'
        return r

    def __init__(self, filename, reactive_indices=None, debug_logfunction=None):
        '''
        Initializing class properties: reading conformational ensemble file, aligning
        conformers to first and centering them in origin.

        :params filename:           Input file name. Can be anything, .xyz preferred
        :params reactive_indices:     Index of atoms that will link during the desired reaction.
                                    May be either int or list of int.
        '''

        if not os.path.isfile(filename):
            if '.' in filename:
                raise SyntaxError((f'Molecule {filename} cannot be read. Please check your syntax.'))

            try:
                filename = smi_to_3d(filename, "generated_3D_coords")
                print(f"--> Embedded SMILES string to 3D structure, saved as {filename}")
                
            except Exception:
                raise SyntaxError((f'The program is trying to read something that is not a valid molecule input ({filename}). ' +
                            'If this looks like a keyword, it is probably faulted by a syntax error.'))

        self.rootname = filename.split('.')[0]
        self.filename = filename
        self.debug_logfunction = debug_logfunction
        self.constraints = []

        if isinstance(reactive_indices, np.ndarray):
            self.reactive_indices = reactive_indices
        else:
            self.reactive_indices = np.array(reactive_indices) if isinstance(reactive_indices, (tuple, list)) else ()

        ccread_object = read_xyz(filename)

        if ccread_object is None:
            raise CCReadError(f'Cannot read file {filename}')

        coordinates = np.array(ccread_object.atomcoords)
        
        self.atomnos = ccread_object.atomnos
        self.position = np.array([0,0,0], dtype=float)  # used in Embedder class
        self.rotation = np.identity(3)                  # used in Embedder class - rotation matrix

        assert all([len(coordinates[i])==len(coordinates[0]) for i in range(1, len(coordinates))]), 'Ensembles must have constant atom number.'
        # Checking that ensemble has constant length
        if self.debug_logfunction is not None:
            debug_logfunction(f'DEBUG: Hypermolecule Class __init__ ({filename}) - Initializing object {filename}, read {len(coordinates)} structures with {len(coordinates[0])} atoms')

        self.centroid = np.sum(np.sum(coordinates, axis=0), axis=0) / (len(coordinates) * len(coordinates[0]))

        self.atomcoords = coordinates - self.centroid
        self.graph = graphize(self.atomcoords[0], self.atomnos)
        # show_graph(self)

        self.atoms = np.array([atom for structure in self.atomcoords for atom in structure])       # single list with all atomic positions
        
    def compute_orbitals(self, override=None):
        '''
        Computes orbital positions for atoms in self.reactive_atoms
        '''

        if self.reactive_indices is None:
            return

        self.sp3_sigmastar, self.sigmatropic = None, None

        self._inspect_reactive_atoms(override=override)
        # sets reactive atoms properties

        # self.atomcoords = align_structures(self.atomcoords, self.get_alignment_indices())
        self.sigmatropic = [is_sigmatropic(self, c) for c, _ in enumerate(self.atomcoords)]
        self.sp3_sigmastar = is_vicinal(self)

        for c, _ in enumerate(self.atomcoords):
            for index, reactive_atom in self.reactive_atoms_classes_dict[c].items():
                reactive_atom.init(self, index, update=True, conf=c)
                # update properties into reactive_atom class.
                # Since now we have mol.sigmatropic and mol.sigmastar,
                # We can update, that is set the reactive_atom.center attribute

    def _set_reactive_indices(self, filename):
        '''
        Manually set the molecule reactive atoms from the ASE GUI, imposing
        constraints on the desired atoms.

        '''
        from ase import Atoms
        from ase.gui.gui import GUI
        from ase.gui.images import Images

        data = read_xyz(filename)
        coords = data.atomcoords[0]
        labels = ''.join([pt[i].symbol for i in data.atomnos])

        atoms = Atoms(labels, positions=coords)

        while atoms.constraints == []:
            print(('\nPlease, manually select the reactive atom(s) for molecule %s.'
                    '\nRotate with right click and select atoms by clicking. Multiple selections can be done by Ctrl+Click.'
                    '\nWith desired atom(s) selected, go to Tools -> Constraints -> Constrain, then close the GUI.') % (filename))

            GUI(images=Images([atoms]), show_bonds=True).run()

        return list(atoms.constraints[0].get_indices())
    
    def get_alignment_indices(self):
        '''
        Return the indices to align the molecule to, given a list of
        atoms that should be reacting. List is composed by reactive atoms
        plus adjacent atoms.
        :param coords: coordinates of a single molecule
        :param reactive atoms: int or list of ints
        :return: list of indices
        '''
        if len(self.reactive_indices) == 0:
            return None

        indices = set()
        for atom in self.reactive_indices:
            indices |= set(list([(a, b) for a, b in self.graph.adjacency()][atom][1].keys()))

        if self.debug_logfunction is not None:
            self.debug_logfunction(f'DEBUG: Hypermolecule.get_aligment_indices {self.filename} - Alignment indices are {list(indices)})')

        return list(indices)

    def _inspect_reactive_atoms(self, override=None):
        '''
        Control the type of reactive atoms and sets the class attribute self.reactive_atoms_classes_dict
        '''
        self.reactive_atoms_classes_dict = {c:{} for c, _ in enumerate(self.atomcoords)}
        
        for c, _ in enumerate(self.atomcoords):
            for index in self.reactive_indices:
                symbol = pt[self.atomnos[index]].symbol

                atom_type = get_atom_type(self.graph, index, override=override)()

                # setting the reactive_atom class type
                atom_type.init(self, index, conf=c)

                # understanding the type of reactive atom in order to align the ensemble correctly and build the correct pseudo-orbitals
                self.reactive_atoms_classes_dict[c][index] = atom_type

                if self.debug_logfunction is not None:
                    self.debug_logfunction(f'DEBUG: Hypermolecule._inspect_reactive_atoms {self.filename} - Reactive atom {index+1} is a {symbol} atom of {atom_type} type. It is bonded to {len(neighbors(self.graph, index))} neighbor(s): {atom_type.neighbors_symbols}')

    def _scale_orbs(self, value):
        '''
        Scale each orbital dimension according to value.
        '''
        for c, _ in enumerate(self.atomcoords):
            for index, atom in self.reactive_atoms_classes_dict[c].items():
                orb_dim = norm_of(atom.center[0]-atom.coord)
                atom.init(self, index, update=True, orb_dim=orb_dim*value, conf=c)

    def get_r_atoms(self, c):
        '''
        c: conformer number
        '''
        return list(self.reactive_atoms_classes_dict[c].values())
    
    def get_centers(self, c):
        '''
        c: conformer number
        '''
        return np.array([[v for v in atom.center] for atom in self.get_r_atoms(c)])

    # def calc_positioned_conformers(self):
    #     self.positioned_conformers = np.array([[self.rotation @ v + self.position for v in conformer] for conformer in self.atomcoords])

    def _compute_hypermolecule(self):
        '''
        '''

        self.energies = [0 for _ in self.atomcoords]

        self.hypermolecule_atomnos = []
        clusters = {i:{} for i, _ in enumerate(self.atomnos)}  # {atom_index:{cluster_number:[position,times_found]}}
        for i, atom_number in enumerate(self.atomnos):
            atoms_arrangement = [conformer[i] for conformer in self.atomcoords]
            cluster_number = 0
            clusters[i][cluster_number] = [atoms_arrangement[0], 1]  # first structure has rel E = 0 so its weight is surely 1
            self.hypermolecule_atomnos.append(atom_number)
            radii = pt[atom_number].covalent_radius
            for j, atom in enumerate(atoms_arrangement[1:]):

                weight = np.exp(-self.energies[j+1] * 503.2475342795285 / self.T)
                # print(f'Atom {i} in conf {j+1} weight is {weight} - rel. E was {self.energies[j+1]}')

                for cluster_number, reference in deepcopy(clusters[i]).items():
                    if norm_of(atom - reference[0]) < radii:
                        clusters[i][cluster_number][1] += weight
                    else:
                        clusters[i][max(clusters[i].keys())+1] = [atom, weight]
                        self.hypermolecule_atomnos.append(atom_number)

        self.weights = [[] for _ in self.atomnos]
        self.hypermolecule = []

        for i, _ in enumerate(self.atomnos):
            for _, data in clusters[i].items():
                self.weights[i].append(data[1])
                self.hypermolecule.append(data[0])

        self.hypermolecule = np.asarray(self.hypermolecule)
        self.weights = np.array(self.weights).flatten()
        self.weights = np.array([weights / np.sum(weights) for weights in self.weights])
        self.weights = flatten(self.weights)

        self.dimensions = (max([coord[0] for coord in self.hypermolecule]) - min([coord[0] for coord in self.hypermolecule]),
                           max([coord[1] for coord in self.hypermolecule]) - min([coord[1] for coord in self.hypermolecule]),
                           max([coord[2] for coord in self.hypermolecule]) - min([coord[2] for coord in self.hypermolecule]))

    def write_hypermolecule(self):
        '''
        '''

        hyp_name = self.rootname + '_hypermolecule.xyz'
        with open(hyp_name, 'w') as f:
            for c, _ in enumerate(self.atomcoords):
                f.write(str(sum([len(atom.center) for atom in self.reactive_atoms_classes_dict[c].values()]) + len(self.atomcoords[0])))
                f.write(f'\FIRECODE Hypermolecule {c} for {self.rootname} - reactive indices {self.reactive_indices}\n')
                orbs =np.vstack([atom_type.center for atom_type in self.reactive_atoms_classes_dict[c].values()]).ravel()
                orbs = orbs.reshape((int(len(orbs)/3), 3))
                for i, atom in enumerate(self.atomcoords[c]):
                    f.write('%-5s %-8s %-8s %-8s\n' % (pt[self.atomnos[i]].symbol, round(atom[0], 6), round(atom[1], 6), round(atom[2], 6)))
                for orb in orbs:
                    f.write('%-5s %-8s %-8s %-8s\n' % ('X', round(orb[0], 6), round(orb[1], 6), round(orb[2], 6)))

    def get_orbital_length(self, index):
        '''
        index: reactive atom index
        '''
        if index not in self.reactive_indices:
            raise NoOrbitalError(f'Index provided must be a molecule reactive index ({index}, {self.filename})')

        r_atom = self.reactive_atoms_classes_dict[0][index]
        return norm_of(r_atom.center[0] - r_atom.coord)

class Pivot:
    '''
    (Cyclical embed)
    Pivot object: vector connecting two lobes of a
    molecule, starting from v1 (first reactive atom in
    mol.reacitve_atoms_classes_dict) and ending on v2.

    For molecules involved in chelotropic reactions,
    that is molecules that undergo a cyclical embed
    while having only one reactive atom, pivots are
    built on that single atom.
    '''
    def __init__(self, c1, c2, a1, a2, index1, index2):
        '''
        c: centers (orbital centers)
        v: vectors (orbital vectors, non-normalized)
        i: indices (of coordinates, in mol.center)
        '''
        self.start = c1
        self.end = c2

        self.start_atom = a1
        self.end_atom = a2

        self.pivot = c2 - c1
        self.meanpoint = np.mean((c1, c2), axis=0)
        self.index = (index1, index2)
        # the pivot starts from the index1-th
        # center of the first reactive atom
        # and to the index2-th center of the second

    def __repr__(self):
        return f'Pivot object - index {self.index}, norm {round(norm_of(self.pivot), 3)}, meanpoint {self.meanpoint}'
