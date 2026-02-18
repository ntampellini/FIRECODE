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

from collections import defaultdict
from itertools import permutations, product
from time import perf_counter
from typing import List, Tuple, Union

import numpy as np
from prism_pruner.algebra import dihedral
from prism_pruner.pruner import prune_by_moment_of_inertia
from prism_pruner.utils import time_to_string
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from firecode.algebra import point_angle
from firecode.utils import write_xyz

norm_of = np.linalg.norm


def rdkit_search_operator(filename, embedder, attempts=1000):

    embedder.log(f'--> Performing an RDKit ETKDGv3 conformational search on {filename} ({attempts} attempts)')

    rdkit_mol = Chem.MolFromXYZFile(filename)

    # XYZ files have no bond info — perceive connectivity and sanitize
    rdkit_mol = Chem.RWMol(rdkit_mol)
    rdDetermineBonds.DetermineBonds(rdkit_mol, charge=0)
    Chem.SanitizeMol(rdkit_mol)

    # Add hydrogens after sanitization so valence is known
    rdkit_mol = Chem.AddHs(rdkit_mol)
    rdkit_atoms = np.array([atom.GetSymbol() for atom in rdkit_mol.GetAtoms()])

    t_start = perf_counter()

    # Generate conformers
    conf_ids = AllChem.EmbedMultipleConfs(
        rdkit_mol,
        numConfs=embedder.options.max_confs,
        numThreads=0,  # Use all available threads
        maxAttempts=attempts,
        pruneRmsThresh=0.5,  # Prune conformers that are too similar
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,

        )

    conformers = []
    for conf_id in conf_ids:
        conf = rdkit_mol.GetConformer(conf_id)
        coords = conf.GetPositions()
        conformers.append(coords)

    conformers = np.array(conformers)

    embedder.log(f'--> RDKit ETKDGv3 generated {len(conformers)} conformers ({time_to_string(perf_counter() - t_start)})')

    # quickly prune the ensemble with PRISM
    conformers, mask = prune_by_moment_of_inertia(
        conformers,
        rdkit_atoms,
        )

    embedder.log(f'--> PRISM MOI pruning: kept {len(conformers)}/{len(mask)}')

    outname = filename[:-4] + '_rdkit_confs.xyz'

    with open(outname, 'w') as f:
        for i, conformer in enumerate(conformers):
            write_xyz(rdkit_atoms, conformer, f, title=f'RDKit conformer {i}')

    embedder.log(f'--> RDKit conformers written to {outname}\n')

    return outname


def get_atom_environment(mol: Chem.Mol, atom_idx: int, depth: int = 4) -> str:
    """Generate a string representation of an atom's local environment.
    
    Parameters
    ----------
    mol : Chem.Mol
        RDKit Molecule containing the atom
    atom_idx : int
        Index of the atom (0-based)
    depth : int
        Number of bonds to traverse in characterizing the environment
        
    Returns
    -------
    str
        A string encoding the local chemical environment

    """
    atom = mol.GetAtomWithIdx(atom_idx)

    # Get initial atom properties (matching OpenBabel's properties)
    env = [
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetTotalDegree(),
        atom.GetDegree(),  # Heavy atom degree (non-H neighbors)
        atom.GetTotalNumHs(includeNeighbors=True) - atom.GetTotalDegree() + atom.GetDegree(),  # Implicit H count
        int(atom.GetHybridization())
    ]

    # Traverse bonds up to specified depth
    visited = {atom_idx}
    current_layer = {atom_idx}
    environment = []

    for _ in range(depth):
        next_layer = set()
        layer_info = []

        for current_idx in current_layer:
            current_atom = mol.GetAtomWithIdx(current_idx)

            # Collect neighbor information
            neighbors = []
            for neighbor in current_atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    neighbors.append((
                        neighbor.GetAtomicNum(),
                        neighbor.GetTotalDegree()
                    ))
                    next_layer.add(neighbor_idx)
                    visited.add(neighbor_idx)

            if neighbors:
                # Sort neighbors for consistent ordering
                neighbors.sort()
                layer_info.extend(neighbors)

        if layer_info:
            environment.extend(layer_info)
        current_layer = next_layer

        if not current_layer:
            break

    return str(env + environment)


def find_symmetric_atoms(mol: Chem.Mol, match: Tuple[int, ...]) -> List[List[int]]:
    """Find groups of symmetric atoms within a match.
    
    Parameters
    ----------
    mol : Chem.Mol
        RDKit Molecule containing the matched atoms
    match : Tuple[int, ...]
        Tuple of atom indices (0-based) in the match
        
    Returns
    -------
    List[List[int]]
        List of lists, where each inner list contains indices (positions in the match)
        of symmetric atoms

    """
    # Group atoms by their atomic number first
    atoms_by_element = defaultdict(list)
    for pos, atom_idx in enumerate(match):
        atomic_num = mol.GetAtomWithIdx(atom_idx).GetAtomicNum()
        atoms_by_element[atomic_num].append((pos, atom_idx))

    symmetric_groups = []

    # For each element type, check for symmetry
    for element_atoms in atoms_by_element.values():
        if len(element_atoms) < 2:
            continue

        # Group by environment
        env_groups = defaultdict(list)
        for pos, atom_idx in element_atoms:
            env = get_atom_environment(mol, atom_idx)
            env_groups[env].append(pos)

        # Add groups of symmetric atoms
        for positions in env_groups.values():
            if len(positions) > 1:
                symmetric_groups.append(positions)

    return symmetric_groups


def match_smarts_pattern(
    molecule_input: Union[str, Tuple[np.ndarray, np.ndarray]],
    smarts_pattern: str,
    symmetric_atoms: List[List[int]] | None = None,
    auto_symmetry: bool = True,
    input_format: str = 'xyz',
    single_match_expected: bool = False,
) -> List[List[Tuple[int, ...]]]:
    """Match a SMARTS pattern against a molecule using RDKit.
    Returns all possible symmetric permutations of the matches.
    
    Parameters
    ----------
    molecule_input : Union[str, Tuple[np.ndarray, np.ndarray]]
        Either:
        - Path to the molecule file (str)
        - Tuple of (coordinates, atomic_numbers) where:
          * coordinates is a numpy array of shape (n_atoms, 3)
          * atomic_numbers is a numpy array of shape (n_atoms,)
    smarts_pattern : str
        The SMARTS pattern to match. Can include multiple fragments separated by dots
    symmetric_atoms : List[List[int]], optional
        Manual specification of symmetric atoms. Each inner list contains indices
        within the SMARTS pattern that are symmetric
    auto_symmetry : bool, optional
        Whether to automatically detect symmetric atoms. If True and symmetric_atoms
        is provided, will combine both manual and automatic symmetries
    input_format : str, optional
        File format if molecule_input is a file path. Default is 'xyz'
    single_match_expected : bool, optional
        If True, will error out if more than one match is found
        
    Returns
    -------
    List[List[Tuple[int, ...]]]
        List where each element contains all symmetric versions of a match.
        Each match is a tuple of atom indices (0-based).
    
    Examples
    --------
    >>> # Automatic symmetry detection
    >>> matches = match_smarts_pattern(mol, "[CX3](=[OX1])[OX1-]")
    
    >>> # Combined manual and automatic symmetry detection
    >>> matches = match_smarts_pattern(
    ...     mol, 
    ...     "[NH2].[OH].[OH]",
    ...     symmetric_atoms=[[3, 5]],  # Manual specification for OH groups
    ...     auto_symmetry=True  # Will also detect NH2 hydrogens
    ... )

    """
    try:
        # Create RDKit molecule object based on input type
        if isinstance(molecule_input, str):
            # Read molecule from file
            if input_format.lower() == 'xyz':
                mol = Chem.MolFromXYZFile(molecule_input)
            elif input_format.lower() == 'mol':
                mol = Chem.MolFromMolFile(molecule_input)
            elif input_format.lower() == 'mol2':
                mol = Chem.MolFromMol2File(molecule_input)
            elif input_format.lower() == 'pdb':
                mol = Chem.MolFromPDBFile(molecule_input)
            else:
                raise ValueError(f"Unsupported input format: {input_format}")

            if mol is None:
                raise ValueError(f"Could not read molecule from {molecule_input}")
        else:
            coords, atomic_nums = molecule_input

            # Create editable molecule
            mol = Chem.RWMol()

            # Add atoms
            for atom_num in atomic_nums:
                atom = Chem.Atom(int(atom_num))
                mol.AddAtom(atom)

            # Set up 3D coordinates
            conf = Chem.Conformer(len(atomic_nums))
            for i, pos in enumerate(coords):
                conf.SetAtomPosition(i, tuple(pos))
            mol.AddConformer(conf)

            # Determine connectivity and bond orders
            # RDKit's approach to determining bonds from coordinates
            mol = mol.GetMol()
            Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
            AllChem.DetermineBonds(mol)

        # Split pattern into fragments
        fragment_patterns = [p.strip() for p in smarts_pattern.split('.')]

        # Find matches for each fragment
        fragment_matches = []
        for pattern in fragment_patterns:
            smarts = Chem.MolFromSmarts(pattern)
            if smarts is None:
                raise ValueError(f"Invalid SMARTS pattern: {pattern}")

            matches = mol.GetSubstructMatches(smarts)
            if not matches:
                raise Exception('Found no SMARTS matches!')

            fragment_matches.append(list(matches))

        # Generate initial combinations of fragment matches
        base_matches = []
        for match_combination in product(*fragment_matches):
            flat_match = sum(match_combination, ())
            # Ensure no atom is used twice
            if len(set(flat_match)) == len(flat_match):
                base_matches.append(flat_match)

        if not base_matches:
            raise Exception('Found no SMARTS matches!')

        # Check single match expectation
        if single_match_expected:
            assert len(base_matches) == 1, f'Found {len(base_matches)} matches instead of 1'

        # Combine manual and automatic symmetry detection
        all_symmetric_groups = []
        if symmetric_atoms:
            all_symmetric_groups.extend(symmetric_atoms)

        if auto_symmetry:
            for match in base_matches:
                auto_groups = find_symmetric_atoms(mol, match)
                # Merge with existing groups, avoiding duplicates
                for group in auto_groups:
                    if group not in all_symmetric_groups:
                        all_symmetric_groups.append(group)

        if not all_symmetric_groups:
            return [[match] for match in base_matches]

        # Generate all symmetric permutations for each base match
        all_results = []
        for base_match in base_matches:
            symmetric_versions = set()
            symmetric_versions.add(base_match)

            for sym_group in all_symmetric_groups:
                new_versions = set()
                for match in symmetric_versions:
                    match_list = list(match)
                    sym_atoms = [match_list[i] for i in sym_group]
                    for perm in permutations(sym_atoms):
                        new_match = match_list.copy()
                        for idx, atom in zip(sym_group, perm):
                            new_match[idx] = atom
                        new_versions.add(tuple(new_match))
                symmetric_versions.update(new_versions)

            all_results.append(list(symmetric_versions))

        return all_results

    except Exception as e:
        raise RuntimeError(f"Error matching SMARTS pattern: {e!s}")


def convert_constraint_with_smarts(self, coords, atomnos, smarts):
    """Converts self.indices from being relative to a SMARTS
    pattern to being the effective molecular indices.
    Since more matches could be present, the one that is
    the closest to satisfying the desired constraint value
    is chosen.

    """
    match_indices_list = match_smarts_pattern((coords, atomnos), smarts)

    if self.type == 'B':
        a, b = self.indices
        deltas = [abs(norm_of(coords[match[a]]-coords[match[b]])-self.value) for match in match_indices_list]
        best_match_indices = match_indices_list[deltas.index(min(deltas))]

    if self.type == 'A':
        a, b, c = self.indices
        deltas = [abs(point_angle(coords[match[a]],
                                    coords[match[b]],
                                    coords[match[c]])-self.value) for match in match_indices_list]
        best_match_indices = match_indices_list[deltas.index(min(deltas))]

    if self.type == 'D':
        a, b, c, d = self.indices
        deltas = [abs(dihedral((coords[match[a]],
                                coords[match[b]],
                                coords[match[c]],
                                coords[match[d]]))-self.value) for match in match_indices_list]
        best_match_indices = match_indices_list[deltas.index(min(deltas))]

    old_indices = self.indices[:]
    for i, index in enumerate(old_indices):
        self.indices[i] = best_match_indices[index]
