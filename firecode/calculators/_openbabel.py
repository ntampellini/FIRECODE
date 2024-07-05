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

# VERSION 0.4.4:
# THIS MODULE IS NOT INTERFACED WITH THE MAIN PROGRAM EMBEDDER ANYMORE.
# IT IS LEFT HERE AS AN EXTERNAL UTILITY TOOL AND FOR POTENTIAL FUTURE
# USE AS A FASTER, LESS ROBUST ALTERNATIVE TO THE XTB FF IMPLEMENTATION.

from firecode.utils import clean_directory, scramble_check, write_xyz, read_xyz
from firecode.algebra import norm, norm_of
from openbabel import openbabel as ob

def openbabel_opt(
                    structure,
                    atomnos,
                    constrained_indices,
                    constrained_distances=None,
                    tight_constraint=True,
                    graphs=None,
                    check=False,
                    method='UFF',
                    nsteps=1000,
                    title='temp_ob',
                    **kwargs,
                ):
        '''
        tight_constraint: False uses the native implementation,
                          True uses a more accurate recursive one 
        return : MM-optimized structure (UFF/MMFF94)
        '''

        assert not check or graphs is not None, 'Either provide molecular graphs or do not check for scrambling.'
        assert method in ('UFF', 'MMFF94', 'Ghemical', 'GAFF'), 'OpenBabel implements only the UFF, MMFF94, Ghemical and GAFF Force Fields.'

        # If we have any target distance to impose,
        # the most accurate way to do it is to manually
        # move the second atom and then freeze both atom
        # in place during optimization. If we would have
        # to move the second atom too much we do that in
        # small steps of 0.2 A, recursively, to avoid having
        # openbabel come up with weird bonding topologies,
        # ending in scrambling.

        if constrained_distances is not None and tight_constraint:
            for target_d, (a, b) in zip(constrained_distances, constrained_indices):
                d = norm_of(structure[b] - structure[a])
                delta = d - target_d

                if abs(delta) > 0.2:
                    sign = (d > target_d)
                    recursive_c_d = [d + 0.2 * sign for d in constrained_distances]

                    structure, _, _ = openbabel_opt(
                                                    structure,
                                                    atomnos,
                                                    constrained_indices,
                                                    constrained_distances=recursive_c_d,
                                                    tight_constraint=True, 
                                                    graphs=graphs,
                                                    check=check,
                                                    method=method,
                                                    nsteps=nsteps,
                                                    title=title,
                                                    **kwargs,
                                                )

                d = norm_of(structure[b] - structure[a])
                delta = d - target_d
                structure[b] -= norm(structure[b] - structure[a]) * delta

        filename=f'{title}_in.xyz'

        with open(filename, 'w') as f:
            write_xyz(structure, atomnos, f)
        # input()
        outname = f'{title}_out.xyz'

        # Standard openbabel molecule load
        conv = ob.OBConversion()
        conv.SetInAndOutFormats('xyz','xyz')
        mol = ob.OBMol()
        conv.ReadFile(mol, filename)
        i = 0

        # Define constraints
        constraints = ob.OBFFConstraints()

        for i, (a, b) in enumerate(constrained_indices):

            # Adding a distance constraint does not lead to accurate results,
            # so the backup solution is to freeze the atoms in place
            if tight_constraint:
                constraints.AddAtomConstraint(int(a+1))
                constraints.AddAtomConstraint(int(b+1))

            else:
                if constrained_distances is None:
                    first_atom = mol.GetAtom(int(a+1))
                    length = first_atom.GetDistance(int(b+1))
                else:
                    length = constrained_distances[i]
                
                constraints.AddDistanceConstraint(int(a+1), int(b+1), length)       # Angstroms

                # constraints.AddAngleConstraint(1, 2, 3, 120.0)      # Degrees
                # constraints.AddTorsionConstraint(1, 2, 3, 4, 180.0) # Degrees

        # Setup the force field with the constraints
        forcefield = ob.OBForceField.FindForceField(method)
        forcefield.Setup(mol, constraints)

        # Set the strictness of the constraint
        forcefield.SetConstraints(constraints)

        # Do a nsteps conjugate gradient minimization
        # (or less if converges) and save the coordinates to mol.
        forcefield.ConjugateGradients(nsteps)
        forcefield.GetCoordinates(mol)
        energy = forcefield.Energy() * 0.2390057361376673 # kJ/mol to kcal/mol

        # Write the mol to a file
        conv.WriteFile(mol,outname)
        conv.CloseOutFile()

        opt_coords = read_xyz(outname).atomcoords[0]

        clean_directory((f'{title}_in.xyz', f'{title}_out.xyz'))
        
        if check:
            success = scramble_check(opt_coords, atomnos, constrained_indices, graphs)
        else:
            success = True

        return opt_coords, energy, success