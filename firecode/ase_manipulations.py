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
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import (CalculationFailed,
                                        PropertyNotImplementedError)
from ase.calculators.gaussian import Gaussian
from ase.calculators.mopac import MOPAC
from ase.calculators.orca import ORCA
from ase.constraints import FixInternals
from ase.dyneb import DyNEB
from ase.optimize import BFGS, LBFGS
from ase.vibrations import Vibrations
from sella import Sella

from firecode.algebra import dihedral, norm, norm_of, point_angle
from firecode.graph_manipulations import findPaths, graphize, neighbors
from firecode.rmsd import get_alignment_matrix
from firecode.settings import COMMANDS, MEM_GB
from firecode.solvents import get_solvent_line
from firecode.utils import (HiddenPrints, align_structures, clean_directory,
                            get_double_bonds_indices, molecule_check,
                            scramble_check, time_to_string, write_xyz)


class Spring:
    '''
    ASE Custom Constraint Class
    Adds an harmonic force between a pair of atoms.
    Spring constant is very high to achieve tight convergence,
    but maximum force is dampened so as not to ruin structures.
    '''
    def __init__(self, i1, i2, d_eq, k=300):
        self.i1, self.i2 = i1, i2
        self.d_eq = d_eq
        self.k = k

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_forces(self, atoms, forces):

        direction = atoms.positions[self.i2] - atoms.positions[self.i1]
        # vector connecting atom1 to atom2

        spring_force = self.k * (norm_of(direction) - self.d_eq)
        # absolute spring force (float). Positive if spring is overstretched.

        
        spring_force = np.clip(spring_force, -50, 50)
        # force is clipped at 50 eV/A ()

        forces[self.i1] += (norm(direction) * spring_force)
        forces[self.i2] -= (norm(direction) * spring_force)
        # applying harmonic force to each atom, directed toward the other one

    def __repr__(self):
        return f'Spring - ids:{self.i1}/{self.i2} - d_eq:{self.d_eq}, k:{self.k}'

class HalfSpring:
    '''
    ASE Custom Constraint Class
    Adds an harmonic force between a pair of atoms,
    only if those two atoms are at least d_max
    Angstroms apart.
    '''
    def __init__(self, i1, i2, d_max, k=1000):
        self.i1, self.i2 = i1, i2
        self.d_max = d_max
        self.k = k

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_forces(self, atoms, forces):

        direction = atoms.positions[self.i2] - atoms.positions[self.i1]
        # vector connecting atom1 to atom2

        if norm_of(direction) > self.d_max:

            spring_force = self.k * (norm_of(direction) - self.d_max)
            # absolute spring force (float). Positive if spring is overstretched.

            spring_force = np.clip(spring_force, -50, 50)
            # force is clipped at 50 eV/A

            forces[self.i1] += (norm(direction) * spring_force)
            forces[self.i2] -= (norm(direction) * spring_force)
            # applying harmonic force to each atom, directed toward the other one

    def __repr__(self):
        return f'Spring - ids:{self.i1}/{self.i2} - d_max:{self.d_max}, k:{self.k}'

class PlanarAngleSpring:
    '''
    ASE Custom Constraint Class
    Adds an harmonic force among a triad of atoms.
    Spring constant is dinamycally adjusted to achieve tight convergence,
    but maximum force is dampened so as not to ruin structures.
    eq_angle: in degrees
    '''
    def __init__(self, i1, i2, i3, eq_angle):
        self.i1, self.i2, self.i3 = i1, i2, i3
        self.eq_angle = eq_angle

        self.k_min = 0.2
        self.k_max = 2
        self.theta_mid = 3

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_forces(self, atoms, forces):

        # Get positions
        p1 = atoms.positions[self.i1]
        p2 = atoms.positions[self.i2]
        p3 = atoms.positions[self.i3]

        # Calculate vectors
        v21 = p1 - p2  # Vector from atom 2 to 1
        v23 = p3 - p2  # Vector from atom 2 to 3
        
        # Get bond lengths
        r1 = norm_of(v21)
        r3 = norm_of(v23)

        # Calculate current angle
        current_angle = point_angle(p1, p2, p3)
        delta_angle = self.eq_angle - current_angle

        # Calculate unit vectors
        e21 = v21 / r1
        e23 = v23 / r3

        # Normal to the plane
        normal = np.cross(e21, e23)
        if norm_of(normal) > 1e-10:  # Check if atoms aren't collinear
            
            normal = norm(normal)
            
            # Calculate perpendicular directions in the plane
            f1_direction = np.cross(e21, normal)
            f3_direction = np.cross(normal, e23)
            
            # Dynamically adjust force constant: k_min for large delta_angles,
            # up to k_max for smaller ones. Sigmoid-like function centered on self.theta_mid.
            sigmoid = 1-(delta_angle-self.theta_mid)/(1+np.abs(delta_angle-self.theta_mid))
            k_dynamic = (self.k_max-self.k_min)/2 * (sigmoid) + self.k_min
                
            force_magnitude = np.clip(delta_angle * k_dynamic, -2, 2)

            # Calculate forces perpendicular to bonds
            f1 = force_magnitude * f1_direction
            f3 = force_magnitude * f3_direction
            
            # The central force needs to maintain both force and torque balance
            f2 = -(f1 + f3)

            forces[self.i1] += f1
            forces[self.i2] += f2
            forces[self.i3] += f3

    def __repr__(self):
        return f'PlanarAngleSpring - ids:{self.i1}/{self.i2}/{self.i3} - eq_angle:{self.eq_angle}'

class DihedralSpring:
    '''
    ASE Custom Constraint Class
    Adds a harmonic force to control a dihedral angle between four atoms.
    Improved handling of stiff bonds and out-of-plane rotations.
    
    Parameters:
    -----------
    i1, i2, i3, i4 : int
        Indices of the four atoms defining the dihedral angle
    eq_angle : float
        Target dihedral angle in degrees
    k_min : float, optional
        Minimum force constant (default: 0.1)
    k_max : float, optional
        Maximum force constant (default: 1)
    '''
    def __init__(self, i1, i2, i3, i4, eq_angle, k_min=0.1, k_max=1):
        self.i1, self.i2, self.i3, self.i4 = i1, i2, i3, i4
        self.eq_angle = eq_angle
        
        self.k_min = k_min
        self.k_max = k_max
        self.theta_mid = 30.0

    def adjust_positions(self, atoms, newpositions):
        pass

    def get_delta_angle(self, current, target):
        """
        Calculate the smallest angle difference considering periodicity.
        Returns the difference in degrees.
        """
        diff = ((target - current + 180) % 360) - 180
        return diff

    def adjust_forces(self, atoms, forces):
        # Get positions         
        p1 = atoms.positions[self.i1].copy()
        p2 = atoms.positions[self.i2].copy()
        p3 = atoms.positions[self.i3].copy()
        p4 = atoms.positions[self.i4].copy()

        # Calculate current dihedral angle
        current_angle =dihedral((p1, p2, p3, p4))
        
        # Calculate periodic angle difference
        delta_angle = self.get_delta_angle(current_angle, self.eq_angle)
        
        # Calculate central bond length (important for force scaling)
        v23 = p3 - p2
        bond_length = norm_of(v23)
        
        # Dynamic force constant calculation - quadratic scaling near target
        rel_angle = np.abs(delta_angle) / self.theta_mid
        k_dynamic = self.k_max * np.exp(-rel_angle) + self.k_min
        
        # Scale force by bond length (shorter bonds need more force)
        force_scale = 1.0 / bond_length
        
        # Calculate force magnitude with scaled clipping
        force_magnitude = np.clip(delta_angle * k_dynamic * force_scale, -5.0, 5.0)
        
        # Calculate force directions
        v21 = p1 - p2
        v34 = p4 - p3
        
        # Calculate perpendicular components for force application
        n1 = np.cross(v21, v23)
        n2 = np.cross(v23, v34)
        
        n1_norm = norm_of(n1)
        n2_norm = norm_of(n2)
        
        if n1_norm < 1e-6 or n2_norm < 1e-6:
            return
            
        f1_direction = n1 / n1_norm
        f4_direction = n2 / n2_norm
        
        # Apply forces with improved balance
        f1 = force_magnitude * f1_direction
        f4 = force_magnitude * f4_direction
        
        # Distribute forces to maintain total force and torque
        forces[self.i1] += f1
        forces[self.i4] += f4
        forces[self.i2] += -0.75 * f1
        forces[self.i3] += -0.75 * f4
        # Remaining -0.25 * (f1 + f4) is distributed equally
        center_force = -0.25 * (f1 + f4)
        forces[self.i2] += center_force
        forces[self.i3] += center_force

    def __repr__(self):
        return f'DihedralSpring(ids: {self.i1}/{self.i2}/{self.i3}/{self.i4}, eq_angle: {self.eq_angle})'

def get_ase_calc(embedder):
    '''
    Attach the correct ASE calculator
    to the ASE Atoms object.
    embedder: either a firecode embedder object or
    a 4-element strings tuple containing
    (calculator, method, procs, solvent)
    '''
    if isinstance(embedder, tuple):
        calculator, method, procs, solvent = embedder

    else:
        calculator = embedder.options.calculator
        method = embedder.options.theory_level
        procs = embedder.procs
        solvent = embedder.options.solvent

    if calculator == 'XTB':
        try:
            from xtb.ase.calculator import XTB
        except ImportError:
            raise Exception(('Cannot import xtb python bindings. Install them with:\n'
                             '>>> conda install -c conda-forge xtb-python\n'
                             '(See https://github.com/grimme-lab/xtb-python)'))

        from firecode.solvents import (solvent_synonyms, xtb_solvents,
                                       xtb_supported)
                                     
        solvent = solvent_synonyms[solvent] if solvent in solvent_synonyms else solvent
        solvent = 'none' if solvent is None else solvent

        if solvent not in xtb_solvents:
            raise Exception(f'Solvent \'{solvent}\' not supported by XTB. Supported solvents are:\n{xtb_supported}')

        return XTB(method=method, solvent=solvent)
    
    if calculator == 'AIMNET2':
        try:
            from aimnet2_firecode.interface import get_aimnet2_calc
        except ImportError:
            raise Exception(('Cannot import AIMNET2 python bindings. Install them with:\n'
                             '>>> pip install aimnet2_firecode\n'))
        
        if hasattr(embedder.dispatcher, "aimnet2_calc"):
            return embedder.dispatcher.aimnet2_calc
        
        else:
            return get_aimnet2_calc(method=method, logfunction=embedder.log)
    
    command = COMMANDS[calculator]

    if calculator == 'MOPAC':

        if solvent is not None:
            method = method + ' ' + get_solvent_line(solvent, calculator, method)

        return MOPAC(label='temp',
                    command=f'{command} temp.mop > temp.cmdlog 2>&1',
                    method=method+' GEO-OK')

    if calculator == 'ORCA':

        orcablocks = ''

        if procs > 1:
            orcablocks += f'%pal nprocs {procs} end'

        if solvent is not None:
            orcablocks += get_solvent_line(solvent, calculator, method)

        return ORCA(label='temp',
                    command=f'{command} temp.inp "--oversubscribe" > temp.out 2>&1',
                    orcasimpleinput=method,
                    orcablocks=orcablocks)

    if calculator == 'GAUSSIAN':

        if solvent is not None:
            method = method + ' ' + get_solvent_line(solvent, calculator, method)

        mem = str(MEM_GB)+'GB' if MEM_GB >= 1 else str(int(1000*MEM_GB))+'MB'

        calc = Gaussian(label='temp',
                        command=f'{command} temp.com',
                        method=method,
                        nprocshared=procs,
                        mem=mem,
                        )

        if 'g09' in command:

            from ase.io import read
            def g09_read_results(self=calc):
                output = read(self.label + '.out', format='gaussian-out')
                self.calc = output.calc
                self.results = output.calc.results

            calc.read_results = g09_read_results

            # Adapting for g09 outputting .out files instead of g16 .log files.
            # This is a bad fix and the issue should be corrected in the ASE
            # source code: merge request on GitHub pending to be written

            return calc

def ase_saddle(embedder, coords, atomnos, constrained_indices=None, mols_graphs=None, title='temp', logfile=None, traj=None, freq=False, maxiterations=200):
    '''
    Runs a first order saddle optimization through the ASE package
    '''
    atoms = Atoms(atomnos, positions=coords)

    atoms.calc = get_ase_calc(embedder)
    
    t_start = time.perf_counter()
    with HiddenPrints():
        with Sella(atoms,
                    logfile=logfile,
                    order=1,
                    trajectory=traj) as opt:

            opt.run(fmax=0.05, steps=maxiterations)
            iterations = opt.nsteps

    if logfile is not None:
        t_end_berny = time.perf_counter()
        elapsed = t_end_berny - t_start
        exit_str = 'converged' if iterations < maxiterations else 'stopped'
        logfile.write(f'{title} - {exit_str} in {iterations} steps ({time_to_string(elapsed)})\n')

    new_structure = atoms.get_positions()
    energy = atoms.get_total_energy() * 23.06054194532933 #eV to kcal/mol

    if mols_graphs is not None:
        success = scramble_check(new_structure, atomnos, constrained_indices, mols_graphs, max_newbonds=embedder.options.max_newbonds)
    else:
        success = molecule_check(coords, new_structure, atomnos, max_newbonds=embedder.options.max_newbonds)

    return new_structure, energy, success

def ase_vib(embedder, coords, atomnos, logfunction=None, title='temp'):
    '''
    Calculate frequencies through ASE - returns frequencies and number of negatives (not in use)
    '''
    atoms = Atoms(atomnos, positions=coords)
    atoms.calc = get_ase_calc(embedder)
    vib = Vibrations(atoms, name=title)

    if os.path.isdir(title):
        os.chdir(title)
        for f in os.listdir():
            os.remove(f)
        os.chdir(os.path.dirname(os.getcwd()))
    else:
        os.mkdir(title)

    os.chdir(title)

    t_start = time.perf_counter()

    with HiddenPrints():
        vib.run()

    # freqs = vib.get_frequencies()
    freqs = vib.get_energies()* 8065.544 # from eV to cm-1 

    if logfunction is not None:
        elapsed = time.perf_counter() - t_start
        logfunction(f'{title} - frequency calculation completed ({time_to_string(elapsed)})')
    
    os.chdir(os.path.dirname(os.getcwd()))

    return freqs, np.count_nonzero(freqs.imag > 1e-3)

def ase_neb(embedder, reagents, products, atomnos, ts_guess=None, n_images=6, mep_override=None, title='temp', optimizer=LBFGS, logfunction=None, write_plot=False, verbose_print=False):
    '''
    embedder: firecode embedder object
    reagents: coordinates for the atom arrangement to be used as reagents
    products: coordinates for the atom arrangement to be used as products
    atomnos: 1-d array of atomic numbers
    n_images: number of optimized images connecting reag/prods
    title: name used to write the final MEP as a .xyz file
    optimizer: ASE optimizer to be used in 
    logfile: filename to dump the optimization data to. If None, no file is written.

    return: 3- element tuple with coodinates of highest point along the MEP, its
    energy in kcal/mol and a boolean value indicating success.
    '''
    reagents, products = align_structures(np.array([reagents, products]))
    first = Atoms(atomnos, positions=reagents)
    last = Atoms(atomnos, positions=products)

    if mep_override is not None:

        images = [Atoms(atomnos, positions=coords) for coords in mep_override]
        neb = DyNEB(images, fmax=0.05, climb=False, method='eb', scale_fmax=1, allow_shared_calculator=True)

    elif ts_guess is None:
        images =  [first]
        images += [first.copy() for _ in range(n_images)]
        images += [last]

        neb = DyNEB(images, fmax=0.05, climb=False, method='eb', scale_fmax=1, allow_shared_calculator=True)
        neb.interpolate(method='idpp')

    else:
        ts_guess = Atoms(atomnos, positions=ts_guess)

        images_1 = [first] + [first.copy() for _ in range(round((n_images-3)/2))] + [ts_guess]
        interp_1 = DyNEB(images_1)
        interp_1.interpolate(method='idpp')

        images_2 = [ts_guess] + [last.copy() for _ in range(n_images-len(interp_1.images)-1)] + [last]
        interp_2 = DyNEB(images_2)
        interp_2.interpolate(method='idpp')

        images = interp_1.images + interp_2.images[1:]

        neb = DyNEB(images, fmax=0.05, climb=False, method='eb', scale_fmax=1, allow_shared_calculator=True)

    if mep_override is None:
        ase_dump(f'{title}_MEP_guess.xyz', images, atomnos)

    if verbose_print and logfunction is not None and mep_override is None:
        logfunction(f'\n\n--> Saved interpolated MEP guess to {title}_MEP_guess.xyz\n')
    
    # Set calculators for all images
    for _, image in enumerate(images):
        image.calc = get_ase_calc(embedder)

    t_start = time.perf_counter()

    # Set the optimizer and optimize
    try:
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # ignore runtime warnings from the NEB module:
            # if something went wrong, we will deal with it later

            with optimizer(neb, maxstep=0.1, logfile=None if not verbose_print else 'neb_opt.log') as opt:

                if verbose_print and logfunction is not None:
                    logfunction(f'\n--> Running NEB-CI through ASE ({embedder.options.theory_level} via {embedder.options.calculator})')

                opt.run(fmax=0.05, steps=20)
                while neb.get_residual() > 0.1:
                    opt.run(fmax=0.05, steps=10+opt.nsteps)
                    # some free relaxation before starting to climb
                    if opt.nsteps > 500 or opt.converged:
                        break

                if verbose_print and logfunction is not None:
                    logfunction('--> fmax below 0.1: Activated Climbing Image and smaller maxstep')

                ase_dump(f'{title}_MEP_start_of_CI.xyz', neb.images, atomnos)

                optimizer.maxstep = 0.01
                neb.climb = True

                opt.run(fmax=0.05, steps=250+opt.nsteps)

                iterations = opt.nsteps
                exit_status = 'CONVERGED' if iterations < 279 else 'MAX ITER'

        # success = True if exit_status == 'CONVERGED' else False

    except (PropertyNotImplementedError, CalculationFailed):
        if logfunction is not None:
            logfunction(f'    - NEB for {title} CRASHED ({time_to_string(time.perf_counter()-t_start)})\n')
            try:
                ase_dump(f'{title}_MEP_crashed.xyz', neb.images, atomnos)
            except Exception():
                pass
        return None, None, None, False

    except KeyboardInterrupt:
        exit_status = 'ABORTED BY USER'

    if logfunction is not None:
        logfunction(f'    - NEB for {title} {exit_status} ({time_to_string(time.perf_counter()-t_start)})\n')

    energies = [image.get_total_energy() * 23.06054194532933 for image in images] # eV to kcal/mol
    
    ts_id = energies.index(max(energies))
    # print(f'TS structure is number {ts_id}, energy is {max(energies)}')

    if mep_override is None:
        os.remove(f'{title}_MEP_guess.xyz')
    ase_dump(f'{title}_MEP.xyz', images, atomnos, energies)
    # Save the converged MEP (minimum energy path) to an .xyz file

    if write_plot:

        plt.figure()
        plt.plot(
            range(1,len(images)+1),
            np.array(energies)-min(energies),
            color='tab:blue',
            label='Image energies',
            linewidth=3,
        )

        plt.plot(
            [ts_id+1],
            [energies[ts_id]-min(energies)],
            color='gold',
            label='TS guess',
            marker='o',
            markersize=3,
        )

        plt.legend()
        plt.title(title)
        plt.xlabel('Image number')
        plt.ylabel('Rel. E. (kcal/mol)')
        plt.savefig(f'{title.replace(" ", "_")}_plt.svg')

    return images[ts_id].get_positions(), energies[ts_id], energies, exit_status

class OrbitalSpring:
    '''
    ASE Custom Constraint Class
    Adds a series of forces based on a pair of orbitals, that is
    virtual points "bonded" to a given atom.

    :params i1, i2: indices of reactive atoms
    :params orb1, orb2: 3D coordinates of orbitals
    :params neighbors_of_1, neighbors_of_2: lists of indices for atoms bonded to i1/i2
    :params d_eq: equilibrium target distance between orbital centers
    '''
    def __init__(self, i1, i2, orb1, orb2, neighbors_of_1, neighbors_of_2, d_eq, k=1000):
        self.i1, self.i2 = i1, i2
        self.orb1, self.orb2 = orb1, orb2
        self.neighbors_of_1, self.neighbors_of_2 = neighbors_of_1, neighbors_of_2
        self.d_eq = d_eq
        self.k = k

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_forces(self, atoms, forces):

        # First, assess if we have to move atoms 1 and 2 at all

        sum_of_distances = (norm_of(atoms.positions[self.i1] - self.orb1) +
                            norm_of(atoms.positions[self.i2] - self.orb2) + self.d_eq)

        reactive_atoms_distance = norm_of(atoms.positions[self.i1] - atoms.positions[self.i2])

        orb_direction = self.orb2 - self.orb1
        # vector connecting orb1 to orb2

        spring_force = self.k * (norm_of(orb_direction) - self.d_eq)
        # absolute spring force (float). Positive if spring is overstretched.

        # spring_force = np.clip(spring_force, -50, 50)
        # # force is clipped at 5 eV/A

        force_direction1 = np.sign(spring_force) * norm(np.mean((norm(+orb_direction),
                                                                    norm(self.orb1-atoms.positions[self.i1])), axis=0))

        force_direction2 = np.sign(spring_force) * norm(np.mean((norm(-orb_direction),
                                                                    norm(self.orb2-atoms.positions[self.i2])), axis=0))

        # versors specifying the direction at which forces act, that is on the
        # bisector of the angle between vector connecting atom to orbital and
        # vector connecting the two orbitals

        if np.abs(sum_of_distances - reactive_atoms_distance) > 0.2:

            forces[self.i1] += (force_direction1 * spring_force)
            forces[self.i2] += (force_direction2 * spring_force)
            # applying harmonic force to each atom, directed toward the other one

        # Now applying to neighbors the force derived by torque, scaled to match the spring_force,
        # but only if atomic orbitals are more than two Angstroms apart. This improves convergence.

        if norm_of(orb_direction) > 2:
            torque1 = np.cross(self.orb1 - atoms.positions[self.i1], force_direction1)
            for i in self.neighbors_of_1:
                forces[i] += norm(np.cross(torque1, atoms.positions[i] - atoms.positions[self.i1])) * spring_force

            torque2 = np.cross(self.orb2 - atoms.positions[self.i2], force_direction2)
            for i in self.neighbors_of_2:
                forces[i] += norm(np.cross(torque2, atoms.positions[i] - atoms.positions[self.i2])) * spring_force

def PreventScramblingConstraint(graph, atoms, double_bond_protection=False, fix_angles=False):
    '''
    graph: NetworkX graph of the molecule
    atoms: ASE atoms object

    return: FixInternals constraint to apply to ASE calculations
    '''
    angles_deg = None
    if fix_angles:
        allpaths = []

        for node in graph:
            allpaths.extend(findPaths(graph, node, 2))

        allpaths = {tuple(sorted(path)) for path in allpaths}

        angles_deg = []
        for path in allpaths:
            angles_deg.append([atoms.get_angle(*path), list(path)])

    bonds = []
    for bond in [[a, b] for a, b in graph.edges if a != b]:
        bonds.append([atoms.get_distance(*bond), bond])

    dihedrals_deg = None
    if double_bond_protection:
        double_bonds = get_double_bonds_indices(atoms.positions, atoms.get_atomic_numbers())
        if double_bonds != []:
            dihedrals_deg = []
            for a, b in double_bonds:
                n_a = neighbors(graph, a)
                n_a.remove(b)

                n_b = neighbors(graph, b)
                n_b.remove(a)

                d = [n_a[0], a, b, n_b[0]]
                dihedrals_deg.append([atoms.get_dihedral(*d), d])

    return FixInternals(dihedrals_deg=dihedrals_deg, angles_deg=angles_deg, bonds=bonds, epsilon=1)

def ase_popt(
                embedder,
                coords,
                atomnos,

                constrained_indices=None,
                constrained_distances=None,

                constrained_angles_indices=None,
                constrained_angles_values=None,

                constrained_dihedrals_indices=None,
                constrained_dihedrals_values=None,

                steps=500,
                safe=False,
                safe_mask=None,
                traj=None,
                logfunction=None,
                title='temp',
        ):
    '''
    embedder: firecode embedder object
    coords: 
    atomnos: 
    constrained_indices:
    safe: if True, adds a potential that prevents atoms from scrambling
    safe_mask: bool array, with False for atoms to be excluded when calculating bonds to preserve
    traj: if set to a string, traj is used as a filename for the bending trajectory.
    not only the atoms will be printed, but also all the orbitals and the active pivot.
    '''
    atoms = Atoms(atomnos, positions=coords)
    atoms.calc = get_ase_calc(embedder)
    constraints = []

    if constrained_indices is not None:
        constrained_distances = constrained_distances or [None for _ in constrained_indices]
        for i, c in enumerate(constrained_indices):
            i1, i2 = c
            tgt_dist = constrained_distances[i] or norm_of(coords[i1]-coords[i2])
            constraints.append(Spring(i1, i2, tgt_dist))

    if constrained_angles_indices is not None:
        constrained_angles_values = constrained_angles_values or [None for _ in constrained_angles_indices]
        for i, c in enumerate(constrained_angles_indices):
            i1, i2, i3 = c
            tgt_angle = constrained_angles_values[i] or point_angle(coords[i1], coords[i2], coords[i3])
            constraints.append(PlanarAngleSpring(i1, i2, i3, tgt_angle))

    if constrained_dihedrals_indices is not None:
        constrained_dihedrals_values = constrained_dihedrals_values or [None for _ in constrained_dihedrals_indices]
        for i, c in enumerate(constrained_dihedrals_indices):
            i1, i2, i3, i4 = c
            tgt_angle = constrained_dihedrals_values[i] or dihedral((coords[i1], coords[i2], coords[i3], coords[i4]))
            constraints.append(DihedralSpring(i1, i2, i3, i4, tgt_angle))

    if safe:
        constraints.append(PreventScramblingConstraint(graphize(coords, atomnos, safe_mask),
                                                        atoms,
                                                        double_bond_protection=embedder.options.double_bond_protection,
                                                        fix_angles=embedder.options.fix_angles_in_deformation))

    atoms.set_constraint(constraints)

    t_start_opt = time.perf_counter()
    with LBFGS(atoms, maxstep=0.1, logfile=None, trajectory=traj) as opt:
        opt.run(fmax=0.05, steps=steps)
        iterations = opt.nsteps

    new_structure = atoms.get_positions()
    success = (iterations < 499)

    if logfunction is not None:
        exit_str = 'REFINED' if success else 'MAX ITER'
        logfunction(f'    - {title} {exit_str} ({iterations} iterations, {time_to_string(time.perf_counter()-t_start_opt)})')

    energy = atoms.get_total_energy() * 23.06054194532933 #eV to kcal/mol

    return new_structure, energy, success

def ase_bend(embedder, original_mol, conf, pivot, threshold, title='temp', traj=None, check=True):
    '''
    embedder: firecode embedder object
    original_mol: Hypermolecule object to be bent
    conf: index of conformation in original_mol to be used
    pivot: pivot connecting two Hypermolecule orbitals to be approached/distanced
    threshold: target distance for the specified pivot, in Angstroms
    title: name to be used for referring to this structure in the embedder log
    traj: if set to a string, traj+\'.traj\' is used as a filename for the bending trajectory.
    not only the atoms will be printed, but also all the orbitals and the active pivot.
    check: if True, after bending checks that the bent structure did not scramble.
    If it did, returns the initial molecule.
    '''

    identifier = np.sum(original_mol.atomcoords[conf])

    if hasattr(embedder, "ase_bent_mols_dict"):
        cached = embedder.ase_bent_mols_dict.get((identifier, tuple(sorted(pivot.index)), round(threshold, 3)))
        if cached is not None:
            return cached

    if traj is not None:

        from ase.io.trajectory import Trajectory

        def orbitalized(atoms, orbitals, pivot=None):
            positions = np.concatenate((atoms.positions, orbitals))

            if pivot is not None:
                positions = np.concatenate((positions, [pivot.start], [pivot.end]))

            symbols = list(atoms.numbers) + [0 for _ in orbitals]

            if pivot is not None:
                symbols += [9 for _ in range(2)]
            # Fluorine (9) represents active orbitals
    
            new_atoms = Atoms(symbols, positions=positions)
            return new_atoms

        try:
            os.remove(traj)
        except FileNotFoundError:
            pass

    i1, i2 = original_mol.reactive_indices

    neighbors_of_1 = neighbors(original_mol.graph, i1)
    neighbors_of_2 = neighbors(original_mol.graph, i2)

    mol = deepcopy(original_mol)
    final_mol = deepcopy(original_mol)

    for p in mol.pivots[conf]:
        if p.index == pivot.index:
            active_pivot = p
            break
    
    dist = norm_of(active_pivot.pivot)

    atoms = Atoms(mol.atomnos, positions=mol.atomcoords[conf])

    atoms.calc = get_ase_calc(embedder)
    
    if traj is not None:
        traj_obj = Trajectory(traj + f'_conf{conf}.traj',
                                mode='a',
                                atoms=orbitalized(atoms,
                                                np.vstack([atom.center for atom in mol.reactive_atoms_classes_dict[0].values()]),
                                                active_pivot))
        traj_obj.write()

    unproductive_iterations = 0
    break_reason = 'MAX ITER'
    t_start = time.perf_counter()

    for iteration in range(500):

        atoms.positions = mol.atomcoords[0]

        orb_memo = {index:norm_of(atom.center[0]-atom.coord) for index, atom in mol.reactive_atoms_classes_dict[0].items()}

        orb1, orb2 = active_pivot.start, active_pivot.end

        c1 = OrbitalSpring(i1, i2, orb1, orb2, neighbors_of_1, neighbors_of_2, d_eq=threshold)

        c2 = PreventScramblingConstraint(mol.graph,
                                            atoms,
                                            double_bond_protection=embedder.options.double_bond_protection,
                                            fix_angles=embedder.options.fix_angles_in_deformation)

        atoms.set_constraint([
                                c1,
                                c2,
                                ])

        opt = BFGS(atoms, maxstep=0.2, logfile=None, trajectory=None)

        try:
            opt.run(fmax=0.5, steps=1)
        except ValueError:
            # Shake did not converge
            break_reason = 'CRASHED'
            break

        if traj is not None:
            traj_obj.atoms = orbitalized(atoms, np.vstack([atom.center for atom in mol.reactive_atoms_classes_dict[0].values()]))
            traj_obj.write()

        # check if we are stuck
        if np.max(np.abs(norm_of(atoms.get_positions() - mol.atomcoords[0], axis=1))) < 0.01:
            unproductive_iterations += 1

            if unproductive_iterations == 10:
                break_reason = 'STUCK'
                break

        else:
            unproductive_iterations = 0

        mol.atomcoords[0] = atoms.get_positions()

        # Update orbitals and get temp pivots
        for index, atom in mol.reactive_atoms_classes_dict[0].items():
            atom.init(mol, index, update=True, orb_dim=orb_memo[index])
            # orbitals positions are calculated based on the conformer we are working on

        temp_pivots = embedder._get_pivots(mol)[0]

        for p in temp_pivots:
            if p.index == pivot.index:
                active_pivot = p
                break
        # print(active_pivot)

        dist = norm_of(active_pivot.pivot)
        # print(f'{iteration}. {mol.filename} conf {conf}: pivot is {round(dist, 3)} (target {round(threshold, 3)})')

        if dist - threshold < 0.1:
            break_reason = 'CONVERGED'
            break
        # else:
            # print('delta is ', round(dist - threshold, 3))

    embedder.log(f'    {title} - conformer {conf} - {break_reason}{" "*(9-len(break_reason))} ({iteration+1}{" "*(3-len(str(iteration+1)))} iterations, {time_to_string(time.perf_counter()-t_start)})', p=False)

    if check:
        if not molecule_check(original_mol.atomcoords[conf], mol.atomcoords[0], mol.atomnos, max_newbonds=1):
            mol.atomcoords[0] = original_mol.atomcoords[conf]
        # keep the bent structures only if no scrambling occurred between atoms

    final_mol.atomcoords[conf] = mol.atomcoords[0]

    # Now align the ensembles on the new reactive atoms positions

    reference, *targets = final_mol.atomcoords
    reference = np.array(reference)
    targets = np.array(targets)

    r = reference - np.mean(reference[final_mol.reactive_indices], axis=0)
    ts = np.array([t - np.mean(t[final_mol.reactive_indices], axis=0) for t in targets])

    output = []
    output.append(r)
    for target in ts:
        matrix = get_alignment_matrix(r, target)
        output.append([matrix @ vector for vector in target])

    final_mol.atomcoords = np.array(output)

    # Update orbitals and pivots
    for conf_, _ in enumerate(final_mol.atomcoords):
        for index, atom in final_mol.reactive_atoms_classes_dict[conf_].items():
            atom.init(final_mol, index, update=True, orb_dim=orb_memo[index])

    embedder._set_pivots(final_mol)

    # add result to cache (if we have it) so we avoid recomputing it
    if hasattr(embedder, "ase_bent_mols_dict"):
        embedder.ase_bent_mols_dict[(identifier, tuple(sorted(pivot.index)), round(threshold, 3))] = final_mol

    clean_directory()

    return final_mol

def ase_dump(filename, images, atomnos, energies=None):

    if energies is None:
        energies = ["" for _ in images]
    else:
        energies = np.array(energies)
        energies -= np.min(energies)

    with open(filename, 'w') as f:
        for i, (image, energy) in enumerate(zip(images, energies)):
            e = f" Rel.E = {round(energy, 3)} kcal/mol" if energy != "" else ""
            coords = image.get_positions()
            write_xyz(coords, atomnos, f, title=f'STEP {i+1} - {filename[:-4]}_image_{i+1}{e}')