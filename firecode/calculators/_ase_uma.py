import os
import sys
import time

from ase import Atoms
from ase.optimize import FIRE, LBFGS
from prism_pruner.algebra import dihedral
from prism_pruner.utils import time_to_string

from firecode.algebra import norm_of, point_angle
from firecode.ase_manipulations import (DihedralSpring, PlanarAngleSpring,
                                        Spring)
from firecode.calculators.__init__ import NewFolderContext
from firecode.calculators._xtb import xtb_gsolv
from firecode.settings import UMA_MODEL_PATH


def uma_opt(
                atoms,
                coords,
                ase_calc=None,
                method='omol',

                constrained_indices=None,
                constrained_distances=None,

                constrained_angles_indices=None,
                constrained_angles_values=None,

                constrained_dihedrals_indices=None,
                constrained_dihedrals_values=None,

                charge=0,
                mult=1,
                solvent=None,
                maxiter=None,
                conv_thr='tight',
                traj=None,
                logfunction=None,
                title='temp',

                optimizer='LBFGS',
                debug=False,
                **kwargs,
            ):
    '''
    coords: 
    atoms: 
    constrained_indices:
    safe: if True, adds a potential that prevents atoms from scrambling
    safe_mask: bool array, with False for atoms to be excluded when calculating bonds to preserve
    traj: if set to a string, traj is used as a filename for the bending trajectory.
    not only the atoms will be printed, but also all the orbitals and the active pivot.
    '''

    maxiter = maxiter or 500

    # create working folder and cd into it
    with NewFolderContext(title, delete_after=(not debug)):

        ase_calc = ase_calc or get_uma_calc(method)
        # ase_calc.do_reset()
        # ase_calc.set_charge(charge)

        atoms = Atoms(atoms, positions=coords)
        atoms.info.update({'charge':charge, 'spin':mult})
        atoms.calc = ase_calc
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

        atoms.set_constraint(constraints)

        fmax = {
            'tight' : 0.05,
            'loose' : 0.1,
        }[conv_thr]

        t_start_opt = time.perf_counter()
        optimizer_class = {'LBFGS':LBFGS, 'FIRE':FIRE}[optimizer]

        try:
            with optimizer_class(atoms, maxstep=0.05, logfile=None, trajectory=traj) as opt:
                opt.run(fmax=fmax, steps=maxiter)
                iterations = opt.nsteps

        except KeyboardInterrupt:
            print('KeyboardInterrupt requested by user. Quitting.')
            sys.exit()

        except TypeError as e:
            if logfunction is not None:
                logfunction(f'{title} in uma_opt CRASHED')
                logfunction(e)
            return coords, None, False 

        new_structure = atoms.get_positions()
        success = (iterations < 499)

        if logfunction is not None:
            exit_str = 'REFINED' if success else 'MAX ITER'
            logfunction(f'    - {title} {exit_str} ({iterations} iterations, {time_to_string(time.perf_counter()-t_start_opt)})')

        energy = atoms.get_total_energy() * 23.06054194532933 #eV to kcal/mol

        if traj is not None:
            os.system(f"ase convert {traj} {title}_trj.xyz")

        # try:
        #     os.remove('temp.traj')
            
        # except FileNotFoundError:
        #     pass

        if solvent is not None:
            gsolv = xtb_gsolv(
                                atoms,
                                new_structure,
                                model='alpb',
                                charge=charge,
                                mult=mult,
                                solvent=solvent,
                                title=title,
                                assert_convergence=True,
                            )
            energy += gsolv

    return new_structure, energy, success

def get_uma_calc(method="omol", logfunction=None):
    '''
    Load UMA model from disk and return the ASE calculator object
    '''

    try:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from torch import cuda

    except ImportError as err:
        print(err)
        raise ImportError('To run the UMA models, please install fairchem:\n    >>> pip install fairchem-core')

    gpu_bool = cuda.is_available()

    if gpu_bool:
        if logfunction is not None:
            logfunction(f'--> {cuda.device_count()} CUDA devices detected: loading model on GPU')

    else:
        if logfunction is not None:
            logfunction('--> No CUDA devices detected: loading model on CPU')

    if logfunction is not None:
            logfunction(f'--> Loading UMA/{method.upper()} model from file')
    
    if UMA_MODEL_PATH[0] == '.':
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.basename(UMA_MODEL_PATH))
    else:
        path = UMA_MODEL_PATH
        
    try:
        predictor = load_predict_unit(path, device='cuda' if gpu_bool else 'cpu')

    except FileNotFoundError:
        raise FileNotFoundError(f'UMA model at path {path} does not found.')

    ase_calc = FAIRChemCalculator(predictor, task_name=method.lower())
    return ase_calc

