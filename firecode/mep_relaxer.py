import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import CalculationFailed
from ase.dyneb import DyNEB
from ase.optimize import LBFGS

from firecode.ase_manipulations import (PreventScramblingConstraint, ase_dump,
                                      get_ase_calc)
from firecode.utils import align_structures, time_to_string


def ase_mep_relax(
        embedder,
        structures,
        atomnos,
        n_images=None,
        maxiter=200,
        title='temp',
        optimizer=LBFGS,
        logfunction=None,
        write_plot=False,
        verbose_print=False,
        safe=False,
    ):
    '''
    embedder: firecode embedder object
    structures: array of coordinates to be used as starting points
    atomnos: 1-d array of atomic numbers
    n_images: total number of optimized images connecting reag/prods
    maxiter: maximum number of ensemble optimization steps
    title: name used to write the final MEP as a .xyz file
    optimizer: ASE optimizer to be used
    logfunction: filename to dump the optimization data to. If None, no file is written.
    write_plot: bool, prints a matplotlib plot with energy information

    return: 3- element tuple with coodinates of the MEP, energy of the structures in
    kcal/mol and a boolean value indicating success.
    '''

    if n_images is None:
        n_images = 10

    if len(structures) < n_images:
        # images = interpolate_structures(align_structures(structures), atomnos, n=n_images)

        # # If any molecule exploded, try linear interpolation
        # if any([True in np.isnan(image.get_positions()) for image in images]) or (
        #     np.max([image.get_positions() for image in images]) > 100):

        #     if logfunction is not None:
        #         logfunction(f'\n--> IDPP interpolation of structures failed, falling back to linear interpolation.')

        images = interpolate_structures(align_structures(structures), atomnos, n=n_images, method='linear')

        if logfunction is not None:
            logfunction(f'\n--> Interpolation of structures successful ({len(images)} images)')

    else:
        images = [Atoms(atomnos, positions=coords) for coords in align_structures(structures)]

    ase_dump('interpolated_MEP_guess.xyz', images, atomnos)

    neb = DyNEB(images,
                k=0.1,
                fmax=0.05,
                climb=False,
                # parallel=True,
                remove_rotation_and_translation=True,
                method='aseneb',
                scale_fmax=1,
                allow_shared_calculator=True,
                )
  
    # Set calculators for all images
    for _, image in enumerate(images):
        image.calc = get_ase_calc(embedder)

        if safe:
            bond_constr = PreventScramblingConstraint(embedder.objects[0].graph, image)
            image.set_constraint([bond_constr])


    t_start = time.perf_counter()

    # Set the optimizer and optimize
    try:
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # ignore runtime warnings from the NEB module:
            # if something went wrong, we will deal with it later

            with optimizer(neb, maxstep=0.1, logfile=None if not verbose_print else 'mep_relax_opt.log') as opt:

                if logfunction is not None:
                    logfunction(f'--> Running MEP relaxation through ASE ({embedder.options.theory_level} via {embedder.options.calculator})')

                for ss in range(1, maxiter//10+1):
                    opt.run(fmax=0.05, steps=maxiter//10*ss)

                    if logfunction is not None:
                        logfunction(f'--> Ran {maxiter//10*ss} steps, wrote partially optimized traj to {title}_MEP.xyz')

                    ase_dump(f'{title}_MEP.xyz', images, atomnos, [image.get_total_energy() * 23.06054194532933 for image in images])

                iterations = opt.nsteps
                exit_status = 'CONVERGED' if iterations < maxiter-1 else 'MAX ITER'

    except (CalculationFailed):
        if logfunction is not None:
            logfunction(f'    - MEP relax for {title} CRASHED ({time_to_string(time.perf_counter()-t_start)})\n')
            try:
                ase_dump(f'{title}_MEP_crashed.xyz', neb.images, atomnos)
            except Exception():
                pass
        return None, None, False

    except KeyboardInterrupt:
        exit_status = 'ABORTED BY USER'

    if logfunction is not None:
        logfunction(f'    - NEB for {title} {exit_status} ({time_to_string(time.perf_counter()-t_start)})\n')
  
    energies = [image.get_total_energy() * 23.06054194532933 for image in images] # eV to kcal/mol
    
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

        plt.legend()
        plt.title(title)
        plt.xlabel('Image number')
        plt.ylabel('Rel. E. (kcal/mol)')
        plt.savefig(f'{title.replace(" ", "_")}_plt.svg')

    mep = np.array([image.get_positions() for image in images])

    return mep, energies, exit_status

def interpolate_structures(structures, atomnos, n, method='idpp'):
    '''
    Return n interpolated structures from the
    first to the last present in structures
    as a list of ASE image objects.
    
    '''
    if len(structures) == 2:
        images = [None for _ in range(n)]
        images[0] = Atoms(atomnos, positions=structures[0])
        images[-1] = Atoms(atomnos, positions=structures[-1])
        group_ranges = [(0,n-1)]

    else:
        # calculate the expansion ratio between what we have and what we want
        ratio = n/len(structures)

        # calculate where original structures will be mapped in the final set
        mappings = [round(i*ratio) for i, _ in enumerate(structures)]
        mappings[-1] = len(structures)

        # initialize output container with initial structures mapped in
        images = [Atoms(atomnos, positions=structures[mappings.index(i)])
                    if i in mappings else None for i in range(n)]
        images[-1] = Atoms(atomnos, positions=structures[-1])

        # calculate ranges to fill
        group_ranges = [(mappings[i], mappings[i+1]) for i, _ in enumerate(mappings[:-1])
                            if mappings[i+1] - mappings[i] > 1]
        group_ranges.append((max(mappings), n-1))

    # fill them by interpolating nearby images
    for (ref_1, ref_2) in group_ranges:

        struc_ref1 = images[ref_1]
        struc_ref2 = images[ref_2]

        images_temp = [struc_ref1] + [struc_ref1.copy() for _ in range(ref_2-ref_1-1)] + [struc_ref2]
        interp_temp = DyNEB(images_temp)
        interp_temp.interpolate(method=method)

        # replace previous blanks with interpolated structures
        for i in range(ref_1+1, ref_2):
            images[i] = interp_temp.images[i-ref_1]

    return images