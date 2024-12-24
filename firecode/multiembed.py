import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations
from shutil import copy, rmtree

import numpy as np

from firecode.errors import InputError, ZeroCandidatesError
from firecode.utils import (cartesian_product, suppress_stdout_stderr,
                          time_to_string, timing_wrapper)


def multiembed_dispatcher(embedder):
    '''
    Calls the appropriate multiembed subfunction
    based on embedder attributes.
    '''
    
    if len(embedder.objects) == 2:
        return multiembed_bifunctional(embedder)
    
    raise InputError('The multiembed requested is currently unavailable.')


def multiembed_bifunctional(embedder):
    '''
    Run multiple concurrent bifunctional cyclical embeds
    exploring all relative arrangement of each pair of
    reactive_indices between the two molecules.
    '''
    
    mol1, mol2 = embedder.objects

    # get every possible combination of indices in the two molecules
    pairs = cartesian_product(mol1.reactive_indices, mol2.reactive_indices)

    # get every arrangement of interacting pairs not insisting on the same atom twice
    arrangements = [((ix_1, ix_2), (iy_1, iy_2)) for ((ix_1, ix_2), (iy_1, iy_2)) in permutations(pairs, 2) if ix_1 != iy_1 and ix_2 != iy_2]

    structures_out, constr_ids, processes = [], [], []

    embedder.t_start_run = time.perf_counter()
    embedder.log()

    max_workers = embedder.avail_cpus or 1
    embedder.log(f'--> Multiembed: running {len(arrangements)} embeds on {max_workers} threads')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        # for each arrangement, perform a dedicated embed 
        for i, arrangement in enumerate(arrangements):

            process = executor.submit(
                                        timing_wrapper,
                                        run_child_embedder,
                                        mol1.filename,
                                        mol2.filename,
                                        constrained_indices=arrangement,
                                        i=i,
                                        options=embedder.options,
                                    )
            processes.append(process)  

        for i, process in enumerate(as_completed(processes)):

            (structures, constrained_indices), elapsed = process.result()

            embedder.log(f'--> Child process {i+1:3}/{len(arrangements):3}: generated {len(structures):4} candidates in {time_to_string(elapsed, verbose=True)}.')

            if len(structures) > 0:
                structures_out.append(structures)
                constr_ids.append(constrained_indices)
                embedder.write_structures('embedded', energies=False)

    structures_out = np.concatenate(structures_out)

    embedder.log(f'\n--> Multiembed completed: generated {len(structures_out)} candidates in {time_to_string(time.perf_counter() - embedder.t_start_run, verbose=True)}.')
    
    # only get interaction constraints, as the internal will be added later during refinement
    embedder.constrained_indices = np.concatenate(constr_ids)

    return structures_out

def run_child_embedder(
                        mol1_name,
                        mol2_name,
                        constrained_indices,
                        i,
                        options,
                    ):

    from firecode.embedder import Embedder, RunEmbedding

    start_dir = os.getcwd()
    foldername = f'firecode_embed{i+1}'
    (ix_1, ix_2), (iy_1, iy_2) = constrained_indices

    # create a dedicated folder
    if not os.path.isdir(os.path.join(os.getcwd(), foldername)):
        os.mkdir(foldername)

    # copy structure files into it
    copy(os.path.join(os.getcwd(), mol1_name),
            os.path.join(os.getcwd(), foldername))
    copy(os.path.join(os.getcwd(), mol2_name),
            os.path.join(os.getcwd(), foldername))

    os.chdir(foldername)
    child_name = f'embed{i+1}_input.txt'

    with open(child_name, 'w') as f:
        extra = ''
        extra += ' debug' if options.debug else ''
        extra += ' simpleorbitals' if options.simpleorbitals else ''
        extra += f' shrink={options.shrink_multiplier}' if options.shrink else ''

        f.write(f'noopt rigid{extra}\n')
        f.write(f'{mol1_name} {ix_1}x {iy_1}y\n')
        f.write(f'{mol2_name} {ix_2}x {iy_2}y\n')

    with suppress_stdout_stderr():

        child_name = os.path.join(os.getcwd(), child_name)
        child_embedder = Embedder(child_name, f'embed{i+1}')
        child_embedder = RunEmbedding(child_embedder)

        child_embedder._set_reactive_atoms_cumnums()
        child_embedder.write_mol_info()
        child_embedder.log(f'\n--> FIRECODE multiembed child process - arrangement {i+1}')
        child_embedder.t_start_run = time.perf_counter()

        try:
            child_embedder.generate_candidates()
            child_embedder.compenetration_refining()
            child_embedder.fitness_refining()
            child_embedder.similarity_refining(rmsd=False, verbose=True)
            child_embedder.write_structures('unoptimized', energies=False)

        except ZeroCandidatesError:
            child_embedder.structures = []

        child_embedder.log(f'\n--> Child process terminated ({time_to_string(time.perf_counter() - child_embedder.t_start_run, verbose=True)})')

        os.chdir(start_dir)
        if not options.debug:
            rmtree(os.path.join(os.getcwd(), foldername))

    return child_embedder.structures, child_embedder.constrained_indices