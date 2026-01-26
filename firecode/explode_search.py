
import numpy as np
from prism_pruner.graph_manipulations import d_min_bond, graphize
from prism_pruner.utils import time_to_string

from firecode.ase_manipulations import (NewBondPreventer, Spring,
                                        ase_popt_lite, ase_tblite_opt)
from firecode.utils import (loadbar, molecule_check, read_xyz, timing_wrapper,
                            write_xyz)


def explode_search(filename, embedder, n=10, iterations=3):
    '''
    Perform a conformational search with the "explode" method.
    Will generate and optimize n exploded structures.
    
    '''
    return explode_search_serial(filename, embedder=embedder, n=n, iterations=iterations)


def _get_scrambled_coords(coords, n):
    '''
    Returns an n-structure array of scrambled coordinates.
    
    '''

    # center structure at the origin
    coords -= np.mean(coords, axis=0)

    # initialize output container
    output = np.zeros((n, coords.shape[0], coords.shape[1]))

    # define a heat range, which will be used
    # to compute how much to move atoms
    heats = np.linspace(0, 1, num=n)
    heats[0:int(n/2)] = 0.5

    # compute maximum displacement
    max_disp = 2 * np.max(np.linalg.norm(coords, axis=1))

    # radial expansion of every atom relative to center
    for i, heat in enumerate(heats):
        weights = np.random.rand(coords.shape[0]) * heat
        for v, vec in enumerate(coords):
            output[i,v] = vec * np.min((1+weights[v], max_disp))

    return output, heats

### DEBUG

def explode_search_serial(filename, embedder, n=10, iterations=3):
    '''
    Perform a conformational search with the "explode" method.
    Will generate and optimize n exploded structures.
    
    '''

    if embedder.options.calculator not in ("TBLITE", "XTB", "AIMNET2"):
        raise NotImplementedError('This routine is only implemented via the XTB, TBLITE and AIMNET2 calculators.')
    
    if len(embedder.objects) > 1 or len(embedder.objects[0].coords) > 1:
        raise NotImplementedError('This routine can only be run on a single input conformer.')

    # read input structure
    mol = read_xyz(filename)
    graph = graphize(mol.atoms, mol.coords[0])
    bonds = [bond for bond in graph.edges if bond[0] != bond[1]]

    # create HalfSpring Constraints objects
    bond_constraints = []
    for bond in bonds:
        i1, i2 = bond
        e1, e2 = mol.atoms[i1], mol.atoms[i2]
        d_eq = d_min_bond(e1, e2, factor=1.0)
        # d_max = 1.1 * d_eq

        hs = Spring(i1, i2, d_eq=d_eq, k=50)
        bond_constraints.append(hs)

    nbp = NewBondPreventer(mol.atoms, mol.coords[0], graph.edges)

    # set the number of cores per job
    if embedder.options.theory_level == 'GFN-FF':
        procs = 1
    else:
        procs = min(4, embedder.avail_cpus)

    # set initial starting point
    base_coords = mol.coords[0]

    for iteration in range(iterations):

        # compute scrambled coordinates
        scrambledcoords, sc_info = _get_scrambled_coords(base_coords, n=n)

        # save them if debugging
        # if embedder.options.debug:
        if True:
            with open(f'firecode_debug_explode_search_{iteration+1}.xyz', 'w') as f:
                write_xyz(mol.atoms, base_coords, f, title='original')
                for s, h in zip(scrambledcoords, sc_info):
                    write_xyz(mol.atoms, s, f, title=f'heat = {h:2f}')

        cum_time = 0
        output_structures, output_energies = [], []

        ase_calc = embedder.dispatcher.get_ase_calc(embedder.options.theory_level, embedder.options.solvent)
        opt_func = ase_popt_lite if embedder.options.calculator == 'XTB' else ase_tblite_opt
        
        max_workers = min(int(np.floor(embedder.avail_cpus/procs)), n)


        embedder.log((f'--> Explode Search round {iteration+1}/{iterations}: running {len(scrambledcoords)} optimizations ' + 
                    f'({embedder.options.theory_level}{f"/{embedder.options.solvent}" if embedder.options.solvent is not None else ""} ' + 
                    f'level via {embedder.options.calculator}, {max_workers} thread{"s" if max_workers>1 else ""})'))

        for i, (sc, sci) in enumerate(zip(scrambledcoords, sc_info)):
                            
            loadbar(i, len(scrambledcoords), prefix=f'Optimizing structure {i+1}/{len(scrambledcoords)} ')

            ((
                new_structure,
                new_energy,
                exit_status
            ),
            # from optimization function
                
            (
                sci,
            ),
            # from payload
            
                t_struct
            # from timing_wrapper

            ) = timing_wrapper(
                                        opt_func,

                                        mol.atoms,
                                        sc,
                                        ase_calc=ase_calc,
                                        method=embedder.options.theory_level,

                                        ase_constraints=bond_constraints,

                                        charge=embedder.options.charge,
                                        mult=embedder.options.mult,
                                        solvent=embedder.options.solvent,
                                        procs=procs,
                                        maxiter=500,
                                        conv_thr='tight',
                                        traj=f'explode_search_{i}_heat_{sci:.3f}.traj',
                                        logfunction=None,
                                        title=f'explode_search_{i}_heat_{sci:.3f}',

                                        debug=embedder.options.debug,
                                        dummy_first=True,
                                        new_bond_preventer=nbp,

                                        payload=(
                                                sci,
                                                )
                                    )

            # assert that the structure did not scramble during optimization
            if exit_status:
                
                exit_status = molecule_check(
                                                mol.atoms,
                                                mol.coords[0],
                                                new_structure,
                                                max_newbonds=embedder.options.max_newbonds,
                                            )
                    
                cum_time += t_struct

            if exit_status and (new_energy is not None):
                output_structures.append(new_structure)
                output_energies.append(new_energy)

            if embedder.options.debug:
                exit_status = 'REFINED  ' if exit_status else 'SCRAMBLED'
                embedder.debuglog(f'DEBUG: explode_search  - ExplodedStruct_{i+1} - {exit_status} {time_to_string(t_struct, digits=3)}')
            
        output_structures, output_energies = zip(*sorted(zip(output_structures, output_energies), key=lambda x: x[1]))

        # log results for the iteration
        embedder.log(f'--> ScrambleSearch completed - generated {len(output_structures)} structures out of {n} attempts ({time_to_string(cum_time)})')
        
        # sort structures energetically
        embedder.structures = np.array(output_structures)
        embedder.energies = np.array(output_energies)

        # remove high energy structures
        embedder.energy_pruning(verbose=False)

        # remove duplicates
        embedder.similarity_refining(verbose=False)

        # reset initial structure
        base_coords = embedder.structures[0]

    return embedder.write_structures('exploded_optimized', energies=True)


if __name__ == '__main__':
    mol = read_xyz('/mnt/c/Users/Nick/Downloads/catH+_prod.xyz')
    scrambled, heats = _get_scrambled_coords(mol.coords[0], n=20)

    with open('exploded_structs.xyz', 'w') as f:
        write_xyz(mol.atoms, mol.coords[0], f, title='original')
        for s, h in zip(scrambled, heats):
            write_xyz(mol.atoms, s, f, title=f'heat = {h:2f}')




    