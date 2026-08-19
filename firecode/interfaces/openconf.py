"""FIRECODE interface to Rowan's openconf."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from openconf import ConformerConfig, generate_conformers_from_pose
from prism_pruner.utils import align_structures, flatten, time_to_string
from rdkit import Chem
from rdkit.Chem import MolFromXYZFile, SanitizeMol
from rdkit.Chem.rdDetermineBonds import DetermineBonds

from firecode.ensemble import Ensemble
from firecode.errors import ZeroCandidatesError
from firecode.utils import compenetration_check

if TYPE_CHECKING:
    from firecode.embedder import Embedder


def openconf_operator(filename: str, embedder: Embedder) -> str:
    """Performs a (constrained) conformational search with openconf.

    See https://github.com/rowansci/openconf
    """
    t_start = perf_counter()

    mol = MolFromXYZFile(filename)
    DetermineBonds(mol, charge=embedder.mols[filename].charge)
    SanitizeMol(mol)

    if embedder.options.debug:
        embedder.debuglog(f"DEBUG: openconf/RDKit SMILES:{Chem.MolToSmiles(mol)}")

    # Fix atoms involved in constraints
    constraints = embedder._get_internal_constraints(filename)
    constrained_indices = frozenset(flatten(constraints, typefunc=int))

    # define options
    config = ConformerConfig(
        max_out=1000,
        n_steps=500,
        energy_window_kcal=20.0,
        do_final_refine=True,
        minimize_batch_size=8,
        torsion_multitry_attempts=8,
        parent_strategy="softmax",
        final_select="diverse",
        patience=0,  # avoid quitting early
    )

    embedder.log(
        f"--> Generating conformers with openconfs [{len(constraints)} constraints = {len(constrained_indices)} fixed atoms]"
    )

    # generate conformers and temporarily save output
    ensemble = generate_conformers_from_pose(
        mol, constrained_atoms=constrained_indices, config=config
    )
    n_generated = ensemble.n_conformers

    # convert openconf ensemble to FIRECODE ensemble
    tempname = filename[:-4] + "_openconfs_unpruned.xyz"
    ensemble.to_xyz(tempname)
    final_ensemble = Ensemble.from_xyz(tempname)

    embedder.log(
        f"Generated {n_generated} conformers in {time_to_string(perf_counter() - t_start)}"
    )

    # now remove clashing structures
    mask = np.ones(n_generated, dtype=bool)
    for s, structure in enumerate(final_ensemble.coords):
        mask[s] = compenetration_check(
            structure,
            graph=embedder.mols[filename].graph,
            max_clashes=len(constraints),
        )

    kept = np.count_nonzero(mask)
    embedder.log(f"Discarded {n_generated - kept} compenetrated structures ({kept} left)\n")

    if kept == 0:
        raise ZeroCandidatesError()

    # remove clashing and similar structures
    final_ensemble.coords = final_ensemble.coords[mask]
    final_ensemble.coords = align_structures(final_ensemble.coords)
    final_ensemble.similarity_pruning()

    outname = filename[:-4] + "_openconfs.xyz"
    final_ensemble.to_xyz(outname)

    return outname
