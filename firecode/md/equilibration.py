from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Sequence

import numpy as np
from ase import Atoms, units
from ase.constraints import FixAtoms
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import neighbor_list
from networkx import Graph, connected_components
from prism_pruner.algebra import normalize
from prism_pruner.utils import time_to_string

from firecode.ase_manipulations import (
    ASEConstraint,
    get_ase_constraints_from_embedder,
    optimizer_dict,
    set_charge_and_mult_on_ase_atoms,
)
from firecode.dispatcher import Opt_func_dispatcher
from firecode.errors import FatalError
from firecode.solvents import solvent_data
from firecode.utils import NewFolderContext, read_xyz

if TYPE_CHECKING:
    from firecode.embedder import Embedder
    from firecode.typing_ import Array1D_float, Array1D_str, Array2D_float


class MultipleSpring(ASEConstraint):
    """ASE Custom Constraint Class
    Adds an harmonic force between multiple pairs of atoms.
    The equilibrium distance used is the current one in the
    ASE atoms, taking the MIC convention for molecules that
    might be positioned across a box boundary.
    Spring constant is high to achieve tight convergence,
    but maximum force is dampened to avoid ruining structures.
    """

    def __init__(
        self, bonds: Sequence[Sequence[int]], atoms: Atoms, k: float = 50.0, fmax: float = 10.0
    ) -> None:
        self.bonds = bonds
        self.eq_dists = [atoms.get_distance(i1, i2, mic=True) for (i1, i2) in bonds]  # type: ignore[no-untyped-call]
        self.k = k
        self.fmax = fmax

    def adjust_positions(self, atoms: Atoms, newpositions: Array1D_float) -> None:
        pass

    def adjust_forces(self, atoms: Atoms, forces: Array1D_float) -> None:
        for (i1, i2), d_eq in zip(self.bonds, self.eq_dists):
            # vector connecting atom1 to atom2
            direction = atoms.get_distances(i1, [i2], mic=True, vector=True)[0]  # type: ignore[no-untyped-call]

            # Positive if spring is overstretched.
            delta_x = np.linalg.norm(direction) - d_eq

            # gated linear force: force is clipped at fmax eV/A
            spring_force = np.clip(self.k * delta_x, -self.fmax, self.fmax)

            # applying harmonic force to each atom, directed toward the other one
            forces[i1] += normalize(direction) * spring_force
            forces[i2] -= normalize(direction) * spring_force

    def __repr__(self) -> str:
        return f"MultipleSpring - n_bonds:{len(self.bonds)} - k:{self.k}, fmax:{self.fmax}"


def _get_solvent_constraint(
    atoms: Atoms,
    solute_atoms_ids: list[int],
) -> ASEConstraint:
    """Return a FixInternals object preventing solvent scrambling."""
    # 1. Get all bonds based on a simple distance cutoff
    i, j = neighbor_list("ij", atoms, cutoff=1.85, self_interaction=False)  # type: ignore[no-untyped-call]

    # 2. constrain the ones belonging to solvent molecules
    solvent_bonds = [
        (int(i1), int(i2))
        for (i1, i2) in zip(i, j)
        if i1 not in solute_atoms_ids and i2 not in solute_atoms_ids
    ]

    return MultipleSpring(bonds=solvent_bonds, atoms=atoms)


def run_md_equilibration(
    symbols: Array1D_str,
    positions: Array2D_float,
    embedder: Embedder,
    constraints: list[ASEConstraint] | None = None,
    title: str = "temp",
    time_step_fs: float = 0.5,
    nvt_time_ps: float = 200.0,
    npt_time_ps: float = 300.0,
) -> str:
    """Runs NVT and NPT equilibration on a solvated system."""
    with NewFolderContext(title + "_md_equilibration", delete_after=False):
        # 1. Initialize Atoms object
        atoms = Atoms(symbols=symbols, positions=positions)
        dispatcher = Opt_func_dispatcher(calculator=embedder.options.calculator)
        ase_calc = dispatcher.get_ase_calc(
            embedder.options.theory_level, solvent=None, force_reload=True, logfunction=embedder.log
        )
        atoms.calc = ase_calc
        optimizer = optimizer_dict[dispatcher.get_optimizer_str()][0]

        set_charge_and_mult_on_ase_atoms(
            atoms, charge=embedder.options.charge, mult=embedder.options.mult
        )

        # set up PBC and cell
        box_size = embedder.options.md_data["box_size"]
        atoms.set_cell([box_size, box_size, box_size])  # type: ignore[no-untyped-call]
        atoms.set_pbc(True)  # type: ignore[no-untyped-call]
        atoms.center()  # type: ignore[no-untyped-call]
        atoms.wrap(pbc=True)  # type: ignore[no-untyped-call]

        # Fix the solute in place and optimize the solvent alone first
        solute_atoms_ids = embedder.options.md_data["solute_atoms_ids"]
        solute_constraint = FixAtoms(solute_atoms_ids)  # type: ignore[no-untyped-call]
        solvent_constraint = _get_solvent_constraint(
            atoms,
            solute_atoms_ids=embedder.options.md_data["solute_atoms_ids"],
        )
        atoms.set_constraint(
            [
                solute_constraint,
                solvent_constraint,
            ]
        )  # type: ignore[no-untyped-call]

        # do a targeted relaxation of the solvent only first
        embedder.log(
            f"--> Relaxing the solvent first ({len(symbols) - len(solute_atoms_ids)}/{len(symbols)}) atoms"
        )
        t_start = perf_counter()
        with optimizer(atoms, maxstep=0.2, trajectory=title + "_solvent_preopt_0K.traj") as opt:  # type: ignore[operator]
            opt.run(fmax=0.2, steps=500)

        if not check_solvent_scramble(
            atoms,
            solvent_n_atoms=embedder.options.md_data["solvent_n_atoms"],
            solute_atoms_ids=embedder.options.md_data["solute_atoms_ids"],
        ):
            raise FatalError("MD equilibration of the solvent molecules of the run scrambled.")

        embedder.log(
            f"Solvent optimization completed in {opt.nsteps} iterations "
            f"({time_to_string(perf_counter() - t_start)})"
        )
        opt.nsteps = 0

        # now set the proper constraints from the embedder
        atoms.set_constraint(constraints or [])  # type: ignore[no-untyped-call]

        # do a global relaxation of the global geometry before heating it up
        embedder.log(f"--> Relaxing the solvent first {len(solute_atoms_ids)}/{len(symbols)} atoms")
        t_start = perf_counter()
        with optimizer(atoms, maxstep=0.2, trajectory=title + "_global_preopt_0K.traj") as opt:  # type: ignore[operator]
            opt.run(fmax=0.2, steps=500)

        embedder.log(
            f"Global 0 K optimization converged in {opt.nsteps} iterations "
            f"({time_to_string(perf_counter() - t_start)})"
        )
        opt.nsteps = 0

        # Set simulation values
        nvt_steps = 1000 * nvt_time_ps / time_step_fs
        npt_steps = 1000 * npt_time_ps / time_step_fs

        compressibility = solvent_data[embedder.options.solvent].get("compressibility")
        if compressibility is None:
            compressibility = 10e-5
            embedder.log(
                f"--> ATTENTION: Compressibility for {embedder.options.solvent} not defined in solvents.py: using 10e-5 bar(^-1)"
            )

        # Set initial velocities to the target temperature
        MaxwellBoltzmannDistribution(atoms, temperature_K=embedder.options.T)

        # 2. NVT Equilibration (Canonical Ensemble)
        # Using Langevin thermostat
        embedder.log(
            f"--> Running NVT Equilibration ({time_step_fs:.1f} fs timestep, {nvt_steps:.1e} steps ({nvt_time_ps:.1} ps))"
        )
        dyn_nvt = Langevin(
            atoms, 2 * units.fs, temperature_K=embedder.options.T, friction=0.01 / units.fs
        )

        traj_nvt = Trajectory("nvt_equil.traj", "w", atoms)  # type: ignore[no-untyped-call]
        dyn_nvt.attach(traj_nvt.write, interval=50)  # type: ignore[no-untyped-call]
        dyn_nvt.run(nvt_steps)  # type: ignore[no-untyped-call]

        # 3. NPT Equilibration (Isothermal-Isobaric Ensemble)
        # Using Berendsen barostat
        embedder.log(
            f"--> Running NPT Equilibration ({time_step_fs:.1f} fs timestep, {npt_steps:.1e} steps ({npt_time_ps:.1} ps))"
        )
        dyn_npt = NPTBerendsen(
            atoms,
            timestep=2 * units.fs,
            temperature_K=embedder.options.T,
            taut=0.5 * 1000 * units.fs,  # Time constant for temperature coupling
            pressure_au=(embedder.options.P or 1.0) * 1.01325 * units.bar,  # 1 atm
            taup=1.0 * 1000 * units.fs,  # Time constant for pressure coupling
            compressibility_au=compressibility / units.bar,
        )

        traj_npt = Trajectory("npt_equil.traj", "w", atoms)  # type: ignore[no-untyped-call]
        dyn_npt.attach(traj_npt.write, interval=50)  # type: ignore[no-untyped-call]
        dyn_npt.run(npt_steps)  # type: ignore[no-untyped-call]

        embedder.log("Equilibration Complete.")

        outname = f"{title}_equilibrated.xyz"
        atoms.write(outname)  # type: ignore[no-untyped-call]

    return outname


def equilibrate_operator(filename: str, embedder: Embedder) -> str:
    """Run a NPT -> NVT MD equilibration of a structure."""
    mol = read_xyz(filename)

    if len(mol.coords) > 1:
        raise NotImplementedError

    # get all constraints belonging to this molecule from the embedder
    constraints = get_ase_constraints_from_embedder(filename, embedder)

    assert embedder.options.md_data != {}, (
        "Solvate the molecule with the `packmol>` operator before running `equilibrate>`."
    )

    outname = run_md_equilibration(
        mol.atoms,
        mol.coords[0],
        embedder=embedder,
        constraints=constraints,
        title=mol.basename,
    )

    return outname


def check_solvent_scramble(
    atoms: Atoms, solvent_n_atoms: int, solute_atoms_ids: Sequence[int]
) -> bool:
    """Asserts that solvent molecules did not scramble during optimization."""
    # 1. Get all bonds based on a simple distance cutoff
    i, j = neighbor_list("ij", atoms, cutoff=1.85, self_interaction=False)  # type: ignore[no-untyped-call]
    solvent_bonds = [
        (i1, i2)
        for (i1, i2) in zip(i, j)
        if i1 not in solute_atoms_ids and i2 not in solute_atoms_ids
    ]

    # 2. Build the Graph
    G = Graph()
    G.add_nodes_from([i for i in range(len(atoms)) if i not in solute_atoms_ids])
    G.add_edges_from(solvent_bonds)
    components = list(connected_components(G))

    # 3. Analyze Connected Components
    for component in components:
        if len(component) != solvent_n_atoms:
            return False

    return True
