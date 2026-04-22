import subprocess
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDKitMol

from firecode.context_managers import NewFolderContext
from firecode.pt import pt
from firecode.typing_ import Array1D_float, Array1D_str, Array2D_float
from firecode.units import AVOGADRO_NA
from firecode.utils import read_xyz, write_xyz


def solvate_molecule(
    solute_atoms: Array1D_str,
    solute_coords: Array2D_float,
    solvent_name: str,
    solvent_data: dict[str, dict[str, Any]],
    title: str = "temp",
    solvation_shells: float = 2.5,
    num_solvent_molecules: int | None = None,
    tolerance: float = 2.5,
    seed: int = 42,
    logfunction: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Create and run a PACKMOL input file to solvate a molecule.

    Args:
        solute_coords: numpy array of shape (N, 4) with columns [x, y, z, atom_type]
                           or a path to an XYZ file
        solvent_name: string key to look up in solvent_data
        solvent_data: dictionary with structure:
                     {
                         'solvent_name': {
                             'smiles': 'SMILES_string',
                             'density': float,  # g/cm³
                             'MW': float  # g/mol
                         },
                         ...
                     }
        title: basename of the solute
        solvation_shells: number of solvation shells to maintain (default: 2.5)
        num_solvent_molecules: number of solvent molecules to use (overrides solvation_shells)
        tolerance: PACKMOL tolerance parameter (default: 2.5 Å)
        seed: random seed for PACKMOL (default: 42)

    Returns:
        dict with keys:
            'output_xyz': path to output XYZ file
            'input_file': path to PACKMOL input file
            'solvent_xyz': path to solvent XYZ file
            'solute_xyz': path to solute XYZ file
            'num_solvent': number of solvent molecules used
            'box_size': size of cubic box in Angstroms
            'box_min': minimum coordinates of box
            'box_max': maximum coordinates of box
            'target_density': target density used

    """
    # Validate solvent exists in dictionary
    if solvent_name not in solvent_data:
        raise ValueError(f"Solvent '{solvent_name}' not found in solvent_data")

    solvent_info = solvent_data[solvent_name]
    smiles = solvent_info["smiles"]
    solvent_density = solvent_info["density"]
    solvent_mass = solvent_info["MW"]

    # define outnames
    solute_xyz = f"{title}.xyz"
    solvent_xyz = f"{solvent_name}.xyz"

    solute_mass = sum([pt.mass(s) for s in solute_atoms])
    work_in = Path(title + "_packmol")

    logfunction(f"--> PACKMOL interface: solvating {title}")

    with NewFolderContext(str(work_in), delete_after=False):
        # Step 1: Create solute XYZ file
        with open(f"{title}.xyz", "w") as f:
            write_xyz(solute_atoms, solute_coords, f)

        solute_center = np.mean(solute_coords, axis=0)

        # Step 2: Generate solvent molecule with RDKit
        _generate_solvent_xyz(smiles, f"{solvent_name}.xyz")

        # Step 3: Calculate solvent diameter from structure
        solvent_diameter = _calculate_shell_thickness(solvent_xyz, logfunction=logfunction)

        # Step 4: Calculate number of solvent molecules if not specified
        if num_solvent_molecules is None:
            num_solvent_molecules = _calculate_num_solvents_from_structure(
                solute_coords=solute_coords,
                solvent_mass=solvent_mass,
                target_density=solvent_density,
                solute_mass=solute_mass,
                solvent_diameter=solvent_diameter,
                solvation_shells=solvation_shells,
                logfunction=logfunction,
            )

        box_size = _calculate_box_size(
            num_solute=1,
            num_solvent=num_solvent_molecules,
            solute_mass=solute_mass if solute_mass else 100,
            solvent_mass=solvent_mass,
            target_density=solvent_density,
        )

        # Center the box around the solute
        box_min = solute_center - box_size / 2
        box_max = solute_center + box_size / 2

        # Step 5: Create PACKMOL input file
        input_file = "packmol_input.inp"
        output_xyz = f"{title}_solvated.xyz"

        _create_packmol_input(
            solute_xyz=solute_xyz,
            solvent_xyz=solvent_xyz,
            num_solvents=num_solvent_molecules,
            box_min=box_min,
            box_max=box_max,
            output_xyz=output_xyz,
            input_file=input_file,
            tolerance=tolerance,
            seed=seed,
            logfunction=logfunction,
        )

        # Step 5: Run PACKMOL
        logfunction(f"Running PACKMOL with {num_solvent_molecules} {solvent_name} molecules...")
        _run_packmol(input_file, logfunction=logfunction)

        solvent_n_atoms = len(read_xyz(solvent_xyz).atoms)
        total_n_atoms = len(solute_atoms) + solvent_n_atoms * num_solvent_molecules

    logfunction(
        f"Final numbuer of atoms: {len(solute_atoms)} ({title}) + {solvent_n_atoms} "
        f"* {num_solvent_molecules} ({solvent_name}) = {total_n_atoms}\n"
    )

    return {
        "output_xyz": str(work_in / output_xyz),
        "input_file": str(work_in / input_file),
        "solvent_xyz": str(work_in / solvent_xyz),
        "solute_xyz": str(work_in / solute_xyz),
        "num_solvent": num_solvent_molecules,
        "box_size": box_size,
        "box_min": box_min,
        "box_max": box_max,
        "target_density": solvent_density,
        "solvent_n_atoms": solvent_n_atoms,
        "solute_atoms_ids": list(range(len(solute_atoms))),
    }


def _generate_solvent_xyz(smiles: str, output_path: str) -> None:
    """Generate a 3D structure of solvent from SMILES and save as XYZ."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)  # type: ignore[attr-defined]
    AllChem.MMFFOptimizeMolecule(mol)  # type: ignore

    # Write to XYZ
    _mol_to_xyz(mol, output_path)


def _mol_to_xyz(mol: RDKitMol, output_path: str = "temp.xyz") -> None:
    """Convert RDKit molecule to XYZ format."""
    conf = mol.GetConformer()
    atoms = mol.GetAtoms()  # type: ignore[no-untyped-call]

    with open(output_path, "w") as f:
        # Write number of atoms
        f.write(f"{mol.GetNumAtoms()}\n")
        # Write comment line
        f.write("Generated by RDKit\n")
        # Write atoms
        for atom in atoms:
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            pos = conf.GetAtomPosition(idx)
            f.write(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")


def _calculate_box_size(
    num_solute: int,
    num_solvent: int,
    solute_mass: float,
    solvent_mass: float,
    target_density: float,
) -> float:
    """Calculate cubic box size based on desired density."""
    # Total mass in grams
    total_mass_g = (num_solute * solute_mass + num_solvent * solvent_mass) / AVOGADRO_NA

    # Volume in cm³
    volume_cm3 = total_mass_g / target_density

    # Convert to Angstroms³
    volume_angstrom3 = volume_cm3 * 1e24

    # Cube root for cubic box
    box_size = volume_angstrom3 ** (1 / 3)

    return cast("float", box_size)


def _calculate_shell_thickness(
    solvent_xyz: str, logfunction: Callable[[str], None] = print
) -> float:
    """Calculate the effective diameter of a solvent molecule from its structure.

    Args:
        solvent_xyz: path to solvent XYZ file

    Returns:
        float: effective diameter in Angstroms

    """
    solvent_coords = read_xyz(solvent_xyz).coords[0]
    center = np.mean(solvent_coords, axis=0)
    distances = np.linalg.norm(solvent_coords - center, axis=0)
    radius = float(np.max(distances))
    diameter = 2 * radius

    logfunction(f"Solvent diameter ({solvent_xyz}): {diameter:.2f} Å")
    return diameter


def _calculate_num_solvents_from_structure(
    solute_coords: Array2D_float,
    solvent_mass: float,
    target_density: float,
    solute_mass: float,
    solvent_diameter: float,
    solvation_shells: float = 2.5,
    logfunction: Callable[[str], None] = print,
) -> int:
    """Calculate the number of solvent molecules needed based on solute structure.
    Ensures at least N solvation shells around the solute.

    Args:
        solute_coords: numpy array of solute atomic coordinates
        solvent_mass: molecular weight of solvent in g/mol
        target_density: target system density in g/cm³
        solute_mass: molecular weight of solute in g/mol
        solvent_diameter: diameter of solvent molecule in Angstroms
        solvation_shells: number of solvation shells to maintain (default: 2)

    Returns:
        int: number of solvent molecules needed

    """
    # Get solute radius (distance from center to farthest atom)
    center = np.mean(solute_coords, axis=0)
    distances = np.linalg.norm(solute_coords - center, axis=1)
    solute_radius = np.max(distances)

    # Calculate outer radius including solvation shells
    outer_radius = solute_radius + (solvation_shells * solvent_diameter)

    # Volume of solvation sphere
    solvation_volume_angstrom3 = (4 / 3) * np.pi * (outer_radius**3)
    projected_box_side = solvation_volume_angstrom3 ** (1 / 3)

    # Convert to cm³
    solvation_volume_cm3 = solvation_volume_angstrom3 / 1e24

    # Calculate mass of solvent needed to fill solvation sphere at target density
    # Mass = density * volume - solute_mass
    total_mass_needed_g = target_density * solvation_volume_cm3
    solute_mass_g = solute_mass / AVOGADRO_NA
    solvent_mass_needed_g = max(0, total_mass_needed_g - solute_mass_g)

    # Convert to number of molecules
    num_solvents = int(solvent_mass_needed_g * AVOGADRO_NA / solvent_mass)

    logfunction(f"Solute radius: {solute_radius:.2f} Å")
    logfunction(
        f"Outer solvation radius ({solvation_shells} shells @ {solvent_diameter:.2f} Å/shell): {outer_radius:.2f} Å"
    )
    logfunction(
        f"Solvation volume: {solvation_volume_angstrom3:.4f} Å³ (cubic box of {projected_box_side:.2f} Å)"
    )
    logfunction(f"Calculated solvent molecules: {num_solvents}")

    return max(1, num_solvents)


def _create_packmol_input(
    solute_xyz: str,
    solvent_xyz: str,
    num_solvents: int,
    box_min: Array1D_float,
    box_max: Array1D_float,
    output_xyz: str,
    input_file: str,
    tolerance: float = 2.5,
    seed: int = 42,
    logfunction: Callable[[str], None] = print,
) -> None:
    """Create the PACKMOL input file."""
    content = f"""filetype xyz
tolerance {tolerance}
seed {seed}
output {output_xyz}

structure {solute_xyz}
  number 1
  fixed 0 0 0 0 0 0
end structure

structure {solvent_xyz}
  number {num_solvents}
  inside box {box_min[0]:.2f} {box_min[1]:.2f} {box_min[2]:.2f} {box_max[0]:.2f} {box_max[1]:.2f} {box_max[2]:.2f}
end structure
"""

    with open(input_file, "w") as f:
        f.write(content)

    logfunction(f"Created PACKMOL input file: {input_file}")


def _run_packmol(input_name: str, logfunction: Callable[[str], None] = print) -> None:
    """Execute PACKMOL with the given input file."""
    try:
        with open("packmol.out", "w") as f:
            result = subprocess.run(
                f"packmol < {input_name}",
                shell=True,
                text=True,
                check=True,
                stdout=f,
            )
        logfunction("PACKMOL execution completed successfully")
        return

    except subprocess.CalledProcessError as e:
        logfunction(f"PACKMOL error: {e.stderr}")
        raise

    except FileNotFoundError:
        raise RuntimeError(
            "PACKMOL not found. Please ensure PACKMOL is installed and in your PATH. Install it with:\n"
            ">>> uv pip install packmol"
        )
