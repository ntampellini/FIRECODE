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

import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Self

import numpy as np
from prism_pruner.graph_manipulations import graphize
from prism_pruner.pruner import prune_by_moment_of_inertia, prune_by_rmsd, prune_by_rmsd_rot_corr
from prism_pruner.utils import time_to_string

from firecode.pt import pt
from firecode.typing_ import Array1D_bool, Array1D_float, Array1D_int, Array1D_str, Array3D_float
from firecode.units import EH_TO_KCAL


@dataclass
class Ensemble:
    """Dataclass representing a conformer ensemble, with pruning and energy sorting."""

    atoms: Array1D_str
    coords: Array3D_float
    filename: str = ""
    atomnos: Array1D_int = field(default_factory=lambda: np.array([], dtype=int))
    energies: Array1D_float = field(default_factory=lambda: np.array([], dtype=float))
    logfunction: Callable[[str], None] | None = print

    @classmethod
    def from_xyz(cls, file: Path | str, read_energies: bool = False) -> Self:
        """Generate ensemble from a multiple conformer xyz file."""
        coords = []
        atoms = []
        energies = []
        with Path(file).open() as f:
            for num in f:
                try:
                    if not num.strip():
                        continue

                    if read_energies:
                        energy = next(re.finditer(r"-*\d+\.\d+", next(f))).group()
                        energies.append(float(energy))
                    else:
                        _comment = next(f)

                    conf_atoms = []
                    conf_coords = []
                    for _ in range(int(num)):
                        atom, *xyz = next(f).split()
                        conf_atoms.append(atom)
                        conf_coords.append([float(x) for x in xyz])

                    atoms.append(conf_atoms)
                    coords.append(conf_coords)

                except StopIteration:
                    pass

        atomnos = np.array([pt.number(letter) for letter in atoms[0]])

        return cls(
            atoms=np.array(atoms[0]),
            coords=np.array(coords),
            filename=str(file),
            atomnos=atomnos,
            energies=np.array(energies),
        )

    def read_energies(
        self,
        verbose: bool = True,
    ) -> bool:
        """Read energies from a .xyz file. Sets self.energies
        (in kcal/mol), returning a success bool.
        """
        from firecode.utils import read_xyz_energies

        energies = read_xyz_energies(self.filename, verbose=verbose, logfunction=self.logfunction)

        if energies is None:
            return False

        self.energies = np.array(energies) * EH_TO_KCAL
        return True

    def energy_pruning(
        self,
        kcal_thr: float = 10.0,
        verbose: bool = True,
    ) -> None:
        """Remove high energy structures above kcal_thr."""
        energy_thr = self.dynamic_energy_thr(kcal_thr, verbose=verbose)
        mask = self.rel_energies < energy_thr

        self.apply_mask(("coords", "energies"), mask)

        if False in mask and verbose and self.logfunction is not None:
            self.logfunction(
                f"Discarded {len([b for b in mask if not b])} candidates for energy ({np.count_nonzero(mask)} left, "
                + f"{round(100 * np.count_nonzero(mask) / len(mask), 1)}% kept, threshold {energy_thr:.1f} kcal/mol)"
            )

    def dynamic_energy_thr(
        self,
        kcal_thr: float = 10.0,
        keep_min: float = 0.1,
        verbose: bool = True,
    ) -> float:
        """Returns an energy threshold that is dynamically adjusted
        based on the distribution of energies around the lowest,
        so that at least (keep_min * 100)% of the structures are retained.

        keep_min: float, minimum percentage of structures to keep
        verbose: bool, prints comments with logfunction.

        """
        active = len(self.coords)
        keep = np.count_nonzero(self.rel_energies < kcal_thr)

        # if the standard threshold keeps enough structures, use that
        if keep / active > keep_min:
            return kcal_thr

        # if not, iterate on the relative energy values as
        # thresholds until we keep enough structures
        for thr in (energy for energy in self.rel_energies if energy > kcal_thr):
            keep = np.count_nonzero(self.rel_energies < thr)

            if keep / active > keep_min:
                if verbose and self.logfunction is not None:
                    self.logfunction(
                        f"--> Dynamically adjusted energy threshold to {thr:.1f} kcal/mol to retain at least {(keep / active) * 100:.2f}% of structures."
                    )
                return float(thr)

        # we won't actually get here, but
        # this keeps the type checker happy
        return kcal_thr

    @property
    def rel_energies(self) -> Array1D_float:
        return self.energies - np.min(self.energies)

    def apply_mask(self, attributes: Iterable[str], mask: Array1D_bool) -> None:
        """Applies in-place masking of Ensemble attributes."""
        for attr in attributes:
            if hasattr(self, attr):
                try:
                    new_attr = getattr(self, attr)[mask]
                    setattr(self, attr, new_attr)
                except IndexError:
                    pass

    def similarity_pruning(
        self,
        moi: bool = True,
        rmsd: bool = True,
        rmsd_rot_corr: bool = False,
        verbose: bool = True,
    ) -> None:
        """Removes structures that are too similar to each other."""
        if verbose and self.logfunction is not None:
            self.logfunction("--> Similarity Processing")

        before = len(self.coords)

        # atrtibutes that the pruning mask
        # should propagate to on top of self.coords
        attr = ("energies",)

        use_energies_in_pruning = len(self.energies) == len(self.coords)
        max_dE = 1.0

        if moi:
            ### Prune by moment of inertia

            before3 = len(self.coords)

            t_start = perf_counter()
            self.coords, mask = prune_by_moment_of_inertia(
                self.coords,
                self.atoms,
                energies=self.energies if use_energies_in_pruning else None,
                max_dE=max_dE,
            )

            self.apply_mask(attr, mask)

            if before3 > len(self.coords) and verbose and self.logfunction is not None:
                self.logfunction(
                    f"Discarded {len([b for b in mask if not b])} candidates for MOI similarity ({len([b for b in mask if b])} left, {time_to_string(perf_counter() - t_start)})"
                )

        if rmsd:
            before1 = len(self.coords)

            t_start = perf_counter()

            self.coords, mask = prune_by_rmsd(
                self.coords,
                self.atoms,
                energies=self.energies if use_energies_in_pruning else None,
                max_dE=max_dE,
            )

            self.apply_mask(attr, mask)

            if before1 > len(self.coords) and verbose and self.logfunction is not None:
                self.logfunction(
                    f"Discarded {len([b for b in mask if not b])} candidates for RMSD similarity ({len([b for b in mask if b])} left, {time_to_string(perf_counter() - t_start)})"
                )

            ### Second step: again but symmetry-corrected (unless we have too many structures)

            if rmsd_rot_corr:
                if len(self.coords) <= 1e3:
                    before2 = len(self.coords)

                    graph = graphize(self.atoms, self.coords[0])

                    t_start = perf_counter()
                    self.coords, mask = prune_by_rmsd_rot_corr(
                        self.coords,
                        self.atoms,
                        graph,
                        energies=self.energies if use_energies_in_pruning else None,
                        max_dE=max_dE,
                        logfunction=self.logfunction,
                    )

                    self.apply_mask(attr, mask)

                    if before2 > len(self.coords) and verbose and self.logfunction is not None:
                        self.logfunction(
                            f"Discarded {len([b for b in mask if not b])} candidates for symmetry-corrected RMSD similarity ({len([b for b in mask if b])} left, {time_to_string(perf_counter() - t_start)})"
                        )

                elif verbose and self.logfunction is not None:
                    self.logfunction("Skipped rotationally-corrected RMSD pruning (>1k structures)")

        if verbose and len(self.coords) == before and self.logfunction is not None:
            self.logfunction(f"All structures passed the similarity check.{' ' * 15}")

        if verbose and self.logfunction is not None:
            self.logfunction("")

    def sort_by_energy(self) -> None:
        """Sort the structures by ascending self.energies."""
        sorted_indices = np.argsort(self.energies)
        self.energies = self.energies[sorted_indices]
        self.coords = self.coords[sorted_indices]
