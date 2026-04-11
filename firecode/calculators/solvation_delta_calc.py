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

import os
from typing import Any, cast

from ase import Atoms
from ase.calculators.calculator import Calculator as ASECalculator
from ase.calculators.calculator import all_changes
from tblite.ase import TBLite
from tblite.exceptions import TBLiteValueError

from firecode.solvents import epsilon_dict, to_xtb_solvents
from firecode.typing_ import Array1D_str, Array2D_float
from firecode.units import EV_TO_KCAL
from firecode.utils import HiddenPrints


class SolvationDeltaCalculator:
    """Unified TBLITE solvation calculator, for single-point calculations only.

    returns [E_xTB(solvent) - E_xTB(gas)]

    Solvation is provided via TBLITE (CPCM or ALPB models).
    """

    def __init__(
        self,
        solvent: str = "ch2cl2",
        method: str = "GFN2-xTB",
        solv_model: str | None = None,
        maxiter: int = 250,
        accuracy: float = 1.0,
        **kwargs: Any,
    ) -> None:

        # set up gas phase calculator
        self.gas_calc = TBLite(method=method, max_iterations=maxiter, accuracy=accuracy)

        solv_model = solv_model or os.environ["FIRECODE_SOLV_METHOD_FOR_ML"]

        # set up solution phase calculator
        match solv_model.lower():
            case "alpb":
                try:
                    # translate if needed
                    xtb_solvent_name = to_xtb_solvents.get(solvent, solvent)

                    # add ALPB solvation via solvent name
                    self.solv_calc = TBLite(
                        method=method,
                        max_iterations=maxiter,
                        alpb_solvation=xtb_solvent_name,
                        accuracy=accuracy,
                    )

                except TBLiteValueError:
                    raise TBLiteValueError(
                        "TBLITE was not able to set up ALPB solvation correctly - "
                        f'"{solvent}" not recognized by TBLITE. A workaround might be to use '
                        "CPCM solvation, if the corresponding epsilon value is known in firecode.solvents."
                    )

            case "cpcm":
                try:
                    epsilon = epsilon_dict[solvent]

                    # add CPCM solvation via solvent name
                    self.solv_calc = TBLite(
                        method=method,
                        max_iterations=maxiter,
                        cpcm_solvation=epsilon,
                        accuracy=accuracy,
                    )

                except KeyError:
                    raise KeyError(
                        f'Solvent "{solvent}" is not present in epsilon_dict. (FIRECODE does not know its epsilon value).'
                    )

            case _:
                raise ValueError(
                    "solv_model or FIRECODE_SOLV_METHOD_FOR_ML (currently "
                    f'{solv_model}) must be either ("alpb" or "cpcm").'
                )

    def get_solvation_delta(
        self,
        atoms: Array1D_str,
        coords: Array2D_float,
    ) -> float:
        """Returns the solvation energy delta in kcal/mol."""
        for calc in [self.gas_calc, self.solv_calc]:
            calc.reset()
            calc.results = {}

        gas_atoms = Atoms(atoms, positions=coords)
        solv_atoms = Atoms(atoms, positions=coords)

        with HiddenPrints():
            self.gas_calc.atoms = gas_atoms
            e_gas = self.gas_calc.get_potential_energy()  # type: ignore[no-untyped-call]

            self.solv_calc.atoms = solv_atoms
            e_solv = self.solv_calc.get_potential_energy()  # type: ignore[no-untyped-call]

        return cast("float", (e_solv - e_gas) * EV_TO_KCAL)


class SolvatedMLCalculator(ASECalculator):
    """Unified solvation-corrected ML calculator, compatible with Sella.

    E = E_ML + [E_xTB(solvent) - E_xTB(gas)]
    F = F_ML + [F_xTB(solvent) - F_xTB(gas)]


    Solvation is provided via TBLITE (CPCM or ALPB models).
    Sub-calculator calculate() methods are called directly (not via
    get_potential_energy()), bypassing their individual caching layers
    and avoiding the stale-cache pathology with Sella's FD Hessian
    evaluations.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        ml_calc: ASECalculator,
        solvent: str = "ch2cl2",
        method: str = "GFN2-xTB",
        solv_model: str | None = None,
        maxiter: int = 250,
        accuracy: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[no-untyped-call]

        # save ML calculator
        self.ml_calc = ml_calc

        # set up gas phase calculator
        self.gas_calc = TBLite(method=method, max_iterations=maxiter, accuracy=accuracy)

        solv_model = solv_model or os.environ["FIRECODE_SOLV_METHOD_FOR_ML"]

        # set up solution phase calculator
        match solv_model.lower():
            case "alpb":
                try:
                    # translate if needed
                    xtb_solvent_name = to_xtb_solvents.get(solvent, solvent)

                    # add ALPB solvation via solvent name
                    self.solv_calc = TBLite(
                        method=method,
                        max_iterations=maxiter,
                        alpb_solvation=xtb_solvent_name,
                        accuracy=accuracy,
                    )

                except TBLiteValueError:
                    raise TBLiteValueError(
                        "TBLITE was not able to set up ALPB solvation correctly - "
                        f'"{solvent}" not recognized by TBLITE. A workaround might be to use '
                        "CPCM solvation, if the corresponding epsilon value is known in firecode.solvents."
                    )

            case "cpcm":
                try:
                    epsilon = epsilon_dict[solvent]

                    # add CPCM solvation via solvent name
                    self.solv_calc = TBLite(
                        method=method,
                        max_iterations=maxiter,
                        cpcm_solvation=epsilon,
                        accuracy=accuracy,
                    )

                except KeyError:
                    raise KeyError(
                        f'Solvent "{solvent}" is not present in epsilon_dict. (FIRECODE does not know its epsilon value).'
                    )

            case _:
                raise ValueError(
                    "solv_model or FIRECODE_SOLV_METHOD_FOR_ML (currently "
                    f'{solv_model}) must be either ("alpb" or "cpcm").'
                )

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list[str]:
        return all_changes

    def calculate(  # type: ignore[override]
        self,
        atoms: Atoms,
        properties: list[str] = ["energy", "forces"],
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)  # type: ignore[no-untyped-call]

        # Always compute both energy AND forces regardless of what
        # properties contains — prevents stale forces being served
        # during Sella's energy-only FD Hessian and rho evaluations
        _props = ["energy", "forces"]

        for calc in [self.ml_calc, self.gas_calc, self.solv_calc]:
            calc.reset()  # type: ignore[no-untyped-call]
            calc.results = {}

        # Call calculate() directly on each sub-calculator with all_changes,
        # bypassing their internal caches. This is the same pattern used by
        # ASE's own SumCalculator, but without the intermediate caching layer.
        self.ml_calc.calculate(atoms, _props, all_changes)  # type: ignore[no-untyped-call]
        e_ml = self.ml_calc.results["energy"]
        f_ml = self.ml_calc.results["forces"].copy()

        with HiddenPrints():
            self.gas_calc.calculate(atoms, _props, all_changes)
            e_gas = self.gas_calc.results["energy"]
            f_gas = self.gas_calc.results["forces"].copy()

            self.solv_calc.calculate(atoms, _props, all_changes)
            e_solv = self.solv_calc.results["energy"]
            f_solv = self.solv_calc.results["forces"].copy()

        self.results["energy"] = e_ml + (e_solv - e_gas)
        self.results["forces"] = f_ml + (f_solv - f_gas)
