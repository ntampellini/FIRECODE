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

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, cast

from ase.calculators.calculator import Calculator as ASECalculator

from firecode.calculators.xtb import xtb_opt
from firecode.solvents import epsilon_dict, to_xtb_solvents
from firecode.typing_ import MaybeNone

if TYPE_CHECKING:
    from firecode.calculators.solvation_delta_calc import SolvationDeltaCalculator


class Opt_func_dispatcher:
    """Dispatcher for optimization functions."""

    solvent: str | None = None
    _solv_calc: SolvationDeltaCalculator | None = None

    def __init__(self, calculator: str) -> None:
        """Init method."""
        self.calculator = calculator
        self.ase_calc: ASECalculator | None = None

        from firecode.ase_manipulations import ase_popt

        optimization_functions_dict = {
            "ORCA": ase_popt,
            "XTB": xtb_opt,
            "TBLITE": ase_popt,
            "UMA": ase_popt,
            "AIMNET2": ase_popt,
        }

        assert os.environ["FIRECODE_SOLV_IMPLEM_FOR_ML"].lower() in ("post", "opt"), (
            "Settings variable FIRECODE_SOLV_IMPLEM_FOR_ML (currently "
            f'"{os.environ["FIRECODE_SOLV_IMPLEM_FOR_ML"]}") can only be "post" or "opt".'
        )

        self.opt_func = optimization_functions_dict[self.calculator]

    def get_optimizer_str(self, override: str | None = None) -> str:
        """Get the string corresponding to an ASE optimizer."""
        from firecode.ase_manipulations import optimizer_dict

        optimizer = (
            override
            or os.environ.get(
                f"FIRECODE_DEFAULT_ASE_OPTIMIZER_{str(self.calculator).upper()}",
                os.environ["FIRECODE_FALLBACK_ASE_OPTIMIZER"],  # fallback value
            )
        ).upper()
        if optimizer not in optimizer_dict:
            raise NameError(
                f'Optimizer "{optimizer}" is unknown. Options: {list(optimizer_dict.keys())}'
            )

        return optimizer

    def get_ase_calc(
        self,
        method: str | None = None,
        solvent: str | None = None,
        force_reload: bool = False,
        raise_err: bool = True,
        logfunction: Callable[[str], None] | None = print,
    ) -> ASECalculator | MaybeNone:
        self.solvent = solvent

        if self.ase_calc is not None and not force_reload:
            return self.ase_calc

        elif self.calculator == "ORCA":
            self.ase_calc = self.load_orca_calc(method)

        elif self.calculator == "AIMNET2":
            self.ase_calc = self.load_aimnet2_calc(method, logfunction)

        elif self.calculator == "TBLITE":
            self.ase_calc = self.load_tblite_calc(method)

        elif self.calculator == "XTB":
            self.ase_calc = self.load_xtb_calc(method)

        elif self.calculator == "UMA":
            self.ase_calc = self.load_uma_calc(method, logfunction)

        elif raise_err:
            raise NotImplementedError(
                f"Calculator {self.calculator} not known. Options are AIMNET2, TBLITE, XTB and UMA."
            )

        if logfunction is not None:
            optimizer_str = os.environ.get(
                f"FIRECODE_DEFAULT_ASE_OPTIMIZER_{self.calculator.upper()}",
                os.environ["FIRECODE_FALLBACK_ASE_OPTIMIZER"],
            ).upper()
            logfunction(
                f"--> Loaded {self.calculator.upper()} ASE Calculator. Default geometry optimizer: {optimizer_str}"
            )

        return self.ase_calc

    def load_orca_calc(self, method: str | None) -> ASECalculator:
        raise NotImplementedError("Open an issue on GitHub if you would like to use ORCA via ASE.")

    def load_aimnet2_calc(
        self,
        theory_level: str | None,
        logfunction: Callable[[str], None] | None = print,
    ) -> ASECalculator:
        try:
            import torch
            from aimnet.calculators import AIMNet2ASE

        except ImportError as err:
            print(err)
            raise Exception(
                (
                    "Cannot import AIMNet2 python bindings for FIRECODE. Install them with:\n"
                    "    >>> uv pip install aimnet[ase]\n"
                    'or alternatively, install the "aimnet2" version of firecode:\n'
                    "    >>> uv pip install firecode[aimnet2]\n"
                )
            )

        gpu_bool = torch.cuda.is_available()
        self.ase_calc = cast("ASECalculator", AIMNet2ASE("aimnet2"))

        if logfunction is not None:
            logfunction(f"--> AIMNet2 calculator loaded on {'GPU' if gpu_bool else 'CPU'}.")

        if self.solvent is None:
            return self.ase_calc

        match os.environ["FIRECODE_SOLV_IMPLEM_FOR_ML"].lower():
            case "post":
                pass

            case "opt":
                from firecode.calculators.solvation_delta_calc import SolvatedMLCalculator

                self.ase_calc = SolvatedMLCalculator(
                    self.ase_calc,
                    solvent=self.solvent,
                    solv_model=os.environ["FIRECODE_SOLV_METHOD_FOR_ML"].lower(),
                )

            case _:
                raise SyntaxError(
                    "Settings variable FIRECODE_SOLV_IMPLEM_FOR_ML (currently "
                    f'"{os.environ["FIRECODE_SOLV_IMPLEM_FOR_ML"]}") can only be "post" or "scf".'
                )

        if logfunction is not None:
            logfunction(
                f"--> Delta solvation: {os.environ['FIRECODE_SOLV_METHOD_FOR_ML'].upper()}"
                f"({self.solvent.upper()}, mode: {os.environ['FIRECODE_SOLV_IMPLEM_FOR_ML'].upper()})"
                f" solvation via TBLITE added to {self.calculator}"
            )
        return self.ase_calc

    def load_tblite_calc(self, method: str | None) -> ASECalculator:
        try:
            from tblite.ase import TBLite
            from tblite.exceptions import TBLiteValueError

        except ImportError as e:
            print(e)
            raise Exception(
                (
                    "Cannot import tblite python bindings for FIRECODE. Install them with conda, (or better yet, mamba):\n"
                    ">>> conda install -c conda-forge mamba\n"
                    ">>> mamba install -c conda-forge tblite tblite-python\n"
                )
            )

        method = method or str(os.environ.get("FIRECODE_DEFAULT_LEVEL_TBLITE"))

        # tblite is picky with names
        synonyms = {
            "GFN1-XTB": "GFN1-xTB",
            "GFN2-XTB": "GFN2-xTB",
            "G-XTB": "g-xTB",
        }

        method = synonyms.get(method, method)

        if self.solvent is None:
            self.ase_calc = TBLite(method=method)
            return self.ase_calc

        try:
            match os.environ["FIRECODE_TBLITE_SOLV_METHOD"].lower():
                case "alpb":
                    try:
                        # translate if needed
                        xtb_solvent_name = to_xtb_solvents.get(self.solvent, self.solvent)

                        # add ALPB solvation via solvent name
                        self.ase_calc = TBLite(method=method, alpb_solvation=xtb_solvent_name)

                    except TBLiteValueError:
                        raise TBLiteValueError(
                            "TBLITE was not able to set up ALPB solvation correctly - "
                            f'"{self.solvent}" not recognized by TBLITE. A workaround might be to use '
                            "CPCM solvation, if the corresponding epsilon value is known in firecode.solvents."
                        )

                case "cpcm":
                    try:
                        epsilon = epsilon_dict[self.solvent]

                        # add CPCM solvation via solvent name
                        self.ase_calc = TBLite(method=method, cpcm_solvation=epsilon)

                    except KeyError:
                        raise KeyError(
                            f'solvent (currently "{self.solvent}" is not present '
                            "in epsilon_dict. (FIRECODE does not know its epsilon value)."
                        )

                case _:
                    raise ValueError(
                        "FIRECODE_TBLITE_SOLV_METHOD (currently "
                        f'"{os.environ["FIRECODE_TBLITE_SOLV_METHOD"].lower()}") '
                        'must be either ("alpb" or "cpcm").'
                    )

        except TBLiteValueError:
            print(
                "--> WARNING: TBLITE was not able to set up ALPB solvation correctly. Defaulted to vacuum."
            )
            self.ase_calc = TBLite(method=method)

        return self.ase_calc

    def load_xtb_calc(self, method: str | None) -> ASECalculator:
        raise NotImplementedError(
            "The XTB ASE calculator is deprecated. Use the TBLITE calculator."
        )

    def load_uma_calc(
        self,
        method: str | None,
        logfunction: Callable[[str], None] | None = print,
    ) -> ASECalculator:
        from firecode.calculators.ase_uma import get_uma_calc

        self.ase_calc = cast("ASECalculator", get_uma_calc(method, logfunction=logfunction))

        if self.solvent is None:
            return self.ase_calc

        match os.environ["FIRECODE_SOLV_IMPLEM_FOR_ML"].lower():
            case "post":
                pass

            case "opt":
                from firecode.calculators.solvation_delta_calc import (
                    SolvatedMLCalculator,
                )

                self.ase_calc = SolvatedMLCalculator(
                    self.ase_calc,
                    solvent=self.solvent,
                    solv_model=os.environ["FIRECODE_SOLV_METHOD_FOR_ML"].lower(),
                )

            case _:
                raise SyntaxError(
                    "Settings variable FIRECODE_SOLV_IMPLEM_FOR_ML (currently "
                    f'"{os.environ["FIRECODE_SOLV_IMPLEM_FOR_ML"]}") can only be "post" or "scf".'
                )

        if logfunction is not None:
            logfunction(
                f"--> Delta solvation: {os.environ['FIRECODE_SOLV_METHOD_FOR_ML'].upper()}"
                f"({self.solvent.upper()}, mode: {os.environ['FIRECODE_SOLV_IMPLEM_FOR_ML'].upper()})"
                f" solvation via TBLITE added to {self.calculator}"
            )

        return self.ase_calc

    def _load_solv_calc(self) -> SolvationDeltaCalculator:
        """Loads the TBLITE delta solvation calculator if not already loaded, and returns it."""
        if self._solv_calc is not None:
            return self._solv_calc

        if self.solvent is None:
            raise ValueError(
                "Cannot load TBLITE delta solvation calculator without solvent specified. "
                "Set the solvent via the get_ase_calc method or by setting the dispatcher.solvent attribute."
            )

        from firecode.calculators.solvation_delta_calc import SolvationDeltaCalculator

        self._solv_calc = SolvationDeltaCalculator(
            solvent=self.solvent, solv_model=os.environ["FIRECODE_SOLV_METHOD_FOR_ML"].lower()
        )

        return self._solv_calc

    @property
    def solv_calc(self) -> SolvationDeltaCalculator:
        """Returns the TBLITE delta solvation calculator."""
        return self._load_solv_calc()
