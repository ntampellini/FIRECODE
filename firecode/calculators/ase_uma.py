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
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from fairchem.core import FAIRChemCalculator


def get_uma_calc(
    method: str | None = None, logfunction: Callable[[str], None] | None = None
) -> FAIRChemCalculator:
    """Load UMA model from disk and return the ASE calculator object"""
    try:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from torch import cuda

    except ImportError as err:
        print(err)
        raise ImportError(
            "To run the UMA models, please install fairchem:\n"
            "    >>> uv pip install fairchem-core\n"
            'or alternatively, install the "uma" version of firecode:\n'
            "    >>> uv pip install firecode[uma]\n"
        )

    gpu_bool = cuda.is_available()
    method = method or str(os.environ.get("FIRECODE_DEFAULT_LEVEL_UMA"))

    if gpu_bool:
        if logfunction is not None:
            logfunction(f"--> {cuda.device_count()} CUDA devices detected: loading model on GPU")

    elif logfunction is not None:
        logfunction("--> No CUDA devices detected: loading model on CPU")
        logfunction(f"--> Loading UMA/{method.upper()} model from file")

    model_path = os.environ.get("FIRECODE_PATH_TO_UMA_MODEL") or ""

    # make relative path absolute
    if model_path.startswith("."):
        model_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.path.basename(model_path)
        )

    try:
        predictor = load_predict_unit(model_path, device="cuda" if gpu_bool else "cpu")

    except FileNotFoundError:
        raise FileNotFoundError(
            f'Invalid path for UMA model: FIRECODE_PATH_TO_UMA_MODEL="{model_path}".'
        )

    ase_calc = FAIRChemCalculator(predictor, task_name=method.lower())
    return ase_calc
