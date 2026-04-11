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

# IF YOU MANUALLY EDIT THIS FILE, BE SURE NOT TO
# CHANGE IDENTATION/WHITESPACES/NEWLINES!

FF_OPT_BOOL = False
# Whether to run Force Field optimization
# prior to the final one.

FF_CALC: str | None = None
# Calculator to perform Force Field optimizations.
# Possibilites are:
# 'XTB' : GFN-FF method

DEFAULT_FF_LEVELS = {
    "XTB": "GFN-FF",
}
# Default levels used to run calculations, overridden by FFLEVEL keyword

CALCULATOR = "TBLITE"
# Default calculator used to run geometry optimization.
# Possibilites are (see default levels below)

FORCE_SINGLE_THREAD = True
# Enforce the use of a single thread in multimolecular optimization.
# Multithread optimization is possible but may suffer from performance
# issues in specific cases.

CHECKPOINT_EVERY = 50
# Save a checkpoint every this many geometry optimizations

UMA_MODEL_PATH = "(set with ``firecode -s``)"
# Path of UMA model to load, either relative (to firecode/calculators/) or absolute

DEFAULT_LEVELS = {
    "XTB": "GFN2-xTB",
    "TBLITE": "GFN2-xTB",
    "AIMNET2": "wB97M-D3",
    "UMA": "OMOL",
    "ORCA": "PM3",
}
# Default levels used to run calculations, overridden by LEVEL keyword

COMMANDS = {
    "ORCA": "orca",
    "XTB": "xtb",
}
# Command with which certain calculators will be called from the command line

PROCS = 4
# Number of processors (cores) per job to be used by XTB or ORCA (0 is auto)

MEM_GB = 4
# Memory allocated for each job (ORCA)

# these need to start with FIRECODE_ to ensure uniqueness
ENV_VARS = dict(
    FIRECODE_DEFAULT_ASE_OPTIMIZER_XTB="LBFGS",
    FIRECODE_DEFAULT_ASE_OPTIMIZER_TBLITE="LBFGS",
    FIRECODE_DEFAULT_ASE_OPTIMIZER_ORCA="LBFGS",
    FIRECODE_DEFAULT_ASE_OPTIMIZER_AIMNET2="LBFGS",
    FIRECODE_DEFAULT_ASE_OPTIMIZER_UMA="LBFGS",
    FIRECODE_FALLBACK_ASE_OPTIMIZER="LBFGS",
    FIRECODE_SOLV_METHOD_FOR_ML="alpb",  # model of solvation via TBLITE: "alpb" or "cpcm"
    FIRECODE_SOLV_IMPLEM_FOR_ML="post",  # Implementation of ALPB solvation via TBLITE:
    # - "post" is post-optimization,
    # - "opt" adds the energy and gradients to the ASE calculator
    #   at each step during the optimization.
    FIRECODE_TBLITE_SOLV_METHOD="alpb",  # model of solvation of TBLITE: "alpb" or "cpcm"
    FIRECODE_SELLA_INTERNAL_OVERRIDE="",  # if "false" enforces Sella to use cartesian coordinates (constrained optimizations not possible!)
)
