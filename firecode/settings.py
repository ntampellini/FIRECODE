# coding=utf-8
'''
FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
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

'''

# IF YOU MANUALLY EDIT THIS FILE, BE SURE NOT TO
# CHANGE IDENTATION/WHITESPACES/NEWLINES!

FF_OPT_BOOL = True
# Whether to run Force Field optimization
# prior to the final one.

FF_CALC = 'XTB'
# Calculator to perform Force Field optimizations.
# Possibilites are:
# 'XTB' : GFN-FF method

DEFAULT_FF_LEVELS = {
    'XTB':'GFN-FF',
}
# Default levels used to run calculations, overridden by FFLEVEL keyword

CALCULATOR = 'XTB'
# Default calculator used to run geometry optimization.
# Possibilites are:

SINGLE_THREAD_BOOL = True
# Enforce the use of a single thread in multimolecular optimization.
# Multithread optimization is only possible with XTB, TBLITE and ORCA calculators
# but may suffer from performance issues on some machines.

UMA_MODEL_PATH = "./uma-s-1p1.pt"
# Path of UMA model to load, either relative (to firecode/calculators/) or absolute 

DEFAULT_LEVELS = {
    'ORCA':'PM3',
    'XTB':'GFN2-xTB',
    'XTB':'GFN2-xTB',
    'AIMNET2':'wB97M-D3',
    'UMA':'OMOL',
}
# Default levels used to run calculations, overridden by LEVEL keyword

COMMANDS = {
    'ORCA':'/vast/palmer/apps/avx.grace/software/ORCA/5.0.4-gompi-2020b/bin/orca',
    'XTB':'xtb',
}
# Command with which calculators will be called from the command line

PROCS = 4
# Number of processors (cores) per job to be used by XTB or ORCA (0 is auto)

MEM_GB = 4
# Memory allocated for each job (ORCA)