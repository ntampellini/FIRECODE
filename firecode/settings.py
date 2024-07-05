# coding=utf-8
'''
FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2024 Nicolò Tampellini

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
# 'GAUSSIAN' : FF methods supported by Gaussian (UFF, MMFF)
# 'XTB' : GFN-FF method

DEFAULT_FF_LEVELS = {
    ### DO NOT REMOVE
    ### THESE TWO LINES
    'GAUSSIAN':'UFF',
    'XTB':'GFN-FF',
}
# Default levels used to run calculations, overridden by FFLEVEL keyword

CALCULATOR = 'XTB'
# Calculator used to run geometry optimization.
# Possibilites are:
# 'MOPAC' : Semiempirical MOPAC2016 (PM7, PM6-DH3, ...)
# 'ORCA' : All methods supported by ORCA
# 'GAUSSIAN' : All methods supported by Gaussian
# 'XTB' : All methods supported by XTB
# 'AIMNET2' : wB97M-D3

DEFAULT_LEVELS = {
    'MOPAC':'PM7',
    'ORCA':'PM3',
    'GAUSSIAN':'PM6',
    'XTB':'GFN2-xTB',
    'AIMNET2':'wB97M-D3',
}
# Default levels used to run calculations, overridden by LEVEL keyword

COMMANDS = {
    'MOPAC':'MOPAC2016.exe',
    'ORCA':'/vast/palmer/apps/avx.grace/software/ORCA/5.0.4-gompi-2020b/bin/orca',
    'GAUSSIAN':'g09.exe',
    'XTB':'xtb',
}
# Command with which calculators will be called from the command line

PROCS = 0
# Number of processors (cores) per job to be used by XTB, ORCA and/or Gaussian (0 is auto)

MEM_GB = 8
# Memory allocated for each job (Gaussian/ORCA)