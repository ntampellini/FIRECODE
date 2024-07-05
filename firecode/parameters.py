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

orb_dim_dict = {
    'H Single Bond' : 0.85,
    'C Single Bond' : 1,
    'O Single Bond' : 1,
    'N Single Bond' : 1,
    'F Single Bond' : 1,
    'Cl Single Bond' : 1.5,
    'Br Single Bond' : 1.5,
    'I Single Bond' : 2,

    'C sp' : 1,
    'N sp' : 1,

    'B sp2' : 0.8,
    'C sp2' : 1.1,
    'N sp2' : 1,

    'B sp3' : 1,
    'C sp3' : 1,
    'Br sp3' : 1,

    'O Ether' : 1,
    'S Ether' : 1,

    'O Ketone': 0.85,
    'S Ketone': 1,

    'N Imine' : 1,

    'C bent carbene' : 1,

    'Metal' : 2.5,

    'Fallback' : 1
                }       
# Half-lenght of the transition state bonding distance involving a given atom

nci_dict={
    # tag in alphabetical order (i.e. 'IN' and not 'NI')
    # maximum distance for a given non-covalent interaction

    # Hydrogen Bonds
    'HO'  :(2.2,'O-H hydrogen bond'),
    'HN'  :(2.2,'N-H hydrogen bond'),

    # Aromatics and Stacking
    'HPh' :(2.8,'H-Ar non-conventional hydrogen bond'), # taken from https://doi.org/10.1039/C1CP20404A
    'PhPh':(3.8, 'pi-stacking interaction'),            # guessed from https://doi.org/10.1039/C2SC20045G

    # Halogens
    'FF'  :(3.5,'F-F interaction'),
    # 'FPh' :(0.0, 'F-Ar halogen-bonding interaction'),
    # 'ClPh':(0.0, 'Cl-Ar halogen-bonding interaction'),
    # 'BrPh':(0.0, 'Br-Ar halogen-bonding interaction'),
    # 'IPh' :(0.0, 'I-Ar halogen-bonding interaction'),
    # 'IN'  :(0.0, 'I-N halogen-bonding interaction'),
    # 'IO'  :(0.0, 'I-O halogen-bonding interaction'),
    # 'BrN' :(0.0, 'Br-N halogen-bonding interaction'),
    # 'BrO' :(0.0, 'Br-O halogen-bonding interaction'),
}
# non covalent interaction threshold and types for atomic pairs