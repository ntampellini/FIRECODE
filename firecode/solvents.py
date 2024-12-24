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

import sys

xtb_solvents = [
    'acetone',
    'acetonitrile',
    'aniline',
    'benzaldehyde',
    'benzene',
    'ch2cl2',
    'chcl3',
    'cs2',
    'dioxane',
    'dmf',
    'dmso',
    'ether',
    'ethylacetate',
    'furane',
    'hexadecane',
    'hexane',
    'methanol',
    'nitromethane',
    'octanol',
    'octanolwet',
    'phenol',
    'toluene',
    'thf',
    'water',

    'none', # This is required for the ASE get_calc function
]

xtb_solvents = xtb_solvents + ['' for _ in range(3-len(xtb_solvents)%3)]
gap = 18
xtb_supported = ''.join([(f'{xtb_solvents[i]}{" "*(gap-len(xtb_solvents[i]))}'
                          f'{xtb_solvents[i+1]}{" "*(gap-len(xtb_solvents[i+1]))}'
                          f'{xtb_solvents[i+2]}\n')
                          for i, _ in enumerate(xtb_solvents) if i%3==0])

epsilon_dict = {
    'aceticacid':6.15,
    'acetone':20.7,
    'acetonitrile':37.5,
    'aniline':7.06,
    'benzaldehyde':17.9,
    'benzene':2.28,
    'chloroform':4.8,
    'cs2':2.63,
    'ch2cl2':8.93,
    'dioxane':2.25,
    'dmf':36.71,
    'dmso':46.68,
    'et2o':4.27,
    'dimethylether':6.18,
    'ethanol':24.3,
    'methanol':32.63,
    'ethylacetate':6.02,
    'furan':2.94,
    'hexadecane':2.05,
    'octanol':10.30,
    'phenol':12.4,
    'toluene':2.38,
    'thf':7.58,
    'water':80.1,
}

solvent_synonyms = {
    'ch3cooh':'aceticacid',
    'ch3cn':'acetonitrile',
    'chcl3':'chloroform',
    'dcm':'ch2cl2',
    'dichloromethane':'ch2cl2',
    'carbondisuphide':'cs2',
    'carbondisulfide':'cs2',
    'diethylether':'et2o',
    'etoh':'ethanol',
    'ch3oh':'methanol',
    'meoh':'methanol',
    'h2o':'water',
}

new_theory_level = {
    'MOPAC':lambda theory_level, solvent: f'EPS={epsilon_dict[solvent]}',
    'GAUSSIAN':lambda theory_level, solvent: f'scrf=(cpcm,solvent={solvent})',
    'ORCA':lambda theory_level, solvent: f'! CPCM\n%cpcm\nepsilon {epsilon_dict[solvent]}\nend',
    # 'XTB':lambda theory_level, _: '',
}

def get_solvent_line(solvent, calculator, theory_level):

    if solvent is None:
        return ''

    if solvent in solvent_synonyms:
        solvent = solvent_synonyms[solvent]

    if solvent not in epsilon_dict:
        print(f'Solvent \'{solvent}\' not recognized. Implemented solvents are:')
        for s in epsilon_dict:
            print('    '+s)
        print('Please note that not all solvents will work with all calculators.')
        sys.exit()

    return new_theory_level[calculator](theory_level, solvent)