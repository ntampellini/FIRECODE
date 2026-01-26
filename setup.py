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
__version__ = '1.4.0'
from setuptools import setup, find_packages

long_description = ('## FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles.\nEnsemble optimizer. Systematic generation of multimolecular arrangements for ' +
                'mono/bi/trimolecular molecular assemblies. Numerous utilities for conformational exploration, selection, pruning and constrained ensemble optimization.')

with open('CHANGELOG.md', 'r') as f:
    long_description += '\n\n'
    long_description += f.read()

setup(
    name='firecode',
    version=__version__,
    description='Computational chemistry general purpose ensemble optimizer and molecular assemblies builder',
    keywords=['computational chemistry', 'ASE', 'transition state', 'xtb', 'AIMNET2', 'TBLITE'],

    # package_dir={'':'firecode'},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: OS Independent',
    ],

    long_description=long_description,
    long_description_content_type='text/markdown',

    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'networkx',
        'ase',
        'prettytable',
        'typing-extensions', # 4.14.1 also ok
        'llvmlite',#==0.41.1',
        'importlib-metadata',#==7.0.1',
        'psutil',#==5.9.6',
        'setuptools', #==75.3.0',
        'rich',
        'inquirerpy',
        'prism-pruner',
    ],

    url='https://www.github.com/ntampellini/firecode',
    author='Nicolò Tampellini',
    author_email='nicolo.tampellini@yale.edu',

    packages=find_packages(),
    python_requires=">=3.12",
)