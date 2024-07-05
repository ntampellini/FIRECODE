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
class ZeroCandidatesError(Exception):
    '''
    Raised at any time during run if all
    candidates are discarded.
    '''

class InputError(Exception):
    '''
    Raised when reading the input file if
    something is wrong.
    '''

class TriangleError(Exception):
    '''
    Raised from polygonize if it cannot build
    a triangle with the given side lengths.
    '''

class CCReadError(Exception):
    '''
    Raised when CCRead cannot read
    the provided filename.
    '''

class MopacReadError(Exception):
    '''
    Thrown when reading MOPAC output files fails for some reason.
    '''

class SegmentedGraphError(Exception):
    '''
    Thrown by Clustered CSearch when graph has more than one connected component.
    '''

class NoOrbitalError(Exception):
    '''
    Thrown when trying to access orbital data when they are not present
    '''

class FatalError(Exception):
    '''
    Thrown when a molecule optimization crashed or scrambled fatally
    '''