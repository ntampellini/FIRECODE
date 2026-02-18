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
from dataclasses import dataclass

from prism_pruner.periodic_table import INDEX_TABLE, MASSES_TABLE, RADII_TABLE


@dataclass
class PeriodicTable:

    def covalent_radius(self, symbol):
        return RADII_TABLE[symbol]

    def mass(self, symbol):
        return MASSES_TABLE[symbol]

    def number(self, symbol):
        return INDEX_TABLE[symbol]

pt = PeriodicTable()
