"""FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2026 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""

import numpy as np
from prism_pruner.algebra import normalize
from scipy.spatial.distance import cdist

from firecode.typing_ import Array1D_float, Array2D_float


def point_angle(p1: Array1D_float, p2: Array1D_float, p3: Array1D_float) -> float:
    """Returns the planar angle between three points in space, in degrees."""
    return np.arccos(np.clip(normalize(p1 - p2) @ normalize(p3 - p2), -1.0, 1.0)) * 180 / np.pi  # type: ignore[no-any-return]


def align_vec_pair(ref: Array2D_float, tgt: Array2D_float) -> Array2D_float:
    """ref, tgt: iterables of two 3D vectors each

    return: rotation matrix that when applied to tgt,
            optimally aligns it to ref
    """
    B = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            tot = 0
            for j in range(2):
                tot += ref[j][i] * tgt[j][k]
            B[i, k] = tot

    u, s, vh = np.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    if np.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]

    return np.ascontiguousarray(np.dot(u, vh))


def count_clashes(coords: Array2D_float) -> int:
    """Returns the number of atomic distances between 0 and 0.5 Å."""
    return int(np.count_nonzero((cdist(coords, coords) < 0.5) & (cdist(coords, coords) > 0)))
