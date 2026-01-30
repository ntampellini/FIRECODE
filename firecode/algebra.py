'''

FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2026 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''
import numpy as np
from prism_pruner.algebra import normalize

norm_of = np.linalg.norm


def point_angle(p1, p2, p3):
    '''
    Returns the planar angle between three points in space, in degrees.
    '''
    return np.arccos(np.clip(normalize(p1 - p2) @ normalize(p3 - p2), -1.0, 1.0))*180/np.pi


def kronecker_delta(i, j) -> int:
    if i == j:
        return 1
    return 0


def align_vec_pair(ref, tgt):
    '''
    ref, tgt: iterables of two 3D vectors each
    
    return: rotation matrix that when applied to tgt,
            optimally aligns it to ref
    '''
    
    B = np.zeros((3,3))
    for i in range(3):
        for k in range(3):
            tot = 0
            for j in range(2):
                tot += ref[j][i]*tgt[j][k]
            B[i,k] = tot

    u, s, vh = np.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    if np.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]

    return np.ascontiguousarray(np.dot(u, vh))


def cart_prod_idx(sizes: np.ndarray):
    """Generates ids tuples for a cartesian product"""
    assert len(sizes) >= 2
    tuples_count  = np.prod(sizes)
    tuples = np.zeros((tuples_count, len(sizes)), dtype=np.int32)
    tuple_idx = 0
    # stores the current combination
    current_tuple = np.zeros(len(sizes))
    while tuple_idx < tuples_count:
        tuples[tuple_idx] = current_tuple
        current_tuple[0] += 1
        # using a condition here instead of including this in the inner loop
        # to gain a bit of speed: this is going to be tested each iteration,
        # and starting a loop to have it end right away is a bit silly
        if current_tuple[0] == sizes[0]:
            # the reset to 0 and subsequent increment amount to carrying
            # the number to the higher "power"
            current_tuple[0] = 0
            current_tuple[1] += 1
            for i in range(1, len(sizes) - 1):
                if current_tuple[i] == sizes[i]:
                    # same as before, but in a loop, since this is going
                    # to get run less often
                    current_tuple[i + 1] += 1
                    current_tuple[i] = 0
                else:
                    break
        tuple_idx += 1
    return tuples


def vector_cartesian_product(x, y):
    '''
    Cartesian product, but with vectors instead of indices
    '''
    indices = cart_prod_idx(np.asarray((x.shape[0], y.shape[0]), dtype=np.int32))
    dim = x.shape[-1] if len(x.shape) > 1 else 1
    new_arr = np.zeros((*indices.shape, dim), dtype=x.dtype)
    for i, (x_, y_) in enumerate(indices):
        new_arr[i][0] = x[x_]
        new_arr[i][1] = y[y_]
    return np.ascontiguousarray(new_arr)


def transform_coords(coords, rot, pos):
    '''
    Returns the rotated and tranlated
    coordinates. Slightly faster than
    Numpy, uses memory-contiguous arrays.
    '''
    t = np.transpose(coords)
    m = rot @ t
    f = np.transpose(m)
    return f + pos
