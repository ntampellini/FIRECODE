'''

FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2024 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''
from math import sqrt

import numba as nb
import numpy as np


@nb.njit
def dihedral(p):
    '''
    Returns dihedral angle in degrees from 4 3D vecs
    Praxeolitic formula: 1 sqrt, 1 cross product
    
    '''
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= norm_of(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    
    return np.degrees(np.arctan2(y, x))

@nb.njit
def vec_angle(v1, v2):
    v1_u = norm(v1)
    v2_u = norm(v2)
    return np.arccos(clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/np.pi

@nb.njit
def clip(n, lower, higher):
    '''
    jittable version of np.clip for single values
    '''
    if n > higher:
        return higher
    elif n < lower:
        return lower
    else:
        return n

# @nb.njit
def point_angle(p1, p2, p3):
    '''
    Returns the planar angle between three points in space, in degrees.
    '''
    return np.arccos(np.clip(norm(p1 - p2) @ norm(p3 - p2), -1.0, 1.0))*180/np.pi

@nb.njit
def norm(vec):
    '''
    Returns the normalized vector.
    Reasonably faster than Numpy version.
    Only for 3D vectors.
    '''
    return vec / sqrt((vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]))

@nb.njit
def norm_of(vec):
    '''
    Returns the norm of the vector.
    Faster than Numpy version, but 
    only compatible with 3D vectors.
    '''
    return sqrt((vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]))

@nb.njit(fastmath=True)
def all_dists(A, B):
    assert A.shape[1]==B.shape[1]
    C=np.empty((A.shape[0],B.shape[0]),A.dtype)
    I_BLK=32
    J_BLK=32
    
    #workaround to get the right datatype for acc
    init_val_arr=np.zeros(1,A.dtype)
    init_val=init_val_arr[0]
    
    #Blocking and partial unrolling
    #Beneficial if the second dimension is large -> computationally bound problem 
    # 
    for ii in nb.prange(A.shape[0]//I_BLK):
        for jj in range(B.shape[0]//J_BLK):
            for i in range(I_BLK//4):
                for j in range(J_BLK//2):
                    acc_0=init_val
                    acc_1=init_val
                    acc_2=init_val
                    acc_3=init_val
                    acc_4=init_val
                    acc_5=init_val
                    acc_6=init_val
                    acc_7=init_val
                    for k in range(A.shape[1]):
                        acc_0+=(A[ii*I_BLK+i*4+0,k] - B[jj*J_BLK+j*2+0,k])**2
                        acc_1+=(A[ii*I_BLK+i*4+0,k] - B[jj*J_BLK+j*2+1,k])**2
                        acc_2+=(A[ii*I_BLK+i*4+1,k] - B[jj*J_BLK+j*2+0,k])**2
                        acc_3+=(A[ii*I_BLK+i*4+1,k] - B[jj*J_BLK+j*2+1,k])**2
                        acc_4+=(A[ii*I_BLK+i*4+2,k] - B[jj*J_BLK+j*2+0,k])**2
                        acc_5+=(A[ii*I_BLK+i*4+2,k] - B[jj*J_BLK+j*2+1,k])**2
                        acc_6+=(A[ii*I_BLK+i*4+3,k] - B[jj*J_BLK+j*2+0,k])**2
                        acc_7+=(A[ii*I_BLK+i*4+3,k] - B[jj*J_BLK+j*2+1,k])**2
                    C[ii*I_BLK+i*4+0,jj*J_BLK+j*2+0]=np.sqrt(acc_0)
                    C[ii*I_BLK+i*4+0,jj*J_BLK+j*2+1]=np.sqrt(acc_1)
                    C[ii*I_BLK+i*4+1,jj*J_BLK+j*2+0]=np.sqrt(acc_2)
                    C[ii*I_BLK+i*4+1,jj*J_BLK+j*2+1]=np.sqrt(acc_3)
                    C[ii*I_BLK+i*4+2,jj*J_BLK+j*2+0]=np.sqrt(acc_4)
                    C[ii*I_BLK+i*4+2,jj*J_BLK+j*2+1]=np.sqrt(acc_5)
                    C[ii*I_BLK+i*4+3,jj*J_BLK+j*2+0]=np.sqrt(acc_6)
                    C[ii*I_BLK+i*4+3,jj*J_BLK+j*2+1]=np.sqrt(acc_7)
        #Remainder j
        for i in range(I_BLK):
            for j in range((B.shape[0]//J_BLK)*J_BLK,B.shape[0]):
                acc_0=init_val
                for k in range(A.shape[1]):
                    acc_0+=(A[ii*I_BLK+i,k] - B[j,k])**2
                C[ii*I_BLK+i,j]=np.sqrt(acc_0)
    
    #Remainder i
    for i in range((A.shape[0]//I_BLK)*I_BLK,A.shape[0]):
        for j in range(B.shape[0]):
            acc_0=init_val
            for k in range(A.shape[1]):
                acc_0+=(A[i,k] - B[j,k])**2
            C[i,j]=np.sqrt(acc_0)
            
    return C

@nb.njit
def kronecker_delta(i, j) -> int:
    if i == j:
        return 1
    return 0

@nb.njit
def get_inertia_moments(coords, masses):
    '''
    Returns the diagonal of the diagonalized inertia tensor, that is
    a shape (3,) array with the moments of inertia along the main axes.
    (I_x, I_y and largest I_z last)
    '''
    
    coords -= center_of_mass(coords, masses)
    inertia_moment_matrix = np.array([[0.,0.,0.],
                                      [0.,0.,0.],
                                      [0.,0.,0.]])

    for i in range(3):
        for j in range(3):
            k = kronecker_delta(i,j)
            inertia_moment_matrix[i][j] = sum([masses[n]*((norm_of(coords[n])**2)*k - coords[n][i]*coords[n][j])
                                                  for n, _ in enumerate(coords)])

    inertia_moment_matrix = diagonalize(inertia_moment_matrix)

    return np.diag(inertia_moment_matrix)

@nb.njit
def diagonalize(A):
    eigenvalues_of_A, eigenvectors_of_A = np.linalg.eig(A)
    B = eigenvectors_of_A[:,np.abs(eigenvalues_of_A).argsort()]   
    diagonal_matrix= np.dot(np.linalg.inv(B), np.dot(A, B))
    return diagonal_matrix

@nb.njit
def center_of_mass(coords, masses):
    '''
    Returns the center of mass for the atomic system.
    '''
    total_mass = sum([masses[i] for i in range(len(coords))])
    w = np.array([0.,0.,0.])
    for i in range(len(coords)):
        w += coords[i]*masses[i]
    return w / total_mass

@nb.njit
def internal_mean(arr):
    '''
    same as np.mean(arr, axis=1), but jitted
    since numba does not support kwargs in np.mean
    '''
    assert len(arr.shape) == 3

    out = np.zeros((arr.shape[0], arr.shape[2]), dtype=arr.dtype)
    dim = arr.shape[1]
    for i, vecs in enumerate(arr):
        for v in vecs:
            out[i] += v
    return out / dim

@nb.njit
def vec_mean(arr):
    '''
    same as np.mean(arr, axis=0), but jitted
    since numba does not support kwargs in np.mean
    '''
    assert len(arr.shape) == 2

    out = np.zeros(arr.shape[1], dtype=arr.dtype)
    dim = arr.shape[0]

    for v in arr:
        out += v
        
    arr /= dim

    return out

@nb.njit
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

@nb.njit
def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
    This rotation matrix converts a point in the local reference 
    frame to a point in the global reference frame.
    """
    # Extract the values from Q (adjusting for scalar last in input)
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return np.ascontiguousarray(rot_matrix)

@nb.njit
def rot_mat_from_pointer(pointer, angle):
    '''
    Returns the rotation matrix that rotates a system around the given pointer
    of angle degrees. The algorithm is based on scipy quaternions.
    :params pointer: a 3D vector
    :params angle: an int/float, in degrees
    :return rotation_matrix: matrix that applied to a point, rotates it along the pointer
    '''
    assert pointer.shape[0] == 3

    pointer = norm(pointer)
    angle *= np.pi/180
    quat = np.array([np.sin(angle/2)*pointer[0],
                     np.sin(angle/2)*pointer[1],
                     np.sin(angle/2)*pointer[2],
                     np.cos(angle/2)])
    # normalized quaternion, scalar last (i j k w)
    
    return quaternion_to_rotation_matrix(quat)

@nb.njit(nb.int32[:,:](nb.int32[:]))
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

@nb.njit
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

@nb.njit
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

@nb.njit
def get_alignment_matrix(p, q):
    '''
    Returns the rotation matrix that aligns
    vectors q to p (Kabsch algorithm)
    Assumes centered vector sets (mean is origin)
    '''

    # calculate the covariance matrix
    cov_mat = np.ascontiguousarray(p.T) @ q
    # cov_mat = np.transpose(p) * q
    
    # Compute the SVD
    v, _, w = np.linalg.svd(cov_mat)
    d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0

    if d:
        v[:, -1] = -v[:, -1]

    # Create Rotation matrix u
    rot_mat = np.dot(v, w)

    return rot_mat