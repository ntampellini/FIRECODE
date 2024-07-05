import numba as nb
import numpy as np

from firecode.algebra import get_alignment_matrix, norm_of


@nb.njit
def np_mean_along_axis(axis, arr):
    '''
    Workaround to specify axis parameters to
    numba functions, adapted from
    https://github.com/numba/numba/issues/1269
    
    '''
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=np.float64)
        for i in range(len(result)):
            result[i] = np.mean(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=np.float64)
        for i in range(len(result)):
            result[i] = np.mean(arr[i, :])
    return result

@nb.njit
def rmsd_and_max_numba(p, q, center=False):
    '''
    Returns a tuple with the RMSD between p and q
    and the maximum deviation of their positions

    '''

    if center:
        # p -= p.mean(axis=0)
        # q -= q.mean(axis=0)
        p -= np_mean_along_axis(0, p)
        q -= np_mean_along_axis(0, q)
    
    # get alignment matrix
    rot_mat = get_alignment_matrix(p, q)

    # Apply it to p
    p = np.ascontiguousarray(p) @ rot_mat

    # Calculate deviations
    diff = p - q

    # Calculate RMSD
    rmsd = np.sqrt((diff * diff).sum() / len(diff))

    # # Calculate max deviation
    # max_delta = np.linalg.norm(diff, axis=1).max()
    max_delta = max([norm_of(v) for v in diff])

    return rmsd, max_delta

def _rmsd_similarity(ref, structures, rmsd_thr=0.5) -> bool:
    '''
    Simple, RMSD similarity eval function.

    '''

    # iterate over target structures
    for structure in structures:
        
        # compute RMSD and max deviation
        rmsd_value, maxdev_value = rmsd_and_max_numba(ref, structure)

        if rmsd_value < rmsd_thr and maxdev_value < 2 * rmsd_thr:
            return True
            
    return False