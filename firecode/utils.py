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

import os
import sys
import time
from _tkinter import TclError
from shutil import rmtree
from subprocess import DEVNULL, STDOUT, CalledProcessError, check_call, run

import numpy as np
from cclib.io import ccread
from numpy.linalg import LinAlgError

from firecode.algebra import get_alignment_matrix, norm_of, rot_mat_from_pointer
from firecode.errors import TriangleError
from firecode.graph_manipulations import graphize
from firecode.pt import pt


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def clean_directory(to_remove=None):

    if to_remove:
        for name in to_remove:
            try:
                os.remove(name)
            except IsADirectoryError:
                rmtree(os.path.join(os.getcwd(), name))
            except FileNotFoundError:
                pass

    for f in os.listdir():
        if f.split('.')[0] == 'temp':
            try:
                os.remove(f)
            except IsADirectoryError:
                rmtree(os.path.join(os.getcwd(), f))
            except FileNotFoundError:
                pass
        elif f.startswith('temp_'):
            try:
                os.remove(f)
            except IsADirectoryError:
                rmtree(os.path.join(os.getcwd(), f))
            except FileNotFoundError:
                pass

def run_command(command:str, p=False):
    if p:
        print("Command: {}".format(command))
    result = run(command.split(), shell=False, capture_output=True)
    if result.stderr:
        raise CalledProcessError(
                returncode = result.returncode,
                cmd = result.args,
                stderr = result.stderr
                )
    if p and result.stdout:
        print("Command Result: {}".format(result.stdout.decode('utf-8')))
    return result

def align_structures(structures:np.array, indices=None, **kwargs):
    '''
    Aligns molecules of a structure array (shape is (n_structures, n_atoms, 3))
    to the first one, based on the indices. If not provided, all atoms are used
    to get the best alignment. Return is the aligned array.

    '''

    reference = structures[0]
    targets = structures[1:]
    if isinstance(indices, (list, tuple)):
        indices = np.array(indices)

    indices = slice(0,len(reference)) if (indices is None or len(indices) == 0) else indices.ravel()

    reference -= np.mean(reference[indices], axis=0)
    for t, _ in enumerate(targets):
        targets[t] -= np.mean(targets[t,indices], axis=0)

    output = np.zeros(structures.shape)
    output[0] = reference

    for t, target in enumerate(targets):

        try:
            matrix = get_alignment_matrix(reference[indices], target[indices])

        except LinAlgError:
        # it is actually possible for the kabsch alg not to converge
            matrix = np.eye(3)
        
        # output[t+1] = np.array([matrix @ vector for vector in target])
        output[t+1] = (matrix @ target.T).T

    return output

def write_xyz(coords:np.array, atomnos:np.array, output, title='temp'):
    '''
    output is of _io.TextIOWrapper type

    '''
    assert atomnos.shape[0] == coords.shape[0]
    assert coords.shape[1] == 3
    string = ''
    string += str(len(coords))
    string += f'\n{title}\n'
    for i, atom in enumerate(coords):
        string += '%s     % .6f % .6f % .6f\n' % (pt[atomnos[i]].symbol, atom[0], atom[1], atom[2])
    output.write(string)

def read_xyz(filename):
    '''
    Wrapper for ccread. Raises an error if unsuccessful.

    '''
    mol = ccread(filename)
    assert mol is not None, f'Reading molecule {filename} failed - check its integrity.'
    return mol

def time_to_string(total_time: float, verbose=False, digits=1):
    '''
    Converts totaltime (float) to a timestring
    with hours, minutes and seconds.
    '''
    timestring = ''

    names = ('days', 'hours', 'minutes', 'seconds') if verbose else ('d', 'h', 'm', 's')

    if total_time > 24*3600:
        d = total_time // (24*3600)
        timestring += f'{int(d)} {names[0]} '
        total_time %= (24*3600)

    if total_time > 3600:
        h = total_time // 3600
        timestring += f'{int(h)} {names[1]} '
        total_time %= 3600

    if total_time > 60:
        m = total_time // 60
        timestring += f'{int(m)} {names[2]} '
        total_time %= 60

    timestring += f'{round(total_time, digits):{2+digits}} {names[3]}'

    return timestring

def pretty_num(n):
    if n < 1e3:
        return str(n)
    if n < 1e6:
        return str(round(n/1e3, 2)) + ' k'
    return str(round(n/1e6, 2)) + ' M'

def loadbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def cartesian_product(*arrays):
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.

    """
    assert vec1.shape == (3,)
    assert vec2.shape == (3,)

    a, b = (vec1 / norm_of(vec1)).reshape(3), (vec2 / norm_of(vec2)).reshape(3)
    v = np.cross(a, b)
    if norm_of(v) != 0:
        c = np.dot(a, b)
        s = norm_of(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
    
    # if the cross product is zero, then vecs must be parallel or perpendicular
    if norm_of(a + b) == 0:
        pointer = np.array([0,0,1])
        return rot_mat_from_pointer(pointer, 180)
        
    return np.eye(3)

def polygonize(lengths):
    '''
    Returns coordinates for the polygon vertices used in cyclical TS construction,
    as a list of vector couples specifying starting and ending point of each pivot 
    vector. For bimolecular TSs, returns vertices for the centered superposition of
    two segments. For trimolecular TSs, returns triangle vertices.

    :params vertices: list of floats, used as polygon side lenghts.
    :return vertices_out: list of vectors couples (start, end)
    '''
    assert len(lengths) in (2,3)

    arr = np.zeros((len(lengths),2,3))

    if len(lengths) == 2:

        arr[0,0] = np.array([-lengths[0]/2,0,0])
        arr[0,1] = np.array([+lengths[0]/2,0,0])
        arr[1,0] = np.array([-lengths[1]/2,0,0])
        arr[1,1] = np.array([+lengths[1]/2,0,0])

        vertices_out = np.vstack(([arr],[arr]))
        vertices_out[1,1] *= -1

    else:
        
        if not all([lengths[i] < lengths[i-1] + lengths[i-2] for i in (0,1,2)]): 
            raise TriangleError(f'Impossible to build a triangle with sides {lengths}')
            # check that we can build a triangle with the specified vectors

        arr[0,1] = np.array([lengths[0],0,0])
        arr[1,0] = np.array([lengths[0],0,0])

        a = np.power(lengths[0], 2)
        b = np.power(lengths[1], 2)
        c = np.power(lengths[2], 2)
        x = (a-b+c)/(2*a**0.5)
        y = (c-x**2)**0.5

        arr[1,1] = np.array([x,y,0])
        arr[2,0] = np.array([x,y,0])

        vertices_out = np.vstack(([arr],[arr],[arr],[arr],
                                  [arr],[arr],[arr],[arr]))

        swaps = [(1,2),(2,1),(3,1),(3,2),(4,0),(5,0),(5,1),(6,0),(6,2),(7,0),(7,1),(7,2)]

        for t,v in swaps:
            # triangle, vector couples to be swapped
            vertices_out[t,v][[0,1]] = vertices_out[t,v][[1,0]]

    return vertices_out

def ase_view(mol):
    '''
    Display an Hypermolecule instance from the ASE GUI
    '''
    from ase import Atoms
    from ase.gui.gui import GUI
    from ase.gui.images import Images

    if hasattr(mol, 'reactive_atoms_classes_dict'):
        images = []

        for c, coords in enumerate(mol.atomcoords):
            centers = np.vstack([atom.center for atom in mol.reactive_atoms_classes_dict[c].values()])
            atomnos = np.concatenate((mol.atomnos, [0 for _ in centers]))
            totalcoords = np.concatenate((coords, centers))
            images.append(Atoms(atomnos, positions=totalcoords))

    else:
        images = [Atoms(mol.atomnos, positions=coords) for coords in mol.atomcoords]
        
    try:
        GUI(images=Images(images), show_bonds=True).run()
    except TclError:
        print('--> GUI not available from command line interface. Skipping it.')

double_bonds_thresholds_dict = {
    'CC':1.4,
    'CN':1.3,
}

def get_double_bonds_indices(coords, atomnos):
    '''
    Returns a list containing 2-elements tuples
    of indices involved in any double bond
    '''
    mask = (atomnos != 1)
    numbering = np.arange(len(coords))[mask]
    coords = coords[mask]
    atomnos = atomnos[mask]
    output = []

    for i1, _ in enumerate(coords):
        for i2 in range(i1+1, len(coords)):
            dist = norm_of(coords[i1] - coords[i2])
            tag = ''.join(sorted([pt[atomnos[i1]].symbol,
                                  pt[atomnos[i2]].symbol]))
            
            threshold = double_bonds_thresholds_dict.get(tag)
            if threshold is not None and dist < threshold:
                output.append((numbering[i1], numbering[i2]))

    return output

def get_scan_peak_index(energies, max_thr=50, min_thr=0.1):
    '''
    Returns the index of the energies iterable that
    corresponds to the most prominent peak.
    '''
    _l = len(energies)
    peaks = [i for i in range(_l) if (

        energies[i-1] < energies[i] >= energies[(i+1)%_l] and
        max_thr > energies[i] > min_thr
        # discard peaks that are too small or too big
    )]

    if not peaks:
        return energies.index(max(energies))
    # if no peaks are present, return the highest

    if len(peaks) == 1:
        return peaks[0]
    # if one is present, return that

    peaks_nrg = [energies[i] for i in peaks]
    return energies.index(max(peaks_nrg))
    # if more than one, return the highest

def molecule_check(old_coords, new_coords, atomnos, max_newbonds=0):
    '''
    Checks if two molecules have the same bonds between the same atomic indices
    '''
    old_bonds = {(a, b) for a, b in list(graphize(old_coords, atomnos).edges) if a != b}
    new_bonds = {(a, b) for a, b in list(graphize(new_coords, atomnos).edges) if a != b}

    delta_bonds = (old_bonds | new_bonds) - (old_bonds & new_bonds)

    if len(delta_bonds) > max_newbonds:
        return False

    return True

def scramble_check(TS_structure, TS_atomnos, excluded_atoms, mols_graphs, max_newbonds=0, logfunction=None, title=None) -> bool:
    '''
    Check if a multimolecular arrangement has scrambled during some optimization
    steps. If more than a given number of bonds changed (formed or broke) the
    structure is considered scrambled, and the method returns False.
    '''
    assert len(TS_structure) == sum([len(graph.nodes) for graph in mols_graphs])

    bonds = set()
    for i, graph in enumerate(mols_graphs):

        pos = sum([len(other_graph.nodes) for j, other_graph in enumerate(mols_graphs) if j < i])
        
        for bond in [tuple(sorted((a+pos, b+pos))) for a, b in list(graph.edges) if a != b]:
            bonds.add(bond)
    # creating bond set containing all bonds present in the desired transition state

    new_bonds = {tuple(sorted((a, b))) for a, b in list(graphize(TS_structure, TS_atomnos).edges) if a != b}
    delta_bonds = (bonds | new_bonds) - (bonds & new_bonds)
    # delta_bonds -= {tuple(sorted(pair)) for pair in constrained_indices}

    for bond in delta_bonds.copy():
        for a in excluded_atoms:
            if a in bond:
                delta_bonds -= {bond}
    # removing bonds involving constrained atoms: they are not counted as scrambled bonds

    if len(delta_bonds) > max_newbonds:
        if logfunction is not None:
            logfunction(f"{title}, scramble_check - found {len(delta_bonds)} extra bonds: {delta_bonds}")
        return False

    return True

def rotate_dihedral(coords, dihedral, angle, mask=None, indices_to_be_moved=None):
    '''
    Rotate a molecule around a given bond.
    Atoms that will move are the ones
    specified by mask or indices_to_be_moved.
    If both are None, only the first index of
    the dihedral iterable is moved.

    angle: angle, in degrees
    '''

    i1, i2, i3 ,_ = dihedral

    if indices_to_be_moved is not None:
        mask = np.array([i in indices_to_be_moved for i, _ in enumerate(coords)])

    if mask is None:
        mask = i1

    axis = coords[i2] - coords[i3]
    mat = rot_mat_from_pointer(axis, angle)
    center = coords[i3]

    coords[mask] = (mat @ (coords[mask] - center).T).T + center

    return coords

def flatten(array, typefunc=float):
    out = []
    def rec(_l):
        for e in _l:
            if type(e) in [list, tuple, np.ndarray]:
                rec(e)
            else:
                out.append(typefunc(e))
    rec(array)
    return out

def auto_newline(string, max_line_len=50, padding=2):
    string = str(string)

    out = [' '*padding]
    line_len = 0
    for word in string.split():
        out.append(word)
        line_len += len(word) + 1

        if line_len >= max_line_len:
            out.append('\n'+' '*padding)
            line_len = 0

    return ' '.join(out)

def smi_to_3d(smi, new_filename):
    with open("temp_smi.txt", "w") as f:
        f.write(smi)

    check_call(f'obabel -i smi temp_smi.txt -o xyz -O {new_filename}.xyz -h --gen3d'.split(), stdout=DEVNULL, stderr=STDOUT)
    # data = read_xyz(f"{new_filename}.xyz")
    clean_directory(["temp_smi.txt"])

    return new_filename + ".xyz"

def timing_wrapper(function, *args, payload=None, **kwargs):
    '''
    Generic function wrapper that appends the
    execution time at the end of return.
    If payload is not None, appends it at the end
    of the function return, before the elapsed time.
    
    '''
    start_time = time.perf_counter()
    func_return = function(*args, **kwargs)
    elapsed = time.perf_counter() - start_time

    if payload is None:
        return func_return, elapsed
    
    return func_return, payload, elapsed

def _saturation_check(atomnos, charge=0):

    transition_metals = [
                    "Sc", "Ti", "V", "Cr", "Mn", "Fe",
                    "Co", "Ni", "Cu", "Zn", "Y", "Zr",
                    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
                    "Ag", "Cd", "La", "Ce", "Pr", "Nd",
                    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
                    "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
                    "Ta", "W", "Re", "Os", "Ir", "Pt",
                    "Au", "Hg", "Th", "Pa", "U", "Np",
                    "Pu", "Am", 
    ]

    # if we have any transition metal, it's hard to tell
    # if the structure looks ok: in this case we assume it is.
    organometallic = any([pt[a].symbol in transition_metals for a in atomnos])

    odd_valent = [  #1 valent
                    "H", "Li", "Na", "K", "Rb", "Cs",
                    "F", "Cl", "Br", "I", "At",

                    # 3/5 valent
                    "N", "P", "As", "Sb", "Bi",
                    "B", "Al", "Ga", "In", "Tl",
                 ]

    n_odd_valent = sum([1 for a in atomnos if pt[a].symbol in odd_valent])
    looks_ok = ((n_odd_valent + charge) / 2) % 1 < 0.001 if not organometallic else True

    return looks_ok
