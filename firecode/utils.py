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

import os
import re
import shutil
import sys
import time
from shutil import rmtree
from subprocess import CalledProcessError, getoutput, run

import numpy as np
from networkx import shortest_path
from prism_pruner.algebra import normalize, rot_mat_from_pointer
from prism_pruner.conformer_ensemble import ConformerEnsemble
from prism_pruner.graph_manipulations import graphize
from prism_pruner.rmsd import rmsd_and_max

from firecode.algebra import norm_of, point_angle
from firecode.errors import TriangleError
from firecode.pt import pt
from firecode.units import EH_TO_KCAL


class Constraint:
    """Constraint class with indices, type and value attributes.
    
    """

    def __init__(self, indices, value=None):

        self.indices = indices

        self.type = {
            2 : 'B',
            3 : 'A',
            4 : 'D',
        }[len(indices)]

        self.value = value

class suppress_stdout_stderr(object):
    """A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    """

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

def clean_directory(
        to_remove: list | None =None,
        to_remove_startswith: list | None = None,
        to_remove_endswith: list | None = None,
        to_remove_contains: list | None = None,
    ) -> None:
    """Cleans the current directory from temporary files created during a run.

    """
    if to_remove is not None:
        for name in to_remove:
            try:
                os.remove(name)
            except IsADirectoryError:
                rmtree(os.path.join(os.getcwd(), name))
            except FileNotFoundError:
                pass

    to_remove_startswith = to_remove_startswith or []
    to_remove_endswith = to_remove_endswith or []
    to_remove_contains = to_remove_contains or []
    for f in os.listdir():
        if (
            f.startswith(('temp', *to_remove_startswith)) or
            f.endswith(('temp', *to_remove_endswith)) or
            any([s in f for s in to_remove_contains])
            ):
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

def write_xyz(atoms:np.array, coords:np.array, output, title='temp'):
    """Output is of _io.TextIOWrapper type

    """
    assert atoms.shape[0] == coords.shape[0]
    assert coords.shape[1] == 3
    string = ''
    string += str(len(coords))
    string += f'\n{title}\n'
    for atom, coord in zip(atoms, coords):
        string += '%s     % .6f % .6f % .6f\n' % (atom, coord[0], coord[1], coord[2])
    output.write(string)

def read_xyz(filename):
    """Wrapper for PRISM's ConformerEnsemble xyz reader, adding FIRECODE support.
    
    Raises an error if unsuccessful.

    """
    mol = ConformerEnsemble.from_xyz(filename)
    mol.atomnos = np.array([pt.number(letter) for letter in mol.atoms])

    assert mol is not None, f'Reading molecule {filename} failed - check its integrity.'
    return mol

def read_xyz_energies(filename, verbose=True):
    """Read energies from a .xyz file. Returns None or an array of floats (in Hartrees).
    """
    energies = None

    # get lines right after the number of atom, which should contain the energy
    comment_lines = getoutput(f'grep -A1 "^[[:space:]]*[0-9]\\+$" {filename} | grep -v "^[[:space:]]*[0-9]\\+$" | grep -v "^--$"').split("\n")

    if len(comment_lines[0].split()) == 1:
        if set(comment_lines[0].split()[0]).issubset('0123456789.-'):
            # only one energy found with no UOM, assume it's in Eh
            energies = [float(e.split()[0].strip()) for e in comment_lines]

            if verbose:
                print(f'--> Read {len(energies)} energies from {filename} (single number, no UOM: assuming Eh units).')

        elif verbose:
            print(f'--> Could not parse energies for {filename} - skipping.')

    else:
        # multiple energies found, parse units
        hartree_matches = re.findall(r'-*\d+.\d+\sEH', comment_lines[0].upper())
        kcal_matches = re.findall(r'-*\d+.\d+\sKCAL/MOL', comment_lines[0].upper())
        number_matches = re.findall(r'-*\d+.\d+', comment_lines[0])

        if hartree_matches:
            energies = [float(re.findall(r'-*\d+.\d+\sEH', e.upper())[0].split()[0].strip()) for e in comment_lines]
            if verbose:
                print(f'--> Read {len(comment_lines)} energies from {filename} (first number followed by Eh units).')

        elif kcal_matches:
            energies = [float(re.findall(r'-*\d+.\d+\sKCAL/MOL', e.upper())[0].split()[0].strip())/EH_TO_KCAL for e in comment_lines]
            if verbose:
                print(f'--> Read {len(comment_lines)} energies from {filename} (first number followed by kcal/mol units).')

        # last resort, parse the first thing that looks like an energy and assume it's in Eh
        elif number_matches:
            energies = [float(re.findall(r'-*\d+.\d+', e)[0].strip()) for e in comment_lines]
            if verbose:
                print(f'--> Read {len(comment_lines)} energies from {filename} (first number, no UOM: assuming Eh units).')

        elif verbose:
            print(f'--> Could not parse energies for {filename} - skipping.')

    return energies

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
    """Find the rotation matrix that aligns vec1 to vec2
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
    """Returns coordinates for the polygon vertices used in cyclical TS construction,
    as a list of vector couples specifying starting and ending point of each pivot 
    vector. For bimolecular TSs, returns vertices for the centered superposition of
    two segments. For trimolecular TSs, returns triangle vertices.

    :params vertices: list of floats, used as polygon side lenghts.
    :return vertices_out: list of vectors couples (start, end)
    """
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
    """Display an Hypermolecule instance from the ASE GUI
    """
    from ase import Atoms
    from ase.gui.gui import GUI
    from ase.gui.images import Images

    if hasattr(mol, 'reactive_atoms_classes_dict'):
        images = []

        for c, coords in enumerate(mol.coords):
            centers = np.vstack([atom.center for atom in mol.reactive_atoms_classes_dict[c].values()])
            totalcoords = np.concatenate((coords, centers))
            images.append(Atoms(mol.atoms, positions=totalcoords))

    else:
        images = [Atoms(mol.atoms, positions=coords) for coords in mol.coords]

    try:
        GUI(images=Images(images), show_bonds=True).run()
    # except TclError:
    except Exception:
        print('--> GUI not available from command line interface. Skipping it.')

def get_scan_peak_index(energies, max_thr=50, min_thr=0.1):
    """Returns the index of the energies iterable that
    corresponds to the most prominent peak.
    """
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

def molecule_check(atoms, old_coords, new_coords, max_newbonds=0):
    """Checks if two molecules have the same bonds between the same atomic indices
    """
    old_bonds = {(a, b) for a, b in list(graphize(atoms, old_coords).edges) if a != b}
    new_bonds = {(a, b) for a, b in list(graphize(atoms, new_coords).edges) if a != b}

    delta_bonds = (old_bonds | new_bonds) - (old_bonds & new_bonds)

    if len(delta_bonds) > max_newbonds:
        return False

    return True

def scramble_check(embedded_atoms, embedded_structure, excluded_atoms, mols_graphs, max_newbonds=0, logfunction=None, title=None) -> bool:
    """Check if a multimolecular arrangement has scrambled during some optimization
    steps. If more than a given number of bonds changed (formed or broke) the
    structure is considered scrambled, and the method returns False.
    """
    assert len(embedded_structure) == sum([len(graph.nodes) for graph in mols_graphs])

    bonds = set()
    for i, graph in enumerate(mols_graphs):

        pos = sum([len(other_graph.nodes) for j, other_graph in enumerate(mols_graphs) if j < i])

        for bond in [tuple(sorted((a+pos, b+pos))) for a, b in list(graph.edges) if a != b]:
            bonds.add(bond)
    # creating bond set containing all bonds present in the desired molecular assembly

    new_bonds = {tuple(sorted((a, b))) for a, b in list(graphize(embedded_atoms, embedded_structure).edges) if a != b}
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

def set_planar_angle(coords, indices, target, graph):
    """Modifies a planar angle, setting the angle
    value to target degrees. Moves the part
    of the molecule attached to the last of
    the three indices defining the angle.

    """
    assert len(indices) == 3

    # define points, axis of rotation and center
    i1, i2 ,i3 = indices
    p1 ,p2, p3 = coords[np.array(indices)]
    delta = target - point_angle(p1, p2, p3)

    rot_axis = np.cross(p1-p2, p3-p2)
    rot_mat = rot_mat_from_pointer(rot_axis, delta)
    center = p2

    # define indices to be moved through graph connectivity
    # (faster to modify graph than copying it)
    graph.remove_edge(i2, i3)

    # get all indices reachable from i3 not going through i2-i3
    indices_to_be_moved = shortest_path(graph, i3).keys()

    # restore modified graph
    graph.add_edge(i2, i3)

    # get rotation mask
    mask = np.array([i in indices_to_be_moved for i, _ in enumerate(coords)])

    # center coordinates, rotate around axis, revert centering
    coords[mask] = (rot_mat @ (coords[mask] - center).T).T + center

    return coords

def set_distance(coords, indices, target, graph):
    """Modifies a distance, setting the
    value to target Angström. Moves the part
    of the molecule attached to the last of
    the two indices defining the distance.

    """
    assert len(indices) == 2

    # define points, axis of rotation and center
    i1, i2 = indices
    p1 ,p2 = coords[np.array(indices)]
    delta = target - norm_of(p1-p2)
    versor = normalize(p2-p1)

    # define indices to be moved through graph connectivity
    # (faster to modify graph than copying it)
    graph.remove_edge(i1, i2)

    # get all indices reachable from i2 not going through i1-i2
    indices_to_be_moved = shortest_path(graph, i2).keys()

    # restore modified graph
    graph.add_edge(i1, i2)

    # get mask
    mask = np.array([i in indices_to_be_moved for i, _ in enumerate(coords)])

    # translate coordinates
    coords[mask] += versor * delta

    return coords

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

def timing_wrapper(function, *args, payload=None, **kwargs):
    """Generic function wrapper that appends the
    execution time at the end of return.
    If payload is not None, appends it at the end
    of the function return, before the elapsed time.
    
    """
    start_time = time.perf_counter()
    func_return = function(*args, **kwargs)
    elapsed = time.perf_counter() - start_time

    if payload is None:
        return func_return, elapsed

    return func_return, payload, elapsed

def saturation_check(atoms, charge=0):

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
    organometallic = any([el in transition_metals for el in atoms])

    odd_valent = [  #1 valent
                    "H", "Li", "Na", "K", "Rb", "Cs",
                    "F", "Cl", "Br", "I", "At",

                    # 3/5 valent
                    "N", "P", "As", "Sb", "Bi",
                    "B", "Al", "Ga", "In", "Tl",
                 ]

    n_odd_valent = sum([1 for a in atoms if a in odd_valent])
    looks_ok = ((n_odd_valent + charge) / 2) % 1 < 0.001 if not organometallic else True

    return looks_ok

def rmsd_similarity(ref, structures, rmsd_thr=0.5) -> bool:
    """Simple, RMSD similarity eval function.

    """
    # iterate over target structures
    for structure in structures:

        # compute RMSD and max deviation
        rmsd_value, maxdev_value = rmsd_and_max(ref, structure)

        if rmsd_value < rmsd_thr and maxdev_value < 2 * rmsd_thr:
            return True

    return False


class NewFolderContext:
    """Context manager: creates a new directory and moves into it on entry.
    
    On exit, moves out of the directory and deletes it if instructed to do so.
     
    """

    def __init__(self, new_folder_name, delete_after=True):
        self.new_folder_name = new_folder_name
        self.delete_after = delete_after

    def __enter__(self):
        # create working folder and cd into it
        if self.new_folder_name in os.listdir():
            shutil.rmtree(os.path.join(os.getcwd(), self.new_folder_name))

        os.mkdir(self.new_folder_name)
        os.chdir(os.path.join(os.getcwd(), self.new_folder_name))

    def __exit__(self, *args):
        # get out of working folder
        os.chdir(os.path.dirname(os.getcwd()))

        # and eventually delete it
        if self.delete_after:
            shutil.rmtree(os.path.join(os.getcwd(), self.new_folder_name))


class FolderContext:
    """Context manager: works in the specified directory and moves back after.
        
    """

    def __init__(self, target_folder):
        self.target_folder = os.path.join(os.getcwd(), target_folder)
        self.initial_folder = os.getcwd()

    def __enter__(self):
        # move into folder
        if os.path.isdir(self.target_folder):
            os.chdir(self.target_folder)

        else:
            raise NotADirectoryError(self.target_folder)

    def __exit__(self, *args):
        # get out of working folder
        os.chdir(self.initial_folder)
