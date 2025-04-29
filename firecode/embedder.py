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
import logging
import os
import pickle
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from getpass import getuser
from itertools import groupby

import numpy as np
from psutil import virtual_memory

from firecode.__main__ import __version__
from firecode.algebra import dihedral, norm_of, point_angle
from firecode.ase_manipulations import ase_saddle
from firecode.calculators._xtb import (xtb_metadyn_augmentation, xtb_opt,
                                       xtb_pre_opt)
from firecode.embedder_options import Options, OptionSetter, keywords_dict
from firecode.embeds import (_get_monomolecular_reactive_indices,
                             cyclical_embed, monomolecular_embed, string_embed)
from firecode.errors import (InputError, NoOrbitalError, SegmentedGraphError,
                             ZeroCandidatesError)
from firecode.graph_manipulations import get_sum_graph
from firecode.hypermolecule_class import Hypermolecule, Pivot, align_by_moi
from firecode.multiembed import multiembed_dispatcher
from firecode.nci import get_nci
from firecode.numba_functions import (compenetration_check, count_clashes,
                                      prune_conformers_tfd, scramble)
from firecode.operators import operate
from firecode.optimization_methods import Opt_func_dispatcher, fitness_check
from firecode.parameters import orb_dim_dict
from firecode.pruning import (prune_by_moment_of_inertia, prune_by_rmsd,
                              prune_by_rmsd_rot_corr)
from firecode.pt import pt
from firecode.references import references
from firecode.rmsd import rmsd_and_max_numba
from firecode.settings import DEFAULT_LEVELS, PROCS
from firecode.torsion_module import _get_quadruplets, csearch
from firecode.utils import (Constraint, _saturation_check, align_structures,
                            ase_view, auto_newline, cartesian_product,
                            clean_directory, graphize, loadbar, scramble_check,
                            time_to_string, timing_wrapper, write_xyz)


class Embedder:
    '''
    Embedder class, containing all methods to set attributes,
    options and initialize the calculation
    '''

    def __init__(self, filename, stamp=None, procs=None):
        '''
        Initialize the Embedder object by reading the input filename (.txt).
        Sets the Option dataclass properties to default and then updates them
        with the user-requested keywords, if there are any.

        '''

        self.t_start_run = time.perf_counter()
        
        parent_dir = os.path.dirname(filename)
        if parent_dir != '':
            os.chdir(parent_dir)

        if stamp is None:
            self.stamp = time.ctime().replace(' ','_').replace(':','-')[4:-8]
            # replaced ctime yields 'Sun_May_23_18-53-47_2021', only keeping 'May_23_18-53'

        else:
            self.stamp = stamp

        self.avail_cpus = len(os.sched_getaffinity(0))
        self.avail_mem_gb = virtual_memory().available/1E9

        try:
            from torch.cuda import device_count
            self.avail_gpus = device_count()
        except ImportError:
            self.avail_gpus = 'N/A'

        self.procs = int(procs) if procs is not None else PROCS or 4

        try:
            os.remove(f'firecode_{self.stamp}.log')

        except FileNotFoundError:
            pass

        log_filename = f'firecode_{self.stamp}.log'
        self.logfile = open(log_filename, 'a', buffering=1, encoding="utf-8")
        logging.basicConfig(filename=log_filename, filemode='a')

        try:

            self.write_banner_and_info()
            # Write banner to log file

            self.options = Options()
            # initialize option subclass

            self.embed = None
            self.warnings = []
            # initialize embed type variable and warnings list

            inp = self._parse_input(filename)
            # collect information about molecule files

            self.objects = [Hypermolecule(name, c_ids) for name, c_ids in inp]
            # load designated molecular files

            # self.objects.sort(key=lambda obj: len(obj.atomcoords[0]), reverse=True)
            # sort them in descending number of atoms (not for now - messes up pairings)

            self.ids = np.array([len(mol.atomnos) for mol in self.objects])
            # Compute length of each molecule coordinates. Used to divide molecules in TSs

            self.graphs = [mol.graph for mol in self.objects]
            # Store molecular graphs

            self._read_pairings()
            # read imposed pairings from input file [i.e. mol1(6)<->mol2(45)]

            self.check_objects_compenetration()
            # make sure the input structures look alright

            self._set_options(filename)
            # read the keywords line and set the relative options
            # then read the operators and store them 

            self.check_saturation()
            # make sure that structures look nice and correct

            self._calculator_setup()
            # initialize default or specified calculator

            self._print_references()
            # based on the data collected from setup

            self._apply_operators()
            # execute the operators, replacing the self.objects molecule

            self._setup()
            # setting embed type and getting ready to embed (if needed)

            if self.options.debug:
                for mol in self.objects:
                    if hasattr(mol, 'reactive_atoms_classes_dict'):
                        if len(mol.reactive_atoms_classes_dict[0]) > 0:
                            mol.write_hypermolecule()
                            self.debuglog(f'DEBUG: written hypermolecule file for ({mol.filename})')
                self.log()

            if self.options.check_structures:
                self._inspect_structures()

        except Exception as e:
            logging.exception(e)
            raise e

    def log(self, string="", p=True):
        if p:
            print(string)
        string += '\n'
        self.logfile.write(string)

    def debuglog(self, string=""):
        if self.options.debug:
            string += '\n'
            # self.logfile.write(string)
            self.debug_logfile.write(string)

    def warn(self, string):
        self.warnings.append(string)
        self.log(string)

    def write_banner_and_info(self):
        '''
        Write banner to log file, containing program and run info
        '''
        
        banner = '''
                 .           .           *             *
                                                    ▒           ..
       *      .     ▒     ..      *       ▒        ▒░▒     * 
                   ▒░▒                 .           ▒░░▒    .  .      ..
 *    .      ▒    ▒░░░▒       ▒                ▒    ▒░░▒
   +        ▒  ▒   ▒░░░░▒           .   ..  ▒          ▒░░▒           *     +
        .         ▒░ ░░▒  ▒              ▒░▒     *   ▒░░░▒    ▒
    *         ▒  ▒░░░░░░▒▒░▒    *      ▒▒░░▒   ▒     ▒░░░░▒  ▒        ..      .
 .        ▒      ▒░░  ░░░░ ▒          ▒░░░▒   ▒░▒▒  ▒░░ ░░░▒▒▒▒
    .   ▒▒  ▒  ▒░░░░░░░░░░░▒    ..  ▒░ ░▒   ▒░░░░▒ ▒░░░░░░░░░ ▒▒  *         +
       ▒▒░░▒    ▒░░░░ ░▒░░░░▒       ▒░░░░░▒ ▒░░ ░░░▒░░░░░▒░░░░▒▒▒      *
 *     ▒▒░░░░▒  ▒░░░░░░░░░░░░░▒  .   ▒░░░░░░▒░░░░░░░░░░░  ░░░░░░░░▒▒       +
      ▒░░░░░░░▒░░░  ░░░░░░░░░░░▒ ▒ ▒░ ░░░░░░░  ░░░░▒░░░ ░░░░░░░ ░▒   
    *▒░░░░ ░░░░░░░░░░▒░░░░░ ░░░░▒░▒░░░░░░░▒░░░ ░░░░░░░  ░░░▒░░░░░▒    ▒
 .  ▒░ ░░░░░░░░░░░ ░░░░░░░░░░░░░░░░░░░ ░░░░░░░░░░ ░░░░░░░░░░░░░░░░░▒  .░▒  .   +  
    ▒░░███████╗██╗██████╗░███████╗░█████╗░░█████╗░██████╗░███████╗░░░░░▒
   ▒ ░░██╔════╝██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝░░░░▒      +
▒  ▒░░░█████╗░░██║██████╔╝█████╗░░██║░░╚═╝██║░░██║██║░░██║█████╗░░░░░▒     *
   ▒░ ░██╔══╝░░██║██╔══██╗██╔══╝░░██║░░██╗██║░░██║██║░░██║██╔══╝░░░░▒
   ▒░░░██║░░░░░██║██║░░██║███████╗╚█████╔╝╚█████╔╝██████╔╝███████╗░░░▒
.  ▒░░░╚═╝░░ ░░╚═╝╚═╝░░╚═╝╚══════╝░╚════╝░░╚════╝░╚═════╝░╚══════╝░░░▒ * ▒  ..  
    ▒ ░░░░░░ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ░░░░░░░░░░░░░░▒  .▒░▒   .  
  ▒ ▒░░░░░ ░░▒░░░ Filtering Refiner and Embedder for ░░░ ░░▒░░ ░░░░░░░▒░░▒                         
    ▒░░░░▒░░░░░░░░ Conformationally Dense Ensembles ░░ ░░░░░░░  ░ ░░░░░░░░▒
     .▒░░░░░░░░ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ░▒░░░ ░░░░▒ *  ▒
.      ▒░░░▒░╔═════════════════════════════════════════════╗░░░ ░░░ ░░░░▒    
         ▒░ ░║                                             ║░▒░░ ░░░░░▒ 
 +▒    ▒░░░░░║        nicolo.tampellini@yale.edu           ║░░░░░░░░▒    ▒
        ▒░░░░║                                             ║░░ ░░░▒      
 ..  ▒  ▒░▒░░║    Version        🔥{0:^24}║░▒░░░ ░░░▒    *      .               
   .  ▒ ░░░░░║    User           🔥{1:^24}║░░░░░░░░▒  +             
       ▒░░ ▒░║    Current Time   🔥{2:^24}║░░ ░░▒    ▒    +    ..                          
 .. ▒ * ▒░ ░░║    Avail CPUs     🔥{3:^24}║░░░░░░░▒           *
    .   ▒░ ░░║    Avail GPUs     🔥{4:^24}║ ░░░░▒    .     ..
      .▒░▒▒░░║    Avail Memory   🔥{5:^24}║░░▒              
  +  .. ▒░░ ░║                                             ║░▒  .. .   
    .    ▒ ░░╚═════════════════════════════════════════════╝░▒     +     
 .     *  ▒░░░░░░  ░░▒░░░░░▒▒░░░▒▒ ░░░░░░░▒░░ ░░░▒░░▒░░░░░▒░░░░▒      .
     .      ▒░░▒░▒░░░ ░░░▒░░▒░ ░░░░░░▒░░ ░░░░░░▒▒▒░ ░░▒░░░░░░▒ .    
               ▒░ ░░░░░ ░░░░░▒░░░░▒░░░░░░░ ▒░░░░░░   ░░░░▒
                    ░░░░   ░░░  ░      ░░     ░ ░     ░░
               
               '''.format(__version__,
                      getuser(),
                      time.ctime()[0:-8],
                      self.avail_cpus,
                      self.avail_gpus,
                      str(round(self.avail_mem_gb, 1))+' GB')
        # 🔥
        
        
#         banner = '''
#        +   .     ____________________________________ .     .    
#     *    .   .. /────────────────────────────────────\   *     .  
#  .     ..   +  /▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒ \ .   .   +  
#    +       ▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒ . ..   .  
#      .  ▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒  .   *  
#        ▒░████████╗░██████╗░█████╗░░█████╗░██████╗░███████╗░░░▒  .  .  
#    +   ▒░╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝░░░▒   ..  
#  ..  . ▒░░░░██║░░░╚█████╗░██║░░╚═╝██║░░██║██║░░██║█████╗░░░░░▒ *    +   
#    .   ▒░░░░██║░░░░╚═══██╗██║░░██╗██║░░██║██║░░██║██╔══╝░░░░░▒   .   .
# .       ▒░░░██║░░░██████╔╝╚█████╔╝╚█████╔╝██████╔╝███████╗░░░▒ ..   +   
#  *  .  / ▒░░╚═╝░░░╚═════╝░░╚════╝░░╚════╝░╚═════╝░╚══════╝░░▒ \ .  ..  
#   ..  /   ▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒   \   .   
# .    /    ▒░░╔══════════════════════════════════════════╗░░▒    \ +    
#     /      ▒░║  Transition State Conformational Docker  ║░▒      \ ..  
#  +  \\\     ▒░║        nicolo.tampellini@yale.edu        ║░▒     //    .  
#      \\\    ▒░║                                          ║░▒    //  .       
#  ..   \\\   ▒░║     Version    >{0:^25}║░▒   // .  *                                    
#    .   \\\  ▒░║      User      >{1:^25}║░▒  //   .                                     
#         \\\ ▒░║      Time      >{2:^25}║░▒ // *   .                                                      
#  ..   *  \\\▒░║      Procs     >{3:^25}║░▒//   ..            
#     .     \▒░║     Threads    >{4:^25}║░▒/  +              
#       .    ▒░║    Avail CPUs  >{5:^25}║░▒ .   ..                            
#   +  .. .  ▒░╚══════════════════════════════════════════╝░▒  .. .   
#     .       ▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒     .     
#  .     *  +   \\\ ▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒ //  .      .
#      .      .  \\\____________________________________// .   .    
#            '''

        # ⏣█▓▒░ banner art adapted from https://fsymbols.com/generators/tarty/

        self.log(banner)

    def _print_references(self):
        '''
        Print relevant literature references based on the run settings

        '''

        self.log('\n--> If you use FIRECODE in your publication, please cite this reference in the main text:\n' +
                 f'    {references["FIRECODE"]}')
        
        cite_ff = self.options.ff_calc == "XTB"
        cite_gfn2 = self.options.calculator == "XTB"
        cite_crest = any(("mtd>" in op or "mtd_search>" in op) for op in self.options.operators)

        if any((cite_ff, cite_gfn2, cite_crest)):
            s = ''
            s += f"    GFN-FF : {references['GFN-FF']}\n" if cite_ff else ""
            s += f"    GFN2-XTB : {references['GFN2-XTB']}\n" if cite_gfn2 else ""
            s += f"    CREST : {references['CREST']}\n" if cite_crest else ""

            self.log(f'\n--> Your run also makes use of this other software: please cite these references as well.\n{s}')

    def _parse_input(self, filename):
        '''
        Reads a textfile and sets the Embedder properties for the run.
        Keywords are read from the first non-comment(#), non-blank line
        if there are any, and molecules are read afterward.

        '''

        with open(filename, 'r') as f:
            lines = f.readlines()

        # write a formatted copy of the input file to the log
        self.log(f'--> Input file: {filename}\n')
        longest = max(len(line.rstrip('\n')) for line in lines)
        self.log('    '+'-'*(longest+6))
        for _l, line in enumerate(lines):
            self.log(f'{_l+1:2}> | '+line.rstrip('\n').ljust(longest)+'   |')
        self.log('    '+'-'*(longest+6)+'\n')

        # Remove comments
        lines = [line.split('#')[0].rstrip() for line in lines]

        # start parsing: get rid of comment lines and blank lines
        lines = [line.replace(', ',',') for line in lines if line != '' and line[0] not in ('\n')]
        
        def _remove_internal_constraints(string):
            numbers = [int(re.sub('[^0-9]', '', i)) for i in string]
            letters = [re.sub('[^A-Za-z]', '', i) for i in string]
            count = [letters.count(_l) if (_l != '') else 1 for _l in letters]
            return tuple([n for n, c in zip(numbers, count) if c == 1])

        try:

            keywords = [_l.split('=')[0] if '(' not in _l else _l.split('(')[0] for _l in lines[0].split()]
            if any(k.upper() in keywords_dict.keys() for k in keywords):
                self.kw_line, *self.mol_lines = lines
            else:
                self.mol_lines = lines

            inp = []
            for _l, line in enumerate(self.mol_lines):

                # if line starts with a space, it must be a constraint or an attribute.
                # Skip for now, we'll deal with it later in _read_pairings
                if line[0] == " ":
                    continue

                if '>' in line:
                    self.options.operators_dict[_l] = [op.rstrip().lstrip() for op in reversed(line.rstrip('\n').split('>')[:-1])]
                    self.options.operators.append(line.rstrip('\n'))
                    line = line.split('>')[-1].lstrip()
                    # record that we will need to perform these operations before the run

                filename, *reactive_atoms = line.split()

                if reactive_atoms:
                    # remove attributes from reactive indices (i.e. charge=1)
                    reactive_atoms = [fragment for fragment in reactive_atoms if '=' not in fragment]

                    # remove internal constraints from reactive indices
                    reactive_indices = _remove_internal_constraints(reactive_atoms)
                else:
                    reactive_indices = None

                inp.append((filename, reactive_indices))

            return inp
            
        except Exception as e:
            print(e)
            raise InputError(f'Error in reading molecule input for {filename}. Please check your syntax.')

    def check_saturation(self):
        '''
        Check each loaded object and make sure it looks nice and correct
        
        '''
        self.log()
        for mol in self.objects:
            charge = int(mol.charge) if hasattr(mol, "charge") else self.options.charge
            
            if _saturation_check(mol.atomnos, charge):
                self.log(f"--> {mol.filename}: saturation check passed (even saturation index)")

            else:
                self.warn(f"--> WARNING! {mol.filename}: saturation check failed. Odd saturation index (charge={charge}). Radical or bad input geometry?")

    def check_objects_compenetration(self):
        '''
        Checks that the input molecules look alright
        
        '''
        for mol in self.objects:
            for c, coords in enumerate(mol.atomcoords):
                if not compenetration_check(coords):
                    clashes = count_clashes(coords)
                    self.warn(f"--> WARNING! {mol.filename}, conformer {c+1}, looks compenetrated ({clashes} interatomic distance{'s' if clashes > 1 else ''} < 0.5 A)")

    def _set_options(self, filename):
        '''
        Set the options dataclass parameters through the OptionSetter class,
        from a list of given keywords. These will be used during the run to
        vary the search depth and/or output.
        '''
      
        try:
            option_setter = OptionSetter(self)
            option_setter.set_options()

        except SyntaxError as e:
            raise e

        except Exception as e:
            print(e)
            raise InputError(f'Error in reading keywords from {filename}. Please check your syntax.')

    def _set_reactive_atoms_cumnums(self):

        if self.embed in ('cyclical', 'chelotropic', 'string'):
            for i, mol in enumerate(self.objects):

                if not hasattr(mol, 'reactive_atoms_classes_dict'):
                    mol.compute_orbitals(override='Single' if self.options.simpleorbitals else None)

                for c, _ in enumerate(mol.atomcoords):
                    for r_atom in mol.reactive_atoms_classes_dict[c].values():
                        r_atom.cumnum = r_atom.index
                        if i > 0:
                            r_atom.cumnum += sum(self.ids[:i])

    def _read_pairings(self):
        '''
        Reads atomic pairings to be respected from the input file, if any are present.
        '''

        parsed = []
        unlabeled_list = []
        self.pairings_dict = {i:{} for i, _ in enumerate(self.objects)}

        # removing constraint lines from mol_lines, saving constraints
        mol_and_constr_lines = deepcopy(self.mol_lines)
        self.mol_lines = []

        for line in mol_and_constr_lines:
            if line[0] == " ":
                mol = self.objects[len(self.mol_lines)-1]
                try:
                    parts = line.split()
                    indices = [int(i) for i in parts[:-1]]

                    # if value is auto, take current value
                    if parts[-1] == 'auto':
                        if len(indices) == 3:
                            target = point_angle(*mol.atomcoords[0][np.array(indices)])
                        elif len(indices) == 4:
                            target = dihedral(mol.atomcoords[0][np.array(indices)])
                    else:
                        target = float(parts[-1])

                    c = Constraint(indices, target)
                    mol.constraints.append(c)

                except Exception:
                    raise Exception(f'Error while parsing line \"{line}\". Constrain syntax: \"i1 i2 i3 [i4] value\".')
            else:
                self.mol_lines.append(line)

        # now i is also the molecule index in self.objects and we can parse inter-file pairings/constraints
        for i, line in enumerate(self.mol_lines):

            fragments = line.split('>')[-1].split()[1:]
            # remove operators (if present) and the molecule name, keeping pairs only ['2a','5b']

            # store custom variables
            for fragment in deepcopy(fragments):
                if '=' in fragment:
                    parts = fragment.split('=')

                    if len(parts) != 2:
                        raise InputError(f'Error reading attribute \'{fragment}\'. Syntax: \'var=value\'')
                    
                    attr_name, attr_value = parts
                    setattr(self.objects[i], attr_name, attr_value)

                    fragments.remove(fragment)

                    self.log(f'--> Set attribute \'{attr_name}\' of {self.objects[i]} to \'{attr_value}\'.')

            self.log()

            unlabeled = []
            pairings = []

            for fragment in fragments:

                if not fragment.lower().islower(): # if all we have is a number
                    unlabeled.append(int(fragment))

                else:
                    index, letters = [''.join(g) for _, g in groupby(fragment, str.isalpha)]

                    for letter in letters:
                        pairings.append([int(index), letter])

            # appending pairing to dict before
            # calculating their cumulative index
            # If a pairing is already present, add the number
            # (refine>/REFINE runs)
            for index, letter in pairings:

                if self.pairings_dict[i].get(letter) is not None:
                    prev = self.pairings_dict[i][letter]
                    self.pairings_dict[i][letter] = (prev, index)

                else:
                    self.pairings_dict[i][letter] = index

            if i > 0:
                for z in pairings:
                    z[0] += sum(self.ids[:i])

                if unlabeled != []:
                    for z in unlabeled:
                        z += sum(self.ids[:i])
                        unlabeled_list.append(z)
            else:
                if unlabeled != []:
                    for z in unlabeled:
                        unlabeled_list.append(z)
                    
            # getting the cumulative index rather than the molecule index

            for cumulative_pair in pairings:
                parsed.append(cumulative_pair)
        # parsed looks like [[1, 'a'], [9, 'a']] where numbers are
        # cumulative indices for embeddings

        links = {j:[] for j in set([i[1] for i in parsed])}
        for index, tag in parsed:
            links[tag].append(index)
        # storing couples into a dictionary

        pairings = sorted(list(links.items()), key=lambda x: x[0])
        # sorting values so that 'a' is the first pairing

        self.pairings_table = {i[0]:sorted(i[1]) for i in pairings}
        # cumulative, looks like {'a':[3,45]}

        letters = tuple(self.pairings_table.keys())

        for letter, ids in self.pairings_table.items():

            if len(ids) == 1:
                raise SyntaxError(f'Letter \'{letter}\' is only specified once. Please flag the second reactive/constrained atom.')

            if len(ids) > 2:
                raise SyntaxError(f'Letter \'{letter}\' is specified more than two times. Please remove the extra instances of the letter.')
        
        if len(self.mol_lines) == 3:
        # adding third pairing if we have three molecules and user specified two pairings
        # (used to adjust distances for trimolecular TSs)
            if len(unlabeled_list) == 2:
                third_constraint = list(sorted(unlabeled_list))
                self.pairings_table['?'] = third_constraint
        
        elif len(self.mol_lines) == 2:
        # adding second pairing if we have two molecules and user specified one pairing
        # (used to adjust distances for bimolecular TSs)
            if len(unlabeled_list) == 2:
                second_constraint = list(sorted(unlabeled_list))
                self.pairings_table['?'] = second_constraint

        # Now record the internal constraints, that is the intramolecular
        # distances/angles to freeze and later relax or enforce to the imposed values
        self.internal_constraints = []

        # making sure we set the kw_line attribute
        self.kw_line = self.kw_line if hasattr(self, 'kw_line') else ''
        
        for letter, pair in self.pairings_table.items():
            for mol_id in self.pairings_dict:
                if isinstance(self.pairings_dict[mol_id].get(letter), tuple):

                    # They are internal constraints only if we have a distance 
                    # to impose later on. We are checking this way because the
                    # set_options function is still to be called at this stage
                    if f'{letter}=' in self.kw_line:
                        self.internal_constraints.append([pair])

        self.internal_constraints = np.concatenate(self.internal_constraints) if self.internal_constraints else []

        # finally, define the internal angle/dihedral constraints for the embedder
        self.internal_angle_dih_constraints = []
        for i, mol in enumerate(self.objects):
            n = 0 if i == 0 else sum(self.ids[:i])
            for constraint in mol.constraints:
                emb_ids = [i+n for i in constraint.indices]
                emb_constr = Constraint(emb_ids, constraint.value)
                self.internal_angle_dih_constraints.append(emb_constr)

    def _set_custom_orbs(self, orb_string):
        '''
        Update the reactive_atoms classes with the user-specified orbital distances.
        :param orb_string: string that looks like 'a=2.345,b=3.456,c=2.22'

        '''
        for mol in self.objects:
            if not hasattr(mol, 'reactive_atoms_classes_dict'):
                mol.compute_orbitals(override='Single' if self.options.simpleorbitals else None)

        self.pairing_dists = {piece.split('=')[0] : float(piece.split('=')[1]) for piece in orb_string.split(',')}

        # Set the new orbital center with imposed distance from the reactive atom. The imposed distance is half the 
        # user-specified one, as the final atomic distances will be given by two halves of this length.
        for letter, dist in self.pairing_dists.items():

            if letter not in self.pairings_table:
                raise SyntaxError(f'Letter \'{letter}\' is specified in DIST but not present in molecules string.')

            for i, mol in enumerate(self.objects):
                for c, _ in enumerate(mol.atomcoords):

                    r_index = self.pairings_dict[i].get(letter)
                    if r_index is None:
                        continue
                    
                    if isinstance(r_index, int):
                        r_atom = mol.reactive_atoms_classes_dict[c][r_index]
                        r_atom.init(mol, r_index, update=True, orb_dim=dist/2, conf=c)
                    
                    else:
                        for r_i in r_index:
                            r_atom = mol.reactive_atoms_classes_dict[c].get(r_i)
                            if r_atom:
                                r_atom.init(mol, r_i, update=True, orb_dim=dist/2, conf=c)

        # saves the last orb_string executed so that operators can
        # keep the imposed orbital spacings when replacing molecules
        self.orb_string = orb_string
        # self.log(f'DEBUG ---> Updated orb string -> {orb_string}')

    def _set_pivots(self, mol):
        '''
        params mol: Hypermolecule class
        (Cyclical embed) Function that sets the mol.pivots attribute, that is a list
        containing each vector connecting two orbitals on different atoms or on the
        same atom (for single-reactive atom molecules in chelotropic embedding)
        '''
        mol.pivots = self._get_pivots(mol)

        for c, _ in enumerate(mol.atomcoords):
            if self.options.suprafacial:
                if len(mol.pivots[c]) == 4:
                # reactive atoms have two centers each.
                # Applying suprafacial correction, only keeping
                # the shorter two, as they should be the suprafacial ones
                    norms = np.linalg.norm([p.pivot for p in mol.pivots[c]], axis=1)
                    for sample in norms:
                        to_keep = [i for i in norms if sample >= i]
                        if len(to_keep) == 2:
                            mask = np.array([i in to_keep for i in norms])
                            mol.pivots[c] = mol.pivots[c][mask]
                            break

            # if mol is reacting with a sigmastar orbital (two connected reactive Sp3/Single
            # Bond centers) then remove all pivots that are not the shortest. This ensures
            # the "suprafaciality" to the pivots used, preventing the embed of
            # impossible bonding structures
            if hasattr(mol, 'sp3_sigmastar') and mol.sp3_sigmastar:
                pivots_lengths = [norm_of(pivot.pivot) for pivot in mol.pivots[c]]
                shortest_length = min(pivots_lengths)
                mask = np.array([(i - shortest_length) < 1e-5 for i in pivots_lengths])
                mol.pivots[c] = mol.pivots[c][mask]

    def _get_pivots(self, mol):
        '''
        params mol: Hypermolecule class
        (Cyclical embed) Function that yields the molecule pivots. Called by _set_pivots
        and in pre-conditioning (deforming, bending) the molecules in ase_bend.
        '''

        if not hasattr(mol, 'reactive_atoms_classes_dict'):
            return []

        pivots_list = [[] for _ in mol.atomcoords]

        for c, _ in enumerate(mol.atomcoords):

            if len(mol.reactive_atoms_classes_dict[c]) == 2:
            # most molecules: dienes and alkenes for Diels-Alder, conjugated ketones for acid-bridged additions

                indices = cartesian_product(*[range(len(atom.center)) for atom in mol.reactive_atoms_classes_dict[c].values()])
                # indices of vectors in reactive_atom.center. Reactive atoms are 2 and so for one center on atom 0 and 
                # 2 centers on atom 2 we get [[0,0], [0,1], [1,0], [1,1]]

                for i,j in indices:
                    a1, a2 = mol.get_r_atoms(c)
                   
                    c1 = a1.center[i]
                    c2 = a2.center[j]

                    pivots_list[c].append(Pivot(c1, c2, a1, a2, i, j))

            elif len(mol.reactive_atoms_classes_dict[c]) == 1:
            # carbenes, oxygen atom in Prilezhaev reaction, SO2 in classic chelotropic reactions

                indices = cartesian_product(*[range(len(mol.get_r_atoms(c)[0].center)) for _ in range(2)])
                indices = [i for i in indices if i[0] != i[1] and (sorted(i) == i).all()]
                # indices of vectors in reactive_atom.center. Reactive atoms is just one, that builds pivots with itself. 
                # pivots with the same index or inverse order are discarded. 2 centers on one atom 2 yield just [[0,1]]
                
                for i,j in indices:
                    a1 = mol.get_r_atoms(c)[0]
                    # chelotropic embeds have pivots that start/end on the same atom

                    c1 = a1.center[i]
                    c2 = a1.center[j]

                    pivots_list[c].append(Pivot(c1, c2, a1, a1, i, j))

        return [np.array(_l) for _l in pivots_list]

    def _setup(self, p=True):
        '''
        Setting embed type and calculating the number of conformation combinations based on embed type
        '''

        if any('pka>' in op for op in self.options.operators) or (
           any('scan>' in op for op in self.options.operators)
        ):
            self.embed = 'data'
            # If a pka or scan operator is requested, the embed is skipped
            # and data is shown instead
            return

        if any('refine>' in op for op in self.options.operators) or self.options.noembed:
            self.embed = 'refine'

            # in a refine run, the global charge must be the one of the single molecule
            self.options.charge = getattr(self.objects[0], "charge", 0)

            # If the run is a refine>/REFINE one, the self.embed
            # attribute is set in advance by the self._set_options
            # function through the OptionSetter class
            return

        for mol in self.objects:
            if self.options.max_confs < len(mol.atomcoords) and self.embed is not None:
                self.log(f'--> {mol.filename} - kept {self.options.max_confs}/{len(mol.atomcoords)} conformations for the embed (override with CONFS=n)\n')
                mol.atomcoords = mol.atomcoords[0:self.options.max_confs]
        # remove conformers if there are too many

        if all([len(mol.reactive_indices) == 0 for mol in self.objects]):
            self.embed = None
            # Flag the embed type as None if no reactive indices are
            # provided (and the run is not a refine> one)
            return

        if len(self.objects) == 1:
        # embed must be either monomolecular

            mol = self.objects[0]

            if len(mol.reactive_indices) == 2:

                self.embed = 'monomolecular'
                mol.compute_orbitals(override='Single' if self.options.simpleorbitals else None)
                self._set_pivots(mol)

                self.options.only_refined = True
                self.options.fix_angles_in_deformation = True
                # These are required: otherwise, extreme bending could scramble molecules

            else:
                self.embed = 'error'
                # if none of the previous, the program had trouble recognizing the embed to carry.

                return
            
        elif len(self.objects) in (2,3):
        # Setting embed type and calculating the number of conformation combinations based on embed type

            cyclical = all(len(molecule.reactive_indices) == 2 for molecule in self.objects)

            # chelotropic embed should check that the two atoms on one molecule are bonded
            chelotropic = sorted(len(molecule.reactive_indices) for molecule in self.objects) == [1,2]

            string = all(len(molecule.reactive_indices) == 1 for molecule in self.objects) and len(self.objects) == 2

            multiembed = (len(self.objects) == 2 and
                          all(len(molecule.reactive_indices) >= 2 for molecule in self.objects) and 
                          not cyclical)

            if cyclical or chelotropic or multiembed:

                if cyclical:
                    self.embed = 'cyclical'
                elif multiembed:
                    self.embed = 'multiembed'
                else:
                    self.embed = 'chelotropic'
                    for mol in self.objects:
                        mol.compute_orbitals(override='Single' if self.options.simpleorbitals else None)
                        for c, _ in enumerate(mol.atomcoords):
                            for index, atom in mol.reactive_atoms_classes_dict[c].items():
                                orb_dim = norm_of(atom.center[0]-atom.coord)
                                atom.init(mol, index, update=True, orb_dim=orb_dim + 0.2, conf=c)
                    # Slightly enlarging orbitals for chelotropic embeds, or they will
                    # be generated a tad too close to each other for how the cyclical embed works          

                self.options.rotation_steps = 5

                if hasattr(self.options, 'custom_rotation_steps'):
                # if user specified a custom value, use it.
                    self.options.rotation_steps = self.options.custom_rotation_steps

                self.systematic_angles = cartesian_product(*[range(self.options.rotation_steps+1) for _ in self.objects]) \
                            * 2*self.options.rotation_range/self.options.rotation_steps - self.options.rotation_range

                if p:
                # avoid calculating pivots if this is an early call 
                    for molecule in self.objects:
                        self._set_pivots(molecule)

            elif string:
                
                self.embed = 'string'
                self.options.rotation_steps = 36

                for mol in self.objects:
                    if not hasattr(mol, 'reactive_atoms_classes_dict'):
                        mol.compute_orbitals(override='Single' if self.options.simpleorbitals else None)

                if hasattr(self.options, 'custom_rotation_steps'):
                # if user specified a custom value, use it.
                    self.options.rotation_steps = self.options.custom_rotation_steps

                self.systematic_angles = [n * 360 / self.options.rotation_steps for n in range(self.options.rotation_steps)]

            else:
                self.embed = 'error'

            if multiembed:
                # Complex, unspecified embed type - will explore many possibilities concurrently
                self.embed = 'multiembed'
                for mol in self.objects:
                    mol.compute_orbitals(override='Single' if self.options.simpleorbitals else None)

            if self.embed == 'error':

                orbitalstring = ''
                for mol in self.objects:
                    orbitalstring += f' - {mol.filename} ({len(mol.reactive_indices)} reactive indices - {" ".join([str(i) for i in mol.reactive_indices])})\n'

                raise InputError((f'Bad input:\n{orbitalstring}\n'

                                  'The only molecular configurations accepted are:\n' 
                                  '1) One molecule with two reactive centers (monomolecular embed)\n'
                                  '2) One molecule with four indices(dihedral embed)\n'
                                  '3) Two or three molecules with two reactive centers each (cyclical embed)\n'
                                  '4) Two molecules with one reactive center each (string embed)\n'
                                  '5) Two molecules, one with a single reactive center and the other with two (chelotropic embed)\n'
                                  '6) Two molecules with at least two reactive centers each'))
            
            if p:
            # avoid calculating this if this is an early call 

                self._set_reactive_atoms_cumnums()
                # appending to each reactive atom the cumulative
                # number indexing in the TS context

        else:
            raise InputError('Bad input - could not set up an appropriate embed type (too many structures specified?)')

        # Only call this part if it is not an early call
        if p:
            if self.options.shrink:
                for molecule in self.objects:
                    molecule._scale_orbs(self.options.shrink_multiplier)
                    self._set_pivots(molecule)
                self.options.only_refined = True
            # SHRINK - scale orbitals and rebuild pivots

            # if self.options.rmsd is None:
            #     self.options.rmsd = 0.25

            self.candidates = self._get_number_of_candidates()
            _s = self.candidates or 'Many'
            self.log(f'--> Setup performed correctly. {_s} candidates will be generated.\n')

    def _get_number_of_candidates(self):
        '''
        Get the number of structures that will be generated in the run.
        '''
        _l = len(self.objects)
        if _l == 1:
            return int(sum([len(self.objects[0].pivots[c])
                            for c, _ in enumerate(self.objects[0].atomcoords)]))

        if self.embed == 'string':
            return int(self.options.rotation_steps*(
                       np.prod([sum([len(mol.get_r_atoms(conf)[0].center)
                                     for conf, _ in enumerate(mol.atomcoords)]) 
                                for mol in self.objects]))
                      )
        
        if self.embed == 'multiembed':
            return 0

        candidates = 2*len(self.systematic_angles)*np.prod([len(mol.atomcoords) for mol in self.objects])
        
        if _l == 3:
            candidates *= 4
        # Trimolecular there are 8 different triangles originated from three oriented vectors,
        # while only 2 disposition of two vectors (parallel, antiparallel).

        if self.pairings_table:
        # If there is any pairing to be respected, each one reduces the number of
        # candidates to be computed.

            if self.embed == 'cyclical':
                if len(self.objects) == 2:
                # Diels-Alder-like, if we have one (two) pairing(s) only half
                # of the total arrangements are to be checked
                    candidates /= 2

                else: # trimolecular
                    if len(self.pairings_table) == 1:
                        candidates /= 4
                    else: # trimolecular, 2 (3) pairings imposed
                        candidates /= 8

        candidates *= np.prod([len(mol.pivots[0]) for mol in self.objects]) # add sum over len(mol.pivots[c])?
        # The more atomic pivots, the more candidates

        return int(candidates)

    def _get_angle_dih_constraints(self):
        '''
        Gets angle and dihedral constraints in a list format
        from the self.internal_angle_dih_constraints attribute.
        
        '''
        (
            constrained_angles_indices,
            constrained_angles_values,
            constrained_dihedrals_indices,
            constrained_dihedrals_values,
        ) = [], [], [], []

        for constraint in self.internal_angle_dih_constraints:
            if constraint.type == "A":
                constrained_angles_indices.append(constraint.indices)
                constrained_angles_values.append(constraint.value)

            elif constraint.type == "D":
                constrained_dihedrals_indices.append(constraint.indices)
                constrained_dihedrals_values.append(constraint.value)

        return (
                    constrained_angles_indices,
                    constrained_angles_values,
                    constrained_dihedrals_indices,
                    constrained_dihedrals_values,
                )

    def _set_embedder_structures_from_mol(self):
        '''
        Intended for REFINE runs, set the self.structures variable
        (and related) to the confomers of a specific molecuele.
        '''
        self.structures = self.objects[0].atomcoords
        self.atomnos = self.objects[0].atomnos
        self.constrained_indices = _get_monomolecular_reactive_indices(self)
        self.ids = None
        self.energies = np.array([0 for _ in self.structures])
        self.exit_status = np.ones(self.structures.shape[0], dtype=bool)
        self.embed_graph = get_sum_graph([graphize(self.structures[0], self.atomnos)], self.constrained_indices[0])

    def _calculator_setup(self):
        '''
        Set up the calculator to be used with default theory levels.
        '''
        # Checking that calculator is specified correctly
        if self.options.calculator not in ('MOPAC', 'ORCA', 'GAUSSIAN','XTB', 'AIMNET2'):
            raise SyntaxError(f'\'{self.options.calculator}\' is not a valid calculator. Change its value from the parameters.py file or with the CALC keyword.')

        # Setting default theory level if user did not specify it
        if self.options.theory_level is None:
            self.options.theory_level = DEFAULT_LEVELS[self.options.calculator]

        self.dispatcher = Opt_func_dispatcher()

        if self.options.calculator == 'AIMNET2':
            self.dispatcher.load_aimnet2_calc(self.options.theory_level, logfunction=self.log)


    def _apply_operators(self):
        '''
        Replace molecules in self.objects with
        their post-operator ones.
        '''

        # early call to get the self.embed attribute
        self._setup(p=False)

        # for input_string in self.options.operators:
        for index, operators in self.options.operators_dict.items():

            for operator in operators: 

                input_string = f'{operator}> {self.objects[index].filename}'
                outname = operate(input_string, self)
                # operator = input_string.split('>')[0]

                if operator == 'refine':
                    self._set_embedder_structures_from_mol()

                # these operators do not need molecule substitution
                elif operator not in ('pka', 'scan'):

                    # names = [mol.filename for mol in self.objects]
                    # filename = self._extract_filename(input_string)
                    # index = names.index(filename)
                    reactive_indices = self.objects[index].reactive_indices

                    # replacing the old molecule with the one post-operators
                    self.objects[index] = Hypermolecule(outname, reactive_indices)

                    # calculating where the new orbitals are
                    self.objects[index].compute_orbitals(override='Single' if self.options.simpleorbitals else None)

                    # updating orbital size if not default
                    if hasattr(self, 'orb_string'):
                        self._set_custom_orbs(self.orb_string)

                    # updating global docker if necessary
                    if operator in ('rsearch', 'csearch') and self.options.noembed and len(self.objects) == 1:
                        self.structures = self.objects[0].atomcoords
                        self.atomnos = self.objects[0].atomnos
                        self.constrained_indices = _get_monomolecular_reactive_indices(self)
                        self.ids = None
                        self.energies = np.array([0 for _ in self.structures])
                        self.exit_status = np.ones(self.structures.shape[0], dtype=bool)
                        self.embed_graph = get_sum_graph([graphize(self.structures[0], self.atomnos)], self.constrained_indices[0])

        # updating the orbital cumnums for 
        # all the molecules in the run
        self._set_reactive_atoms_cumnums()

        # resetting the attribute
        self.embed = None

    def _extract_filename(self, input_string):
        '''
        Input: 'refine> firecode_unoptimized_comp_check.xyz 5a 36a 0b 43b 33c 60c'
        Output: 'firecode_unoptimized_comp_check.xyz'
        '''
        input_string = input_string.split('>')[-1].lstrip()
        # remove operator and whitespaces after it

        input_string = input_string.split()[0]
        # remove pairing numbers/letters and newline chars

        return input_string

    def _inspect_structures(self):
        '''
        '''

        self.log('--> Structures check requested. Shutting down after last window is closed.\n')

        for mol in self.objects:
            ase_view(mol)
        
        self.close_log_streams()
        os.remove(f'firecode_{self.stamp}.log')

        sys.exit()

    def scramble(self, array, sequence):
        return np.array([array[s] for s in sequence])

    def get_pairing_dist_from_letter(self, letter):
        '''
        Get constrained distance between paired reactive
        atoms, accessed via the associated constraint letter.
        The distance returned is the final one (not affected by SHRINK)
        '''

        if hasattr(self, 'pairing_dists') and self.pairing_dists.get(letter) is not None:
            return self.pairing_dists[letter]

        d = 0
        try:
            for mol_index, mol_pairing_dict in self.pairings_dict.items():
                if r_atom_index := mol_pairing_dict.get(letter):

                    # for refine embeds, one letter corresponds to two indices
                    # on the same molecule
                    if isinstance(r_atom_index, tuple):
                        i1, i2 = r_atom_index
                        return (self.objects[mol_index].get_orbital_length(i1) +
                                self.objects[mol_index].get_orbital_length(i2))

                    # for other runs, it is just one atom per molecule per letter
                    d += self.objects[mol_index].get_orbital_length(r_atom_index)

            if self.options.shrink:
                d /= self.options.shrink_multiplier

            return d

        # If no orbitals were built, return None
        except NoOrbitalError:
            return None

    def get_pairing_dists_from_constrained_indices(self, constrained_pair):
        '''
        Returns the constrained distance
        for a specific constrained pair of indices
        '''
        try:
            letter = next(lett for lett, pair in self.pairings_table.items() if (pair[0] == constrained_pair[0] and      
                                                                                 pair[1] == constrained_pair[1]))
            return self.get_pairing_dist_from_letter(letter)

        except StopIteration:
            return None

    def get_pairing_dists(self, conf):
        '''
        Returns a list with the constrained distances for each embedder constraint
        '''
        if self.constrained_indices[conf].size == 0:
            return None

        constraints = np.concatenate([self.constrained_indices[conf], self.internal_constraints]) if len(self.internal_constraints) > 0 else self.constrained_indices[conf]
        return [self.get_pairing_dists_from_constrained_indices(pair) for pair in constraints]

    def write_structures(
                            self,
                            tag, 
                            indices=None, 
                            energies=True, 
                            relative=True, 
                            extra='', 
                            align='indices', 
                            p=True,
                        ):
        '''
        Writes structures to file.

        '''

        align_functions = {
            'indices' : align_structures,
            'moi' : align_by_moi,
        }

        if energies:
            rel_e = self.energies

            if relative:
                rel_e -= np.min(self.energies)

        # truncate if there are too many (embed debug first dump)
        if len(self.structures) > 10000 and not self.options.let:
            self.log(f'Truncated {tag} output structures to 10000 (from {len(self.structures)} - keyword LET to override).')
            output_structures = self.structures[0:10000]
        else:
            output_structures = self.structures

        self.outname = f'firecode_{tag}_{self.stamp}.xyz'
        with open(self.outname, 'w') as f:        

            for i, structure in enumerate(align_functions[align](output_structures, atomnos=self.atomnos, indices=indices)):
                title = f'Strucure {i+1} - {tag}'

                if energies:
                    title += f' - Rel. E. = {round(rel_e[i], 3)} kcal/mol '
                
                title += extra

                write_xyz(structure, self.atomnos, f, title=title)

        if p:
            self.log(f'Wrote {len(output_structures)} {tag} structures to {self.outname} file.\n')

    def write_quote(self):
        '''
        Reads the quote file and writes one in the logfile
        '''
        from firecode.quotes import load_quotes
        quote, author = random.choice(load_quotes()).values()

        self.log('\n' + auto_newline(quote))

        if author != "":
            self.log(f'    - {author}\n')

    def run(self):
        '''
        Run the embedding.
        '''
        try:
            RunEmbedding(self).run()

        except Exception as _e:
            logging.exception(_e)
            raise _e
        
    def normal_termination(self):
        '''
        Terminate the run, printing the total time and the
        relative energies of the first 10 structures, if possible.

        '''
        clean_directory()
        self.log(f'\n--> FIRECODE normal termination: total time {time_to_string(time.perf_counter() - self.t_start_run, verbose=True)}.')
        
        if hasattr(self, "structures"):
            show = 10
            if len(self.structures) > 0 and hasattr(self, "energies"):
                self.energies = self.energies if len(self.energies) <= show else self.energies[0:show]

                # Don't write structure info if there is only one, or all are zero
                if np.max(self.energies - np.min(self.energies)) > 0:

                    self.log(f'\n--> Energies of output structures (first {show}, {self.options.theory_level}/{self.options.calculator}{f"/{self.options.solvent}" if self.options.solvent is not None else ""})\n')

                    self.log(f'> # (total {len(self.structures)})     Rel. E.           RMSD')
                    self.log('-------------------------------------------')
                    for i, energy in enumerate(self.energies-self.energies[0]):

                        rmsd_value = '(ref)' if i == 0 else str(round(rmsd_and_max_numba(self.structures[i], self.structures[0], center=True)[0], 2))+' Å'

                        self.log(f'> Candidate {str(i+1):2}  :  {energy:.2f} kcal/mol  :  {rmsd_value}')

                    if len(self.structures) > show:
                        self.log(f'> ... ({len(self.structures)-show} more)')

                with open('firecode_best.xyz', 'w') as f:
                    write_xyz(self.structures[0], self.atomnos, f)

        self.write_quote()
        self.close_log_streams()
        sys.exit()

    def close_log_streams(self):
        self.logfile.close()

        if hasattr(self, "debug_logfile"):
            self.debug_logfile.close()

class RunEmbedding(Embedder):
    '''
    Class for running embeds, containing all
    methods to embed and refine structures
    '''

    def __init__(self, embedder):
        '''
        Copying all non-callable attributes 
        of the previous embedder.
        '''
        # Copy all the non-callables (variables) into the child class
        for attr in dir(embedder):
            if attr[0:2] != '__' and attr != 'run':
                attr_value = getattr(embedder, attr)
                if not hasattr(attr_value, '__call__'):
                    setattr(self, attr, attr_value)

    def rel_energies(self):
        return self.energies - np.min(self.energies)

    def apply_mask(self, attributes, mask):
        '''
        Applies in-place masking of Embedder attributes
        '''
        for attr in attributes:
            if hasattr(self, attr):
                new_attr = getattr(self, attr)[mask]
                setattr(self, attr, new_attr)

    def zero_candidates_check(self):
        '''
        Asserts that not all structures are being rejected.
        '''
        if len(self.structures) == 0:
            self.log_warnings()
            raise ZeroCandidatesError()

    def generate_candidates(self):
        '''
        Generate a series of candidate structures by the proper embed algorithm.
        '''

        embed_functions = {
            'chelotropic' : cyclical_embed,
            'cyclical' : cyclical_embed,
            'monomolecular' : monomolecular_embed,
            'string' : string_embed,
            'multiembed' : multiembed_dispatcher,
        }

        if self.embed == 'refine':
            self.log('\n')
            return

        # Embed structures and assign them to self.structures
        self.structures = embed_functions[self.embed](self)

        # cumulative list of atomic numbers associated with coordinates
        self.atomnos = np.concatenate([molecule.atomnos for molecule in self.objects])

        # Build the embed graph. This will be used as a future reference.
        # Note that the use of the first constrained_indices pair is irrelevant
        # for the torsion fingerprint outcome, but other future features might
        # rely on the embed_graph to be accurate if conformers have different
        # constrained indices.

        additional_bonds = self.constrained_indices[0]
        if len(self.internal_constraints) > 0:
            additional_bonds = np.concatenate((self.internal_constraints, additional_bonds))

        self.embed_graph = get_sum_graph(self.graphs, additional_bonds)
        
        self.log(f'Generated {len(self.structures)} transition state candidates ({time_to_string(time.perf_counter()-self.t_start_run)})\n')

        # if self.options.debug:
        self.write_structures('embedded', energies=False)

        if self.options.debug:
            self.dump_status('generate_candidates')
            self.debuglog('DEBUG: Dumped emebedder status after generating candidates (\"generate_candidates\")')

    def dump_status(self, outname, only_fixed_constraints=False):
        '''
        Writes structures and energies to [outname].xyz
        and [outname].dat to help debug the current run.
                
        '''

        if hasattr(self, 'energies'):
            with open(f'{outname}_energies.dat', 'w') as _f:
                for i, energy in enumerate(self.energies):
                    print_energy = str(round(energy-np.min(self.energies), 2))+' kcal/mol' if energy != 1E10 else 'SCRAMBLED'
                    _f.write('Candidate {:5} : {}\n'.format(i, print_energy))

        with open(f'{outname}_structures.xyz', 'w') as _f:
            exit_status = self.exit_status if hasattr(self, 'exit_status') else [0 for _ in self.structures]    
            energies = self.rel_energies() if hasattr(self, 'energies') else [0 for _ in self.structures]
            for i, (structure, status, energy) in enumerate(zip(align_structures(self.structures),
                                                                exit_status,
                                                                energies)):

                kind = 'REFINED - ' if status else 'NOT REFINED - '
                write_xyz(structure, self.atomnos, _f, title=f'Structure {i+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol')

        with open(f'{outname}_constraints.dat', 'w') as _f:
            for i, constraints in enumerate(self.constrained_indices):

                if only_fixed_constraints:
                    constraints = np.array([value for key, value in self.pairings_table.items() if key.isupper()])
                
                else:
                    constraints = np.concatenate([constraints, self.internal_constraints]) if len(self.internal_constraints) > 0 else constraints

                c_str = repr(constraints).replace('\n','').replace(',       ',', ')
                d_str = [self.get_pairing_dists_from_constrained_indices(_c) for _c in constraints]
                _f.write('Candidate {:5} : {} -> {}\n'.format(i, c_str, d_str))

        with open(f'{outname}_runembedding.pickle', 'wb') as _f:
            d = {
                'structures' : self.structures,
                'constrained_indices' : self.constrained_indices,
                'graphs' : self.graphs,
                'objects' : self.objects,
                'options' : self.options,
                'atomnos' : self.atomnos,
            }

            if hasattr(self, 'energies'):
                d['energies'] = self.energies

            pickle.dump(d, _f)

    def compenetration_refining(self):
        '''
        Performing a sanity check for excessive compenetration
        on generated structures, discarding the ones that look too bad.
        '''

        if self.embed not in ('string', 'cyclical', 'monomolecular'):
        # these do not need compenetration refining: the 
        # algorithm checks for compenetrations when embedding
            
            self.log('--> Checking structures for compenetrations')

            t_start = time.perf_counter()
            mask = np.zeros(len(self.structures), dtype=bool)
            # num = len(self.structures)
            for s, structure in enumerate(self.structures):
                # if num > 100 and num % 100 != 0 and s % (num % 100) == 99:
                #     loadbar(s, num, prefix=f'Checking structure {s+1}/{num} ')
                mask[s] = compenetration_check(structure, self.ids, max_clashes=self.options.max_clashes, thresh=self.options.clash_thresh)

            # loadbar(1, 1, prefix=f'Checking structure {len(self.structures)}/{len(self.structures)} ')

            self.apply_mask(('structures', 'constrained_indices'), mask)
            t_end = time.perf_counter()

            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for compenetration ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
            else:
                self.log(f'All {len(mask)} structures passed the compenetration check')
            self.log()

            self.zero_candidates_check()

        # initialize embedder values for the active structures
        # that survived the compenetration check
        self.energies = np.full(len(self.structures), 1E10)
        self.exit_status = np.zeros(len(self.structures), dtype=bool)
    
    def fitness_refining(self, threshold=5, verbose=False):
        '''
        Performing a distance check on generated structures, 
        discarding the ones that do not respect the imposed pairings.
        Internal constraints are ignored.

        threshold : rejection happens when the sum of the deviations from the
        intended spacings is greater than threshold.

        '''
        if verbose:
            self.log(' \n--> Fitness pruning - removing inaccurate structures')

        mask = np.ones(len(self.structures), dtype=bool)
        
        for s, (structure, constraints) in enumerate(zip(self.structures, self.constrained_indices)):

            constrained_distances = tuple(self.get_pairing_dists_from_constrained_indices(_c) for _c in constraints)

            mask[s] = fitness_check(structure,
                                    constraints,
                                    constrained_distances,
                                    threshold=threshold)

        attr = (
            'structures',
            'energies',
            'constrained_indices',
            'exit_status',
        )

        self.apply_mask(attr, mask)

        if False in mask:
            self.log(f'Discarded {len([b for b in mask if not b])} candidates for unfitness ({len([b for b in mask if b])} left)')
        else:
            if verbose:
                self.log('All candidates meet the imposed criteria.')
        self.log()

        self.zero_candidates_check()

    def similarity_refining(self, tfd=True, moi=True, rmsd=True, verbose=False):
        '''
        If possible, removes structures with similar torsional profile (TFD-based).
        Removes structures that are too similar to each other (RMSD-based).
        '''

        if verbose:
            self.log('--> Similarity Processing')

        before = len(self.structures)
        attr = ('constrained_indices', 'energies', 'exit_status')

        if (
            tfd and 
            len(self.objects) > 1 and 
            hasattr(self, 'embed_graph') and
            self.embed_graph.is_single_molecule
        ):

            t_start = time.perf_counter()

            quadruplets = _get_quadruplets(self.embed_graph)
            if len(quadruplets) > 0:
                self.structures, mask = prune_conformers_tfd(self.structures, quadruplets, verbose=verbose)
                
            self.apply_mask(attr, mask)
            
            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} structures for TFD similarity ({len([b for b in mask if b])} left, {time_to_string(time.perf_counter()-t_start)})')

        if moi:

            if len(self.structures) <= 1E5:

                ### Now again, based on the moment of inertia

                before3 = len(self.structures)

                t_start = time.perf_counter()
                self.structures, mask = prune_by_moment_of_inertia(self.structures, self.atomnos, debugfunction=self.debuglog)

                self.apply_mask(attr, mask)

                if before3 > len(self.structures):
                    self.log(f'Discarded {int(len([b for b in mask if not b]))} candidates for MOI similarity ({len([b for b in mask if b])} left, {time_to_string(time.perf_counter()-t_start)})')

            else:
                self.log('Skipped MOI pruning (>100k structures)')

        if rmsd and len(self.structures) <= 1E5:

            before1 = len(self.structures)

            t_start = time.perf_counter()
            
            self.structures, mask = prune_by_rmsd(self.structures, self.atomnos, self.options.rmsd, debugfunction=self.debuglog)

            self.apply_mask(attr, mask)

            if before1 > len(self.structures):
                self.log(f'Discarded {int(len([b for b in mask if not b]))} candidates for RMSD similarity ({len([b for b in mask if b])} left, {time_to_string(time.perf_counter()-t_start)})')

            ### Second step: again but symmetry-corrected (unless we have too many structures)

            if len(self.structures) <= 1E3 and hasattr(self, 'embed_graph'):

                before2 = len(self.structures)

                t_start = time.perf_counter()
                self.structures, mask = prune_by_rmsd_rot_corr(
                                                self.structures,
                                                self.atomnos,
                                                self.embed_graph,
                                                max_rmsd=self.options.rmsd, 
                                                logfunction=(self.log if verbose else None),
                                                debugfunction=self.debuglog,
                                            )

                self.apply_mask(attr, mask)

                if before2 > len(self.structures):
                    self.log(f'Discarded {int(len([b for b in mask if not b]))} candidates for symmetry-corrected RMSD similarity ({len([b for b in mask if b])} left, {time_to_string(time.perf_counter()-t_start)})')

            elif hasattr(self, 'embed_graph'):
                self.log('Skipped rotationally-corrected RMSD pruning (>1k structures)')

        else:
            self.log('Skipped RMSD pruning (>100k structures)')

        if verbose and len(self.structures) == before:
            self.log(f'All structures passed the similarity check.{" "*15}')

        self.log()

    def force_field_refining(self, conv_thr="tight", only_fixed_constraints=False, prevent_scrambling=False):
        '''
        Performs structural optimizations with the embedder force field caculator.
        Only structures that do not scramble during FF optimization are updated,
        while the rest are kept as they are.
        conv_thr: convergence threshold, passed to calculator
        only_fixed_constraints: only uses fixed (UPPERCASE) constraints in optimization
        prevent_scrambling: preserves molecular identities constraining bonds present in graphs (XTB only)
        '''

        ################################################# CHECKPOINT BEFORE FF OPTIMIZATION

        self.outname = f'firecode_checkpoint_{self.stamp}.xyz'
        if not only_fixed_constraints:
            with open(self.outname, 'w') as f:        
                for i, structure in enumerate(align_structures(self.structures)):
                    write_xyz(structure, self.atomnos, f, title=f'TS candidate {i+1} - Checkpoint before FF optimization')
            self.log(f'\n--> Checkpoint output - Wrote {len(self.structures)} unoptimized structures to {self.outname} file before FF optimization.\n')

        ################################################# GEOMETRY OPTIMIZATION - FORCE FIELD
        
        if only_fixed_constraints:
            task = 'Structure optimization (tight) / relaxing interactions'
        else:
            task = f'Structure {"pre-" if prevent_scrambling else ""}optimization (loose)'

        self.log(f'--> {task} ({self.options.ff_level}{f"/{self.options.solvent}" if self.options.solvent is not None else ""} level via {self.options.ff_calc}, {self.avail_cpus} thread{"s" if self.avail_cpus>1 else ""})')

        t_start_ff_opt = time.perf_counter()

        processes = []
        cum_time = 0

        opt_function = xtb_pre_opt if prevent_scrambling else xtb_opt

        # Running as many threads as we have procs
        # since FF does not parallelize well with more cores
        with ProcessPoolExecutor(max_workers=self.avail_cpus) as executor:

            for i, structure in enumerate(deepcopy(self.structures)):

                if only_fixed_constraints:
                    constraints = np.array([value for key, value in self.pairings_table.items() if key.isupper()])
                
                else:
                    constraints = np.concatenate([self.constrained_indices[i], self.internal_constraints]) if len(self.internal_constraints) > 0 else self.constrained_indices[i]

                pairing_dists = [self.get_pairing_dists_from_constrained_indices(_c) for _c in constraints]

                (
                    constrained_angles_indices,
                    constrained_angles_values,
                    constrained_dihedrals_indices,
                    constrained_dihedrals_values,
                ) = self._get_angle_dih_constraints()

                process = executor.submit(
                                            timing_wrapper,
                                            opt_function,
                                            structure,
                                            self.atomnos,
                                            graphs=self.graphs,
                                            calculator=self.options.ff_calc,
                                            method=self.options.ff_level,
                                            solvent=self.options.solvent,
                                            charge=self.options.charge,
                                            maxiter=None,
                                            conv_thr=conv_thr,

                                            constrained_indices=constraints,
                                            constrained_distances=pairing_dists,

                                            constrained_angles_indices=constrained_angles_indices,
                                            constrained_angles_values=constrained_angles_values,

                                            constrained_dihedrals_indices=constrained_dihedrals_indices,
                                            constrained_dihedrals_values=constrained_dihedrals_values,

                                            procs=2, # FF just needs two per structure
                                            title=f'Candidate_{i+1}',
                                            spring_constant=0.2 if prevent_scrambling else 1,
                                            payload=(
                                                self.constrained_indices[i],
                                                )
                                        )
                processes.append(process)
          
            for i, process in enumerate(as_completed(processes)):
                        
                loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')

                ((
                    new_structure,
                    new_energy,
                    self.exit_status[i]
                ),
                # from optimization function
                 
                (
                    self.constrained_indices[i],
                ),
                # from payload
                
                    t_struct
                # from timing_wrapper

                ) = process.result()
                
                # assert that the structure did not scramble during optimization
                if self.exit_status[i]:
                    constraints = (np.concatenate([self.constrained_indices[i], self.internal_constraints])
                                   if len(self.internal_constraints) > 0
                                   else self.constrained_indices[i])
                    
                    self.exit_status[i] = scramble_check(new_structure,
                                                        self.atomnos,
                                                        excluded_atoms=constraints.ravel(),
                                                        mols_graphs=self.graphs,
                                                        max_newbonds=self.options.max_newbonds,
                                                        logfunction=self.log if self.options.debug else None,
                                                        title=f"Candidate_{i+1}")
                    
                cum_time += t_struct

                if self.options.debug:
                    exit_status = 'REFINED  ' if self.exit_status[i] else 'SCRAMBLED'
                    self.debuglog(f'DEBUG: force_field_refining ({conv_thr}) - Candidate_{i+1} - {exit_status} {time_to_string(t_struct, digits=3)}')
                
                if self.exit_status[i] and new_energy is not None:
                    self.structures[i] = new_structure
                    self.energies[i] = new_energy

                else:
                    self.energies[i] = 1E10

                ### Update checkpoint every (20*max_workers) optimized structures, and give an estimate of the remaining time
                chk_freq = self.avail_cpus * self.options.checkpoint_frequency
                if i % chk_freq == chk_freq-1:

                    with open(self.outname, 'w') as f:        
                        for j, (structure, status, energy) in enumerate(zip(align_structures(self.structures),
                                                                            self.exit_status,
                                                                            self.rel_energies())):

                            kind = 'REFINED - ' if status else 'NOT REFINED - '
                            write_xyz(structure, self.atomnos, f, title=f'Structure {j+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({self.options.ff_level})')

                    elapsed = time.perf_counter() - t_start_ff_opt
                    average = (elapsed)/(i+1)
                    time_left = time_to_string((average) * (len(self.structures)-i-1))
                    speedup = cum_time/elapsed
                    self.log(f'    - Optimized {i+1:>4}/{len(self.structures):>4} structures - updated checkpoint file (avg. {time_to_string(average)}/struc, {round(speedup, 1)}x speedup, est. {time_left} left)', p=False)
        
        loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')

        elapsed = time.perf_counter() - t_start_ff_opt
        average = (elapsed)/(len(self.structures))
        speedup = cum_time/elapsed

        self.log(f'{self.options.ff_calc}/{self.options.ff_level} optimization took {time_to_string(elapsed)} (~{time_to_string(average)} per structure, {round(speedup, 1)}x speedup)')
        
        ################################################# EXIT STATUS

        self.log(f'Successfully optimized {len([b for b in self.exit_status if b])}/{len(self.structures)} candidates at {self.options.ff_level} level.')
        
        ################################################# PRUNING: ENERGY

        _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
        self.energies = self.scramble(self.energies, sequence)
        self.structures = self.scramble(self.structures, sequence)
        self.constrained_indices = self.scramble(self.constrained_indices, sequence)
        # sorting structures based on energy

        if self.options.debug:
            self.dump_status(f'force_field_refining_{conv_thr}', only_fixed_constraints=only_fixed_constraints)
            self.debuglog(f'DEBUG: Dumped emebedder status after generating candidates (\"force_field_refining_{conv_thr}\")')
  
        mask = self.rel_energies() < 1E10
        self.apply_mask(('structures', 'constrained_indices', 'energies', 'exit_status'), mask)

        if False in mask:
            self.log(f'Discarded {len([b for b in mask if not b])} scrambled candidates ({np.count_nonzero(mask)} left)')

        ################################################# PRUNING: FITNESS (POST FORCE FIELD OPT)

        self.fitness_refining(threshold=2)

        ################################################# PRUNING: SIMILARITY (POST FORCE FIELD OPT)

        self.zero_candidates_check()
        self.similarity_refining()

        ################################################# CHECKPOINT AFTER FF OPTIMIZATION
        
        s = f'--> Checkpoint output - Updated {len(self.structures)} optimized structures to {self.outname} file'

        if self.options.optimization and (self.options.ff_level != self.options.theory_level) and conv_thr != "tight":
            s += f' before {self.options.calculator} optimization.'

        else:
            self.outname = f'firecode_{"ensemble" if self.embed == "refine" else "poses"}_{self.stamp}.xyz'
            # if the FF optimization was the last one, call the outfile accordingly


        self.log(s+'\n')

        with open(self.outname, 'w') as f:        
            for i, (structure, status, energy) in enumerate(zip(align_structures(self.structures),
                                                                self.exit_status,
                                                                self.rel_energies())):

                kind = 'REFINED - ' if status else 'NOT REFINED - '
                write_xyz(structure, self.atomnos, f, title=f'Structure {i+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({self.options.ff_level})')

        # do not retain energies for the next optimization step if optimization was not tight
        if not only_fixed_constraints:
            self.energies.fill(0)

    def _set_target_distances(self):
        '''
        Called before TS refinement to compute all
        target bonding distances. These are only returned
        if that pairing is not a non-covalent interaction,
        that is if pairing was not specified with letters
        "x", "y" or "z".
        '''
        self.target_distances = {}

        # grab the atoms we want to extract information from
        r_atoms = {}
        for mol in self.objects:
            for letter, r_atom in mol.reactive_atoms_classes_dict[0].items():
                cumnum = r_atom.cumnum if hasattr(r_atom, 'cumnum') else r_atom.index
                if letter not in ("x", "y", "z"):
                    r_atoms[cumnum] = r_atom

        pairings = self.constrained_indices.ravel()
        pairings = pairings.reshape(int(pairings.shape[0]/2), 2)
        pairings = {tuple(sorted((a,b))) for a, b in pairings}

        active_pairs = [indices for letter, indices in self.pairings_table.items() if letter not in ("x", "y", "z")]

        for index1, index2 in pairings:

            if [index1, index2] in active_pairs:

                if hasattr(self, 'pairing_dists'):
                    letter = list(self.pairings_table.keys())[active_pairs.index([index1, index2])]

                    if letter in self.pairing_dists:
                        self.target_distances[(index1, index2)] = self.pairing_dists[letter]
                        continue
                # if target distance has been specified by user, read that, otherwise compute it

                r_atom1 = r_atoms[index1]
                r_atom2 = r_atoms[index2]

                dist1 = orb_dim_dict.get(r_atom1.symbol + ' ' + str(r_atom1), orb_dim_dict['Fallback'])
                dist2 = orb_dim_dict.get(r_atom2.symbol + ' ' + str(r_atom2), orb_dim_dict['Fallback'])

                self.target_distances[(index1, index2)] = dist1 + dist2
 
    def optimization_refining(self, maxiter=None, conv_thr='tight', only_fixed_constraints=False):
        '''
        Refines structures by constrained optimizations with the active calculator,
        discarding similar ones and scrambled ones.
        maxiter - int, number of max iterations for the optimization
        conv_thr: convergence threshold, passed to calculator
        only_fixed_constraints: only uses fixed (UPPERCASE) constraints in optimization

        '''

        # pytorch parallellization is carried out differently
        # (not parallelized for now)
        if self.options.calculator == 'AIMNET2':
            from aimnet2_firecode.interface import \
                aimnet2_optimization_refining
            return aimnet2_optimization_refining(self, maxiter=maxiter, conv_thr=conv_thr, only_fixed_constraints=only_fixed_constraints)

        self.outname = f'firecode_{"ensemble" if self.embed == "refine" else "poses"}_{self.stamp}.xyz'

        if only_fixed_constraints:
            task = 'Structure optimization (tight) / relaxing interactions'
        else:
            task = 'Structure optimization (loose)'

        max_workers = {
                        'XTB' : int(self.avail_cpus//4),
                        'ORCA' : int(self.avail_cpus//self.procs),
                        'GAUSSIAN' : int(self.avail_cpus//self.procs),
                    }[self.options.calculator]

        self.log(f'--> {task} ({self.options.theory_level}{f"/{self.options.solvent}" if self.options.solvent is not None else ""}' +
                 f' level via {self.options.calculator}, {max_workers} thread{"s" if max_workers>1 else ""})')

        self.energies.fill(0)
        # Resetting all energies since we changed theory level

        t_start = time.perf_counter()
        processes = []
        cum_time = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            opt_func = self.dispatcher.opt_funcs_dict[self.options.calculator]

            for i, structure in enumerate(deepcopy(self.structures)):
                loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')

                if only_fixed_constraints:
                    constraints = np.array([value for key, value in self.pairings_table.items() if key.isupper()])
                
                else:
                    constraints = np.concatenate([self.constrained_indices[i], self.internal_constraints]) if len(self.internal_constraints) > 0 else self.constrained_indices[i]

                pairing_dists = [self.get_pairing_dists_from_constrained_indices(_c) for _c in constraints]

                (  
                    constrained_angles_indices,
                    constrained_angles_values,
                    constrained_dihedrals_indices,
                    constrained_dihedrals_values,
                ) = self._get_angle_dih_constraints()

                process = executor.submit(
                                            timing_wrapper,
                                            opt_func,
                                            structure,
                                            self.atomnos,
                                            method=self.options.theory_level,
                                            solvent=self.options.solvent,
                                            charge=self.options.charge,
                                            maxiter=maxiter,
                                            conv_thr=conv_thr,

                                            constrained_indices=constraints,
                                            constrained_distances=pairing_dists,
                                            
                                            constrained_angles_indices=constrained_angles_indices,
                                            constrained_angles_values=constrained_angles_values,

                                            constrained_dihedrals_indices=constrained_dihedrals_indices,
                                            constrained_dihedrals_values=constrained_dihedrals_values,
                                            
                                            procs=self.procs,
                                            title=f'Candidate_{i+1}',
                                            spring_constant=2 if only_fixed_constraints else 1,

                                            payload=(
                                                self.constrained_indices[i],
                                                )
                                        )
                                        
                processes.append(process)

            for i, process in enumerate(as_completed(processes)):
                        
                loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')

                (   (
                    new_structure,
                    new_energy,
                    self.exit_status[i]
                    ),
                # from optimization function
                 
                    (
                    self.constrained_indices[i],
                    ),
                # from payload
                
                    t_struct
                # from timing_wrapper

                ) = process.result()

                # assert that the structure did not scramble during optimization
                if self.exit_status[i]:
                    constraints = (np.concatenate([self.constrained_indices[i], self.internal_constraints])
                                   if len(self.internal_constraints) > 0
                                   else self.constrained_indices[i])
                    
                    self.exit_status[i] = scramble_check(new_structure,
                                                        self.atomnos,
                                                        excluded_atoms=constraints.ravel(),
                                                        mols_graphs=self.graphs,
                                                        max_newbonds=0)
                    
                cum_time += t_struct

                if self.options.debug:
                    exit_status = 'REFINED  ' if self.exit_status[i] else 'SCRAMBLED'
                    self.debuglog(f'DEBUG: optimzation_refining ({conv_thr}) - Candidate_{i+1} - {exit_status if new_energy is not None else "CRASHED"} {time_to_string(t_struct, digits=3)}')
                
                if self.exit_status[i] and new_energy is not None:
                    self.structures[i] = new_structure
                    self.energies[i] = new_energy

                else:
                    self.energies[i] = 1E10

                ### Update checkpoint every (20*max_workers) optimized structures, and give an estimate of the remaining time
                chk_freq = int(self.avail_cpus//4) * self.options.checkpoint_frequency
                if i % chk_freq == chk_freq-1:

                    with open(self.outname, 'w') as f:        
                        for j, (structure, status, energy) in enumerate(zip(align_structures(self.structures),
                                                                            self.exit_status,
                                                                            self.rel_energies())):

                            kind = 'REFINED - ' if status else 'NOT REFINED - '
                            write_xyz(structure, self.atomnos, f, title=f'Structure {j+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({self.options.theory_level})')

                    elapsed = time.perf_counter() - t_start
                    average = (elapsed)/(i+1)
                    time_left = time_to_string((average) * (len(self.structures)-i-1))
                    speedup = cum_time/elapsed
                    self.log(f'    - Optimized {i+1:>4}/{len(self.structures):>4} structures - updated checkpoint file (avg. {time_to_string(average)}/struc, {round(speedup, 1)}x speedup, est. {time_left} left)', p=False)

        loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')
        
        elapsed = time.perf_counter() - t_start
        average = (elapsed)/(len(self.structures))
        speedup = cum_time/elapsed

        self.log((f'{self.options.calculator}/{self.options.theory_level} optimization took '
                f'{time_to_string(elapsed)} (~{time_to_string(average)} per structure, {round(speedup, 1)}x speedup)'))

        ################################################# EXIT STATUS

        self.log(f'Successfully optimized {len([b for b in self.exit_status if b])}/{len(self.structures)} structures. Non-optimized ones will {"not " if not self.options.only_refined else ""}be discarded.')

        if self.options.only_refined:
    
            mask = self.exit_status
            self.apply_mask(('structures', 'constrained_indices', 'energies', 'exit_status'), mask)

            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for unsuccessful optimization ({np.count_nonzero(mask)} left')

        ################################################# PRUNING: ENERGY

        _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
        self.energies = self.scramble(self.energies, sequence)
        self.structures = self.scramble(self.structures, sequence)
        self.constrained_indices = self.scramble(self.constrained_indices, sequence)
        # sorting structures based on energy

        if self.options.debug:
            self.dump_status(f'optimization_refining_{conv_thr}', only_fixed_constraints=only_fixed_constraints)
            self.debuglog(f'DEBUG: Dumped emebedder status after generating candidates (\"optimization_refining_{conv_thr}\")')

        if self.options.kcal_thresh is not None and only_fixed_constraints:
    
            # mask = self.rel_energies() < self.options.kcal_thresh
            energy_thr = self.dynamic_energy_thr()
            mask = self.rel_energies() < energy_thr

            self.apply_mask(('structures', 'constrained_indices', 'energies', 'exit_status'), mask)

            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for energy ({np.count_nonzero(mask)} left, ' +
                            f'{round(100*np.count_nonzero(mask)/len(mask), 1)}% kept, threshold {round(energy_thr, 1)} kcal/mol)')
    
        ################################################# PRUNING: FITNESS (POST SEMIEMPIRICAL OPT)

        self.fitness_refining(threshold=2)

        ################################################# PRUNING: SIMILARITY (POST SEMIEMPIRICAL OPT)

        self.zero_candidates_check()
        self.similarity_refining()

        ################################################# CHECKPOINT AFTER SE OPTIMIZATION      

        with open(self.outname, 'w') as f:        
            for i, (structure, status, energy) in enumerate(zip(align_structures(self.structures),
                                                                self.exit_status,
                                                                self.rel_energies())):

                kind = 'REFINED - ' if status else 'NOT REFINED - '
                write_xyz(structure, self.atomnos, f, title=f'Structure {i+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({self.options.ff_level})')

        self.log(f'--> Wrote {len(self.structures)} optimized structures to {self.outname}')

        # do not retain energies for the next optimization step if optimization was not tight
        if not only_fixed_constraints:
            self.energies.fill(0)

    def dynamic_energy_thr(self, keep_min=0.1, verbose=True):
        '''
        Returns an energy threshold that is dynamically adjusted
        based on the distribution of energies around the lowest,
        so that at least 10% of the structures are retained.

        keep_min: float, minimum percentage of structures to keep
        verbose: bool, prints comments in self.log

        '''
        active = len(self.structures)
        keep = np.count_nonzero(self.rel_energies() < self.options.kcal_thresh)

        # if the standard threshold keeps enough structures, use that
        if keep/active > keep_min:
            return self.options.kcal_thresh
        
        # if not, iterate on the relative energy values as
        # thresholds until we keep enough structures
        for thr in (energy for energy in self.rel_energies() if energy > self.options.kcal_thresh):
            keep = np.count_nonzero(self.rel_energies() < thr)

            if keep/active > keep_min:
                if verbose:
                    self.log(f"--> Dynamically adjusted energy threshold to {round(thr, 1)} kcal/mol to retain at least {round(thr*100, 1)}% of structures.")
                return thr

    def metadynamics_augmentation(self):
        '''
        Runs a metadynamics simulation (MTD) through
        the XTB program for each structure in self.structure.
        New structures are obtained from the simulations, minimized
        in energy and added to self. structures.
        '''

        self.log('--> Performing XTB Metadynamic augmentation of TS candidates')

        before = len(self.structures)
        t_start_run = time.perf_counter()

        for s, (structure, constrained_indices) in enumerate(zip(deepcopy(self.structures), deepcopy(self.constrained_indices))):

            loadbar(s, before, f'Running MTD {s+1}/{before} ')
            t_start = time.perf_counter()

            new_structures = xtb_metadyn_augmentation(structure,
                                                      self.atomnos,
                                                      constrained_indices=constrained_indices,
                                                      new_structures=5,
                                                      title=s)

            self.structures = np.concatenate((self.structures, new_structures))
            self.energies = np.concatenate((self.energies, [0 for _ in new_structures]))
            self.constrained_indices = np.concatenate((self.constrained_indices, [constrained_indices for _ in new_structures]))
        
            self.log(f'   - Structure {s+1} - {len(new_structures)} new conformers ({time_to_string(time.perf_counter()-t_start)})', p=False)

        loadbar(before, before, f'Running MTD {before}/{before} ')
        self.exit_status = np.array([True for _ in self.structures], dtype=bool)

        self.log(f'Metadynamics augmentation completed - found {len(self.structures)-before} new conformers ({time_to_string(time.perf_counter()-t_start_run)})\n')

    def csearch_augmentation(self, text='', max_structs=1000):
        '''
        Runs a conformational search for each structure in self.structure,
        preserving the current reactive atoms pose and HB interactions.
        New structures geometries are optimized and added to self.structures.
        '''

        self.warn("--> WARNING! csearch_augmentation is an experimental routine and has not been fully tested yet.")
        self.log(f'--> Performing conformational augmentation of candidates {text}')

        before = len(self.structures)
        t_start_run = time.perf_counter()
        n_out = 100 if len(self.structures)*100 < max_structs else round(max_structs/len(self.structures))
        n_out = max((1, n_out))

        for s, (structure, constrained_indices) in enumerate(zip(self.structures, self.constrained_indices)):

            loadbar(s, before, f'Performing CSearch {s+1}/{before} ', suffix=f'({len(self.structures)-before} generated)')
            t_start = time.perf_counter()

            if self.options.debug:
                dump = open(f'Candidate_{s+1}_csearch_log.txt', 'w', buffering=1)

            try:
                new_structures = csearch(
                                            structure,
                                            self.atomnos,
                                            constrained_indices=constrained_indices,
                                            keep_hb=True,
                                            mode=2,
                                            n_out=n_out,
                                            logfunction=lambda s: dump.write(s+'\n') if self.options.debug else None,
                                            title=f'Candidate_{s+1}',
                                            interactive_print=False,
                                            write_torsions=self.options.debug,
                                        )

            # if CSearch cannot be performed, just go on
            except SegmentedGraphError:
                new_structures = []

            if self.options.debug:
                dump.close()

            if len(new_structures) != 0: # could be either array or list, so have to check this way
                self.structures = np.concatenate((self.structures, new_structures))
                self.energies = np.concatenate((self.energies, [1E10 for _ in new_structures]))
                self.constrained_indices = np.concatenate((self.constrained_indices, [constrained_indices for _ in new_structures]))
        
            self.log(f'   - Candidate {s+1} - {len(new_structures)} new conformers ({time_to_string(time.perf_counter()-t_start)})', p=False)

        loadbar(before, before, f'Performing CSearch {before}/{before} ', suffix=f'{" "*15}')
        self.exit_status = np.array([True for _ in self.structures], dtype=bool)

        self.similarity_refining(rmsd=False)

        self.log(f'Conformational augmentation completed - generated {len(self.structures)-before} new conformers ({time_to_string(time.perf_counter()-t_start_run)})\n')

    def csearch_augmentation_routine(self):
        '''
        '''

        if self.options.csearch_aug:

            csearch_func = self.csearch_augmentation

            null_runs = 0

            for i in range(3):

                min_e = np.min(self.energies)

                csearch_func(text=f'(step {i+1}/3)', max_structs=self.options.max_confs)
                self.force_field_refining()

                if np.min(self.energies) < min_e:
                    delta = min_e - np.min(self.energies)
                    self.log(f'--> Lower minima found: {round(delta, 2)} kcal/mol below previous best\n')

                    if self.options.debug:
                        with open(f'best_of_aug_run_{i}.xyz', 'w') as f:
                            e, s = zip(*sorted(zip(self.energies, self.structures), key=lambda x: x[0]))
                            write_xyz(s[0], self.atomnos, f, title=f'Energy = {round(e[0], 6)}')

                    min_e = np.min(self.energies)

                else:
                    self.log('--> No new minima found.\n')
                    null_runs += 1

                if null_runs == 2:
                    break

    def saddle_refining(self):
        '''
        Performs a first order saddle optimization for each structure.

        '''
        self.log(f'--> Saddle optimization ({self.options.theory_level} level)')
        t_start = time.perf_counter()

        for i, structure in enumerate(self.structures):

            loadbar(i, len(self.structures), prefix=f'Performing saddle opt {i+1}/{len(self.structures)} ')

            try:

                self.structures[i], self.energies[i], self.exit_status[i] = ase_saddle(self,
                                                                            structure,
                                                                            self.atomnos,
                                                                            self.constrained_indices[i],
                                                                            mols_graphs=self.graphs if self.embed != 'monomolecular' else None,
                                                                            title=f'Saddle opt - Structure {i+1}',
                                                                            logfile=self.logfile,
                                                                            traj=f'Saddle_opt_{i+1}.traj',
                                                                            maxiterations=200)

            except ValueError:
                # Thrown when an ASE read fails (during saddle opt)
                self.exit_status[i] = False

        loadbar(1, 1, prefix=f'Performing saddle opt {len(self.structures)}/{len(self.structures)} ')
        t_end = time.perf_counter()
        self.log(f'{self.options.calculator} {self.options.theory_level} saddle optimization took {time_to_string(t_end-t_start)} ({time_to_string((t_end-t_start)/len(self.structures))} per structure)')
        self.log(f'Saddle opt completed for {len([i for i in self.exit_status if i])}/{len(self.structures)} structures')

        mask = self.exit_status

        self.apply_mask(('structures', 'energies', 'exit_status'), mask)

        ################################################# PRUNING: SIMILARITY (POST SADDLE OPT)

        if len(self.structures) != 0:

            t_start = time.perf_counter()
            self.structures, mask = prune_by_rmsd(self.structures, self.atomnos, max_rmsd=self.options.rmsd, debugfunction=self.debuglog)
            self.apply_mask(('energies', 'exit_status'), mask)
            t_end = time.perf_counter()
            
            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
            self.log()

        ################################################# SADDLE OPT EXTRA XYZ OUTPUT

            _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
            self.energies = scramble(self.energies, sequence)
            self.structures = scramble(self.structures, sequence)
            self.constrained_indices = scramble(self.constrained_indices, sequence)
            # sorting structures based on energy

            self.outname = f'firecode_SADDLE_TSs_{self.stamp}.xyz'
            with open(self.outname, 'w') as f:        
                for structure, energy in zip(align_structures(self.structures), self.rel_energies()):
                    write_xyz(structure, self.atomnos, f, title=f'Structure {i+1} - TS - Rel. E. = {round(energy, 3)} kcal/mol')

            self.log(f'Wrote {len(self.structures)} saddle-optimized structures to {self.outname} file\n')

        else:
            self.log()

    def print_nci(self):
        '''
        Prints and logs the non-covalent interactions guesses for final structures.

        '''
        self.log('--> Non-covalent interactions finder (EXPERIMENTAL)')
        self.nci = []

        for i, structure in enumerate(self.structures):

            nci, print_list = get_nci(structure, self.atomnos, self.constrained_indices[i], self.ids)
            self.nci.append(nci)

            if nci != []:
                self.log(f'Structure {i+1}: {len(nci)} interactions')

                for p in print_list:
                    self.log('    '+p)
                self.log()
        
        if not [_l for _l in self.nci if _l != []]:
            self.log('No particular NCIs spotted for these structures\n')

        else:
            unshared_nci = []
            for i, nci_list in enumerate(self.nci):
                for nci in nci_list:
                # for each interaction of each structure

                    if nci not in [n[0] for n in unshared_nci]:
                    # if we have not already done it

                        if not all([nci in structure_nci for structure_nci in self.nci]):
                        # if the interaction is not shared by all structures, take note

                            shared_by = [i for i, structure_nci in enumerate(self.nci) if nci in structure_nci]
                            unshared_nci.append((nci, shared_by))

            if unshared_nci != []:
                self.log('--> Differential NCIs found - these might be the source of selectivity:')
                for nci, shared_by in unshared_nci:
                    nci_type, i1, i2 = nci
                    self.log(f'    {nci_type} between indices {i1}/{i2} is present in {len(shared_by)}/{len(self.structures)} structures {tuple([i+1 for i in shared_by])}')
                self.log()

    def write_mol_info(self):
        '''
        Writes information about the firecode molecules read from the input file.

        '''

        head = ''
        for i, mol in enumerate(self.objects):

            if hasattr(mol, 'reactive_atoms_classes_dict'):

                descs = [atom.symbol+f'({str(atom)} type, {round(norm_of(atom.center[0]-atom.coord), 3)} A, ' +
                        f'{len(atom.center)} center{"s" if len(atom.center) != 1 else ""})' for atom in mol.reactive_atoms_classes_dict[0].values()]

            else:

                descs = [pt[mol.atomnos[i]].symbol for i in mol.reactive_indices]

            t = '\n        '.join([(str(index) + ' ' if len(str(index)) == 1 else str(index)) + ' -> ' + desc for index, desc in zip(mol.reactive_indices, descs)])
            
            mol_line = f' -> {len(mol.atomcoords[0])} atoms, {len(mol.atomcoords)} conformer{"s" if len(mol.atomcoords) != 1 else ""}'
            if hasattr(mol, 'pivots') and len(mol.pivots) > 0:
                mol_line += f', {len(mol.pivots[0])} pivot{"s" if len(mol.pivots[0]) != 1 else ""}'

                if mol.sp3_sigmastar:
                    mol_line += ', sp3_sigmastar'

                if any(mol.sigmatropic):
                    mol_line += ', sigmatropic'
                    if all(mol.sigmatropic):
                        mol_line += ' (all conformers)'
                    else:
                        mol_line += ' (some conformers)'

            head += f'\n    {i+1}. {mol.filename}{mol_line}\n        {t}\n'

        self.log('--> Input structures & reactive indices data:\n' + head)

    def write_options(self):
        '''
        Writes information about the firecode parameters used in the calculation, if applicable to the run.
        '''

        ######################################################################################################## PAIRINGS

        if not self.pairings_table:
            if all([len(mol.reactive_indices) == 2 for mol in self.objects]):
                self.log('--> No atom pairings imposed. Computing all possible dispositions.\n')
                # only print the no pairings statements if there are multiple regioisomers to be computed
        else:
            self.log(f'--> Atom pairings imposed are {len(self.pairings_table)}: {list(self.pairings_table.values())} (Cumulative index numbering)\n')
            
            for i, letter in enumerate(self.pairings_table):
                kind = 'Constraint' if letter.isupper() else 'Interaction'
                internal = any(isinstance(d.get(letter), tuple) for d in self.pairings_dict.values())
                kind += ' (Internal)' if internal else ''
                dist = self.get_pairing_dist_from_letter(letter)

                if self.options.shrink and not internal:
                    dist *= self.options.shrink_multiplier

                if dist is None:
                    kind += ' - will relax'
                elif kind == 'Interaction':
                    kind += f' - embedded at {round(dist, 3)} A - will relax'
                else:
                    kind += f' - constrained to {round(dist, 3)} A'

                if self.options.shrink and not internal:
                    kind += f' (to be shrinked to {round(dist/self.options.shrink_multiplier, 3)} A)'

                s = f'    {i+1}. {letter} - {kind}\n'

                for mol_id, d in self.pairings_dict.items():
                    atom_id = d.get(letter)

                    if atom_id is not None:
                        mol = self.objects[mol_id]

                        if isinstance(atom_id, int):
                            atom_id = [atom_id]
                        
                        for a in atom_id:
                            s += f'       Index {a} ({pt[mol.atomnos[a]].name}) on {mol.rootname}\n'

                self.log(s)

        for m, mol in enumerate(self.objects):
            n = 0 if m == 0 else sum(self.ids[:i])
            if mol.constraints:
                for constraint in mol.constraints:
                    ids = '-'.join([str(i) for i in constraint.indices])
                    cum_ids = '-'.join([str(i+n) for i in constraint.indices])
                    elements = '-'.join([str(pt[mol.atomnos[i]]) for i in constraint.indices])
                    t, uom = {
                        'B' : ('Bond', ' Å'),
                        'A' : ('Angle', '°'),
                        'D' : ('Dihedral', '°'),
                        }[constraint.type]
                    self.log(f'--> Additional constraint ({mol.filename}): {t} - {elements} -> {round(constraint.value, 3)}{uom}')
                    self.log(f'    Molecular ids:  {ids}')
                    self.log(f'    Cumulative ids: {cum_ids}')

        self.log()

        ######################################################################################################## EMBEDDING/CALC OPTIONS

        self.log('--> Calculation options used were:')
        for line in str(self.options).split('\n'):

            if self.embed in ('monomolecular', 'string', 'refine') and line.split()[0] in ('rotation_range',
                                                                                          'rotation_steps',
                                                                                          'rigid',
                                                                                          'suprafacial',
                                                                                          'fix_angles_in_deformation',
                                                                                          'double_bond_protection'):
                continue

            if self.embed == 'refine' and line.split()[0] in ('shrink',
                                                              'shrink_multiplier',
                                                              'fix_angles_in_deformation',
                                                              'double_bond_protection'):
                continue

            if not self.options.optimization and line.split()[0] in ('calculator',
                                                                     'double_bond_protection',
                                                                     'ff_opt',
                                                                     'ff_calc',
                                                                     'ff_level',
                                                                     'fix_angles_in_deformation',
                                                                     'only_refined',
                                                                     'rigid',
                                                                     'theory_level'):
                continue
            
            if self.options.rigid and line.split()[0] in ('double_bond_protection',
                                                          'fix_angles_in_deformation'):
                continue

            if not self.options.shrink and line.split()[0] in ('shrink_multiplier',):
                continue

            if not self.options.ff_opt and line.split()[0] in ('ff_calc', 'ff_level'):
                continue

            self.log(f'    - {line}')

    def log_warnings(self):
        '''
        Logs the non-fatal errors (warnings) at the end of a run.
        
        '''
        if self.warnings:
            self.log()
            self.log("{:*^76}".format("  W  A  R  N  I  N  G  S  "))
            self.log("{:*^76}".format(" your run generated these non-fatal warnings "))
            self.log()

            for warning in self.warnings:
                self.log(auto_newline(warning, max_line_len=65))
                self.log()

            self.log("*"*76)

    def run(self):
        '''
        Run the firecode program.
        
        '''
        self.write_mol_info()

        if self.embed is None:
            self.log('--> No embed requested, exiting.\n')
            self.normal_termination()

        if self.embed == 'error':
            self.log('--> Embed type not recognized, exiting.\n')
            self.normal_termination()

        if self.embed == 'data':
            self.data_termination()

        if not self.options.let and (
               self.embed in ('cyclical', 'chelotropic')) and (
               max([len(mol.atomcoords) for mol in self.objects]) > 100) and (
               not self.options.rigid):

            self.options.rigid = True

            self.log('--> Large embed: RIGID keyword added for efficiency (override with LET)')

        self.write_options()

        if not hasattr(self, "t_start_run"):
            self.t_start_run = time.perf_counter()

        if self.options.dryrun:
            self.log('\n--> Dry run requested: exiting.')
            self.normal_termination()

        try: # except KeyboardInterrupt
            try: # except ZeroCandidatesError()
                self.generate_candidates()
                
                if self.options.bypass:
                    self.write_structures('unoptimized', energies=False)
                    self.normal_termination()

                self.compenetration_refining()
                self.similarity_refining(rmsd=True if self.embed == "refine" else False, verbose=True)

                if self.options.optimization:

                    if self.options.ff_opt:

                        # perform safe optimization only for embeds
                        if len(self.objects) > 1 and self.options.ff_calc == 'XTB':
                            # self.log(f"--> Performing {self.options.calculator} FF pre-optimization (loose convergence, molecular and pairing constraints)\n")
                            self.force_field_refining(conv_thr="loose", prevent_scrambling=True)

                        if len(self.structures) > 500:
                            # self.log(f"--> Performing {self.options.calculator} FF optimization (loose convergence, pairing constraints, step 1/2)\n")
                            self.force_field_refining(conv_thr="loose")

                        # self.log(f"--> Performing {self.options.calculator} FF optimization (tight convergence, fixed constraints only, step 2/2)\n")
                        self.force_field_refining(conv_thr="tight", only_fixed_constraints=True)

                        # self.csearch_augmentation_routine()

                    if not (self.options.ff_opt and self.options.theory_level == self.options.ff_level):
                        # If we just optimized at a (FF) level and the final
                        # optimization level is the same, avoid repeating it

                        if self.options.calculator == "ORCA":
                        # Perform stepwise pruning of the ensemble for more expensive theory levels
                            
                            self.log("--> Performing ORCA optimization (3 iterations, step 1/3)\n")
                            self.optimization_refining(maxiter=3)

                            self.log("--> Performing ORCA optimization (5 iterations, step 2/3)\n")
                            self.optimization_refining(maxiter=5)

                            self.log("--> Performing ORCA optimization (convergence, step 3/3)\n")

                        if len(self.structures) > 500:
                            self.optimization_refining(conv_thr='loose')
                            # final uncompromised optimization (with fixed constraints and interactions active)

                        self.optimization_refining(conv_thr='tight', only_fixed_constraints=True)
                        # final uncompromised optimization (with only fixed constraints active)

                else:
                    self.write_structures('unoptimized', energies=False)
                    # accounting for output in "refine" runs with NOOPT

            except ZeroCandidatesError:
                t_end_run = time.perf_counter()
                s = ('    Sorry, the program did not find any reasonable embedded structure. Are you sure the input indices and pairings were correct? If so, try these tips:\n'
                     '    - If no structure passes the compenetration check, the SHRINK keyword may help (see documentation).\n'
                     '    - Similarly, enlarging the spacing between atom pairs with the DIST keyword facilitates the embed.\n'
                     '    - If no structure passes the fitness check, try adding a solvent with the SOLVENT keyword.\n'
                     '    - Impose less strict compenetration rejection criteria with the CLASHES keyword.\n'
                     '    - Generate more structures with higher STEPS and ROTRANGE values.\n'
                )

                self.log(f'\n--> Program termination: No candidates found - Total time {time_to_string(t_end_run-self.t_start_run)}')
                self.log(s)
                self.close_log_streams()
                clean_directory()
                sys.exit()

            ##################### AUGMENTATION - METADYNAMICS / CSEARCH

            if self.options.metadynamics:

                self.metadynamics_augmentation()
                self.optimization_refining()
                self.similarity_refining()

            ##################### POST FIRECODE - SADDLE, NEB, NCI, VMD

            # if (self.options.optimization or self.options.ff_opt) and not self.options.bypass:
            #     self.write_vmd()

            # if self.options.neb:
            #     self.hyperneb_refining()

            if self.options.saddle:
                self.saddle_refining()
                
            if self.options.nci and self.options.optimization:
                self.print_nci()
            
            self.log_warnings()
            self.normal_termination()

            ################################################ END

        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt requested by user. Quitting.')
            sys.exit()

    def data_termination(self):
        '''
        Type of termination for runs when there is no embedding,
        but some computed data are to be shown in a formatted way.
        '''

        if any('pka>' in op for op in self.options.operators):
            self.pka_termination()

        if len([op for op in self.options.operators if 'scan>' in op]) > 1:
            self.scan_termination()

        self.normal_termination()

    def pka_termination(self):
        '''
        Print data acquired during pKa energetics calculation
        for every molecule in input
        '''

        self.log('\n--> pKa energetics (from best conformers)')
        solv = 'gas phase' if self.options.solvent is None else self.options.solvent

        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ['Name', '#(Symb)', 'Process', 'Energy (kcal/mol)']

        for mol in self.objects:
            if hasattr(mol, 'pka_data'):
                table.add_row([mol.rootname,
                               f'{mol.reactive_indices[0]}({pt[mol.atomnos[mol.reactive_indices[0]]].symbol})',
                               mol.pka_data[0],
                               mol.pka_data[1]])

        # Add pKa column if we were given a reference
        if hasattr(self, 'pka_ref'):

            pkas = []
            for mol in self.objects:
                if mol.filename == self.pka_ref[0]:
                    dG_ref = mol.pka_data[1]
                    break

            for mol in self.objects:
                process, free_energy = mol.pka_data

                dG = free_energy - dG_ref if process == 'HA -> A-' else dG_ref - free_energy
                # The free energy difference has a different sign for acids or bases, since
                # the pKa for a base is the one of its conjugate acid, BH+

                pka = dG / (np.log(10) * 1.9872036e-3 * 298.15) + self.pka_ref[1]
                pkas.append(round(pka, 3))

            table.add_column(f'pKa ({solv}, 298.15 K)', pkas)

        self.log(table.get_string())
        self.log(f'\n  Level used is {self.options.theory_level} via {self.options.calculator}' + 
                 f", using the ALPB solvation model for {self.options.solvent}" if self.options.solvent is not None else "")

        if len(self.objects) == 2:
            mol0, mol1 = self.objects
            if hasattr(mol0, 'pka_data') and hasattr(mol1, 'pka_data'):
                tags = (mol0.pka_data[0],
                        mol1.pka_data[0])
                if 'HA -> A-' in tags and 'B -> BH+' in tags:
                    dG = mol0.pka_data[1] + mol1.pka_data[1]
                    self.log('\n  Equilibrium data:')
                    self.log(f'\n    HA + B -> BH+ + A-    K({solv}, 298.15 K) = {round(np.exp(-dG/(1.9872036e-3 * 298.15)), 3)}')
                    self.log(f'\n                         dG({solv}, 298.15 K) = {round(dG, 3)} kcal/mol')

    def scan_termination(self):
        '''
        Print the unified data and write the cumulative plot
        for the approach of all the molecules in input
        '''
        # import pickle

        import matplotlib.pyplot as plt

        plt.figure()

        for mol in self.objects:
            if hasattr(mol, 'scan_data'):
                plt.plot(*mol.scan_data, label=mol.rootname)

        plt.legend()
        plt.title('Unified scan energetics')
        plt.xlabel('Distance (A)')
        plt.gca().invert_xaxis()
        plt.ylabel('Rel. E. (kcal/mol)')
        plt.savefig(f'{self.stamp}_cumulative_plt.svg')
        # with open(f'{self.stamp}_cumulative_plt.pickle', 'wb') as _f:
        #     pickle.dump(fig, _f)
        
        self.log(f'\n--> Written cumulative scan plot at {self.stamp}_cumulative_plt.svg')