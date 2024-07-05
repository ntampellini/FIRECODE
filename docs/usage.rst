.. _usg:

Usage
=====

The program can be run from terminal, with the command:

::

    python -m firecode myinput.txt -n [custom_title]

A custom name for the run can be optionally provided with the -n flag, otherwise a time
stamp will be used to name the output files.

It is also possible, for simple runs, to avoid creating an input file, and writing
instruction in a string after the ``-cl``/``--command_line`` argument:

::

    python -m firecode -cl "csearch> molecule.xyz"
    python -m firecode --command_line "csearch> molecule.xyz"

In this case, an input file will be written for you by the program.

Input formatting
----------------

The input can be any text file. The extension is arbitrary but I suggest
sticking with ``.txt``.

-  Any blank line will be ignored
-  Any line starting with ``#`` will be ignored
-  Keywords, if present, need to be on first non-blank, non-comment line

Then, molecule files are specified. A molecule line is made up of these elements, in this order:

-  An operator (optional)
-  The molecule file name (required)
-  Indices (numbers) and pairings (letters) for the molecule (optional)

An example with all three is ``opt> butadiene.xyz 6a 8b``.

FIRECODE can work with all molecular formats read by
`cclib <https://github.com/cclib/cclib>`__, but best practice is using
only the ``.xyz`` file format, particularly for multimolecular files
containing different conformers of the same molecule. **Molecule indices
are counted starting from zero!**

Operators
+++++++++

The first step of every run is the execution of the requested operators. See the
:ref:`operators <op_kw>` page to see the full set of tools available.

Embedding runs
++++++++++++++

Then, if the input you provided is consistent with an embedding, one will be carried out.
Embeddings can be of six kinds:

-  **monomolecular** - One molecule, two reactive atoms (*i.e.*
   Cope rearrangements)
-  **dihedral** - One molecule, four reactive atoms (*i.e.*
   racemization of BINOL)
-  **string** - Two molecules, one reactive atom each (*i.e.* SN2
   reactions)
-  **chelotropic** - Two molecules, one with a single reactive
   atom and the other with two reactive atoms (*i.e.* epoxidations)
-  **cyclical** (bimolecular) - Two molecules, two reactive atoms
   each (*i.e.* Diels-Alder reactions)
-  **cyclical** (trimolecular) - Three molecules, two reactive
   atoms each (*i.e.* reactions where two partners are bridged by a
   carboxylic acid like the example above)

Reactive atoms supported include various hybridations of
``C, H, O, N, P, S, F, Cl, Br and I``. Many common metals are also
included (``Li, Na, Mg, K, Ca, Ti, Rb, Sr, Cs, Ba, Zn``), and it is easy
to add more if you need them (from *reactive_atoms_classes.py*). 

.. figure:: /images/embeds.svg
   :alt: Embeds Infographic
   :align: center
   :width: 700px

   *Colored dots represent the imposed atom pairings.*

Pairings
++++++++

After each reactive index, it is possible to specify a pairing letter
representing the "flag" of that atom. If provided, the
program will only yield poses that respect these atom
pairings. It is also possible to specify more than one flag per atom,
useful for chelotropic embeds - *i.e.* the hydroxyl oxygen atom of a peracid, as
``4ab``.

.. If a ``NEB`` calculation is to be performed on a trimolecular transition
.. state, the reactive distance "scanned" is the first imposed (``a``). See
.. ``NEB`` keyword in the keyword section.

Embeddings: good practice and suggested options
-----------------------------------------------

When modeling a reaction through FIRECODE geometrical embedding, I suggest following these
guidelines. Not all of them apply to all embed types, but they will
surely help in leveraging the program in the best way.

1) Assess that the reaction is supported, and plan on what the
input will look like. See Input formatting above for help.

1) Obtain molecular structures in .xyz format. If more conformers are provided,
they must be in a multimolecular ``.xyz`` file, and **atom ordering
must be consistent throughout all structures.** Otherwise, they will just be
skipped by the module used to read molecular files (cclib).

2) If a given molecule is present in the transition state, but it is
not strictly involved in bonds breaking/forming, then that molecule
can be pre-complexed to the moiety with which it is interacting. That is,
the bimolecular complex can be used as a starting point. This can be the
case for multimolecular adducts of non-covalently-bound catalysts.

1) Make sure to use the ``csearch>`` and/or ``csearch_hb>`` operators or provide conformational
ensembles obtained with other software. Note that the CSearch implementation here
is meant to be fast, scalable, and efficient, and is not able to sample ring conformations.
The most effective conformational search tool (although usually slower) is ``mtd_search>``
which runs a metadynamic-based conformational search through CREST.
   
1) Understand what atoms are reacting for each structure and record
their index (**starting from 0!**). If you are unsure of reactive atomic
indices, you can run a test input without indices, and the program will
ask you to manually specify them from the ASE GUI by clicking. This is
not possible if you are running FIRECODE on STPs with no desktop access.
When choosing this option of manually picking atoms, it is not
possible to specify atom pairings. Therefore, I suggest using this
option only to check the reactive atoms indices and then building a
standard input file.

1) Optionally, after specifying reactive indices, the ``CHECK`` keyword
can be used. A series of pop-up ASE GUI windows will be displayed,
showing each molecule with a series of red dots around the reactive
atoms chosen. This can be used to check "orbital" positions or conformer
reading faults (scroll through conformers with page-up and down
buttons). Program will terminate after the last visualization is closed.

1) FIRECODE default parameters are tentatively optimized to yield good results
for the most common situations. However, if you
have more information about your system, specifying details of the pairings
and options for your system is likely to give better results. For example,
embedding trimolecular TSs without imposed pairings generates about 8
times more structures than an embed with defined pairings. Also, if
reactive atoms distances in the transition state is known, using the
``DIST`` keyword can yield structures that are quite close to
ones obtained at higher levels of theory. If no pairing
distances are provided, a distance guess is performed based on the atom type
(defaults are editable in the ``parameters.py`` file).

1) If the reaction involves big molecules, or if a lot of conformations
are to be used, a preliminar run using the ``NOOPT`` keyword may be a
good idea to see how many structures are generated and would require
optimization in a standard run.

1) If FIRECODE does not find any suitable candidate for the given reacion,
this could be bacause of different factors. An option is to try the ``SHRINK`` keyword
(see keywords section).
Note that the use of ``RIGID``, while speeding up cyclical embeds
considerably, could jeopardize finding some transition state arrangements.

1) If you are not sure about what to do, or have any other questions I will be
happy to have a chat with you. Send me an email `here <mailto:nicolo.tampellini@yale.edu>`__!