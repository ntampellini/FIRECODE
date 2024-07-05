.. _introduction:

.. figure:: /images/logo.png
   :alt: FIRECODE logo
   :align: center
   :width: 600px

.. figure:: /images/intro_embed.PNG
   :alt: Embed logic scheme
   :align: center
   :width: 700px


What it is
----------

FIRECODE is a computational chemistry toolbox for the generation, optimization and refinement
of conformational ensembles. It features many flexible and highly customizable workflow utilities
including conformer generation (via `CREST <https://github.com/crest-lab/crest>`_ or FIRECODE),
constrained ensemble optimization through popular calculators like `XTB <https://github.com/grimme-lab/xtb>`_,
`ORCA <https://www.orcasoftware.de/tutorials_orca/>`_, `GAUSSIAN <https://gaussian.com/>`_
and Pytorch Neural Network models via `ASE <https://github.com/rosswhitfield/ase>`_
(`AIMNET2 <https://github.com/isayevlab/AIMNet2>`_). It implements a series of conformational
pruning routines based on inertia tensors, RMSD, symmetry-corrected RMSD, and more. It can
also assemble non-covalent adducts from conformational ensembles (embedding) for fast and
automated generation and evaluation of ground and transtition state-like structures. CPU and
GPU multithreading is implemented throughout the codebase and linear algebra-intensive modules
are compiled at runtime via `Numba <https://github.com/numba/numba>`_.

What it does
------------

**Generate accurately spaced poses** for bimolecular and trimolecular
ground or transition states of organic molecules by assembling combinations of conformations (embedding).
The distance between pairs of atoms can be specified, so as to obtain 
different poses with precise molecular spacings.

**Perform ensemble refinement** by optimizing conformers with the option of applying ensemble-wide
constraints, and rejecting similar structures (rotamers and enantiomers).

First, :ref:`operators<op_kw>` (if provided) are applied to input structures. Then, if more
than one input file is provided and the input format conforms to some embedding algorithm,
a series of poses is created and then refined. Alternatively, it is also
possible to generate and refine an ensemble starting from a single structure or refine
user-provided ensembles (see :ref:`some examples<exs>`).

How the embedding works
-----------------------

Combinations of conformations of individual molecules are arranged in space using
some basic modeling of atomic orbitals and a good dose of linear algebra.

.. figure:: /images/orbitals.png
   :align: center
   :alt: Schematic representation of orbital models used for the embeddings
   :width: 85%

   *Schematic representation of orbital models used for the embeddings*


How the ensemble refinement works
---------------------------------

Ensemble refinement is a combination of free or constrained optimizations and similarity pruning.
Similarity is evaluated in a series of ways:

 - **TFD** (torsion fingerprint deviation) - only for monomolecular embeds/ensembles

 - **MOI** (moment of inertia) - quickly removes enantiomers and rotamers

 - **Heavy-atoms RMSD**

 - **Rotationally-corrected heavy-atoms RMSD** - invariant for periodic rotation of locally symmetrical groups (i.e. tBu, Ph, NMe2)

.. Extra features
.. --------------

.. **Transition state searches**

.. FIRECODE implements routines for locating transition states, both for poses generated
.. through the program and as a standalone functionality. The ``SADDLE`` and ``NEB``
.. keywords and the ``saddle>`` and ``neb>`` operators are available:

.. - With ``SADDLE``, a geometry optimization to the closest energetic maxima is performed
..   on the embedded structures, using the `Sella <https://github.com/zadorlab/sella>`__ library through ASE.

.. - With ``NEB``, a climbing image nudged elastic band (CI-NEB) transition state
..   search is performed on each embedded structure. This tends to perform best with the scan> operator,
..   where the initial minimum energy path is extracted from the distance or dihedral scan points.

.. - The ``saddle>`` and ``neb>`` operators work in the same way on user-provided structures.

.. See the :ref:`operators and keywords page<op_kw>` for more details on their usage.