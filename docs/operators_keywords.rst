.. _op_kw:

Operators and keywords
++++++++++++++++++++++

Operators
=========

Many workflows can be defined by one or more operators acting on an input structure.

.. code-block::

   freq> opt> crest> mol.xyz

Operators act on the adjacent structure file in "inside out" order, in analogy with
mathematical operators. In the example above, ``crest>`` is executed first, and the resulting
output (``mol_crest_confs.xyz``) is piped to ``opt>`` and then ``freq``. All non-terminating operators can be chained
to define a workflow.
Here is a list of the currently available operators:

-  ``opt>`` - Performs an optimization of all conformers of a molecule before
   running the embedding. Generates a new ``molecule_opt.xyz`` file with the optimized
   coordinates. This operator is constraint-aware and will perform constrained
   optimizations obeying constraints and the DIST keyword.

.. code-block::

   saddle> opt> mol.xyz
     B 12 25 1.62 # bond constraint: enforced in "opt>", ignored in "saddle>"

-  ``refine>`` - Very similar to ``opt>``, acts on a multimolecular input file pruning similar structures and optimizing it
   at the chosen theory level. Soft, non-fixed constraints ("interactions") are enforced during a
   first loose optimization, and subsequently relaxed (terminating operator).

-  ``crest_search>`` / ``crest>`` / ``mtd_search>`` / ``mtd>`` - Performs a metadynamics-based conformational
   search on the specified input structure through `CREST <https://crest-lab.github.io/crest-docs/>`__.
   The operator is constraints-aware
   and will constrain the specified distances (both "fixed" and not). Generates a new ``mol_crest_confs.xyz``
   file with the crest-optimized conformers. The default level is GFN2//GFN-FF (see CREST docs).
   Charge and multiplicity are passed from the molecule or keyword line.

.. code-block::

   mtd> molecule.xyz 4A 8A charge=-1

-  ``rdkit_search>`` / ``racerts>`` - Performs a conformational search with the `ETKDG algorithm <https://pubs.acs.org/doi/10.1021/acs.jcim.0c00025>`__
   via RDKit. If constraints are provided, uses the ``racerts`` library to generate TS conformers by fixing the atoms involved in the constraint.
   Generates a new ``mol_rdkit_confs.xyz`` file with the generated conformers.

-  ``goat>`` - Performs a conformational search with the `GOAT algorithm <https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.202500393>`__ via ORCA.
   Generates a new ``mol_goat_confs.xyz`` file with the generated conformers.

-  ``neb>`` - Runs a NEB (Nudged Elastic Band) procedure on the provided structures.
   The implementation is a climbing image nudged elastic band (CI-NEB) TS search.
   The operator should be used on a single file that contains two or more structures.
   The first and last structures will be used as start and endpoints. If three are provided, the intermediate
   structure will be used as a TS guess in the interpolation. If more are provided, they will all be used
   in place of (or to aid) interpolation to reach the desired number of images
   (set with the ``IMAGES=n`` or NEB(IMAGES=n) keywords). Default is 7.

- ``saddle>`` - Runs a saddle point optimization on each conformer provided. If chained after a linear ``scan>``,
  acts on the highest energy structure of the linear scan. All constraints provided are ignored. Can be conveniently chained to
  ``scan>`` (linear scan only), ``opt>`` and ``neb>``. Will automatically perform a vibrational analysis
  on each structure that is successfully optimized (no need to follow it with a ``freq>`` operator).

-  ``freq>`` - Runs a frequency calculation on each input structure at the specified temperature and pressure.

-  ``scan>`` - Runs a scan of the desired distance or dihedral angle, based on the number of indices provided.
   For distance scans, the atomic indices will be approached if they are not bound, and separated if they are.
   This could be chained with, for example, a NEB calculation, where the scan trajectory will be used as a starting
   point for a NEB calculation.

   .. code-block::

      neb> scan> mol.xyz 3 4

   For dihedral scans, the dihedral is rotated the full 360° in both clockwise and counterclockwise directions.
   Maxima above a certain threshold (set with ``KCAL=n``) are scanned again in smaller steps. All maxima for the second mode accurate
   scan are saved to firecode_maxima_mol.xyz. The optimized geometries for the initial scans are saved with the relative
   names. This version of the ``scan>`` operator is terminating and cannot be chained.

Deprecated operators (legacy)
-----------------------------

-  ``firecode_search>`` - Performs a diversity-based, torsionally-clustered conformational
   search on the specified input structure. Only the bonds that do not brake imposed
   constraints are rotated (see examples). Generates a new ``molecule_confs.xyz``
   file with the unoptimized conformers.

-  ``firecode_search_hb>`` - Analogous to ``csearch>``, but recognizes the hydrogen bonds present
   in the input structure and only rotates bonds that keep those hydrogen bonds in place.
   Useful to restrict the conformational space that is explored, and ensures that the final
   poses possess those initial hydrogen bonds.

-  ``firecode_random_search>`` - Performs a random torsion-based conformational
   search on the specified input structure (fast but inaccurate). Only the bonds that do not brake imposed
   constraints are rotated (see examples). Generates a new ``molecule_confs.xyz``
   file with the unoptimized conformers.


General keywords
================

Keywords are case-insensitive and are divided by at least one blank space.
Some of them are self-sufficient (*i.e.* ``NCI``), while some others require an
additional input (*i.e.* ``KCAL=10`` or ``NEB(images=10, ci=true)``).
Commas are used to divide keyword arguments where more than
one is accepted, like in ``NEB``.

.. list-table::
   :header-rows: 1
   :align: center

   * - Keyword
     - Description

   * - CALC
     - Overrides default calculator in ``settings.py``. Available calculators are
       ORCA, XTB, TBLITE, AIMNET2 and UMA. Syntax: ``CALC=ORCA``

   * - CHARGE
     - Specify the charge to be used in optimizations.

   * - NCI
     - `crest>` and `goat>` runs: runs these programs in NCI
       (non-covalent interaction) mode, *i.e.* applying an ellipsoid wall potential to prevent
       unconstrained non-covalent complexes from evaporating.

   * - DEBUG
     - Outputs more intermediate files and information in general.
       Structural adjustments, distance refining and similar processes will
       output ASE ".traj" trajectory files. Legacy: It will also produce
       "hypermolecule" ``.xyz`` files for the first conformer of each
       structure, with dummy atoms (X) in each FIRECODE "orbital" position.

   * - DRYRUN
     - Skips lenghty operations (operators, embedding, refining)
       but retains other functions and printouts. Useful for debugging and
       checking purposes.

   * - IMAGES
     - Number of images to be used in NEB, ``neb>`` and ``mep_relax>`` jobs.

   * - KCAL
     - In refinements (``refine>``), trim output structures to a given value of relative energy from the most stable
       (in kcal/mol, default is 10). In ``scan>`` runs, sets the threshold to consider a local
       energy maxima for further refinement. Syntax: ``KCAL=n``.

   * - LET
     - Overrides safety checks that prevent the program from
       running calculations that seem too large and likely erroneous.

   * - LEVEL
     - Manually set the theory level to be used.
       White spaces, if needed, can be expressed in input files with underscores. Syntax (ORCA):
       ``LEVEL(B3LYP_def2-TZVP)``. Defaults can be found in settings.py, and can be modified by running
       the module with the -s flag (``>>> firecode -s``).

   * - NEB
     - Set NEB options. Syntax and default values: ``NEB(images=7, preopt=true, ci=true)``.

   * - ONLYREFINED
     - Discard structures that do not successfully
       refine bonding distances. Set by default with the ``SHRINK`` keyword
       and for monomolecular TSs.

   * - P
     - Pressure, in atm (default 1.0). Syntax: ``P=100.0``

   * - PKA
     - Specify the reference pKa for a compound in multimolecular
       pKa calculation runs. Syntax: ``PKA(mol.xyz)=11``

   * - PROCS
     - Manually set the number of cores to be used in each
       higher level (non-force field) calculation, overriding the value in
       ``settings.py``. Syntax: ``PROCS=32``

   * - RMSD
     - RMSD threshold (Angstroms) for structure pruning.
       The smaller, the more retained structures (default is 0.25 A).
       Syntax: ``THRESH=n``

   * - T
     - Temperature, in Kelvin (default 298.15). Syntax: ``T=300.0``

   * - T_C
     - Temperature, in Celsius (default 25.0). Syntax: ``T_C=25.0``


Legacy: Embedding-specific keywords
-----------------------------------

.. list-table::
   :header-rows: 1
   :align: center

   * - Keyword
     - Description

   * - BYPASS
     - Debug keyword. Used to skip all pruning steps and
       directly output all the embedded geometries.

   * - CLASHES
     - Manually specify the max number of clashes and/or
       the distance threshold at which two atoms are considered clashing.
       The more forgiving (higher number, smaller dist), the more structures will reach the geometry
       optimization step. Default values are num=0 and dist=1.5 (A). Syntax: ``CLASHES(num=3,dist=1.2)``

   * - CONFS
     - Override the maximum number of conformers to be used
       for the refinement or embedding of each molecule (default is 1000). Syntax: ``CONFS=10000``

   * - DEEP
     - Performs a deeper search, retaining more starting
       points for calculations and smaller turning angles. Equivalent to
       ``THRESH=0.3 STEPS=72 CLASHES=(num=1,dist=1.3)``. **Use with care!**

   * - DIST
     - Manually imposed distance between specified atom
       pairs across the same or different molecules, in Angstroms.
       Syntax uses parenthesis and commas. Spaces are tolerated:
       ``DIST(a=2.345, b=3.67, c=2.1)``

   * - EZPROT
     - Preserve the E or Z configuration of double bonds
       (C=C and C=N) during the embed. Likely to be useful only for
       monomolecular embeds, where molecular distortion is often important, and
       undesired isomerization processes can occur.

   * - NOOPT
     - Skip the optimization steps, directly writing
       structures to file after compenetration and similarity pruning.
       Dihedral embeds: performs rigid scans instead of relaxed ones.

   * - RIGID
     - Only applies to "cyclical"/"chelotropic" embeds.
       Avoid bending structures to better build TSs.

   * - ROTRANGE
     - Only applies to "cyclical"/"chelotropic" embeds.
       Manually specify the rotation range to be explored around the
       structure pivot. Default is 45. Syntax: ``ROTRANGE=90``

   * - SHRINK
     - Exaggerate orbital dimensions during embed, scaling
       them by a specified factor. If used as a single keyword (``SHRINK``),
       orbital dimensions are scaled by a factor of one and a half. A syntax
       like ``SHRINK=3.14`` allows for custom scaling. This scaling makes it
       easier to perform the embed without having molecules clashing one
       into the other. Then, the correct distance between reactive atom
       pairs is achieved as for standard runs by spring constraints during
       optimization. The larger the scaling, the more the program
       is likely to find at least some transition state poses, but the more
       time-consuming the step of distance refinement is going to be. Values
       from 1.5 to 3 are likely to do what this keyword was thought for.

   * - SIMPLEORBITALS
     - Override the automatic orbital assignment, using "Single"
       type orbitals for every reactive atom (faster embeds, less candidates). Ideal
       in conjuction with SHRINK to make up for the less optimal orbital positions.

   * - STEPS
     - Applies to "string", "cyclical" and "chelotropic" embeds. Manually
       specify the number of steps to be taken in scanning rotations. For
       "string" embeds, the range to be explored is the full 360°, and the
       default ``STEPS=24`` will perform 15° turns. For "cyclical" and
       "chelotropic" embeds, the rotation range to be explored is
       +-\ ``ROTRANGE`` degrees. Therefore the default values, equivalent to
       ``ROTRANGE=45 STEPS=5``, will sample five equally spaced positions between
       +45 and -45 degrees (going through zero).

   * - SUPRAFAC
     - Only retain suprafacial orbital configurations in
       cyclical TSs. Thought for Diels-Alder and other cycloaddition
       reactions.
