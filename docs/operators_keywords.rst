.. _op_kw:

Operators and keywords
======================

Operators
+++++++++

Molecule files can be preceded by *operators*, like
``opt> molecule.xyz``. They operate on the input file before it is
fed to FIRECODE embeddings, or modify the behavior of the program to
use some of its functionalities, without running a full embedding.
Here is a list of the currently available operators:

-  ``opt>`` - Performs an optimization of all conformers of a molecule before
   running the embedding. Generates a new ``molecule_opt.xyz`` file with the optimized
   coordinates. This operator is constraint-aware and will perform constrained
   optimizations obeying the distances provided with DIST.

-  ``csearch>`` - Performs a diversity-based, torsionally-clustered conformational
   search on the specified input structure. Only the bonds that do not brake imposed
   constraints are rotated (see examples). Generates a new ``molecule_confs.xyz``
   file with the unoptimized conformers.

-  ``csearch_hb>`` - Analogous to ``csearch>``, but recognizes the hydrogen bonds present
   in the input structure and only rotates bonds that keep those hydrogen bonds in place.
   Useful to restrict the conformational space that is explored, and ensures that the final
   poses possess those initial hydrogen bonds.

-  ``rsearch>`` - Performs a random torsion-based conformational
   search on the specified input structure (fast but not the most accurate). Only the bonds that do not brake imposed
   constraints are rotated (see examples). Generates a new ``molecule_confs.xyz``
   file with the unoptimized conformers.

-  ``mtd_search>``/ ``mtd>`` - Performs a metadynamics-based conformational
   search on the specified input structure through `CREST <https://crest-lab.github.io/crest-docs/>`__
   (slower but best). It is letter constraints-aware
   and will constrain the specified distances. Generates a new ``molecule_mtd_confs.xyz``
   file with the crest-optimized conformers. The default level is GFN2//GFN-FF (see CREST docs).
   It is also possible to pass a charge attribute for the molecule via molecule-line 'charge' attribute.
   
   ::
   
       mtd> molecule.xyz 4A 8A charge=-1

-  ``neb>`` - Runs a NEB (Nudged Elastic Band) procedure on external structures. Useful 
   if working with calculators that do not natively integrate such methods (*i.e.* Gaussian). 
   The implementation is a climbing image nudged elastic band (CI-NEB) TS search with 7 moving images.  
   The operator should be used on a file that contains two or three structures, that will be interpolated as the
   start, end and optional transition state guess for the NEB optimization. Alternatively, an odd number of
   structures can be used as a starting point, overriding interpolation.
   
   A graph with the energy of each image is written, along with the MEP guess 
   and the converged MEP. It is also possible to provide three structures, that will be used as start,
   end and transition state guess, respectively.

-  ``autoneb>`` - Analogous to the ``neb>`` operator, but automatically builds a MEP guess from a single structure.
   At the moment, it is only able to do so for atropisomeric 7-member rings inversions.

-  ``refine>`` - Reads the (multimolecular) input file and treats it as an ensemble generated
   during a FIRECODE embedding. That is the ensemble is pruned removing similar structure, optimized
   at the theory level(s) chosen and again pruned for similarity.

All non-terminating operators can be nested:

::

   refine> rsearch> opt> mol.xyz
   
In this case, they are executed from the inner to the outer, i.e. from right to left.

Keywords
++++++++

Keywords are case-insensitive and are divided by at least one blank space.
Some of them are self-sufficient (*i.e.* ``NCI``), while some others require an
additional input (*i.e.* ``STEPS=10`` or ``DIST(a=1.8,b=2,c=1.34)``). In
the latter case, whitespaces are NOT allowed inside the parenthesis.
Floating point numbers are to be expressed with points like ``3.14``,
while commas are only used to divide keyword arguments where more than
one is accepted, like in ``DIST``.

-  **BYPASS** - Debug keyword. Used to skip all pruning steps and
   directly output all the embedded geometries.

-  **CALC** - Overrides default calculator in ``settings.py``.
   (Gaussian, ORCA, XTB, Syntax: ``CALC=ORCA``

-  **CHARGE** - Specify the charge to be used in optimizations.

-  **CHECK** - Visualize the input molecules through the ASE GUI, to
   check orbital positions or conformers reading faults. *(not available
   from CLI)*

-  **CLASHES** - Manually specify the max number of clashes and/or
   the distance threshold at which two atoms are considered clashing.
   The more forgiving (higher number, smaller dist), the more structures will reach the geometry
   optimization step. Default values are num=0 and dist=1.5 (A). Syntax: ``CLASHES(num=3,dist=1.2)``

-  **CONFS** - Override the maximum number of conformers to be used for
   the embed of each molecule (default is 1000). Syntax: ``CONFS=10000``

-  **CRESTNCI** - mtd>/mtd_search> runs: passes the "--nci" argument to CREST, running
   it in non-covalent interaction mode, *i.e.* applying a wall potential to prevent
   unconstrained non-covalent complexes to evaporate during the metadynamics.

-  **DEBUG** - Outputs more intermediate files and information in general.
   Structural adjustments, distance refining and similar processes will
   output ASE ".traj" trajectory files. It will also produce
   "hypermolecule" ``.xyz`` files for the first conformer of each
   structure, with dummy atoms (X) in each FIRECODE "orbital" position.

-  **DEEP** - Performs a deeper search, retaining more starting
   points for calculations and smaller turning angles. Equivalent to
   ``THRESH=0.3 STEPS=72 CLASHES=(num=1,dist=1.3)``. **Use with care!**

-  **DIST** - Manually imposed distance between specified atom
   pairs, in Angstroms. Syntax uses parenthesis and commas:
   ``DIST(a=2.345,b=3.67,c=2.1)``

-  **DRYRUN** - Skips lenghty operations (operators, embedding, refining)
   but retains other functions and printouts. Useful for debugging and
   checking purposes.

.. -  **ENANTIOMERS** - Do not discard enantiomeric structures.

-  **EZPROT** - Preserve the E or Z configuration of double bonds
   (C=C and C=N) during the embed. Likely to be useful only for
   monomolecular embeds, where molecular distortion is often important, and
   undesired isomerization processes can occur.

-  **FFCALC** - Overrides default force field calculator in ``settings.py``.
   Values can be ``OB``, ``Gaussian``, ``XTB``. Syntax: ``FFCALC=OB``

-  **FFLEVEL** - Manually set the theory level to be used for force field
   calculations. Default is UFF for Openbabel and Gaussian, GFN-FF for XTB.
   Standard values can be modified by running the module with the -s flag
   (>>> python -m firecode -s).

-  **FFOPT** - Manually turn on ``FFOPT=ON`` or off ``FFOPT=OFF`` the force
   field optimization step, overriding the value in ``settings.py``.

-  **IMAGES** - Number of images to be used in NEB, ``neb>`` and ``mep_relax>`` jobs.

-  **KCAL** - In refinements, trim output structures to a given value of relative energy
   (in kcal/mol, default is 10). In ``scan>`` runs, sets the threshold to consider a local
   energy maxima for further refinement. Syntax: ``KCAL=n``.

-  **LET** - Overrides safety checks that prevent the program from
   running too large calculations, and avoids efficiency-oriented trimming
   when writing large files to disk (DEBUG keyword).

-  **LEVEL** - Manually set the theory level to be used. Default is
   PM7 for MOPAC, PM3 for ORCA, PM6 for Gaussian and GFN2-xTB for XTB.
   White spaces, if needed, can be expressed with underscores. Be careful
   to use the syntax of your calculator, as ORCA requires a space between method
   and basis set, while Gaussian a forward slash. Syntax (ORCA):
   ``LEVEL(B3LYP_def2-TZVP)``. Standard values can be modified by running the
   module with the -s flag (>>> python -m firecode -s).
   .. Here ( should be written as [ in input, or it will crash (temporary fix?)

.. -  **MTD** - Augments the conformational sampling of transition
..    state candidates through the `XTB metadynamics
..    implementation <https://xtb-docs.readthedocs.io/en/latest/mtd.html>`__
..    (XTB calculator only, experimental).

.. -  **NCI** - Estimate and print non-covalent interactions present in
..    the generated poses (experimental).

-  **NEB** - Perform an automatic climbing image nudged elastic
   band (CI-NEB) TS search after the partial optimization step,
   inferring reagents and products for each generated TS pose. For scan>
   runs, scan points around the energy maxima are used.

-  **NOOPT** - Skip the optimization steps, directly writing
   structures to file after compenetration and similarity pruning.
   Dihedral embeds: performs rigid scans instead of relaxed ones.

-  **ONLYREFINED** - Discard structures that do not successfully
   refine bonding distances. Set by default with the ``SHRINK`` keyword
   and for monomolecular TSs.

-  **PKA** - Specify the reference pKa for a compound in multimolecular
   pKa calculation runs. Syntax: ``PKA(mol.xyz)=11``

-  **PROCS** - Manually set the number of cores to be used in each
   higher level (non-force field) calculation, overriding the value in
   ``settings.py``. Syntax: ``PROCS=32``

-  **REFINE** - Same as calling ``refine>`` on a multimolecular file. 
   The program does not embed structures, but uses the input ensemble
   as a starting point as if it came out of a FIRECODE embedding.

-  **RIGID** - Only applies to "cyclical"/"chelotropic" embeds.
   Avoid bending structures to better build TSs.

-  **RMSD** - RMSD threshold (Angstroms) for structure pruning.
   The smaller, the more retained structures (default is 0.5 A).
   Two structures are pruned if they have an RMSD value smaller than
   this threshold and the maximum deviation value smaller than double
   this threshold. For smaller systems, a value of 0.3 is better suited, and
   it is set by default for embeds of less than 50 atoms. For dihedral
   embeds, the default value is 0.2 A. Syntax: ``THRESH=n``, where n is
   a number.

-  **ROTRANGE** - Only applies to "cyclical"/"chelotropic" embeds.
   Manually specify the rotation range to be explored around the
   structure pivot. Default is 45. Syntax: ``ROTRANGE=90``

-  **SADDLE** - After embed and refinement, optimize structures to the 
   closest first order saddle point using the `Sella <https://github.com/zadorlab/sella>`__ library through ASE.

-  **SHRINK** - Exaggerate orbital dimensions during embed, scaling
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

-  **SIMPLEORBITALS** - Override the automatic orbital assignment, using "Single"
   type orbitals for every reactive atom (faster embeds, less candidates). Ideal
   in conjuction with SHRINK to make up for the less optimal orbital positions.

-  **STEPS** - Applies to "string", "cyclical" and "chelotropic" embeds. Manually
   specify the number of steps to be taken in scanning rotations. For
   "string" embeds, the range to be explored is the full 360°, and the
   default ``STEPS=24`` will perform 15° turns. For "cyclical" and
   "chelotropic" embeds, the rotation range to be explored is
   +-\ ``ROTRANGE`` degrees. Therefore the default values, equivalent to
   ``ROTRANGE=45 STEPS=5``, will sample five equally spaced positions between 
   +45 and -45 degrees (going through zero).

-  **SUPRAFAC** - Only retain suprafacial orbital configurations in
   cyclical TSs. Thought for Diels-Alder and other cycloaddition
   reactions.