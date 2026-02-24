.. _op_kw:

Operators and keywords
======================

Operators
+++++++++

Molecule files can be preceded by **operators**, *i.e.*
``refine> opt> molecule.xyz``. They operate on the input file in "inside out" order,
as mathematical operators. In the example above, ``opt>`` is executed first, and the resulting
output is piped to ``refine>``. All non-terminating operators can be chained.
Here is a list of the currently available operators:

-  ``opt>`` - Performs an optimization of all conformers of a molecule before
   running the embedding. Generates a new ``molecule_opt.xyz`` file with the optimized
   coordinates. This operator is constraint-aware and will perform constrained
   optimizations obeying the distances provided with DIST.

-  ``crest_search>``/ ``crest>``/ ``mtd_search>``/ ``mtd>`` - Performs a metadynamics-based conformational
   search on the specified input structure through `CREST <https://crest-lab.github.io/crest-docs/>`__.
   The operator is constraints-aware
   and will constrain the specified distances (both "fixed" and not). Generates a new ``mol_crest_confs.xyz``
   file with the crest-optimized conformers. The default level is GFN2//GFN-FF (see CREST docs).
   Charge and multiplicity are passed from the molecule or keyword line.
   
   ::
   
       mtd> molecule.xyz 4A 8A charge=-1

-  ``rdkit_search>`` - Performs a conformational search with the `ETKDGv3 algorithm <https://pubs.acs.org/doi/10.1021/acs.jcim.0c00025>`__
   via RDKit. Generates a new ``mol_rdkit_confs.xyz`` file with the generated conformers.

-  ``refine>`` - Acts on a multimolecular input file, pruning similar structures and optimizing it
   at the chosen theory level. Soft, non-fixed constraints ("interactions") are enforced during a 
   first loose optimization, and subsequently relaxed.

-  ``neb>`` - Runs a NEB (Nudged Elastic Band) procedure on the provided structures.
   The implementation is a climbing image nudged elastic band (CI-NEB) TS search.  
   The operator should be used on a single file that contains two or more structures.
   The first and last structures will be used as start and endpoints. If three are provided, the intermediate
   structure will be used as a TS guess in the interpolation. If more are provided, they will all be used
   in place of (or to aid) interpolation to reach the desired number of images
   (set with the ``IMAGES=n`` or NEB(IMAGES=n) keywords). Default is 7.
   
-  ``scan>`` - Runs a scan of the desired distance or dihedral angle, based on the number of indices provided.
   For distance scans, the atomic indices will be approached if they are not bound, and separated if they are.
   This could be chained with, for example, a NEB calculation, where the scan trajectory will be used as a starting
   point for a NEB calculation.

   ::

       neb> scan> mol.xyz 3 4

   For dihedral scans, the dihedral is rotated the full 360° in both clockwise and counterclockwise directions.
   Maxima above a certain threshold (set with ``KCAL=n``) are scanned again in smaller steps. All maxima for the second mode accurate 
   scan are saved to firecode_maxima_mol.xyz. The optimized geometries for the initial scans are saved with the relative 
   names. This version of the ``scan>`` operator is terminating and cannot be chained.

Deprecated (legacy):

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
++++++++++++++++

Keywords are case-insensitive and are divided by at least one blank space.
Some of them are self-sufficient (*i.e.* ``NCI``), while some others require an
additional input (*i.e.* ``STEPS=10`` or ``DIST(a=1.8, b=2, c=1.34)``). 
Commas are used to divide keyword arguments where more than
one is accepted, like in ``DIST``.

-  **CALC** - Overrides default calculator in ``settings.py``. 
   Available calculators are ORCA, XTB, TBLITE, AIMNET2 and UMA. Syntax: ``CALC=ORCA``

-  **CHARGE** - Specify the charge to be used in optimizations.

-  **CONFS** - Override the maximum number of conformers to be considered 
   for the refinement or embedding of each molecule (default is 1000). Syntax: ``CONFS=10000``

-  **CRESTNCI** - crest_search> runs: passes the "--nci" argument to CREST, running
   it in non-covalent interaction mode, *i.e.* applying a wall potential to prevent
   unconstrained non-covalent complexes to evaporate during the metadynamics.

-  **DEBUG** - Outputs more intermediate files and information in general.
   Structural adjustments, distance refining and similar processes will
   output ASE ".traj" trajectory files. It will also produce
   "hypermolecule" ``.xyz`` files for the first conformer of each
   structure, with dummy atoms (X) in each FIRECODE "orbital" position.

-  **DIST** - Manually imposed distance between specified atom
   pairs, in Angstroms. Syntax uses parenthesis and commas. Spaces are tolerated:
   ``DIST(a=2.345, b=3.67, c=2.1)``

-  **DRYRUN** - Skips lenghty operations (operators, embedding, refining)
   but retains other functions and printouts. Useful for debugging and
   checking purposes.

-  **FFCALC** - Overrides default force field calculator in ``settings.py``.
   Value can only be ``XTB`` or None. Syntax: ``FFCALC=XTB``

-  **FFLEVEL** - Manually set the theory level to be used for force field
   calculations. Default is GFN-FF for XTB. Support for other FF calculators has been discontinued.  
   Standard values can be modified by running the module with the -s flag
   (>>> python -m firecode -s).

-  **FFOPT** - Manually turn on ``FFOPT=ON`` or off ``FFOPT=OFF`` the force
   field optimization step, overriding the value in ``settings.py``.

-  **IMAGES** - Number of images to be used in NEB, ``neb>`` and ``mep_relax>`` jobs.

-  **KCAL** - In refinements (``refine>``), trim output structures to a given value of relative energy from the most stable
   (in kcal/mol, default is 10). In ``scan>`` runs, sets the threshold to consider a local
   energy maxima for further refinement. Syntax: ``KCAL=n``.

-  **LET** - Overrides safety checks that prevent the program from
   running calculations that seem too large and likely erroneous.

-  **LEVEL** - Manually set the theory level to be used.
   White spaces, if needed, can be expressed in input files with underscores. Syntax (ORCA):
   ``LEVEL(B3LYP_def2-TZVP)``. Defaults can be found in settings.py, and can be modified by running 
   the module with the -s flag (``>>> firecode -s``).

-  **NEB** - Set NEB options. Syntax and default values: ``NEB(images=7, preopt=true, ci=true)``.

-  **ONLYREFINED** - Discard structures that do not successfully
   refine bonding distances. Set by default with the ``SHRINK`` keyword
   and for monomolecular TSs.

-  **PKA** - Specify the reference pKa for a compound in multimolecular
   pKa calculation runs. Syntax: ``PKA(mol.xyz)=11``

-  **PROCS** - Manually set the number of cores to be used in each
   higher level (non-force field) calculation, overriding the value in
   ``settings.py``. Syntax: ``PROCS=32``

-  **RMSD** - RMSD threshold (Angstroms) for structure pruning.
   The smaller, the more retained structures (default is 0.25 A). 
   Syntax: ``THRESH=n``


Embedding-specific keywords (legacy)
++++++++++++++++++++++++++++++++++++

-  **BYPASS** - Debug keyword. Used to skip all pruning steps and
   directly output all the embedded geometries.
   
-  **CLASHES** - Manually specify the max number of clashes and/or
   the distance threshold at which two atoms are considered clashing.
   The more forgiving (higher number, smaller dist), the more structures will reach the geometry
   optimization step. Default values are num=0 and dist=1.5 (A). Syntax: ``CLASHES(num=3,dist=1.2)``
   
-  **DEEP** - Performs a deeper search, retaining more starting
   points for calculations and smaller turning angles. Equivalent to
   ``THRESH=0.3 STEPS=72 CLASHES=(num=1,dist=1.3)``. **Use with care!**

-  **EZPROT** - Preserve the E or Z configuration of double bonds
   (C=C and C=N) during the embed. Likely to be useful only for
   monomolecular embeds, where molecular distortion is often important, and
   undesired isomerization processes can occur.
 
-  **NOOPT** - Skip the optimization steps, directly writing
   structures to file after compenetration and similarity pruning.
   Dihedral embeds: performs rigid scans instead of relaxed ones.
   
-  **RIGID** - Only applies to "cyclical"/"chelotropic" embeds.
   Avoid bending structures to better build TSs.

-  **ROTRANGE** - Only applies to "cyclical"/"chelotropic" embeds.
   Manually specify the rotation range to be explored around the
   structure pivot. Default is 45. Syntax: ``ROTRANGE=90``
   
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