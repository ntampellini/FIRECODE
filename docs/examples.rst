.. _exs:

Input formatting
----------------

The input can be any text file, but sticking with ``.txt`` or ``.inp`` is recommended.

-  Any blank line will be ignored.
-  Comments (content of lines after ``#``) will be ignored.
-  Keywords, if present, need to be on **first** non-blank line.

Then, molecule files are specified. FIRECODE works with ``.xyz`` files. A molecule line is made up of these elements, in this order:

-  Zero or more operators (*i.e.* ``csearch>``, ``opt>``, etc.) separated by spaces
-  The molecule file name (required)
-  Optional indices (numbers) and pairings (letters) for the molecule (*i.e.* ``2A 4B 5c``)
-  Optional properties of the molecule (*i.e.* ``charge=1``, ``property=value``)

.. note::
   Molecule indices are zero-based! (counted starting from zero!)

By default, molecular charge is read from the filenames by counting the number of ``+`` and ``-`` characters.
However, this can be overridden by explicitly setting ``charge=n`` in the molecule line.

Uppercase letters specify **fixed** constraints (always enforced) while lowercase letters specify **temporary**
constraints, enforced during all optimization steps and then relaxed at the end.

A molecule line can be followed by one or more constraints: these lines should start with one or more
spaces. The line constraint syntax is ``{type} {i1} {i2} [{i3}] [{i4}] [{target} | "auto" | "ts"]``.
There are three types of constraints implemented: ``B`` (bonds), ``A`` (planar angles), and ``D`` (dihedrals).
These have to be followed by two, three or four indices, respectively. Then, an optional last parameter
specifies the target value. If a number is provided, this will be taken as the target distance in Å or angle
in degrees. Omitting this last input of specifying ``auto`` will use the current value read from the first
conformer. For bond constraints, specifying ``ts`` will set the target distance to 1.35 times the sum of
the elements' covalent radii.

Constraint lines, like molecule lines, can also read properties. The only implemented for now is
"fixed" (default is ``fixed=true``) mirroring the behavior of uppercase and lowercase constraints.

Examples of inputs can be found on the :ref:`examples <exs>` page.

Operators
=========

The core elements of every run are the operators acting on a given molecule. See the
:ref:`operators <op_kw>` page to see the full set of operators available. This should cater for
most common workflow needs, from conformational search protocols to ensemble optimizations or
double-ended TS-search methods like NEB or FSM.

Input examples
==============

This series of examples is meant to give guidance on how to perform a series of workflows
with FIRECODE, hoping that some of these examples will be similar to yours.

For detailed descriptions of the operators and keywords present in the inputs, see :ref:`op_kw`.

Work is in progress to expand this section with more examples.

1. Conformational search and refinement
+++++++++++++++++++++++++++++++++++++++

::

   LEVEL=GFN2-XTB SOLVENT=DMF FREQ T_C=-10
   refine> crest_search> opt> ala_ala.xyz

   # This is a comment line!

   # First row sets the level of theory at the GFN2-XTB level via XTB.
   # If XTB is not set as the default calculator, you can specify it
   # here adding CALC=XTB in the keyword line.

   # The FREQ keyword will perform a frequency calculation at the end
   # of refine>, calculating free energies at the temperature (T_C) of
   # -10 °C (default would be 25 °C).

   # The operators on ala_ala.xyz are applied starting from the inside out:

   # opt> - the structure will be optimized (output will be ala_ala_opt.xyz)

   # crest_search> - a metadynamics-based conformational search via CREST
   # is carried out on ala_ala_opt.xyz, generating ala_ala_opt_crest_confs.xyz
   # The default level for crest is GFN2-XTB//GFN-FF, with an energetic window
   # ("--ewin") of 10 kcal/mol.

   # refine> - takes the ala_ala_opt_crest_confs.xyz ensemble and refines it by
   # removing duplicates and performing geometry optimizations at the GFN2-XTB
   # level, then repeating the similarity pruning step.

2. Complex transition state search routine
++++++++++++++++++++++++++++++++++++++++++

::

   CALC=UMA SOLVENT=CH2Cl2 NEB(IMAGES=9)
   saddle> neb> scan> CH3Br_Br-.xyz 0 2

   # The presence of one "-" in the structure filename will set its charge to -1.
   # This can be always overridden by specifying an attribute:
   # ...> ...> mol+++.xyz charge=0

   # The UMA model does not support implicit solvation natively:
   # A ΔG_solv term will be added at the ALPB level via XTB.

   # CH3Br_Br-.xyz was provided with two "reactive indices": 0 (Br) and 2 (C).

   # scan> - A relaxed linear scan will be conducted between the two reactive indices.
   # The input structure features no bond between index 0 (Br) and 2 (C), therefore
   # the scan will try to form a bond between these two by reducing their distance.
   # The scan will terminate when the atoms are appropriately close.
   # This operator will return a file with all the scanned structures, passed to neb>.

   # neb> - A NEB calculation will be set up with the scan structures, extracting the
   # most spaced apart 9 images from the multi-xyz scan file. The default NEB mode will
   # run a NEB-CI procedure, returning a transition state guess.

   # saddle> - Will perform a saddle point optimization on the NEB transition state
   # guess, and run a vibrational analysis on the converged structure.

3. Constrained conformational search, partial optimization, saddle optimization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

   CALC=UMA T=300 CRESTNCI
   saddle> opt> crest_search> opt> old_ts.xyz
       B 34 144
       B 34 143  auto
       B 1 2     auto fixed=false
       D 1 2 3 4 30.0 fixed=false

   # opt> The old_ts.xyz structure will be relaxed at the default level
   # for the UMA calculator, enforcing all four constraints.

   # crest_search> A constrained conformational will be performed via CREST
   # at the default GFN2-XTB//GFN-FF level (4 constraints), in NCI mode
   # (CRESTNCI, "--nci" keyword for CREST).

   # opt> The resulting ensemble will be relaxed again with UMA,
   # enforcing all 4 constraints.

   # saddle> The structures will be optimized to a first order saddle point
   # with no active constraints, using the two fixed constraints
   # to generate a guess for the saddle eigenvector.
   # The temporary (i.e. non-fixed) constraints will be ignored.

   # Thermochemistry will be calculated at 300 K.

Legacy: systematic conformational embeddings
============================================

1. Trimolecular input
+++++++++++++++++++++

::

    DIST(A=2.135)

    maleimide.xyz 0A 5x
    opt> HCOOH.xyz 4x 1y
    crest_search> dienamine.xyz 6A 23y

    # First pairing (A) is the C-C reactive distance
    # Second and third pairings (x, y) are the
    # hydrogen bonds bridging the two partners.

    # Fixed constraints (A, UPPERCASE letters) will refine to the imposed values (here a=2.135 A)
    # Interaction constraints (x, y, lowercase letters) will relax to an optimal value

    # opt> - structure of HCOOH.xyz will be optimized before running the embedding
    # crest_search> - A conformational search will be performed on dienamine.xyz before running the embedding

.. figure:: /images/trimolecular.png
   :align: center
   :alt: Example output structure
   :width: 75%

   *Best transition state arrangement found by FIRECODE for the above trimolecular input, following imposed atom spacings and pairings*

3. Atropisomer rotation
+++++++++++++++++++++++

::

    KCAL=10
    scan> atropisomer.xyz 1 2 9 10

    # scan> : (four indices specified) performs two dihedral
    # scans (clockwise/anticlockwise) rotating the specified
    # dihedral angle in 10° increments. Then, peaks above
    # 10 kcal/mol (KCAL keyword) form the lowest energy
    # structure are re-scanned at increased accuracy (1°
    # increments).

.. figure:: /images/atropo.png
   :alt: Example output structure
   :width: 75%
   :align: center

   *Best transition state arrangement found for the above input*


.. figure:: /images/plot.svg
   :alt: Example plot
   :width: 75%
   :align: center

   *Plot of energy as a function of the dihedral angle (part of the program output).*

3. Peptide-substrate binding mode
+++++++++++++++++++++++++++++++++

::

    RMSD=0.3
    crest_search> hemiacetal.xyz 34x
    crest_search> peptide.xyz 39x

    # Complex binding mode between a reaction
    # intermediate (hemiacetal) and the catalyst
    # (peptide).

    # RMSD=0.3 reduces the similarity threshold to
    # retain more structures (default 0.5 or 1 A)

    # crest_search> performs a conformational
    # search on hemiacetal.xyz (2 diastereomers,
    # total of 72 conformers)

    # String algorithm: 5.18 M poses checked

.. figure:: /images/peptide_chemdraw.png
   :alt: Input structures
   :width: 75%
   :align: center

   *Input structures for hemiacetal.xyz (left) and peptide.xyz (right)*


.. figure:: /images/peptide.png
   :alt: One of the output poses
   :width: 75%
   :align: center

   *Best pose generated for the above input. The yellow bond is the imposed interaction, dotted lines are hydrogen bonds*

4. Complex embedding with internal and external constraints
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

   DIST(a=2.0, x=1.6, y=1.6) SOLVENT=ch2cl2 KCAL=20
   crest_search> quinazolinedione.xyz 6A 14A 0x 7y
   rdkit_search> peptide.xyz 0x 88y 19z 80z

   # Four pairings provided (A, x, y, z):

   # A - Fixed (UPPERCASE letter), internal to quinazolinedione
   # (green) - kept at 2.0 Å during the entire run

   # x - Interaction (lowercase letter) - will be embedded at 1.6 Å
   # and then relaxed during the ensemble optimization steps (red)

   # y - Interaction (lowercase letter) -  will be embedded at 1.6 Å
   # and then relaxed during the ensemble optimization steps (orange)

   # z - Interaction (lowercase letter), internal to peptide (light blue)
   # No distance provided, will relax during optimization

   # crest_search> - metadynamics-based conformational search through CREST.
   # Note that this is internal constraints-aware, and will treat the "A",
   # "x", "y" and "z" pairings as bonds, retaining the specified distances.

   # The KCAL keyword sets the energy threshold in kcal/mol for both the final
   # ensemble and the metadynamics-based conformational search ("--ewin" in CREST).

.. figure:: /images/complex_embed_cd.png
   :alt: Chemdraw representation of the embed pairings
   :width: 100%
   :align: center

.. figure:: /images/qz_firecode.gif
   :alt: One of the output poses
   :width: 100%
   :align: center

   *One of the poses generated for the above input. Note how fixed constraints were mantained (a=2) while interactions were relaxed (x=1.6, y=1.6, z)*
