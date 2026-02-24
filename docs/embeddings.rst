Embedding runs (legacy, from TSCoDe)
++++++++++++++++++++++++++++++++++++

If the provided input is consistent with an embedding, one will be carried out.
Embeddings (construction of molecular complexes) can be of six kinds:

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


Embeddings: good practice and suggested options
-----------------------------------------------

Here are some guidelines for conformational embedding. Not all of them apply to all embed types, but they will 
help in getting the most out of the program.


1) If a given molecule is present in the transition state, but it is
not strictly involved in bonds breaking/forming, then that molecule
can be pre-complexed to the moiety with which it is interacting. That is,
the bimolecular complex can be used as a starting point. This can be the
case for multimolecular adducts of non-covalently-bound catalysts.

2) Use the ``rdkit_search>`` or ``crest_search>`` operators, or provide conformational
ensembles obtained with other software.
   
3) FIRECODE default parameters are tentatively optimized to yield good results
for the most common situations. However, if you
have more information about your system, specifying details of the pairings
and options for your system is likely to give better results. For example,
embedding trimolecular TSs without imposed pairings generates about 8
times more structures than an embed with defined pairings. Also, if
reactive atoms distances in the transition state are known, using the
``DIST`` keyword can yield structures that are quite close to
ones obtained at higher levels of theory. If no pairing
distances are provided, a distance guess is performed based on the atom type
(defaults are editable in the ``parameters.py`` file).


4) If FIRECODE does not find any suitable candidate for the given reacion,
you could try the ``SHRINK`` keyword (see keywords section).