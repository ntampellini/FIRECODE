
<!-- - Introduced compatibility of SADDLE and NEB keywords for scan> runs with both 2 indices (distance scans) and 4 indices (distance scans) -->

<!-- - ... mep_relax> BETA
- ... IMAGES kw, also implement it for neb>-->

## FIRECODE 1.1.3 🔥 (June 3 2025)
- The CREST MTD operator checks that CREST 2 is installed before trying to run a search. For now, only CREST 2 is supported, as CREST 3 needs some updated interface
- Solved CRETSLEVEL bug.

## FIRECODE 1.1.2 🔥 (April 29 2025)
- Added CRESTLEVEL keyword to specify CREST theory level (i.e. "CRESTLEVEL=GFN2").
- Updated aimnet2 bindings to only useunderscore in the module name calls and not hyphens.
- Fixed bug with the default value for the attribute "charge".

## FIRECODE 1.1.1 🔥 (January 2025)
- Added pretty error traceback with rich.
<!-- SMARTS Constraint class -->


## FIRECODE 1.1.0 🔥 (December 24 2024)
- Implemented planar angle and dihedral constraints in ensemble optimization (only intra-file, but also available in embedding runs). These are propagated to xtb, ASE-mediated calculators and PyTorch models, as well as to operators like mtd> (CREST). These constraints can be specified in new input lines that start with a blank space under the reference molecule line, with the syntax "i1 i2 i3 \[i4\] target(float)" (i.e. "0 1 2 3 90.0" for dihedrals or "0 1 2 0.0" for planar angles). For now, these constraints are enforced at every step (as capital letter constraints, but not associated to any letter).
- Improved comments parsing: possible to comment in any input line (with "#", following a Pyhton-style syntax).
- Tweaked similarity_refining maximum ensemble size thresholds to carry different pruning steps.
- Turned off interactive ASE GUI specification of Sp3 atoms leaving groups for orbital building.

## FIRECODE 1.0.1 🔥 (September 29 2024)
- Minor bugfixes and printout clean-ups.

## FIRECODE 1.0.0 🔥 (July 5 2024)
- Renamed project to better reflect the expanded functionality of the toolbox! 🔥
- New Pruner class: simpler, more homogenized implementation of iterative subset pruning, sharing the same basic architecture across different similarity evaluation functions, with cached calls to the eval similarity function.
- Added new debug dedicated log, that collects all debug messages if the DEBUG keyword is specified.
- Fixed bug with some parts of torsion_module for segmented graphs, mainly for rotationally corrected RMSD calculation.
- General code debug and PEP 8 cleanup (thanks Ruff).
- Informative suggestions: keywords that are not understood can generate suggestion of known keywords to use instead.
- Removed rmsd library dependency for local numba numpy implementation.
- Increased similarity_refining thresholds for the maximum number of structures to be compared: 1E5 for RMSD and MOI, 1E4 for rotationally corrected RMSD. Added printouts in log if we skip a pruning stage because the ensemble is too large.
- Fixed a bug in _get_rotation_mask that would rotate the wrong part of the molecule (spotted because prune_by_rmsd_rot_corr would not reject some molecules that it should have) 
- Added a debug messages printout function for the Pruner class, which by default is set to embedder.debuglog
- Added support for AIMNET2 Pytorch models, including GPU support, with the package aimnet2-firecode. If a solvent is specified, the ALPB ΔGsolv (XTB) is added to the vacuum SP EE.
- Removed support for the THREADS keyword and command line setting, as the number of workers in every multithreaded parallelism is now automated.

# TSCoDe Changelog (old)

## 0.4.16 (April 2, 2024)
- Solved bugfix in check_objects_compenetration.
- Updated compenetration_check warning printouts, specifying the number of compenetrations found.

## 0.4.15 (March 23, 2024)
- Reinstated the only_refined option at the end of optimization_refining.
- Solved bugfix in operators.

## 0.4.14 (March 23, 2024)
- renamed python_functions.py to numba_functions.py
- Decreased compenetration check threshold for close contacts to 0.5 A from 0.95 A. In this way, topological mistakes made by GFN-FF, that bind hydrogen atoms (d~0.89 A) will not automatically flag the structures as compenetrated, as this will most likely optimize fine when the theory level is raised to GFN2 or other.

## 0.4.13 (March 22, 2024)
- Changed library versions

## 0.4.12 (March 22, 2024)
- Changed numpy version to 1.20.3
- Added saturation check

## 0.4.11 (March 10, 2024)
- Various small bugfixes/graphic restyling
- Stuctures are optimized at embedder.options.theory_level before running the mtd_search> operator, and a FatalError is raised if they scramble during this process. This increases the robustness of the workflow by avoiding changes in the molecular graph if the input structure is bad.
- Added a warning printout system to the Embedder class, via logging and appending strings to the embedder.warnings list.
- Added a saturation and compenetration check for input molecules, that warns about potential bad geometries. 

## 0.4.10 (February 24, 2024)
- Reduced/summarized printouts for loose to tight optimization steps. 
<!-- - Removed call to compenetration_refining() on input ensemble: it is usually already pruned for compenetration for embed runs, and other ensembles (for example from "refine> mtd>" runs coming from CREST) benefit from relaxing eventual clashes that are present. -->

## 0.4.9 (February 22, 2024)
- CREST constraints (mtd_search>) are now passed as distance constraints instead of atom fixing (forgot to implement it this way before?).
- CREST metadynamics input/output files are all retained now and not deleted after execution.
- Output names changed from starting with "TSCoDe" to "tscode" for easier typing.
- Modified variable definition of periodic table in tscode.pt to avoid ValueError(s).
- Added the mep_relax> operator (beta, not present in documentation) to relax chains of images along the PES. Useful to get good starting points for higher-level NEB calculations. First, the optimization is performed by retaining all bond distances, which are then relaxed after the first cycle is converged (developed to find atropisomer interconversion pathways).

## 0.4.8 (February 2, 2024)
- Updated library dependency versions.

## 0.4.7 (Januray 28, 2024)
- Restyled copy of input in the log file.
- Added citations to the main program and external modules if they are used (references.py, embedder.py/_print_references).
- In runs with operators, t_start_run is set before running the first operator, and it is not overwritten by RunEmbedding if it was already set. This makes sure that the final time always includes the time spent executing the operators.

## 0.4.6 (January 21, 2024)
- CRESTNCI keyword: passes "--nci" to CREST, making it run in "NCI mode" that is applying a wall potential to prevent unconstrained non-covalent complexes from evaporating during the dynamics.
- If metadynamic conformational searches fail (CalledProcessError) at the default GFN2-XTB//GFN-FF level they are re-launched with pure GFN2-XTB (slower but more stable and works with inorganic ions).

## 0.4.5 (January 15, 2024)
- Added support for passing an energy threshold to mtd_search> CREST runs ("--ewin" in CREST, set with the KCAL keyword on TSCoDe).
- Dynamically adjusted energy pruning threshold above self.options.kcal_thresh in optimization_refining if needed, so that at least a percentage of structures is retained (default minimum set to 10%).

## 0.4.4 (December 17, 2023)
### Discontinued OpenBabel FF support
- Updated prune_conformers_rmsd group criteria, avoiding group numbers that have (on average) less than 20 active structures.
- Disconnected the Openbabel Force Field calculator from the embedder, as the XTB implementation proved uniquely versatile and robust for more sophisticated manipulations. The interface is still present in the code (inside calculators) for reference, external utility purposes and potential future re-adoption for specific tasks. Updated the rest of the documentation accordingly.

## 0.4.3 (November 18, 2023)
### RMSD pruning significative speedup, increased extent of parallelization throughout, keywords priority
- Small bug fixes.
- Added priority to keywords, that dictate their order of execution. Options are set first (priority 1), then priority 2 keywords that modify attributes and depend on priority 1 keywords (for now, just DIST).
- Added a status_dump call after generate_candidates (debug keyword).
- Changed spring stiffness for constrained optimizations to dynamic values (0.25 to 1 for FF, 1 to 2 for SE, in Eh/bohrs). Less scrambling, more accurate poses.
- Multiembeds now are run through self.avail_cpus max_workers instead of half.
- prune_conformers_rmsd is now completely compiled with numba, cached, parallelized and ~30 times faster. The old similarity-graph-based logic was also removed in favor of the removal of any structure at the first instance of a similar one. In the future, all pruning functions should work similarly.
- Cyclical embed now generates fewer candidates, as it discards them if they are too similar to others that share the same pivots and conformation ids (RMSD-based).
- Adjusted default threshold for RMSD similarity from 0.25 back to 0.5 A (benchmarked to retain all methylcyclohexane conformers with the new pruning algorithm).
- Reduced checkpoint dump frequency (embedder.options.checkpoint_frequency) from 20 to 50, as writing large (>20k) structures so often can slow down execution and clutter the logfile.
- Reversed order of this CHANGELOG.md file, to display the most recent updates on top.

## 0.4.2 (October 27, 2023)
### Memory usage, pruning refinements, multiembed improvements, molecule attributes
- Significatively reduced memory usage of the prune_conformers functions, avoiding using the wasteful similarity_mat for a lighter, faster set().
- Reduced default values for ROTRANGE and STEPS for cyclical embeds (from 90, 9 to 45, 5).
- TFD similarity refining is now only carried out for single molecules, not for multimolecular embeds.
- Added SIMPLEORBITALS keyword. All orbitals will be of Single type. Reduces the number of pivots for each molecule, and consequently the number of candidates that will be generated.
- Added generation of orbitals for main embedder in multiembed runs. Aids debugging and allows for a better bird's-eye view in the initial printout.
- Input file: added custom attributes in molecule line (mtd> mol.xyz 2A 3A 7x charge=-1). Any attribute is set directly on the Hypermolecule class, and its declaration is recorded in the logfile. For now, the only active attribute is 'charge' and it is passed to the mtd> operator. The approach should be easily applicable to set future molecule-specific settings.
- Fixed SHRINK printout bug when not specifying embed distances.
- Multiembed child embedders are passed more options now: shrink (with shrink_multiplier) and simpleorbitals

## 0.4.1 (October 26, 2023)
### **CHANGED CONSTRAINTS SPECIFICATION (NON-BACKWARDS COMPATIBLE)**
### RunEmbedding refactoring, stability improvements, internal cleanup of old code
### "Dihedral embeds" are now part of scan>
- Various small fixes for stability purposes and printout beautification.
- write_structures can now also align ensembles based on the moments of inertia (align='moi')
- Renamed output of refine> runs 'ensemble' instead of 'poses', which is now only for when an embed is carried out.
- Cleaned some unused junk code from the past (fast_score, hyperneb, opt_iscans, TS keyword, and other experiments)
- Dihedral "embeds" are now just a part of the scan> operator, as no real embed was carried out.
- Added printout of input file after the banner in the logfile for easier tracebacks.
- Renamed mtd_csearch> to mtd_search> (solves a bug where csearch> was called instead).
- Moved RunEmbedding back to embedder.py, and cleaned up the class inheritance. This allowed less redundant and tidier code at the expense of having a larger embedder.py file (~2300 lines).
- Expanded dump_status to reflect the different constraints at different steps of optimization (all constraints or just fixed). Also added the target distance for each in the printout.
- **CHANGED CONSTRAINTS SPECIFICATION (NON-BACKWARDS COMPATIBLE)** - now fixed constraints are specified with UPPERCASE letters instead of a/b/c, and interaction constraints are specified with lowercase letters instead of x/y/z. Expands the number of each type of constraint that is possible to specify (from 3 to 26).

## 0.4.0 (October 21, 2023)
### CONCURRENT OPTIMIZATIONS AND EMBEDDINGS!
### Crest metadynamic conformational searches with mtd>
- xtb_opt now reads output geometry from the trajectory file rather than from xtbopt.xyz. This was needed for multiprocessing capability as it is not possible to override the default name for the optimized geometry (xtbopt.xyz) from the input section.
- Replaced exit() calls with sys.exit()
- xtb_opt now creates a new folder and works inside of it, making it easier to transition into a multiprocessing workflow.
- xtb_pre_opt passes bond constraints to xtb_opt through a string now (constrain_string) instead of a file.
- Implemented parallel (multiprocess) optimization for force_field_refining (2 cores/thread, as many threads to use all CPUs) and optimization_refining (4 cores/thread, as many threads to use all CPUs). Embedder.threads variable currently not in use.
- Added CHARGE keyword (specify charge to be passed to calculator).
- Removed VMD printout for simple runs (write_vmd) but kept anchor debugging (write_anchor_vmd).
- Added metadynamic conformational search through CREST (mtd_csearch>, or mtd>).
- Changed the way xtb_opt deals with distance constraints, now using very stiff springs instead of exact fixing (better when more than one distance constraint is specified).
- Reinstated fitness_refining after every optimization step, which is now based on a cumulative deviation from the imposed pairing distances.

## 0.3.8 (October 5, 2023)
- Expanded debug functionality (DEBUG keyword): printouts of RunEmbedding status (dump_status)
- Added a xtb_pre_opt wrapper for xtb_opt that retains every bond present in the initial set of graphs, in addition to provided constraints. This will greatly increase the quality of structures coming from embeds into force_field_refining. Added before force_field_opt for embed runs (>1 self.objects) that use XTB as FF calculator.
- Introduced orbital subtypes (atom.subtype) to offer more flexibility for embed types. These do not alter default pairing_dists but allow for specific orbital arrangements around the reactive atom.
- New Ketone atom subtype: 'trilobe' - three lobes at the opposite end of alkoxide/sulfonamide substituents relative to oxygen.
- Added the DRYRUN keyword. Skips lengthy operations (operators, embedding, refining) but retains other functions and printouts. Useful for debugging and checking purposes.
- Compacted output information: individual candidate details only printed with DEBUG keyword, average time per structure and estimate of completion given instead every 20 iterations when saving checkpoints

## 0.3.7 (September 28, 2023)
- Added energy table at the end of runs that generate energetic data
- Fixed bug in force_field_refining that runs with no pairings
- Fixed lack of output in optimization_refining when less than 20 structures were optimized
- Brought back MOI-based pruning, when the ensemble is <200 structures, as it proves fast and beneficial for locally symmetric (dummy) rotations not yet identified by rmsd_rot_corr. Threshold is statically set at 1% (10E-2). In the future, it could be possible to uniform to CREST's approach of adjusting it from 1 to 2.5 % dynamically based on the anisotropy of moments of inertia.
- Adjusted default threshold for RMSD similarity from 0.5 to 0.25 A (benchmarked to retain all methylcyclohexane conformers)
- Cleaned residuals of the ENANTIOMERS keyword and related prune_enantiomers (now prune_by_moment_of_inertia)
- Renamed every "indexes" to "indices", as I should have done a long time ago...
- The function distance_refinement is now just incorporated in force_field_refinement and optimization_refinement as an option (only_fixed_constraints). First, an optimization is done with all specified constraints (fixed and interactions, at loose convergence for force field) and then the interaction constraints are released (tight convergence for FF). This process should be more robust than the one before, as it minimizes scrambling/separation of multimolecular arrangements and avoids the limitations of xtb-python when using XTB as calculator (since everything is now dealt with without ASE). Moreover, energy pruning is only performed after the interaction constraints have been released, so that random fluctuations in interaction distances constraints do not bias conformer selection (imperfect embedding geometries in mind).
- Rotationally corrected RMSD-based pruning in similarity_refining is now only done for molecules with at least one locally symmetric torsion (saves time)

## 0.3.6 (September 24, 2023)
- Implemented nested operators call ('refine> rsearch> opt> mol.xyz 2a 7a'), executed in reversed order (opt>, then rsearch>, then refine>)
- Removed checkpoint automatic restart, as it was not compatible with the new nested operator routine
- To remedy the last point, the last 'checkpoint' ensemble is now not deleted after optimization_refinement
- The opt> operator is now aware of internal constraints and performs constrained optimizations (moved letter/pairing/dist functions from RunEmbedder to Embedder)
- Updated README and ReadTheDocs documentation

## 0.3.5 (September 22, 2023)
- Added rotationally-corrected RMSD pruning, to treat symmetrical rotations and get rid of identical rotamers that only differ from indexing order. The treatment is skipped for ensembles greater than 750 structures, to avoid unnecessarily slowing down the refinement process.
- Renamed clustered_csearch.py as torsion_module.py, since it now contains mostly torsion-based constructs and utilities.
- Resolved a bug for internal constraints preventing paired embeds from being recognized as correct.
- Implemented "multiembed" embeds, able to perform bimolecular cyclical embeds systematically on every arrangement of pairs of multiple atoms (see documentation)
- Fixed dependency requirements (again?)

## 0.3.4 (August 30, 2023)
- Added the ability to recognize interrupted refine> runs and restart from the last checkpoint.
- Moreover, checkpoints now update every 20 optimized structures.
- Layered optimization protocol for optimization_refinement when using ORCA - first round with 3 iterations, then 5 extra, then to convergence.
- XTB calculations that do not reach convergence do not crash the program anymore unless specified with assert_convergence=True.
- XTB force field ensemble optimizations are carried out in two steps, first with loose convergence and then with tight convergence.
- Various small bugfixes and printout beautifications.

## 0.3.3 (July 10, 2023)
- Added "autoneb>" operator, that automatically builds a MEP guess based on input structure. Currently only supporting 7-membered rings inversions.
- The Hypermolecule class can now also accept SMILES strings instead of molecular files. Only embeds with no index to be specified, as "autoneb>"
- Changed run> operator back to refine> (and RUN keyword bask to REFINE).
- Fixed bug with scan> termination called when not required.
- Changed default for conformational search during embeds to false, and NOCSEARCH keyword to CSEARCH.
- Clustered csearch module: added flexibility in some functions to allow external use of them.

## 0.3.2 (November 29, 2022)
- Removed Walrus operators from the code for Python 3.7 compatibility

## 0.3.1 (November 27, 2022)
- Fixed TFD pruning bug for embeds with no rotable bonds.
- approach> operator is now called scan> and automatically infers the scan direction (approaching or separating the atoms based on their distance).
- Fixed bug in specifying indices without letters (in _remove_internal_constraints)
- Minor bug fixes
- Added flexibility in NEB keyword, allowing optimization of start/end points and specifying the number of images (NEB keyword)
- NEB calculations now support two, three or a greater odd number of structures as input, to facilitate computational refinement of MEPs

## 0.3.0 (August 31, 2022)
- Fixed run> operator bug (leftover refine> references)
- Implemented a fast, preliminary torsion fingerprint deviation-based similarity pruning (prune_confs_tfd, faster than prune_confs_rmsd) for similarity_refining
- Added this TFD pruning in the most_diverse_conformers function (this is fast enough, RMSD-based was not) and at the end of clustered_csearch and csearch_aug as well
- Refined distance constraining for XTB optimizations as well, guided by the target length. More accurate distances, so more accurate energies for constrained structures. Also less work for the final refinement step.
- Implemented wider compatibility for internal constraints - intramolecular distances that have to be respected (so that csearch is aware of them) and even enforced to a specific distance (with DIST)

## 0.2.0 (June 26, 2022)
- Implemented pka> operator (xtb calculator only)
- Solved random_csearch bug (when confs < n_out)
- Refined distance constraining for OB optimizations, guided by the target length. More accurate distances, so more accurate UFF energies for constrained structures.
- Reincluded symmetrical aryl ring rotations in clustered_csearch (previously considered dummy and skipped)
- Corrected bug in the approach> operator
- Now multiple molecules are supported for approach> runs and a cumulative output is provided (distance-energy graph)
- Random Csearch now always tries to output n_out conformers (1000 max tries), instead of generating n_out confs and discarding compenetrated ones
- Implemented rsearch> operator
- Added the -cl (command line) argument, to call TSCoDe without writing an inputfile
- Added quote from database at the end of each successful run, courtesy of https://type.fit/api/quotes
- Upgraded startup banner to a benzene-like logo
- Added error logging to file via the logging module
- Solved Profiler bug (adapted to new Embedder architecture)
- Added logging for the exit status on any call to optimize(...) (supporting FF calculations too)
- Added the tscode_procs option for embedder.options, to run python code in parallel (for now just csearch_aug implemented)

## 0.1.0 (May 14, 2022)
- Refined conformational search - better parameters, torsion printout, fixed HB bugs, added secondary amides as rotable, made it faster and more scalable (random sampling overrides KMeans for n>1k)
- Various small bug fixes and print refinements in dihedral embed and ase_neb functions
- Added csearch_hb> operator that allows to keep the current hydrogen bonding situation in conformational sampling
- Added approach> operator that performs automatic linear scans (ideally for locating approach energy maxima)
- Torsional clustering: changed clustering method to DBSCAN over KMeans (better for spatial clusters)
- Added numerous temporary "print" statements to asses code activity at all times
- CSearch now warns when it is fed non-connected complexes of two or more molecules, if no intramolecular HBonds are found
- CSearch now has a random dihedral variant (mode 2) used for faster conformational augmentation
- Implemented conformational augmentation of TS candidates (random dihedral rotations, for three cycles, max 1000 new conformers generated)
- Refined molecule reactive atoms/pairings print statements in log
- Fully implemented the use of constrained indices in prune>/NOEMBED runs (pairing distance can be specified now)
- Maximum number of conformers to use in embed can be specified with the CONFS keyword
- Removed enantiomers pruning for performance reasons
- RIGID keyword is automatically added for cyclical embeds with >100 conformers (override with LET)
- The NOEMBED keyword is now called REFINE, and the prune> operator is called refine>

## 0.0.9 (December 2021)
- Added error message if molecule reading fails
- Operators now support spacing after the > sign
- csearch> operator now works for molecules with more than one conformer
- Improved the speed of the align_vector_pair and rot_mat_from vec functions (numba)
- Periodic table is now in pt.py, removing any cyclical import issue
- Graph manipulations are now organized in their own file, allowing shared use (graph_manipulations.py)
- The old Docker is now called Embedder, leaving the "Docker" name for future docking extension
- If no embed is recognized after applying operators, run is cleanly terminated (embedder.embed attribute in RunEmbedding.run function, run.py)
- NOEMBED keyword still works, but structure pruning now can also be performed calling the prune> operator
- conformational search now discard symmetric rotations involving 6-membered aromatic rings like phenyl, 4-pyridyl, mesityl, ...
- procs == None bugfix
- secondary amides are now considered rotable by the csearch algorithm

## 0.0.8 (October 21, 2021)
- Removed unnecessary for loop in dihedral embed NEB optimizations
- Added pre-optimization before dihedral embed
- Customized text can be inserted in write_structures function
- Dihedral embed structures energies are now relative to equilibrim geometry (direct barrier height)
- Moved "test" folder inside "tscode" (fixes bug)

### 0.0.7 (Oct 20 2021)
- Import bugfix

### 0.0.6 (Oct 19 2021)
- Updated module call adding __main__.py file to tscode/
- All internal imports are now relative to main module (import 'tscode.module' instead of 'import module')
- Maximum number of conformers per molecule is now 10, since the compenetration/similarity pruning algs are faster and more efficient (still overridden by LET keyword)
- Removed strict versioning for python required libraries
- Removed MMFF keyword
- Added NOEMBED, FFOPT, FFCALC, FFLEVEL and TS keywords
- Added a Clustered Conformational Search implementation
- Old csearch> operator is now called confab>
- New csearch> operator is the TSCoDe conformational search engine
- General code optimizations
- Added profiling command line keyword (-p)

### 0.0.5 (Sep 1 2021)
- SADDLE keyword implementation.
- Added keywords print at top of log
- Pairings are now of two types: reactive atoms (a, b, c) or NCIs (x, y, z). The latter are adjusted when specifying distances with DIST but are left free to reach their equilibrium distance (HalfSpring constraint + additional relax).
- Major code cleaning, refactoring and reordering
- Added solvent support for calculators (SOLVENT keyword)
- Dihedral embeds now support both the SADDLE and NEB keywords
- Similar structures are now pruned in a rational way: the best looking is kept (fast_score)

### 0.0.3 (Aug 10 2021)
- setup.py bugfixes.

## 0.0.2 (August 10, 2021)
- If pivots decrease during a bend, an exception is raised. Future versions might have a different behavior in this scenario.

## 0.0.1 (August 10, 2021)
- First release