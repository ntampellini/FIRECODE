
# Environmental variables are set as strings. These will be parsed before use
# (usually via str_to_var) to cast them into the most appropriate Python variable type.
# This means that "" and "none" will be cast as None, "true" as True, and so on.

# If a `.firecoderc` file is found in the submission directory, environmental variables
# defined there will take priority over the ones in this file. The syntax of `.firecoderc`
# should be a simple `key=value` pair per line.

# env vars start with "FIRECODE_" to ensure uniqueness
ENV_VARS = dict(

    # General environment variables
    FIRECODE_CALCULATOR="TBLITE",           # Default calculator

    # Default levels for calculators (overridden by LEVEL keyword)
    FIRECODE_DEFAULT_LEVEL_XTB="GFN2-xTB",
    FIRECODE_DEFAULT_LEVEL_TBLITE="GFN2-xTB",
    FIRECODE_DEFAULT_LEVEL_AIMNET2="wB97M-D3",
    FIRECODE_DEFAULT_LEVEL_UMA="OMOL",
    FIRECODE_DEFAULT_LEVEL_ORCA="GFN2-xTB",

    FIRECODE_PROCS="0",                     # Number of processors (cores) per job to be
                                            # used by XTB and ORCA (0 is auto)

    FIRECODE_CHECKPOINT_EVERY="50",         # Checkpoint frequency during serial
                                            # multimolecular ensemble optimizations

    FIRECODE_FORCE_SINGLE_THREAD="true",    # Enforce the use of a single thread in
                                            # multimolecular optimization.
                                            # Multithread optimization is possible but
                                            # may suffer from performance issues.


    # Full path to calculator binaries.
    # Empty strings will default to the output of `shutil.which`.
    FIRECODE_PATH_TO_ORCA="",
    FIRECODE_PATH_TO_ORCA_LIB="",
    FIRECODE_PATH_TO_XTB="",

    # Full path to UMA model (.pt), either relative (to firecode/calculators/) or absolute
    FIRECODE_PATH_TO_UMA_MODEL="(set with `firecode -s` or in settings.py`)",

    # Default optimizers for a given calculator
    FIRECODE_DEFAULT_ASE_OPTIMIZER_XTB="LBFGS",
    FIRECODE_DEFAULT_ASE_OPTIMIZER_TBLITE="LBFGS",
    FIRECODE_DEFAULT_ASE_OPTIMIZER_ORCA="LBFGS",
    FIRECODE_DEFAULT_ASE_OPTIMIZER_AIMNET2="LBFGS",
    FIRECODE_DEFAULT_ASE_OPTIMIZER_UMA="LBFGS",
    FIRECODE_FALLBACK_ASE_OPTIMIZER="LBFGS",

    # Delta solvation variables
    FIRECODE_SOLV_METHOD_FOR_ML="alpb",     # model of solvation via TBLITE: "alpb" or "cpcm"
    FIRECODE_SOLV_IMPLEM_FOR_ML="post",     # Implementation of ALPB solvation via TBLITE:
                                            # - "post" is post-optimization,
                                            # - "opt" adds the energy and gradients to the ASE
                                            #   calculator at each step during the optimization.

    # TBLITE solvation (non-delta calc)
    FIRECODE_TBLITE_SOLV_METHOD="alpb",     # model of solvation of TBLITE: "alpb" or "cpcm"

    # Sella environmental variables
    FIRECODE_SELLA_INTERNAL_OVERRIDE="",    # Default will use internal coordinates.
                                            # "false" enforces Sella to use cartesian coordinates.
                                            # Constrained optimizations are not possible in cartesian
                                            # coordinates with Sella.
)
