.. _env_vars:

Environment Variables
+++++++++++++++++++++

Some program options can be controlled via environment variables. When the same option
is controlled by a keyword (*i.e.* the main calculator) then the keyword takes precedence.

Default environment variables are defined in ``settings.py`` and can be modified manually
or via ``firecode -s``.

Alternatively, if a file named ``.firecoderc`` is present either in the submission directory
(where the input file is) or in ``~/``, those environment variables will take precedence over
the ones in ``settings.py``. The local ``.firecoderc`` (submission directory) will override the
one in ``~/``, if present.

Example of ``.firecoderc`` specifying an alternate version of ORCA to be used:

::

    FIRECODE_PATH_TO_ORCA=/home/orca/orca_6_1_1_linux_x86-64_shared_openmpi418/orca
    FIRECODE_PATH_TO_ORCA_LIB=/home/orca/orca_6_1_1_linux_x86-64_shared_openmpi418/lib/


What follows is the default ``settings.py`` file.

.. literalinclude:: ../firecode/settings.py
   :language: python
