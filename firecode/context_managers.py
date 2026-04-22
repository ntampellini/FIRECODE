"""Context managers for FIRECODE."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree, which
from typing import (
    Any,
    Generator,
    Optional,
)


class suppress_stdout_stderr(object):
    """A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self) -> None:
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self) -> None:
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_: object) -> None:
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class HiddenPrints:
    def __enter__(self) -> None:
        self._original_stdout: Any = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]
    ) -> None:
        sys.stdout.close()
        sys.stdout = self._original_stdout


class NewFolderContext:
    """Context manager: creates a new directory and moves into it on entry.

    On exit, moves out of the directory and deletes it if instructed to do so.

    """

    def __init__(
        self, new_folder_name: str, delete_after: bool = True, overwrite_if_exists: bool = True
    ) -> None:
        self.new_folder_name = os.path.join(os.getcwd(), new_folder_name)
        self.delete_after = delete_after
        self.overwrite_if_exists = overwrite_if_exists

    def __enter__(self) -> None:
        if self.overwrite_if_exists:
            rmtree(self.new_folder_name, ignore_errors=True)

        if not os.path.isdir(self.new_folder_name):
            # create working folder and cd into it
            new_dir = Path(self.new_folder_name)
            new_dir.mkdir()

        os.chdir(self.new_folder_name)

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        # get out of working folder
        os.chdir(os.path.dirname(os.getcwd()))

        # only delete if instructed to
        # and no unhandled exception occurred
        if self.delete_after and exc_type is None:
            rmtree(self.new_folder_name, ignore_errors=True)


class FolderContext:
    """Context manager: works in the specified directory and moves back after."""

    def __init__(self, target_folder: str) -> None:
        self.target_folder = os.path.join(os.getcwd(), target_folder)
        self.initial_folder = os.getcwd()

    def __enter__(self) -> None:
        """Move into folder on entry."""
        if os.path.isdir(self.target_folder):
            os.chdir(self.target_folder)

        else:
            raise NotADirectoryError(self.target_folder)

    def __exit__(self, *args: object) -> None:
        """Get out of working folder on exit."""
        os.chdir(self.initial_folder)


@contextmanager
def env_override(**kwargs: Any) -> Generator[Any]:
    """Temporarily override environment variables."""
    old = {k: os.environ.get(k) for k in kwargs}
    os.environ.update({k: str(v) for k, v in kwargs.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def sella_env() -> Generator[Any]:
    """Environment optimized for Sella with JAX internal coordinates."""
    avail_cpus = len(os.sched_getaffinity(0))
    sella_threads = str(min(avail_cpus, 4))

    # Build XLA flags additively to avoid clobbering any existing flags
    existing_flags = os.environ.get("XLA_FLAGS", "")
    add_flags = ("--xla_cpu_multi_thread_eigen=true", "--xla_force_host_platform_device_count=1")
    new_flags = ""
    for new_flag in add_flags:
        if new_flag not in existing_flags:
            new_flags = (new_flags + " " + new_flag).strip()

    # take ownership of the sella jax compilation cache so that
    # we are sure to compile and use our own version of it
    jax_comp_cache_dir = Path.home() / ".cache/sella/firecode_jax_cache"

    with env_override(
        # Sella's JAX runs on CPU regardless
        JAX_PLATFORMS="cpu",
        JAX_PLATFORM_NAME="cpu",
        JAX_COMPILATION_CACHE_DIR=str(jax_comp_cache_dir),  # recompile for each
        # JAX/XLA for Sella's internal coordinate calculations (CPU)
        XLA_FLAGS=new_flags,
        OPENBLAS_NUM_THREADS=sella_threads,
        # multithreaded BLAS, with a modest thread count
        OMP_NUM_THREADS=sella_threads,
        MKL_NUM_THREADS=sella_threads,
    ):
        yield


@contextmanager
def orca_env() -> Generator[Any]:
    """Environment for ORCA calculations."""
    orca_path = Path(os.environ.get("FIRECODE_PATH_TO_ORCA", "") or str(which("orca")))
    orca_lib_path = Path(
        os.environ.get("FIRECODE_PATH_TO_ORCA_LIB", "") or orca_path.parent / "lib"
    )

    xtb_path = Path(os.environ.get("FIRECODE_PATH_TO_XTB", "") or str(which("xtb")))

    with env_override(
        RSH_COMMAND="/usr/bin/ssh -x",
        LD_LIBRARY_PATH=f"{orca_lib_path!s}:{orca_path.parent!s}",
        ORCAEXE=str(orca_path),
        XTBEXE=str(xtb_path),
    ):
        yield
