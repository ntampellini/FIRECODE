"""Tests for FIRECODE."""
import os
from pathlib import Path

import pytest

from firecode.embedder import Embedder
from firecode.utils import FolderContext, HiddenPrints, clean_directory

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

def run_firecode_input(name) -> None:
    """Runs a FIRECODE input file and checks that it exits successfully.
    
    """
    with FolderContext(name):
        with pytest.raises(SystemExit) as result:
            with HiddenPrints():
                embedder = Embedder(f'{name}.txt', stamp=name)
                embedder.run()

        assert result.type == SystemExit
        assert result.value.code == 0

        clean_directory(
            to_remove_startswith=['firecode'],
            to_remove_endswith=['.log', '.out', '.svg'],
            to_remove_contains=['clockwise'],
            )

def test_string() -> None:
    """Tests a simple string embed.

    """
    run_firecode_input('embed_string')

def test_cyclical() -> None:
    """Tests a simple cyclical embed.

    """
    run_firecode_input('embed_cyclical')

def test_trimolecular() -> None:
    """Tests a simple trimolecular embed.

    """
    run_firecode_input('embed_trimolecular')

def test_dihedral() -> None:
    """Tests a simple dihedral scan.

    """
    run_firecode_input('scan_dihedral')
