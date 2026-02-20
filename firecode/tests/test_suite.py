"""Tests for FIRECODE."""

from pathlib import Path

import pytest

from firecode.embedder import Embedder
from firecode.optimization_methods import Opt_func_dispatcher
from firecode.utils import FolderContext, HiddenPrints, clean_directory, read_xyz

HERE = Path(__file__).resolve().parent


def run_calculator_test(calculator) -> None:
    """Tests a generic calculator."""
    mol = read_xyz(HERE / "C2H4.xyz")
    dispatcher = Opt_func_dispatcher(calculator)
    _, _, success = dispatcher.opt_func(
        atoms=mol.atoms,
        coords=mol.coords[0],
        calculator=calculator,
        maxiter=5,
    )
    assert success


@pytest.mark.calc
@pytest.mark.codecov
def test_calc_xtb() -> None:
    """Tests the FIRECODE XTB calculator."""
    run_calculator_test("XTB")


@pytest.mark.calc
@pytest.mark.codecov
def test_calc_tblite() -> None:
    """Tests the ASE TBLITE calculator."""
    run_calculator_test("TBLITE")


@pytest.mark.calc
@pytest.mark.codecov
def test_calc_aimnet2() -> None:
    """Tests the ASE AIMNET2 calculator."""
    run_calculator_test("AIMNET2")


@pytest.mark.calc
def test_calc_uma() -> None:
    """Tests the ASE UMA calculator."""
    run_calculator_test("UMA")


def run_firecode_input(name) -> None:
    """Runs a FIRECODE input file and checks that it exits successfully."""
    with FolderContext(HERE / name):
        with pytest.raises(SystemExit) as result:
            with HiddenPrints():
                embedder = Embedder(f"{name}.txt", stamp=name)
                embedder.run()

        assert result.type == SystemExit
        assert result.value.code == 0

        clean_directory(
            to_remove_startswith=["firecode"],
            to_remove_endswith=[".log", ".out", ".svg"],
            to_remove_contains=["clockwise", "_scan", "_confs", "_opt", "_crest"],
        )


@pytest.mark.embed
@pytest.mark.codecov
def test_embed_string() -> None:
    """Tests a simple string embed."""
    run_firecode_input("embed_string")


@pytest.mark.embed
@pytest.mark.codecov
def test_embed_cyclical() -> None:
    """Tests a simple cyclical embed."""
    run_firecode_input("embed_cyclical")


@pytest.mark.embed
@pytest.mark.codecov
def test_trimolecular() -> None:
    """Tests a simple trimolecular embed."""
    run_firecode_input("embed_trimolecular")


# @pytest.mark.embed
# @pytest.mark.codecov
# def test_embed_multimolecular() -> None:
#     """Tests a simple multimolecular embed."""
#     run_firecode_input("embed_multimolecular")


@pytest.mark.scan
@pytest.mark.codecov
def test_scan_linear() -> None:
    """Tests a simple linear scan."""
    run_firecode_input("scan_linear")


@pytest.mark.scan
@pytest.mark.codecov
def test_scan_dihedral() -> None:
    """Tests a simple dihedral scan."""
    run_firecode_input("scan_dihedral")


@pytest.mark.operator
@pytest.mark.codecov
def test_operator_rdkit_search() -> None:
    """Tests the rdkit_search operator."""
    run_firecode_input("operator_rdkit_search")
