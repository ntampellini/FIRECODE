"""Tests for FIRECODE."""

import json
from pathlib import Path

import pytest

from firecode.embedder import Embedder
from firecode.optimization_methods import Opt_func_dispatcher
from firecode.utils import FolderContext, HiddenPrints, NewFolderContext, clean_directory, read_xyz

HERE = Path(__file__).resolve().parent


def cleanup() -> None:
    """Clean the current test directory."""
    clean_directory(
        to_remove_startswith=["firecode"],
        to_remove_endswith=[".log", ".out", ".svg", ".json", "_traj.xyz", "_saddles.xyz"],
        to_remove_contains=[
            "clockwise",
            "_confs",
            "_scan_max.xyz",
            "_scan.xyz",
            "_opt",
            "_thermo",
            "_ts_conf",
            "CREST",
            "NEB",
            "FSM",
        ],
    )


def run_calculator_test(calculator: str) -> None:
    """Tests a generic calculator."""
    mol = read_xyz(str(HERE / "C2H4.xyz"))
    dispatcher = Opt_func_dispatcher(calculator)
    _, _, success = dispatcher.opt_func(  # type: ignore[operator]
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


def run_firecode_input(name: str) -> None:
    """Runs a FIRECODE input file and checks that it exits successfully."""
    with FolderContext(str(HERE / name)):
        with pytest.raises(SystemExit) as result:
            with HiddenPrints():
                embedder = Embedder(f"{name}.txt", stamp=name)
                embedder.run()

        assert result.type == SystemExit
        assert result.value.code == 0

        cleanup()


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


@pytest.mark.operator
@pytest.mark.codecov
def test_scan_linear() -> None:
    """Tests a simple linear scan."""
    run_firecode_input("operator_scan_linear")


@pytest.mark.operator
@pytest.mark.codecov
def test_scan_plus_neb() -> None:
    """Tests a linear scan followed by a NEB."""
    run_firecode_input("operator_scan+neb")


@pytest.mark.operator
@pytest.mark.codecov
def test_scan_dihedral() -> None:
    """Tests a simple dihedral scan."""
    run_firecode_input("operator_scan_dihedral")


@pytest.mark.operator
@pytest.mark.codecov
def test_operator_rdkit_search() -> None:
    """Tests the rdkit_search operator."""
    run_firecode_input("operator_rdkit_search")


@pytest.mark.operator
@pytest.mark.codecov
def test_operator_crest_search() -> None:
    """Tests the crest_search operator."""
    run_firecode_input("operator_crest_search")


@pytest.mark.operator
@pytest.mark.codecov
def test_operator_neb() -> None:
    """Tests the neb operator."""
    run_firecode_input("operator_neb")


@pytest.mark.operator
@pytest.mark.codecov
def test_operator_fsm() -> None:
    """Tests the fsm operator."""
    run_firecode_input("operator_fsm")


@pytest.mark.operator
@pytest.mark.codecov
def test_operator_pka() -> None:
    """Tests the pka operator."""
    run_firecode_input("operator_pka")


@pytest.mark.operator
@pytest.mark.codecov
def test_operator_saddle() -> None:
    """Tests the saddle operator."""
    run_firecode_input("operator_saddle")


@pytest.mark.codecov
def test_solvent_names() -> None:
    """Tests the solvent names."""
    from firecode.solvents import epsilon_dict, solvent_data, solvent_synonyms, to_xtb_solvents

    for solvent in solvent_synonyms:
        if solvent in solvent_data:
            corrected = solvent_synonyms[solvent]
            raise KeyError(f'"{solvent}" in solvent_data should be named "{corrected}"')

    err = []
    for translated_solvent in solvent_synonyms.values():
        if translated_solvent not in to_xtb_solvents:
            if translated_solvent not in epsilon_dict:
                err.append(
                    f'Solvent "{translated_solvent}" has no epsilon value defined in solvent_data: '
                    "it should then be a key in to_xtb_solvents."
                )

    if err:
        raise Exception("\n".join(err))


@pytest.mark.codecov
def test_vib_analysis() -> None:
    """Tests the ase_vib function."""
    from firecode.thermochemistry import ase_vib

    mol = read_xyz(str(HERE / "C2H4.xyz"))
    solvent = "toluene"
    ase_calc = Opt_func_dispatcher("XTB").get_ase_calc("GFN-FF", solvent)
    with NewFolderContext(str(HERE / "temp_C2H4_vib")):
        _ = ase_vib(
            atoms=mol.atoms,
            coords=mol.coords[0],
            ase_calc=ase_calc,
            charge=0,
            mult=1,
            solvent=solvent,
        )

        with open(str(HERE / "temp_C2H4_vib/temp_thermo.json"), "rb") as f:
            thermo_dict = json.loads(f.read())

        assert all([f >= 0 for f in thermo_dict["freqs_cm1"]])
        assert len(thermo_dict["freqs_cm1"]) == len(mol.atoms) * 3
