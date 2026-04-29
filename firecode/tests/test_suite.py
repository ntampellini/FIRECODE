"""Tests for FIRECODE."""

import json
import shutil
from pathlib import Path

import pytest

from firecode.context_managers import FolderContext, HiddenPrints, NewFolderContext
from firecode.dispatcher import Opt_func_dispatcher
from firecode.embedder import Embedder
from firecode.standalone_optimizer import OptimizerOptions, standalone_optimize
from firecode.utils import clean_directory, read_xyz

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
            "_packmol",
            "CREST",
            "NEB",
            "FSM",
        ],
        not_to_remove_endswith=[".txt", ".inp"],
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


@pytest.mark.embed
@pytest.mark.codecov
def test_embed_multiembed() -> None:
    """Tests a simple multiembed."""
    run_firecode_input("embed_multiembed")


@pytest.mark.operator
@pytest.mark.codecov
def test_legacy_csearch() -> None:
    """Tests the two legacy conf search operators."""
    run_firecode_input("operator_firecode_search")


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
def test_operator_racerts() -> None:
    """Tests the racerts operator."""
    run_firecode_input("operator_racerts")


# @pytest.mark.operator
# @pytest.mark.codecov
# def test_operator_pka() -> None:
#     """Tests the pka operator."""
#     run_firecode_input("operator_pka")


@pytest.mark.operator
@pytest.mark.codecov
def test_operator_saddle() -> None:
    """Tests the saddle operator."""
    run_firecode_input("operator_saddle")


@pytest.mark.operator
@pytest.mark.codecov
def test_packmol_operator() -> None:
    """Test the packmol operator."""
    run_firecode_input("operator_packmol")


@pytest.mark.codecov
def test_multithread_refin() -> None:
    """Test multithread refining."""
    run_firecode_input("multithread_refine")


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
    dispatcher = Opt_func_dispatcher("TBLITE")
    dispatcher.get_ase_calc(solvent=solvent)
    with NewFolderContext(str(HERE / "temp_C2H4_vib")):
        _ = ase_vib(
            atoms=mol.atoms,
            coords=mol.coords[0],
            dispatcher=dispatcher,
            charge=0,
            mult=1,
            solvent=solvent,
            tighten_opt_before_vib=True,
        )

        with open(str(HERE / "temp_C2H4_vib/temp_thermo.json"), "rb") as f:
            thermo_dict = json.loads(f.read())

        assert all([f >= 0 for f in thermo_dict["freqs_cm1"]])
        assert len(thermo_dict["freqs_cm1"]) == len(mol.atoms) * 3


@pytest.mark.codecov
def test_alpb_delta_calc() -> None:
    """Verifies that the ALPB delta calc does something."""
    from firecode.ase_manipulations import ase_popt
    from firecode.context_managers import env_override

    mol = read_xyz(str(HERE / "C2H4.xyz"))
    solvent = "toluene"

    with env_override(
        FIRECODE_SOLV_IMPLEM_FOR_ML="opt",
        FIRECODE_SOLV_METHOD_FOR_ML="alpb",
    ):
        dispatcher = Opt_func_dispatcher("AIMNET2")
        dispatcher.get_ase_calc(solvent=None, force_reload=True)

    _, vac_energy, _ = ase_popt(
        atoms=mol.atoms,
        coords=mol.coords[0],
        dispatcher=dispatcher,
        maxiter=0,
    )

    with env_override(
        FIRECODE_SOLV_IMPLEM_FOR_ML="opt",
        FIRECODE_SOLV_METHOD_FOR_ML="alpb",
    ):
        dispatcher = Opt_func_dispatcher("AIMNET2")
        dispatcher.get_ase_calc(solvent=solvent, force_reload=True)

    _, solv_energy, _ = ase_popt(
        atoms=mol.atoms,
        coords=mol.coords[0],
        dispatcher=dispatcher,
        maxiter=0,
    )

    assert abs(vac_energy - solv_energy) > 1e-3, (
        f"solv_energy (ALPB) ~ 0 ({abs(vac_energy - solv_energy):.6f} kcal/mol)"
    )


@pytest.mark.codecov
def test_cpcm_delta_calc() -> None:
    """Verifies that the CPCM delta calc does something."""
    from firecode.ase_manipulations import ase_popt
    from firecode.context_managers import env_override

    mol = read_xyz(str(HERE / "C2H4.xyz"))
    solvent = "toluene"

    with env_override(
        FIRECODE_SOLV_IMPLEM_FOR_ML="opt",
        FIRECODE_SOLV_METHOD_FOR_ML="cpcm",
    ):
        dispatcher = Opt_func_dispatcher("AIMNET2")
        dispatcher.get_ase_calc(solvent=None, force_reload=True)

    _, vac_energy, _ = ase_popt(
        atoms=mol.atoms,
        coords=mol.coords[0],
        dispatcher=dispatcher,
        maxiter=0,
    )

    with env_override(
        FIRECODE_SOLV_IMPLEM_FOR_ML="opt",
        FIRECODE_SOLV_METHOD_FOR_ML="cpcm",
    ):
        dispatcher = Opt_func_dispatcher("AIMNET2")
        dispatcher.get_ase_calc(solvent=solvent, force_reload=True)

    _, solv_energy, _ = ase_popt(
        atoms=mol.atoms,
        coords=mol.coords[0],
        dispatcher=dispatcher,
        maxiter=0,
    )

    assert abs(vac_energy - solv_energy) > 1e-3, (
        f"solv_energy (CPCM) ~ 0 ({abs(vac_energy - solv_energy):.6f} kcal/mol)"
    )


@pytest.mark.codecov
def test_standalone_xtb() -> None:
    """Verifies that the standalone optimizer runs with XTB."""
    filename = str(HERE / "C2H4.xyz")

    options = OptimizerOptions(
        filenames=[filename],
        calc="XTB",
        method="GFN2-xTB",
        solvent="toluene",
    )

    standalone_optimize(options)


@pytest.mark.codecov
def test_standalone_tblite() -> None:
    """Verifies that the standalone optimizer runs with tblite."""
    source = str(HERE / "C2H4.xyz")
    target = str(HERE / "test_standalone_tblite" / "C2H4.xyz")

    with NewFolderContext(
        str(HERE / "test_standalone_tblite"),
        # delete_after=False
    ):
        # copy input structure over
        shutil.copy(source, target)

        # write a SMARTS-based constraint file
        with open("c.txt", "w") as f:
            f.write("SMARTS [#6]([#1])~[#6]([#1])\n0 1 2 3 0.0\n")

        # run writing log to temp.log
        with open("temp.log", "w") as f:
            options = OptimizerOptions(
                filenames=[target],
                calc="TBLITE",
                method="GFN2-xTB",
                solvent="ch2cl2",
                opt=True,
                constraint_file="c.txt",
                logfunction=lambda s: f.write(s + "\n"),  # type: ignore[arg-type]
            )

            standalone_optimize(options)


@pytest.mark.codecov
def test_standalone_aimnet2() -> None:
    """Verifies that the standalone optimizer runs with AIMNET2."""
    filename = str(HERE / "C2H4.xyz")

    options = OptimizerOptions(
        filenames=[filename],
        calc="AIMNET2",
        method="wB97M-D3",
        solvent="dmf",
    )

    standalone_optimize(options)


@pytest.mark.codecov
def test_standalone_saddle() -> None:
    """Verifies that the standalone optimizer can run a saddle optimization."""
    source = str(HERE / "propane_ts.xyz")
    target = str(HERE / "test_standalone_saddle" / "propane_ts.xyz")

    with NewFolderContext(str(HERE / "test_standalone_saddle")):
        shutil.copy(source, target)
        options = OptimizerOptions(
            filenames=[target],
            calc="TBLITE",
            method="GFN2-xTB",
            solvent="toluene",
            saddle=True,
            irc=True,
        )

        standalone_optimize(options)
