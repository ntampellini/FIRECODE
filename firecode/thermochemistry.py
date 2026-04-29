# coding=utf-8
"""FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2026 Nicolò Tampellini

SPDX-License-Identifier: LGPL-3.0-or-later

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see
https://www.gnu.org/licenses/lgpl-3.0.en.html#license-text.

"""

from __future__ import annotations

import json
import math
import os
from shutil import rmtree
from subprocess import getoutput
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import numpy as np
from ase import Atoms
from ase.vibrations import Vibrations
from prettytable import PrettyTable
from prism_pruner.utils import time_to_string

from firecode.ase_manipulations import optimizer_dict, set_charge_and_mult_on_ase_atoms
from firecode.context_managers import HiddenPrints, NewFolderContext
from firecode.dispatcher import Opt_func_dispatcher
from firecode.solvents import solvent_data, solvent_synonyms
from firecode.typing_ import Array1D_float, Array1D_str, Array2D_float, Array3D_float
from firecode.units import (
    AVOGADRO_NA,
    EH_TO_EV,
    EH_TO_KCAL,
    EV_TO_KCAL,
    EV_TO_WAVENUMS,
    KB__J_K,
    A3_TO_mL,
    AMU__kg,
    ANGSTROEM_TO_m,
    C__cm_s,
    KB__eV_K,
    PLANCK_h__J_s,
    theta_per_cm1_K,
)
from firecode.utils import clean_directory, loadbar, write_xyz

if TYPE_CHECKING:
    from firecode.embedder import Embedder


# Frequencies below this are considered belonging to proper
# transition states and excluded from thermochemical analysis,
# while the (_TS_THR_CM_1 < v < 0) range will be treated as positive.
_TS_THR_CM_1: float = -25


def _free_space_mL_per_L(solvent: str | None = None) -> float:
    """Return accessible free space (mL per L) for a solute in bulk solvent.
    Based on Shakhnovich & Whitesides (J. Org. Chem. 1998, 63, 3821) and
    the GoodVibes implementation.

    """
    if not solvent:
        return 1000.0

    # standardize solvent name
    solvent = solvent_synonyms.get(solvent.lower(), solvent.lower())

    if solvent not in solvent_data:
        raise NotImplementedError(
            f'Unknown "{solvent}" passed as solvent. Currently '
            f"parametrized solvents for quasi-RRHO: {list(solvent_data.keys())}"
        )

    molarity = solvent_data[solvent]["molarity"]
    mol_volume = solvent_data[solvent]["molecular_volume"]

    # v_free (Å^3 per molecule) for accessible volume
    v_free = (
        8.0 * ((1e27 / (molarity * AVOGADRO_NA)) ** (1.0 / 3.0) - mol_volume ** (1.0 / 3.0)) ** 3
    )

    # Convert to mL free space per liter of bulk solvent
    freespace_mL_per_L = v_free * molarity * AVOGADRO_NA * A3_TO_mL

    return float(freespace_mL_per_L)


def rotational_constants_cm1_from_I(I_amuA2: Array1D_float) -> Array1D_float:
    I_SI = I_amuA2 * AMU__kg * (ANGSTROEM_TO_m**2)
    B_cm = []
    for I in I_SI:
        B_cm.append(0.0 if I <= 0 else PLANCK_h__J_s / (8.0 * math.pi**2 * C__cm_s * I))
    return np.array([B_cm[0], B_cm[1], B_cm[2]])


def classify_geometry(I_amuA2: Array1D_float) -> str:
    """Return one of: 'atom', 'linear', 'nonlinear'
    - 'atom' if all principal moments are (near) zero
    - 'linear' if one moment is ~0 but not all (diatomics, etc.)
    - 'nonlinear' otherwise
    """
    abs_I_amuA2 = np.abs(I_amuA2)
    imax = float(np.max(abs_I_amuA2))
    # Treat as atom if all moments are essentially zero
    if imax < 1e-12:
        return "atom"
    if (float(np.min(abs_I_amuA2)) / imax) < 1e-6:
        return "linear"
    return "nonlinear"


def rrho_thermo(
    atoms: Atoms,
    freqs_cm1: Array1D_float,
    T_K: float = 298.15,
    P_atm: float | None = 1.0,
    conc_mol_L: float | None = None,
    symmetry_number: int = 1,
    E_el_Eh: float = 0.0,
    charge: int = 0,
    mult: int = 1,
    qrrho: bool = True,
    cutoff_cm1: Optional[float] = None,
    qrrho_ref_cm1: float = 100.0,
    qrrho_alpha: float = 4.0,
    solv: Optional[str] = None,
    assert_gs_or_ts: bool = False,
) -> dict[str, Any]:
    # Remove 3/5/6 zero modes
    I = atoms.get_moments_of_inertia()  # type:ignore[no-untyped-call]

    match geom := classify_geometry(I):
        case "atom":
            start = 3
        case "linear":
            start = 5
        case _:  # 'nonlinear'
            start = 6

    vib_cm_all = []
    for i, f in enumerate(freqs_cm1):
        if i < start:
            continue

        # genuine TS imaginary mode — exclude entirely
        if f < _TS_THR_CM_1:
            continue

        # small negative = numerical noise — treat as positive
        if abs(f) > 1e-3:
            vib_cm_all.append(abs(f))

    if assert_gs_or_ts:
        assert (len(freqs_cm1) - len(vib_cm_all) - start) <= 1, (
            f"Frequency mismatch: expected {len(freqs_cm1) - start} (GS) or "
            f"{len(freqs_cm1) - start - 1} (TS) frequencies, but {len(vib_cm_all)} were read."
        )

    if cutoff_cm1 is None:
        cutoff_cm1 = 1.0 if qrrho else 35.0
    vib_cm = [f for f in vib_cm_all if f > cutoff_cm1]

    mass_amu = float(np.sum(atoms.get_masses()))
    mass_kg = mass_amu * AMU__kg

    B_A, B_B, B_C = rotational_constants_cm1_from_I(I)
    theta = [theta_per_cm1_K * b for b in (abs(B_A), abs(B_B), abs(B_C))]
    theta = [t if t > 0 else 1e-30 for t in theta]

    # vib_e = [f / EV_TO_WAVENUMS for f in vib_cm]
    # ZPE_eV = 0.5 * sum(vib_e) # GoodVibes approximation: all modes contribute equally

    # In Grimme's original formulation (Angew. Chem. 2012), the ZPE should technically
    # be interpolated as well — a free rotor has no zero-point energy, so the
    # contribution should be (w × 0.5 * hν) per mode. The strict expression is:

    # ZPE = Σ  w_i × (½hν_i)

    # Added to vib_cm loop, see below.

    # Rot/trans energies (unchanged)
    Erot_eV = 0.0 if geom == "atom" else ((1.0 if geom == "linear" else 1.5) * KB__eV_K * T_K)
    Etrans_eV = 1.5 * KB__eV_K * T_K

    # ---------- Translational entropy: pressure OR concentration ----------
    #                              (Sackur-Tetrode)
    # S/k = ln( (2π m kT)^{3/2} / (h^3 n) ) + 5/2  where n is number density [1/m^3].
    # - Gas phase:    n = P / (kT)
    # - Solution:     n = (conc [mol/L]) * 1000 [L/m^3] * Na / (free_space_fraction),
    #                 with free-space from Shakhnovich–Whitesides.
    lambda_factor = ((2.0 * math.pi * mass_kg * KB__J_K * T_K) ** 1.5) / (PLANCK_h__J_s**3)

    # if user provided a concentration,
    # reference state is a solution
    if conc_mol_L is not None:
        free_mL_per_L = _free_space_mL_per_L(solv)
        # free-space fraction in a liter:
        free_frac = max(free_mL_per_L / 1000.0, 1e-9)  # avoid zero
        number_density = conc_mol_L * 1000.0 * AVOGADRO_NA / free_frac  # 1/m^3
    else:
        P_atm = P_atm or 1.0
        P_Pa = P_atm * 101325.0
        number_density = P_Pa / (KB__J_K * T_K)

    S_trans_over_k = math.log(lambda_factor / number_density) + 2.5
    TS_trans_eV = KB__eV_K * T_K * S_trans_over_k

    # ---------- Rotational entropy ----------
    if geom == "atom":
        S_rot_over_k = 0.0
    elif geom == "linear":
        theta_rot = theta[1]
        S_rot_over_k = math.log(T_K / (symmetry_number * theta_rot)) + 1.0

        # adding Herzberg linear correction
        S_rot_over_k += math.log(1.0 + theta_rot / (3.0 * T_K))
    else:
        prod_theta = theta[0] * theta[1] * theta[2]

        # classical term
        S_rot_over_k = (
            math.log(math.sqrt(math.pi) * (T_K**1.5) / (symmetry_number * math.sqrt(prod_theta)))
            + 1.5
        )

        # Herzberg formula: Add Euler-Maclaurin correction to ln(q_rot):
        euler_maclaurin = (theta[0] + theta[1] + theta[2]) / (12.0 * T_K)
        S_rot_over_k += math.log(1.0 + euler_maclaurin)

    TS_rot_eV = KB__eV_K * T_K * S_rot_over_k

    # ---------- Vibrational entropy + thermal vibrational energy ----------
    I_SI = I * AMU__kg * (ANGSTROEM_TO_m**2)
    I_av = float(np.mean(I_SI)) if np.any(I_SI > 0) else 1e-46

    ZPE_eV = 0.0
    S_vib_over_k = 0.0
    Evib_corr_eV = 0.0

    for f_cm in vib_cm:
        e_eV = f_cm / EV_TO_WAVENUMS
        x = e_eV / (KB__eV_K * T_K) if T_K > 0 else float("inf")

        if T_K > 0:
            S_HO_over_k = (x / math.expm1(x)) - math.log1p(-math.exp(-x))
            E_th_HO_eV = e_eV / math.expm1(x)
        else:
            S_HO_over_k = 0.0
            E_th_HO_eV = 0.0

        if not qrrho:
            # full weight
            ZPE_eV += 0.5 * e_eV

            S_vib_over_k += S_HO_over_k
            Evib_corr_eV += E_th_HO_eV
            continue

        # qRRHO
        w = 1.0 / (1.0 + (qrrho_ref_cm1 / f_cm) ** qrrho_alpha)

        nu_Hz = C__cm_s * f_cm
        muK = PLANCK_h__J_s / (8.0 * math.pi**2 * nu_Hz)
        muEff = (muK * I_av) / (muK + I_av)

        S_FR_over_k = 0.5 + 0.5 * math.log(
            (8.0 * math.pi**2 * muEff * KB__J_K * T_K) / (PLANCK_h__J_s**2)
        )

        # qRRHO: ZPE interpolated, free rotor contributes no ZPE
        ZPE_eV += w * (0.5 * e_eV)
        S_vib_over_k += w * S_HO_over_k + (1.0 - w) * S_FR_over_k
        Evib_corr_eV += w * E_th_HO_eV + (1.0 - w) * (0.5 * KB__eV_K * T_K)

    # Ignoring electronic entropy: beyond the
    # scope of this implementation
    TS_el_eV = 0.0

    E_el_eV = E_el_Eh * EH_TO_EV

    U_total_eV = E_el_eV + ZPE_eV + Evib_corr_eV + Erot_eV + Etrans_eV
    Hcorr_eV = KB__eV_K * T_K
    H_total_eV = U_total_eV + Hcorr_eV
    TS_vib_eV = KB__eV_K * T_K * S_vib_over_k
    TS_tot_eV = TS_el_eV + TS_vib_eV + TS_rot_eV + TS_trans_eV
    G_total_eV = H_total_eV - TS_tot_eV
    G_minus_Eel_eV = G_total_eV - E_el_eV

    # ---- conversions (unchanged) ----
    ZPE_Eh = ZPE_eV / EH_TO_EV
    Evib_corr_Eh = Evib_corr_eV / EH_TO_EV
    Erot_Eh = Erot_eV / EH_TO_EV
    Etrans_Eh = Etrans_eV / EH_TO_EV
    U_total_Eh = U_total_eV / EH_TO_EV
    H_total_Eh = H_total_eV / EH_TO_EV
    Hcorr_Eh = Hcorr_eV / EH_TO_EV
    TS_el_Eh = TS_el_eV / EH_TO_EV
    TS_vib_Eh = TS_vib_eV / EH_TO_EV
    TS_rot_Eh = TS_rot_eV / EH_TO_EV
    TS_trans_Eh = TS_trans_eV / EH_TO_EV
    G_total_Eh = G_total_eV / EH_TO_EV
    G_minus_Eel_Eh = G_minus_Eel_eV / EH_TO_EV

    # symmetry sweep
    rot_table = []
    for sn in range(1, 13):
        if geom == "atom":
            TS_rot_sn_Eh = 0.0
        elif geom == "linear":
            S_rot_sn = math.log(T_K / (sn * (theta[1]))) + 1.0
            S_rot_sn += math.log(1.0 + theta_rot / (3.0 * T_K))  # adding Herzberg linear correction
            TS_rot_sn_Eh = (KB__eV_K * T_K * S_rot_sn) / EH_TO_EV
        else:
            prod_theta = theta[0] * theta[1] * theta[2]
            S_rot_sn = (
                math.log(math.sqrt(math.pi) * (T_K**1.5) / (sn * math.sqrt(prod_theta))) + 1.5
            )
            S_rot_sn += math.log(1.0 + euler_maclaurin)  # adding Herzberg linear correction
            TS_rot_sn_Eh = (KB__eV_K * T_K * S_rot_sn) / EH_TO_EV
        rot_table.append((sn, TS_rot_sn_Eh))

    return dict(
        freqs_cm1=[float(f) for f in freqs_cm1],
        mass_amu=float(np.sum(atoms.get_masses())),
        rotconsts_cm1=(B_A, B_B, B_C),
        ZPE_Eh=ZPE_Eh,
        Evib_corr_Eh=Evib_corr_Eh,
        Erot_Eh=Erot_Eh,
        Etrans_Eh=Etrans_Eh,
        U_total_Eh=U_total_Eh,
        H_total_Eh=H_total_Eh,
        Hcorr_Eh=Hcorr_Eh,
        TS_el_Eh=TS_el_Eh,
        TS_vib_Eh=TS_vib_Eh,
        TS_rot_Eh=TS_rot_Eh,
        TS_trans_Eh=TS_trans_Eh,
        G_total_Eh=G_total_Eh,
        G_minus_Eel_Eh=G_minus_Eel_Eh,
        rot_table_Eh=rot_table,
        qrrho=qrrho,
        cutoff_cm1=float(cutoff_cm1),
        qrrho_ref_cm1=float(qrrho_ref_cm1) if qrrho else None,
        qrrho_alpha=float(qrrho_alpha) if qrrho else None,
        conc_mol_L=conc_mol_L,
        solv=(solv or "none"),
        mult=int(mult),
        charge=int(charge),
    )


def ase_vib(
    atoms: Array1D_str,
    coords: Array2D_float,
    dispatcher: Opt_func_dispatcher,
    charge: int,
    mult: int,
    optimizer: str | None = None,
    T_K: float = 298.15,
    P_atm: float | None = 1.0,
    C_mol_L: float | None = None,
    solvent: str | None = None,
    title: str = "temp",
    return_gcorr: bool = True,
    tighten_opt_before_vib: bool = False,
    write_log: bool = True,
) -> tuple[Array1D_float, float]:
    """returns: tuple of Array of frequencies and either G(corr) or Free energy, in kcal/mol.

    If pressure is provided, a gas reference state will be used.
    """
    ase_atoms = Atoms(atoms, positions=coords)
    ase_atoms = set_charge_and_mult_on_ase_atoms(ase_atoms, charge=charge, mult=mult)

    ase_atoms.calc = dispatcher.ase_calc  # ensures the calc is pre-loaded
    optimizer = optimizer or dispatcher.get_optimizer_str()
    ase_optimizer, opt_kwargs = optimizer_dict[optimizer]

    vib_name = title + "_vib"
    vib = Vibrations(ase_atoms, name=vib_name, delta=0.005)  # type: ignore[no-untyped-call]

    with open(f"{title}.out", "w") as f:
        f.write("--> FIRECODE ASE Frequency calculation report\n")
        f.write(
            f"charge={charge}, mult={mult}, C={C_mol_L} mol/L, P={P_atm} atm T={T_K:.2f} K ({T_K - 273.15:.2f} °C)\n"
        )
        f.write(
            f"Concentration {'not ' if C_mol_L is None else ''}provided: reference "
            f"state used is {'gas' if C_mol_L is None else 'solution'} phase.\n"
        )

        f.write(
            f"Solvent is {solvent} - accessible Shakhnovich & Whitesides space is {_free_space_mL_per_L(solvent):.2f} mL/L\n\n"
        )

        # tighten convergence to avoid negative frequencies
        if tighten_opt_before_vib:
            t_start = perf_counter()
            f.write("--> Tightening geom. opt. convergence to fmax=1E-2\n")
            opt = ase_optimizer(ase_atoms, **opt_kwargs)  # type: ignore[operator]
            with HiddenPrints():
                opt.run(fmax=1e-2)
            f.write(
                f"Structure optimized to fmax=1E-2 in {time_to_string(perf_counter() - t_start)}\n\n"
            )
            coords = ase_atoms.get_positions()  # type: ignore[no-untyped-call]

        # save structure in thermo folder
        with open(f"{title}.xyz", "w") as ff:
            write_xyz(atoms, coords, ff)

        energy_ev = ase_atoms.get_potential_energy()  # type: ignore[no-untyped-call]

        # add solvation energy if appropriate
        if solvent is not None:
            solv_energy_ev = (
                dispatcher.get_delta_solvation_energy(
                    atoms=atoms,
                    coords=coords,
                )
                / EV_TO_KCAL
            )
            energy_ev += solv_energy_ev

        # remove cache folder
        rmtree(os.path.join(os.getcwd(), "vib"), ignore_errors=True)

        # run vibrational analysis
        t_start = perf_counter()
        with HiddenPrints():
            vib.run()  # type: ignore[no-untyped-call]
        f.write(
            f"Vibrational frequencies calculated in {time_to_string(perf_counter() - t_start)}\n\n"
        )

        write_neg_modes_to_file(vib, title=title)

        # get energies (frequencies)
        freqs_complex = vib.get_energies() * EV_TO_WAVENUMS  # type: ignore[no-untyped-call]

        # clamp small negatives and sort freqs
        freqs = cleanup_freqs(ase_atoms, freqs_complex)

        # Print pretty table with freqs
        table = PrettyTable()
        table.field_names = ["Mode", "Freq. (cm^-1)"]
        for i, freq in enumerate(freqs):
            table.add_row([i, round(freq, 1)])

        f.write("\n" + table.get_string() + "\n")

        # run quasi-RRHO
        thermo_dict = rrho_thermo(
            atoms=ase_atoms,
            freqs_cm1=freqs,
            T_K=T_K,
            P_atm=P_atm,
            symmetry_number=1,  # detect_symmetry_number(atoms)?
            E_el_Eh=energy_ev / EH_TO_EV,
            charge=charge,
            mult=mult,
            conc_mol_L=C_mol_L,
            solv=solvent,
        )

        # save the raw thermo data as .json
        with open(f"{title}_thermo.json", "w") as ff:
            ff.write(json.dumps(thermo_dict, indent=4))

        EE = energy_ev / EH_TO_EV  # was eV, now Eh
        G = thermo_dict["G_total_Eh"]
        Gcorr = thermo_dict["G_minus_Eel_Eh"]
        H = thermo_dict["H_total_Eh"]
        S = (H - G) / T_K  # Eh/K

        f.write("\n--> What follows mocks an ORCA output:\n\n")
        f.write(f"Number of atoms ... {len(atoms)}\n")
        f.write(f"Total Charge ... ... {charge}\n\n")
        f.write(f"Temperature ...: {T_K:.2f} K ({T_K - 273.15:.2f} °C)\n")

        f.write("VIBRATIONAL FREQUENCIES\n")
        f.write("-------------------------------------\n")
        for i, freq in enumerate(freqs):
            f.write(f"  {i:>4}:    {freq:4.2f} cm**-1\n")

        f.write(f"\nFINAL SINGLE POINT ENERGY {EE:.8f} Eh\n")
        f.write(f"FINAL GIBBS FREE ENERGY {G:.8f} Eh\n")
        f.write(f"G-E(el) ... {Gcorr:.8f} Eh     {Gcorr * EH_TO_KCAL:.2f} kcal/mol\n\n")

        kB_times_T = (KB__eV_K / EH_TO_EV) * T_K
        f.write(f"Thermal Enthalpy correction ... {kB_times_T:.8f} Eh\n")
        f.write(f"Total correction {H - EE - kB_times_T:.8f} Eh\n")
        f.write(f"Total enthalpy ... {H:.8f} Eh\n\n")

        f.write(f"Final entropy term ... {S:.8f} Eh/K\n\n")

        f.write("*** ORCA TERMINATED NORMALLY ***\n")

    del vib
    rmtree(vib_name, ignore_errors=True)

    if not write_log:
        os.remove(f"{title}.out")

    if return_gcorr:
        return freqs, cast("float", Gcorr * EH_TO_KCAL)
    return freqs, cast("float", G * EH_TO_KCAL)


def cleanup_freqs(
    atoms: Atoms,
    freqs_complex: list[complex],
    scaling_factor: float = 1.0,
) -> Array1D_float:
    """Return sorted and cleaned up frequencies."""
    # Geometry class -> rigid-body count
    I = atoms.get_moments_of_inertia()  # type:ignore[no-untyped-call]

    match classify_geometry(I):  # "atom", "linear" or "nonlinear"
        case "atom":
            zero_first = 3
        case "linear":
            zero_first = 5
        case _:  # "nonlinear"
            zero_first = 6

    freqs_cm = []
    for f in np.asarray(freqs_complex):
        if np.iscomplexobj(f) and abs(f.imag) > 1e-8:
            freqs_cm.append(-float(abs(f.imag)))  # imaginary -> negative
        else:
            freqs_cm.append(float(np.real(f)))
    freqs = np.array(freqs_cm, dtype=float)

    nmode = len(freqs)
    idx_all = np.arange(nmode)

    # Identify the most negative as "imag" if any
    idx_imag = int(np.argmin(freqs)) if np.any(freqs < 0.0) else None

    # See if one looks like a real TS imgainary freq
    ts = freqs[idx_imag] < _TS_THR_CM_1 if idx_imag is not None else False

    # Choose rigid-body indices as those with the smallest |f|, excluding the chosen imag
    idx_sorted_abs = np.argsort(np.abs(freqs))
    rigid = [i for i in idx_sorted_abs if i != idx_imag][:zero_first]

    # Build permutation:
    #   rigid zeros, then (if TS) the imaginary, then the rest by increasing |f|
    perm = []
    perm.extend(rigid)
    if ts:
        perm.append(idx_imag)
    placed = set(perm)
    rest = [i for i in idx_all if i not in placed]
    rest_sorted = sorted(rest, key=lambda i: abs(freqs[i]))
    perm.extend(rest_sorted)

    # Apply permutation to freqs and modes
    freqs = freqs[perm]
    # modes_mw = modes_mw[:, perm]

    # Post-facto cleanup:
    #   - force the first zero_first freqs to exactly 0.0
    #   - clamp tiny |f| < |_TS_THR_CM_1| to +abs(f) for the rest
    freqs[:zero_first] = 0.0
    for i in range(zero_first, nmode):
        if abs(freqs[i]) < abs(_TS_THR_CM_1):
            freqs[i] = abs(freqs[i])

    # Scale if needed
    if scaling_factor != 1.0:
        freqs *= scaling_factor

    return freqs


def write_neg_modes_to_file(vib: Vibrations, title: str = "temp") -> None:
    """Write negative modes animations to file."""
    # get energies (frequencies)
    freqs_complex = vib.get_energies() * EV_TO_WAVENUMS  # type: ignore[no-untyped-call]

    for i, f in enumerate(freqs_complex):
        if abs(f.imag) > abs(_TS_THR_CM_1):
            # write mode with imaginary freq
            vib.write_mode(n=i, kT=0.02, nimages=30)  # type: ignore[no-untyped-call]

            # convert ase format to .xyz
            getoutput(f"ase convert {vib.name}.{i}.traj {title}_mode_{i + 1:0>3d}.xyz")

            # remove ase file
            os.remove(f"{vib.name}.{i}.traj")


def get_free_energies(
    embedder: Embedder,
    atoms: Array1D_str,
    structures: Array3D_float,
    charge: int,
    mult: int,
    title: str = "temp",
    tighten_opt_before_vib: bool = True,
    logfunction: Callable[[str], None] | None = None,
) -> Array1D_float:
    """Run vibrational analysis on a set of structures."""
    free_energies = []

    with NewFolderContext(title + "_thermo", delete_after=False, overwrite_if_exists=False):
        for s, structure in enumerate(structures):
            loadbar(
                s,
                len(structures),
                f"{title} Performing vibrational analysis {s + 1}/{len(structures)} ",
            )

            t_start = perf_counter()

            freqs, free_energy = ase_vib(
                atoms,
                structure,
                dispatcher=embedder.dispatcher,
                charge=charge,
                mult=mult,
                T_K=embedder.options.T,
                C_mol_L=embedder.options.C,
                solvent=embedder.options.solvent,
                title=f"{title}_conf{s + 1}",
                return_gcorr=False,
                tighten_opt_before_vib=tighten_opt_before_vib,
            )

            free_energies.append(free_energy)

            if logfunction is not None:
                elapsed = perf_counter() - t_start
                match num_neg := np.count_nonzero(freqs < 0.0):
                    case 0:
                        exit_str = "GS"
                        ss = "s., "
                    case 1:
                        exit_str = "TS"
                        ss = ".,  "
                    case _:
                        ss = "s., "
                        exit_str = "??"

                logfunction(
                    f"    - {title} conf{s:0>4d} - {exit_str} ({num_neg} negative freq{ss}{time_to_string(elapsed)})"
                )

        loadbar(
            len(structures),
            len(structures),
            f"{title} Performing vibrational analysis {len(structures)}/{len(structures)} ",
        )

        clean_directory(to_remove=("gfnff_topo",))

    return np.array(free_energies)
