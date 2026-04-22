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

from typing import Any

from firecode.units import AVOGADRO_NA, A3_TO_mL

# from any solvent name to the standard FIRECODE name.
# Convention: short over long (methanol->meoh)
# but common over short (phme->toluene)
# No spaces in solvent name!
solvent_synonyms = {
    "ch3cooh": "acoh",
    "aceticacid": "acoh",
    "ch3cn": "mecn",
    "acetonitrile": "mecn",
    "chloroform": "chcl3",
    "dcm": "ch2cl2",
    "dichloromethane": "ch2cl2",
    "carbondisuphide": "cs2",
    "carbondisulfide": "cs2",
    "diethylether": "et2o",
    "ethanol": "etoh",
    "ch3oh": "meoh",
    "methanol": "meoh",
    "water": "h2o",
    "ethylacetate": "etoac",
    "phme": "toluene",
    "phh": "benzene",
    "dimethylformamide": "dmf",
    "dimethylether": "me2o",
    "triethylamine": "et3n",
    "nitromethane": "meno2",
    "dimethylacetamide": "dmac",
}

# Convert from standard to XTB names.
# do not remove dummy (s : s) entries,
# as keys are used for checking purposes
to_xtb_solvents = {
    "acetone": "acetone",
    "mecn": "acetonitrile",
    "aniline": "aniline",
    "benzaldehyde": "benzaldehyde",
    "benzene": "benzene",
    "ch2cl2": "ch2cl2",
    "chcl3": "chcl3",
    "cs2": "cs2",
    "dioxane": "dioxane",
    "dmf": "dmf",
    "dmso": "dmso",
    "et2o": "ether",
    "etoac": "ethylacetate",
    "furane": "furane",
    "hexadecane": "hexadecane",
    "hexane": "hexane",
    "meoh": "methanol",
    "meno2": "nitromethane",
    "octanol": "octanol",
    "octanolwet": "octanolwet",
    "phenol": "phenol",
    "toluene": "toluene",
    "thf": "thf",
    "h2o": "water",
}

# name: {molarity (M), molecular_volume(Å^3)}
# see https://organicchemistrydata.org/solvents/
solvent_data: dict[str, dict[str, Any]] = {
    "none": {
        "MW": 0.0,  # g/mol
        "density": 0.0,  # g/mL
        "molarity": 1.0,  # mol/L
        "molecular_volume": 1.0,  # Å^3
        "compressibility": 10e-5,  # bar^(-1)
    },
    "h2o": {
        "smiles": "O",
        "molarity": 55.6,
        "molecular_volume": 27.944,
        "epsilon": 80.1,
        "compressibility": 4.57e-5,
    },
    "toluene": {
        "molarity": 9.4,
        "molecular_volume": 149.070,
        "epsilon": 2.38,
    },
    "dmf": {
        "molarity": 12.9,
        "molecular_volume": 77.442,
        "epsilon": 36.71,
    },
    "acoh": {
        "molarity": 17.4,
        "molecular_volume": 86.10,
        "epsilon": 6.15,
    },
    "chcl3": {
        "molarity": 12.5,
        "molecular_volume": 97.0,
        "epsilon": 4.8,
    },
    "acetone": {
        "MW": 58.08,
        "density": 0.7845,
        "epsilon": 20.7,
    },
    "mecn": {
        "MW": 41.052,
        "density": 0.7857,
        "epsilon": 37.5,
        "compressibility": 10.7e-5,
    },
    "benzene": {
        "MW": 78.11,
        "density": 0.8765,
        "epsilon": 2.28,
    },
    "cs2": {
        "MW": 76.13,
        "density": 1.266,
        "epsilon": 2.63,
    },
    "dioxane": {
        "MW": 88.11,
        "density": 1.033,
        "epsilon": 2.25,
    },
    "dmso": {
        "MW": 78.13,
        "density": 1.092,
        "epsilon": 46.68,
        "compressibility": 5.2e-5,
    },
    "et2o": {
        "MW": 74.12,
        "density": 0.713,
        "epsilon": 4.27,
        "compressibility": 18.0e-5,
    },
    "me2o": {
        "MW": 46.07,
        "density": 0.735,
        "epsilon": 6.18,
    },
    "etoh": {
        "MW": 46.07,
        "density": 0.789,
        "epsilon": 24.3,
    },
    "meoh": {
        "MW": 32.04,
        "density": 0.791,
        "epsilon": 32.63,
    },
    "etoac": {
        "MW": 88.11,
        "density": 0.895,
        "epsilon": 6.02,
    },
    "thf": {
        "MW": 72.106,
        "density": 0.8833,
        "epsilon": 7.58,
    },
    "mtbe": {
        "MW": 88.15,
        "density": 0.741,
    },
    "phcf3": {
        "MW": 146.11,
        "density": 1.19,
        "epsilon": 9.18,
    },
    "et3n": {
        "MW": 101.19,
        "density": 0.7255,
        "epsilon": 2.4,
    },
    "ch2cl2": {
        "smiles": "C(Cl)Cl",
        "MW": 84.93,
        "density": 1.33,
        "epsilon": 8.93,
    },
    "dmac": {
        "MW": 87.122,
        "density": 0.937,
        "epsilon": 37.78,
    },
}

# estimate molarity or molecular_volume from
# MW and density if we are missing expt. data
for data_dict in solvent_data.values():
    if data_dict.get("molarity") is None:
        data_dict["molarity"] = 1000 / (data_dict["density"] * data_dict["MW"])

    if data_dict.get("molecular_volume") is None:
        data_dict["molecular_volume"] = (
            data_dict["MW"] / data_dict["density"] / AVOGADRO_NA / A3_TO_mL
        )

epsilon_dict = {
    solvent: data["epsilon"] for solvent, data in solvent_data.items() if "epsilon" in data
}
