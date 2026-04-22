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

import os
import re
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator


def sub_value(string: str, key: str, value: str) -> str:
    """Returns a new string with the settings.py syntax."""
    return re.sub(
        f'{key}=".*"',
        f'{key}="{value}"',
        string,
        count=1,
    )


def run_setup() -> None:
    """Invoked by the command
    > python -m firecode -s (--setup)

    Guides the user in setting up the calculation options
    contained in the settings.py file.
    """
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    properties = {
        "FIRECODE_CALCULATOR": "",
        "FIRECODE_PATH_TO_UMA_MODEL": "",
    }

    print("\nFIRECODE setup:\n")

    #########################################################################################

    # FIRECODE_CALCULATOR

    properties["FIRECODE_CALCULATOR"] = inquirer.select(  # type: ignore[attr-defined]
        message="What main calculator would you like to use?",
        choices=[
            Choice(value="AIMNET2", name="AIMNET2"),
            Choice(value="XTB", name="XTB"),
            Choice(value="TBLITE", name="TBLITE"),
            Choice(value="ORCA", name="ORCA"),
            Choice(value="UMA", name="UMA"),
        ],
        default="TBLITE",
    ).execute()

    #########################################################################################

    # FIRECODE_DEFAULT_LEVEL_{calc}

    kw_name = f"FIRECODE_DEFAULT_LEVEL_{properties['FIRECODE_CALCULATOR']}"
    old_default = str(os.environ.get(kw_name))
    properties[kw_name] = inquirer.text(  # type: ignore[attr-defined]
        message=f"The default level for {properties['FIRECODE_CALCULATOR']} calculations is '{old_default}'.\n"
        + "If you would like to change it, type it here, otherwise press enter:",
        default=old_default,
    ).execute()

    #########################################################################################

    # FIRECODE_PATH_TO_UMA_MODEL

    if properties["FIRECODE_CALCULATOR"] == "UMA":
        old_default = str(Path(str(os.environ.get("FIRECODE_PATH_TO_UMA_MODEL", ""))))
        properties["FIRECODE_PATH_TO_UMA_MODEL"] = inquirer.filepath(  # type: ignore[attr-defined]
            message="Please specify the location of the UMA model:",
            default=old_default,
            validate=PathValidator(is_file=True, message="Please specify a file"),
        ).execute()

    #########################################################################################

    with open("settings.py", "r") as f:
        lines = f.readlines()

    old_lines = lines.copy()

    for _l, line in enumerate(old_lines):
        for key, value in properties.items():
            if key in line:
                lines[_l] = sub_value(line, key, value=value)

    with open("settings.py", "w") as f:
        f.write("".join(lines))

    print("\nFIRECODE setup performed correctly.")

    opt = f"{properties['FIRECODE_CALCULATOR']}/{properties[kw_name]}"

    s = f"  OPT       : {opt}\n"

    if properties["FIRECODE_CALCULATOR"] == "UMA":
        s += f"  UMA MODEL : {properties['FIRECODE_PATH_TO_UMA_MODEL']}"

    s += "\n"

    print(s)


if __name__ == "__main__":
    run_setup()
