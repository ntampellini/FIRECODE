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

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator

from firecode.settings import DEFAULT_FF_LEVELS, DEFAULT_LEVELS, UMA_MODEL_PATH


def run_setup():
    """Invoked by the command
    > python -m firecode -s (--setup)

    Guides the user in setting up the calculation options
    contained in the settings.py file.
    """
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    properties = {
        "FF_OPT_BOOL": False,
        "FF_CALC": None,
        "NEW_FF_DEFAULT": None,
        "CALCULATOR": None,
        "NEW_DEFAULT": None,
        "NEW_COMMAND": None,
        "UMA_MODEL_PATH": None,
        "PROCS": 4,
        "MEM_GB": 4,
    }

    print("\nFIRECODE setup:\n")

    #########################################################################################

    selected_calc = inquirer.select(
        message="What Force Field calculator would you like to use?",
        choices=(
            Choice(value=None, name="None (turn off FF optimization)"),
            Choice(value="XTB", name="XTB"),
        ),
        default=None,
    ).execute()

    if selected_calc is not None:
        properties["FF_CALC"] = selected_calc
        properties["FF_OPT_BOOL"] = True
    else:
        properties["FF_OPT_BOOL"] = False

    properties["CALCULATOR"] = inquirer.select(
        message="What main calculator would you like to use?",
        choices=(
            Choice(value="AIMNET2", name="AIMNET2"),
            Choice(value="XTB", name="XTB"),
            Choice(value="TBLITE", name="TBLITE"),
            Choice(value="ORCA", name="ORCA"),
            Choice(value="UMA", name="UMA"),
        ),
        default="XTB",
    ).execute()

    #########################################################################################

    properties["NEW_DEFAULT"] = inquirer.text(
        message=f"The default level for {properties['CALCULATOR']} calculations is '{DEFAULT_LEVELS[properties['CALCULATOR']]}'.\n"
        + "If you would like to change it, type it here, otherwise press enter:",
        default=DEFAULT_LEVELS[properties["CALCULATOR"]],
    ).execute()

    #########################################################################################

    if properties["CALCULATOR"] == "UMA":
        properties["UMA_MODEL_PATH"] = inquirer.filepath(
            message="Please specify the location of the UMA model:",
            default=UMA_MODEL_PATH,
            validate=PathValidator(is_file=True, message="Please specify a file"),
        ).execute()

    if properties["CALCULATOR"] in ["ORCA", "XTB"]:
        properties["PROCS"] = inquirer.text(
            message=f"How many cores should {properties['CALCULATOR']} jobs run on?:",
            default=str(properties["PROCS"]),
            validate=lambda inp: inp.isdigit(),
            filter=int,
        ).execute()

    if properties["CALCULATOR"] == "ORCA":
        properties["MEM_GB"] = inquirer.text(
            message="How much memory per core should a ORCA job have, in GBs?:",
            default=str(properties["MEM_GB"]),
            validate=lambda inp: inp.isdigit(),
            filter=int,
        ).execute()

    #########################################################################################

    rank = {
        "XTB": 1,
        "TBLITE": 2,
        "AIMNET2": 3,
        "UMA": 4,
        "ORCA": 5,
    }

    q = '"'

    with open("settings.py", "r") as f:
        lines = f.readlines()

    old_lines = lines.copy()

    for _l, line in enumerate(old_lines):
        if "FF_OPT_BOOL =" in line:
            lines[_l] = "FF_OPT_BOOL = " + str(properties["FF_OPT_BOOL"]) + "\n"
            FF_OPT_BOOL = properties["FF_OPT_BOOL"]

        if "FF_CALC =" in line:
            _q = q if properties["FF_CALC"] is not None else ""
            lines[_l] = "FF_CALC = " + _q + str(properties["FF_CALC"]) + _q + "\n"
            FF_CALC = properties["FF_CALC"]

        elif "CALCULATOR =" in line:
            lines[_l] = "CALCULATOR = " + q + properties["CALCULATOR"] + q + "\n"
            CALCULATOR = properties["CALCULATOR"]

        elif "DEFAULT_LEVELS = {" in line:
            if properties["NEW_DEFAULT"] is not None:
                lines[_l + rank[properties["CALCULATOR"]]] = (
                    " " * 4
                    + q
                    + properties["CALCULATOR"]
                    + q
                    + ":"
                    + q
                    + properties["NEW_DEFAULT"]
                    + q
                    + ",\n"
                )
                DEFAULT_LEVELS[CALCULATOR] = properties["NEW_DEFAULT"]

        elif "DEFAULT_FF_LEVELS = {" in line:
            if properties["NEW_FF_DEFAULT"] is not None:
                lines[_l + rank[properties["FF_CALC"]]] = (
                    " " * 4
                    + q
                    + properties["FF_CALC"]
                    + q
                    + ":"
                    + q
                    + properties["NEW_FF_DEFAULT"]
                    + q
                    + ",\n"
                )
                DEFAULT_FF_LEVELS[FF_CALC] = properties["NEW_FF_DEFAULT"]

        elif "COMMANDS = {" in line:
            if properties["NEW_COMMAND"] is not None:
                lines[_l + rank[properties["CALCULATOR"]]] = (
                    " " * 4
                    + q
                    + properties["CALCULATOR"]
                    + q
                    + ":"
                    + q
                    + properties["NEW_COMMAND"]
                    + q
                    + ",\n"
                )

        elif "PROCS =" in line:
            lines[_l] = "PROCS = " + str(properties["PROCS"]) + "\n"
            PROCS = properties["PROCS"]

        elif "MEM_GB =" in line:
            lines[_l] = "MEM_GB = " + str(properties["MEM_GB"]) + "\n"
            MEM_GB = properties["MEM_GB"]

        elif "UMA_MODEL_PATH =" in line:
            lines[_l] = "UMA_MODEL_PATH = " + q + properties["UMA_MODEL_PATH"] + q + "\n"

    with open("settings.py", "w") as f:
        f.write("".join(lines))

    print("\nFIRECODE setup performed correctly.")

    ff = f"{FF_CALC}/{DEFAULT_FF_LEVELS[FF_CALC]}" if FF_OPT_BOOL else "Turned off"
    opt = f"{CALCULATOR}/{DEFAULT_LEVELS[CALCULATOR]}"

    s = f"  FF OPT    : {ff}\n  OPT       : {opt}\n  PROCS     : {PROCS}"
    s += f"\n  MEM       : {MEM_GB} GB"

    if properties["CALCULATOR"] == "UMA":
        s += f"\n  UMA MODEL : {properties['UMA_MODEL_PATH']}"

    s += "\n"

    print(s)


if __name__ == "__main__":
    run_setup()
