# coding=utf-8
'''
FIRECODE: Filtering Refiner and Embedder for Conformationally Dense Ensembles
Copyright (C) 2021-2024 Nicolò Tampellini

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

'''

def run_setup():
    '''
    Invoked by the command
    > python -m firecode -s (--setup)

    Guides the user in setting up the calculation options
    contained in the settings.py file.
    '''

    import os
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    from firecode.settings import DEFAULT_LEVELS, DEFAULT_FF_LEVELS, COMMANDS

    properties = {
        'FF_OPT_BOOL':False,
        'FF_CALC':None,
        'NEW_FF_DEFAULT':None,
        'CALCULATOR':None,
        'NEW_DEFAULT':None,
        'NEW_COMMAND':None,
        'PROCS':1,
        'MEM_GB':1,
    }

    def ask(text, accepted=('y','n'), default='n'):
        text = '--> ' + text
        if accepted is None:
            answer = input(text)
            print()
            return answer
        answer = None
        while answer not in accepted:
            answer = input(text)
            answer = answer if answer != '' else default
        print()
        return answer

    tag_dict = {
        'mop':'MOPAC',
        'orca':'ORCA',
        'gau':'GAUSSIAN',
        'xtb':'XTB',
    }

    print('\FIRECODE setup:\n')

    #########################################################################################

    answer = ask('What Force Field calculator would you like to use?\n- XTB -> xtb\n- Gaussian -> gau\n'
                 '- None -> none\n\nAnswer [xtb]/gau/none: ', accepted=('xtb', 'gau', 'none'), default='xtb')

    if answer == 'xtb':
        properties['FF_OPT_BOOL'] = True
        properties['FF_CALC'] = 'XTB'

    elif answer != 'none':
        properties['FF_OPT_BOOL'] = True
        properties['FF_CALC'] = tag_dict[answer]

        answer = ask((f'The default level for {properties["FF_CALC"]} force field calculations is \'{DEFAULT_FF_LEVELS[properties["FF_CALC"]]}\'. ' +
                       'If you would like to change it, type it here, otherwise press enter : '), accepted=None)
        if answer != '':
            properties['NEW_FF_DEFAULT'] = answer

    #########################################################################################

    answer = ask('What main calculator would you like to use?\n- MOPAC -> mop\n- ORCA -> orca\n- Gaussian -> gau\n- XTB -> xtb\n\nAnswer mop/orca/gau/[xtb]: ',
                accepted=('mop','orca','gau','xtb'), default='xtb')

    properties['CALCULATOR'] = tag_dict[answer]

    #########################################################################################

    answer = ask((f'The default level for {properties["CALCULATOR"]} calculations is \'{DEFAULT_LEVELS[properties["CALCULATOR"]]}\'. ' +
                'If you would like to change it, type it here, otherwise press enter : '), accepted=None)
    if answer != '':
        properties['NEW_DEFAULT'] = answer

    #########################################################################################

    if properties['CALCULATOR'] != 'XTB':
        answer = ask((f'Current command to call {properties["CALCULATOR"]} is {COMMANDS[properties["CALCULATOR"]]}. ' +
                    'If you would like to change it, type it here, otherwise press enter : '), accepted=None)
        if answer != '':
            properties['NEW_COMMAND'] = answer

    #########################################################################################
    
    properties['PROCS'] = ask(f'How many cores should {properties['CALCULATOR']} jobs run on? [4] : ',
                                accepted=[str(n) for n in range(1,100)], default=4)

    #########################################################################################

    # if properties['CALCULATOR'] in ('GAUSSIAN', 'ORCA'):
    properties['MEM_GB'] = int(ask('How much memory per core should a GAUSSIAN/ORCA job have, in GBs? [4] : ',
                                accepted=[str(n) for n in range(1,1000)], default='4'))

    #########################################################################################

    rank = {
        'MOPAC':1,
        'ORCA':2,
        'GAUSSIAN':3,
        'XTB':4,
    }

    q = "\'"

    with open('settings.py', 'r') as f:
        lines = f.readlines()

    old_lines = lines.copy()

    for _l, line in enumerate(old_lines):

        if 'FF_OPT_BOOL =' in line:
            lines[_l] = 'FF_OPT_BOOL = ' + str(properties['FF_OPT_BOOL']) + '\n'
            FF_OPT_BOOL = properties['FF_OPT_BOOL']

        if 'FF_CALC =' in line:
            lines[_l] = 'FF_CALC = ' + q + str(properties['FF_CALC']) + q + '\n'
            FF_CALC = properties['FF_CALC']

        elif 'CALCULATOR =' in line:
            lines[_l] = 'CALCULATOR = ' + q + properties['CALCULATOR'] + q + '\n'
            CALCULATOR = properties['CALCULATOR']

        elif 'DEFAULT_LEVELS = {' in line:
            if properties['NEW_DEFAULT'] is not None:
                lines[_l+rank[properties['CALCULATOR']]] = ' '*4 + q + properties['CALCULATOR'] + q + ':' + q + properties['NEW_DEFAULT'] + q + ',\n'
                DEFAULT_LEVELS[CALCULATOR] = properties['NEW_DEFAULT']

        elif 'DEFAULT_FF_LEVELS = {' in line:
            if properties['NEW_FF_DEFAULT'] is not None:
                lines[_l+rank[properties['FF_CALC']]] = ' '*4 + q + properties['FF_CALC'] + q + ':' + q + properties['NEW_FF_DEFAULT'] + q + ',\n'
                DEFAULT_FF_LEVELS[FF_CALC] = properties['NEW_FF_DEFAULT']

        elif 'COMMANDS = {' in line:
            if properties['NEW_COMMAND'] is not None:
                lines[_l+rank[properties['CALCULATOR']]] = ' '*4 + q + properties['CALCULATOR'] + q + ':' + q + properties['NEW_COMMAND'] + q + ',\n'

        elif 'PROCS =' in line:
            lines[_l] = 'PROCS = ' + str(properties['PROCS']) + '\n'
            PROCS = properties['PROCS']

        elif 'MEM_GB =' in line:
            lines[_l] = 'MEM_GB = ' + str(properties['MEM_GB']) + '\n'
            MEM_GB = properties['MEM_GB']

    with open('settings.py', 'w') as f:
        f.write(''.join(lines))

    print('\nfirecode setup performed correctly.')

    ff = f'{FF_CALC}/{DEFAULT_FF_LEVELS[FF_CALC]}' if FF_OPT_BOOL else 'Turned off'
    opt = f'{CALCULATOR}/{DEFAULT_LEVELS[CALCULATOR]}'
    s = f'  FF      : {ff}\n  OPT     : {opt}\n  PROCS   : {PROCS}'
    s += f'\n  MEM     : {MEM_GB} GB'

    print(s)

if __name__ == '__main__':
    run_setup()