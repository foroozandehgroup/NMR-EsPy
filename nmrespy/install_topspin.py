#!/usr/bin/python3
# install script for nmrespy

import getpass
import os
import sys
import time
import platform
import shutil

import nmrespy as espy
from nmrespy._colors import *
if USE_COLORAMA:
    import colorama

def get_path(first=True):
    """Ask user for non-default path to TopSpin installation"""

    if first:
        msg = f'{O}No TopSpin installation was found in the default' \
              + f' installation location. If you have TopSpin installed in' \
              + f' a non-default location, enter its full path here.' \
              + f' To quit, enter [q]: {END}'

    else:
        msg = f'{O}The path you input does not exist. Please try again,' \
              + f' or enter [q] to quit: {END}'

    custom_path = input(msg)

    if custom_path == 'q' or custom_path == 'Q':
        print(f'{R}Cancelling NMR-EsPy TopSpin installation{END}')
        exit()

    if os.path.exists(custom_path):
        dest = custom_path

    else:
        return get_path(first=False)


def choose_path(paths, first=True):
    """Ask user to choose from more than one path option"""

    if first:
        msg = f'{O}More than one TopSpin installation was detected. Choose' \
              + f' the installation to install the NMR-EsPy app to by' \
              + f' entering the number next to its name. To quit, enter [q]:\n'
        for i, path in enumerate(paths, 1):
            msg += f'{path}  [{i}]\n'
        msg += END

    else:
        msg = f'{O}Invalid input. Please enter an integer from 1 to' \
              + f' {len(paths)}, or enter [q] to quit: {END}'

    choice = input(msg)

    if choice == 'q' or choice == 'Q':
        print(f'{R}Cancelling NMR-EsPy TopSpin installation{END}')
        exit()

    try:
        idx = int(choice) - 1
        return paths[idx]
    except:
        return choose_path(paths, first=False)


if __name__ == "__main__":

    # operating system ('Linux', 'Windows' accepted atm)
    opsys = platform.system()

    topspin_paths = []

    # Linux systems (possibly MacOS as well?)
    # default TopSpin installation in /opt/topspin<x.y.z>/
    if opsys == 'Linux':
        file = 'nmrespy_lin.py'
        path_list = os.listdir('/opt/')
        for entry in path_list:
            if entry[:7] == 'topspin':
                topspin_paths.append(os.path.join('/opt', entry))

    # Windows systems - default TopSpin path is C:/Bruker/TopSpin<x.y.z>
    elif opsys == 'Windows':
        file = 'nmrespy_win.py'
        path_list = os.listdir(r"C:/")
        for entry in path_list:
            if entry[:6] == 'Bruker':
                bruker_list = os.listdir(r"C:/Bruker")
                for subentry in bruker_list:
                    if subentry[:7] == 'TopSpin':
                        topspin_paths.append(os.path.join('C:/Bruker', subentry))

    # If no TopSpin installation found in default place, ask user if they
    # have one in a non-default location.
    if not topspin_paths:
        path = get_path()

    # one TopSpin installation found in default location
    elif len(topspin_paths) == 1:
        path = topspin_paths[0]

    elif len(topspin_paths) > 1:
        path = choose_path(topspin_paths)

    # destination path
    dir = os.path.join(path, 'exp/stan/nmr/py/user')

    if os.path.exists(dir):
        src = os.path.join(os.path.dirname(espy.__file__), f'topspin/{file}')
        shutil.copyfile(src, os.path.join(dir, 'nmrespy.py'))

    else:
        msg = f'{R}The expcted file directory: {dir} doesn\'t exist!{END}'
        raise IOError(msg)
