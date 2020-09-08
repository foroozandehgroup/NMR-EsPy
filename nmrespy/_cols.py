#!/usr/bin/python3

# nmrespy.cols
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Coloured terminal output

import platform

END = '\033[0m'
R = '\033[31m' # red
G = '\033[32m' # green
O = '\033[33m' # orange
B = '\033[34m' # blue
MA = '\033[35m' # magenta (M is reserved for no. of oscillators)
C = '\033[96m' # cyan

USE_COLORAMA = False

# If on windows, enable ANSI colour escape sequences if colorama
# is installed
if platform.system() == 'Windows':
    try:
        import colorama
        USE_COLORAMA = True

    # if colorama not installed, make color attributes empty to prevent
    # bizzare outputs
    except ModuleNotFoundError:
        END = ''
        R = ''
        G = ''
        O = ''
        B = ''
        MA = ''
        C = ''

# “Don't be so humble - you are not that great.”
# —————————————————————————————————Golda Meir———
