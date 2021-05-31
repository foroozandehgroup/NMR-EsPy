# _cols
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Coloured terminal output"""

import importlib
import platform

END = '\033[0m'  # end editing
R = '\033[31m'  # red
G = '\033[32m'  # green
OR = '\033[33m'  # orange
B = '\033[34m'  # blue
MA = '\033[35m'  # magenta (M is reserved for no. of oscillators)
C = '\033[96m'  # cyan

USE_COLORAMA = False

# If on windows, enable ANSI colour escape sequences if colorama
# is installed
if platform.system() == 'Windows':
    colorama_spec = importlib.util.find_spec("colorama")
    if colorama_spec is not None:
        USE_COLORAMA = True
    # If colorama not installed, make color attributes empty to prevent
    # bizzare outputs
    else:
        END = ''
        R = ''
        G = ''
        OR = ''
        B = ''
        MA = ''
        C = ''
