# _colors.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 10:32:30 GMT

from importlib.util import find_spec
from platform import system

END = "\033[0m"  # end editing
RED = "\033[31m"  # red
GRE = "\033[32m"  # green
ORA = "\033[33m"  # orange
BLU = "\033[34m"  # blue
MAG = "\033[35m"  # magenta
CYA = "\033[96m"  # cyan

USE_COLORAMA = False

# If on windows, enable ANSI colour escape sequences if colorama
# is installed
if system() == "Windows":
    if find_spec("colorama"):
        USE_COLORAMA = True
    # If colorama not installed, make color attributes empty to prevent
    # bizzare outputs
    else:
        END = ""
        RED = ""
        GRE = ""
        ORA = ""
        BLU = ""
        MAG = ""
        CYA = ""
