# _topspin.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 23 May 2022 11:40:14 BST

# Jython script for accessing the NMR-EsPy GUI from inside TopSpin
# Should be set to the path topspinx.y.z/exp/stan/nmr/py/user/nmrespy.py
# Runs nmrespy.__main__
#
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

import os
from subprocess import Popen, PIPE

# ---PYTHON EXECUTABLE----------------------------------------------------
# exe should be set as the path to the Python executable that you use for
# nmrespy.
#
# One way to determine this that is general for all OSes is to start an
# interactive Python session from a terminal/command prompt and then enter
# the following:
#
# 	>>> import sys
#   >>> exe = sys.executable.replace('\\', '\\\\')
#   >>> print(f"\"{exe}\"")
#
# Set py_exe as exactly what the output of this is
py_exe = None
# ------------------------------------------------------------------------

# ----PDFLATEX EXECUTABLE-----------------------------------------------
# If you have LaTeX installed and would like to produce PDFs of results,
# the variable `pdflatex` should be set as the full path the pdflatex
# executable.
#
# Find this by entering the following into a Python interpreter:
# Windows:
#    >>> from subprocess import check_output
#    >>> output = check_output("where pdflatex", shell=True)
#    >>> exe = str(output, 'utf-8').rstrip().replace("\\", "\\\\")
#    >>> print(f"\"{exe}\"")
#    "C:\\texlive\\2020\\bin\\win32\\pdflatex.exe"
#
# UNIX:
#    >>> from subprocess import check_output
#    >>> output = check_output("which pdflatex", shell=True)
#    >>> exe = str(output, 'utf-8').rstrip()
#    >>> print(f"\"{exe}\"")
#    "/usr/bin/pdflatex"
#
# Set pdflatex_exe as exactly what the output of this is
pdflatex_exe = "None"
# ------------------------------------------------------------------------

if py_exe is None:
    ERRMSG(
        "The Python 3 binary has not been specified. See the NMR-EsPy GUI "
        "documentation for help.",
        modal=1,
    )
    EXIT()

# Check whether nmrespy exists by importing
# If it exists, $? = 0
# If it does not exist, $? = int > 0
checknmrespy = Popen([py_exe, "-c", '"import nmrespy"'], stdout=PIPE)
checknmrespy.communicate()[0]
if checknmrespy.returncode != 0:
    ERRMSG("Could not find NMR-EsPy in your Python 3 path!", modal=1)
    EXIT()

# get path
curdata = CURDATA()

# curdata will be None if no active data exists.
# Inform user if this is the case.
if curdata is None:
    ERRMSG("Please select a data set to run nmrespy!", modal=1)
    EXIT()

# Full path to the pdata directory
path = os.path.join(curdata[3], curdata[0], curdata[1], "pdata", curdata[2])

Popen(
    [
        py_exe,
        "-m",
        "nmrespy",
        "--estimate",
        path,
        "--topspin",
        "--pdflatex",
        pdflatex_exe,
    ]
)
