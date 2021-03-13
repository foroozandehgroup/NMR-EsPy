# topspin.py
# (renamed to .../topspinx.y.z/exp/stan/nmr/py/user/nmrespy.py when installed)
# simon.hulse@chem.ox.ac.uk
# Jython script for accessing the NMR-EsPy GUI from inside TopSpin
# Calls nmrespy.app.__main__.py

import os
import platform
from subprocess import *

# ------------------------------------------------------------------------
# PATH TO EXECUTABLE
# If you would like to use the default python 3 path (python3 on Linux/py -3
# on Windows), leave this as None.
# Otherwise, set this as the full path to the desired executable binary.
# exe = None
exe = "/home/simon/Documents/DPhil/projects/spectral_estimation/NMR-EsPy/nmrespy-venv/bin/python3.9"
# ------------------------------------------------------------------------

# If `exe` is `None` specify the default python3 command for the OS
if exe is None:
	if platform.system() == 'Windows':
		exe = 'py -3'
	else:
		exe = 'python3'

# Check whether nmrespy exists by importing
# If it exists, $? = 0
# If it does not exist, $? = int > 0
checknmrespy = Popen(
	["%s -c 'import nmrespy' ; echo $?" %exe],
	shell=True,
	stdout=PIPE,
).communicate()[0]

if int(checknmrespy):
	ERRMSG('Could not find nmrespy in your Python 3 PATH!', modal=1)
	EXIT()

# get path
curdata = CURDATA()

# info will be None if no active data exists. Inform user if this is the case
if curdata == None:
	ERRMSG("Please select a data set to run nmrespy!", modal=1)
	EXIT()

# Full path to the pdata directory
path = os.path.join(curdata[3], curdata[0], curdata[1], 'pdata', curdata[2])

Popen(
	["%s -m nmrespy --estimate %s --topspin" %(exe, path)],
	shell=True,
)
