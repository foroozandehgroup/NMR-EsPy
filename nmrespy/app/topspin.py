# Jython script for accessing the NMR-EsPy GUI from inside TopSpin
# Should be set to the path topspinx.y.z/exp/stan/nmr/py/user/nmrespy.py
# Runs nmrespy.app.__main__
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

import os
import platform
from subprocess import *

# ------------------------------------------------------------------------
# exe should be set as the path to the Python binary that you use for
# nmrespy.
#
# If you want to use the default Python 3 executable:
# --> On UNIX, it is likely you should set:
# 	  exe = "python3"
# --> On Windows, it is likely you should set:
#     exe = "py -3"
#
# If you want to use a non-default Python3 binary, set the full path to
# the binary.
exe = None
# ------------------------------------------------------------------------

if exe is None:
	ERRMSG(
		'The Python 3 binary has not been specified. See the NMR-EsPy GUI '
		'documentation for help.',
		modal=1,
	)
	EXIT()

# Check whether nmrespy exists by importing
# If it exists, $? = 0
# If it does not exist, $? = int > 0
checknmrespy = Popen(
	exe.split() + ["-c", "import nmrespy"],
	shell=True,
	stdout=PIPE,
)
checknmrespy.communicate()[0]

if checknmrespy.returncode != 0:
	ERRMSG('Could not find NMR-EsPy in your Python 3 path!', modal=1)
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
	exe.split() + ["-m", "nmrespy", "--estimate", path, "--topspin"],
	shell=True,
)
