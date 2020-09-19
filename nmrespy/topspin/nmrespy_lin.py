# nmrespy.py
# simon.hulse@chem.ox.ac.uk
# Jython script for Spectral Estimation of NMR data within TopSpin
# Calls an external Python3 script for the estimation and GUI

# Linux version

import os
from subprocess import *

# check whether nmrest exists by importing
# if it exists, $? = 0
# if it does not exist, $? = int > 0
checknmrespy = Popen(["python3 -c 'import nmrespy' ; echo $?"], shell=True, stdout=PIPE).communicate()[0]

if int(checknmrespy):
	ERRMSG('Could not find NMR-ESPY in your Python 3 PATH!', modal=1)
	EXIT()

else:
  nmrespy_path = Popen(["python3 -c 'import nmrespy; import os; print(os.path.dirname(nmrespy.__file__))'"], shell=True, stdout=PIPE).communicate()[0]
  nmrespy_path = nmrespy_path.rstrip()
  print nmrespy_path

# get path
info = CURDATA()

# info will be None if no active data exists. Inform user if this is the case
if info == None:
	ERRMSG("Please select a data set to run spectral estimation!", modal=1)
	EXIT()

data_path = '%s/%s/%s' %(info[3], info[0], info[1])

proc_path = '%s/%s/%s/pdata/%s' %(info[3], info[0], info[1], info[2])

f = open('%s/topspin/tmp/info.txt' %nmrespy_path, 'w')

f.write('%s %s' %(data_path, proc_path))

f.close()

script = Popen(['%s/topspin/topspin.py' %nmrespy_path.rstrip()], shell=True)
