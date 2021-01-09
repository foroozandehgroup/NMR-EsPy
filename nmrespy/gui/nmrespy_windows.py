# nmrespy.py
# Simon Hulse, 30/07/2020
# simon.hulse@chem.ox.ac.uk
# Jython script for communication with NMR-ESPY within TopSpin
# WINDOWS VERSION

import os
from subprocess import *

checknmrespy = Popen(['py', '-c', 'import nmrespy; import os; print(os.path.dirname(nmrespy.__file__))'], shell=True, stdout=PIPE).communicate()[0]

if checknmrespy == '':
	ERRMSG('Could not find nmrespy in your Python 3 PATH!\nSee the nmrespy documentation for help', modal=1)
	EXIT()

else:
	nmrespy_path = checknmrespy.rstrip()

# get path
info = CURDATA()

# info will be None if no active data exists. Inform user if this is the case
if info == None:
	ERRMSG("Please select a data set to run spectral estimation!", modal=1)
	EXIT()

data_path = '%s/%s/%s' %(info[3], info[0], info[1])

proc_path = '%s/%s/%s/pdata/%s' %(info[3], info[0], info[1], info[2])

f = open('%s/gui/tmp/info.txt' %nmrespy_path, 'w')

f.write('%s %s' %(data_path, proc_path))

f.close()

script = Popen(['%s/gui/apps.py' %nmrespy_path], shell=True)
