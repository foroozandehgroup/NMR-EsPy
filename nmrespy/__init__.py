import itertools
import os
from pathlib import Path

import nmrespy._cols as cols

from ._version import __version__


NMRESPYPATH = os.path.dirname(__file__)
MFLOGOPATH = os.path.join(NMRESPYPATH, 'images/mf_logo.png')
NMRESPYLOGOPATH = os.path.join(NMRESPYPATH, 'images/nmrespy_full.png')
GITHUBLOGOPATH = os.path.join(NMRESPYPATH, 'images/github.png')
EMAILICONPATH = os.path.join(NMRESPYPATH, 'images/email_icon.png')

GITHUBPATH = 'https://github.com/foroozandehgroup/NMR-EsPy'
MFGROUPPATH = 'http://foroozandeh.chem.ox.ac.uk/home'
MAILTOPATH = 'mailto:simon.hulse@chem.ox.ac.uk'
