# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 11 May 2022 15:17:23 BST

"""NMR-EsPy: Nuclear Magnetic Resonance Estimation in Python."""

import nmrespy._version as nv
__version__ = nv.__version__
from nmrespy.expinfo import ExpInfo
from nmrespy.estimators.onedim import Estimator1D
from nmrespy.estimators.jres import Estimator2DJ
