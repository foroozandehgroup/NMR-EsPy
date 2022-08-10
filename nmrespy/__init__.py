# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 04 Aug 2022 01:18:38 BST

"""NMR-EsPy: Nuclear Magnetic Resonance Estimation in Python."""

import importlib
MATLAB_AVAILABLE = importlib.util.find_spec("matlab") is not None

from nmrespy.expinfo import ExpInfo
from nmrespy.estimators.onedim import Estimator1D
from nmrespy.estimators.jres import Estimator2DJ
