# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 10 Aug 2022 14:10:43 BST

"""NMR-EsPy: Nuclear Magnetic Resonance Estimation in Python."""

import importlib
MATLAB_AVAILABLE = importlib.util.find_spec("matlab") is not None

from nmrespy.expinfo import ExpInfo
from nmrespy.estimators.onedim import Estimator1D
from nmrespy.estimators.seq_onedim import EstimatorSeq1D
from nmrespy.estimators.jres import Estimator2DJ
