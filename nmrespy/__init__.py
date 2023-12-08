# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 08 Dec 2023 03:09:03 PM EST

"""NMR-EsPy: Nuclear Magnetic Resonance Estimation in Python."""

from importlib.metadata import version
from importlib.util import find_spec
import importlib
from pathlib import Path

__version__ = version(__package__)

MATLAB_AVAILABLE = find_spec("matlab") is not None
directory = Path(__file__).parent.resolve()
TOPSPINPATHS = [
    directory / "app/topspin_scripts" / x
    for x in (directory / "app/topspin_scripts").iterdir() if x.is_file()
]

from nmrespy.expinfo import ExpInfo
from nmrespy.estimators import Estimator
from nmrespy.estimators.onedim import Estimator1D
from nmrespy.estimators.jres import Estimator2DJ
