# funcs.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 04 Feb 2022 16:07:33 GMT

from typing import Any, Optional
import numpy as np
from nmrespy import ExpInfo


def check_bool(obj: Any):
    if not isinstance(obj, bool):
        return "Should be a bool."


def check_float(obj: Any):
    if not isinstance(obj, float):
        return "Should be a float."


def check_positive_float(obj: Any):
    if not isinstance(obj, float) and obj > 0:
        return "Should be a positive float."


def check_parameter_array(obj: Any, dim: int) -> Optional[str]:
    if not isinstance(obj, np.ndarray):
        return "Should be a numpy array."
    p = 2 * (dim + 1)
    if not (obj.ndim == 2 and obj.shape[1] == p):
        return f"Should be a 2-dimensional array with shape (M, {p})."


def check_expinfo(obj: Any) -> Optional[str]:
    if not isinstance(obj, ExpInfo):
        return "Should be an instance of nmrespy.ExpInfo."


def check_points(obj: Any, dim: int) -> Optional[str]:
    try:
        iter(obj)
    except TypeError:
        return "Should be iterable."
    n = 0
    for e in obj:
        if not isinstance(e, int):
            return "Every element should be an int."
        n += 1
    if n != dim:
        return f"Should have {dim} elements."


def check_modulation(obj: Any) -> Optional[str]:
    if (obj is not None) and (obj not in ("amp", "phase")):
        return "Should be on of None, \"amp\", or \"phase\"."
