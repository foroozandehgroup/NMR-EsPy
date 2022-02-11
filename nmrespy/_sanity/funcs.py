# funcs.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 11 Feb 2022 14:56:00 GMT

from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple
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


def check_positive_int(obj: Any):
    if not isinstance(obj, int) and obj > 0:
        return "Should be a positive int."


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
        return "Should be one of None, \"amp\", or \"phase\"."


def check_one_of(obj: Any, *args) -> Optional[str]:
    print(args)
    if obj not in args:
        return (
            "Should be one of the following:\n" +
            ", ".join([f"\"{x}\"" for x in args])
        )


def check_path(obj: Any) -> Optional[str]:
    if isinstance(obj, Path):
        directory = obj.resolve().parent
    elif isinstance(obj, str):
        directory = Path(obj).resolve().parent
    else:
        return "Should be a pathlib.Path object or a string specifying a path."
    if not directory.is_dir():
        return f"The parent directory {directory} doesn't exist."


def check_is_path(obj: Any) -> Optional[str]:
    check_path(obj)
    if not Path(obj).resolve().is_file():
        return f"The file {obj} does not exist."


def check_initial_guess(obj: Any, dim: int) -> Optional[str]:
    if isinstance(obj, int):
        return check_positive_int(obj)
    elif isinstance(obj, np.ndarray):
        return check_parameter_array(obj, dim)
    else:
        return "Should be an int, a NumPy array, or None."


def check_region_float(obj: Any, full_region: Iterable[Tuple[float, float]]):
    dim = len(full_region)
    if not isinstance(obj, (list, tuple)):
        return "Should be a list or tuple."
    if len(obj) != dim:
        return f"Should be of length {dim}."

    msg = "Each element should be a list or tuple of 2 floats."
    for axis, full in zip(obj, full_region):
        if not isinstance(axis, (list, tuple)):
            return msg
        if len(axis) != 2:
            return msg
        if not all([isinstance(x, float) for x in axis]):
            return msg
        if not all([full[0] >= x >= full[1] for x in axis]):
            return "At least one specified value lies outside the spectral window."


def check_region_hz(obj: Any, expinfo: ExpInfo) -> Optional[str]:
    full_region = [[sw / 2 + off, -sw / 2 + off]
                   for sw, off in zip(expinfo._sw, expinfo._offset)]
    return check_region_float(obj, full_region)


def check_region_ppm(obj: Any, expinfo: ExpInfo) -> Optional[str]:
    full_region = [[(sw / 2 + off) / sfo, (-sw / 2 + off) / sfo]
                   for sw, off, sfo in zip(expinfo._sw, expinfo._offset, expinfo._sfo)]
    return check_region_float(obj, full_region)
