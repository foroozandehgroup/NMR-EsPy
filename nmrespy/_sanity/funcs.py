# funcs.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 18 Mar 2022 14:11:02 GMT

from pathlib import Path
import re
from typing import Any, Iterable, Optional, Tuple

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

import numpy as np
from nmr_sims.nuclei import Nucleus, supported_nuclei
from nmr_sims.spin_system import SpinSystem

from nmrespy import ExpInfo


def isiter(x: Any) -> bool:
    return isinstance(x, (tuple, list))


def isnum(x: Any) -> bool:
    return isinstance(x, (int, float))


def isfloat(x: Any) -> bool:
    return isinstance(x, float)


def isint(x: Any) -> bool:
    return isinstance(x, int)


def check_bool(obj: Any):
    if not isinstance(obj, bool):
        return "Should be a bool."


def check_float(obj: Any):
    if not isinstance(obj, float):
        return "Should be a float."


def check_float_greater_that_one(obj: Any) -> Optional[str]:
    if not isinstance(obj, float) or obj < 1.0:
        return "Should be a float greater than 1."


def check_float_list(obj: Any, dim: Optional[int] = None) -> Optional[str]:
    if not isiter(obj):
        return "Should be a list or tuple."
    if not all([isfloat(x for x in obj)]):
        return "All elements should be floats."


def check_int_list(obj: Any, dim: Optional[int] = None) -> Optional[str]:
    if not isiter(obj):
        return "Should be a list or tuple."
    if not all([isint(x for x in obj)]):
        return "All elements should be ints."


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


def check_positive_float_list(obj: Any, length: Optional[int] = None) -> Optional[str]:
    if not isinstance(obj, (tuple, list)):
        return f"Should be a tuple or list of length {length}."
    if (length is not None) and (len(obj) != length):
        return f"Should be of length {length}."
    if not all([isinstance(x, float) for x in obj]):
        return "All elements should be floats."
    if not all([x > 0 for x in obj]):
        return "All elements should be positive."


def check_ndarray(
    obj: Any,
    dim: Optional[int] = None,
    shape: Optional[Iterable[Tuple[int, int]]] = None,
) -> Optional[str]:
    if not isinstance(obj, np.ndarray):
        return "Should be a numpy array."
    if dim is not None and obj.ndim != dim:
        return f"Should be a {dim}-dimensional array."
    if shape is not None:
        for (axis, size) in shape:
            if obj.shape[axis] != size:
                return f"Axis {axis} should be of size {size}."


def check_expinfo(obj: Any, dim: Optional[int] = None) -> Optional[str]:
    if not isinstance(obj, ExpInfo):
        return "Should be an instance of nmrespy.ExpInfo."
    if isinstance(dim, int):
        if not obj.unpack("dim") == dim:
            return "Should be {dim}-dimensional."


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
    if obj not in args:
        return (
            "Should be one of the following:\n" +
            ", ".join([f"\"{x}\"" for x in args])
        )


def check_str(obj: Any) -> Optional[str]:
    if not isinstance(obj, str):
        return "Should be a str."


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

    msg = "Each element should be a list or tuple of 2 floats or None."
    for axis, full in zip(obj, full_region):
        if axis is None:
            continue
        if not isinstance(axis, (list, tuple)):
            return msg
        if len(axis) != 2:
            return msg
        if not all([isinstance(x, float) for x in axis]):
            return msg
        if not all([full[0] >= x >= full[1] for x in axis]):
            return "At least one specified value lies outside the spectral window."

    if all([axis is None for axis in obj]):
        return "Should not have all elements as None."


def check_region_hz(obj: Any, expinfo: ExpInfo) -> Optional[str]:
    full_region = [[sw / 2 + off, -sw / 2 + off]
                   for sw, off in zip(expinfo._sw, expinfo._offset)]
    return check_region_float(obj, full_region)


def check_region_ppm(obj: Any, expinfo: ExpInfo) -> Optional[str]:
    if expinfo._sfo is None:
        return "Cannot specify region in ppm. sfo is not defined in expinfo."
    full_region = [[(sw / 2 + off) / sfo, (-sw / 2 + off) / sfo]
                   for sw, off, sfo in zip(expinfo._sw, expinfo._offset, expinfo._sfo)]
    return check_region_float(obj, full_region)


def check_region_idx(obj: Any, pts: Iterable[int]) -> Optional[str]:
    dim = len(pts)
    if not isinstance(obj, (list, tuple)):
        return "Should be a list or tuple"
    if len(obj) != dim:
        return f"Should be of length {dim}."

    msg = "Each element should be a list or tuple of 2 ints."
    for axis, p in zip(obj, pts):
        if not isinstance(axis, (list, tuple)):
            return msg
        if len(axis) != 2:
            return msg
        if not all([isinstance(x, int) for x in axis]):
            return msg
        if not all([0 <= x < p for x in axis]):
            return "At least one specified value lies outside the spectral window."


def check_jres_region_float(
    obj: Any, full_region: Tuple[float, float]
) -> Optional[str]:
    if not isinstance(obj, (list, tuple)):
        return "Should be a list or tuple."
    if len(obj) != 2:
        return "Should be of length 2."
    if not all([isinstance(x, float) for x in obj]):
        return "Each element should be a float."
    if not all([full_region[0] >= x >= full_region[1] for x in obj]):
        return "At least one specified value lies outside the spectral window."


def check_jres_region_hz(obj: Any, expinfo: ExpInfo) -> Optional[str]:
    sw = expinfo._sw[1]
    offset = expinfo._offset[1]
    full_region = [sw / 2 + offset, -sw / 2 + offset]
    return check_jres_region_float(obj, full_region)


def check_jres_region_ppm(obj: Any, expinfo: ExpInfo) -> Optional[str]:
    sw = expinfo._sw[1]
    offset = expinfo._offset[1]
    if expinfo._sfo is None:
        return "Cannot specify region in ppm. sfo is not defined in expinfo."
    sfo = expinfo._sfo[1]
    full_region = [(sw / 2 + offset) / sfo, (-sw / 2 + offset) / sfo]
    return check_jres_region_float(obj, full_region)


def check_ints_less_than_n(obj: Any, n: int) -> Optional[str]:
    if not isinstance(obj, (int, list, tuple)):
        "Should be an int, list or tuple."
    if isinstance(obj, int) and not 0 <= obj < n:
        return f"Should be between 0 and {n - 1}."
    if isinstance(obj, (list, tuple)) and not all([0 <= i < n for i in obj]):
        return f"All elements should be between 0 and {n - 1}."


def check_frequency_unit(obj: Any, ppm_valid: Optional[bool] = True) -> Optional[str]:
    if obj not in ["hz", "ppm"]:
        return "Should be one of \"hz\" and \"ppm\""
    if obj == "ppm" and not ppm_valid:
        return "Cannot process ppm values without sfo specification."


def check_file_format(obj: Any) -> Optional[str]:
    if obj not in ["txt", "pdf", "csv"]:
        return "Should be one of \"txt\" and \"pdf\", \"csv\""


def check_sci_lims(obj: Any) -> Optional[str]:
    if not isinstance(obj, (tuple, list)) or len(obj) != 2:
        return "Should be a tuple/list of length 2."
    if not isinstance(obj[0], int) or obj[0] > 0:
        return "First element should be a negative int."
    if not isinstance(obj[1], int) or obj[1] < 0:
        return "Second element should be a positive int."


def is_mpl_color(obj: Any) -> bool:
    try:
        mcolors.to_hex(obj)
        return True
    except ValueError:
        return False


def check_mpl_color(obj: Any) -> Optional[str]:
    if not is_mpl_color(obj):
        return "Invalid color specification"


def check_oscillator_colors(obj: Any) -> Optional[str]:
    if is_mpl_color(obj):
        return
    if obj in plt.colormaps():
        return
    if isinstance(obj, (list, np.ndarray)):
        for e in obj:
            if not is_mpl_color(e):
                return f"The following is not a valid color: {e}"
        return
    return "Not a valid color, list of colors, or colormap."


def check_spin_system(obj: Any) -> Optional[str]:
    if not isinstance(obj, SpinSystem):
        return "Should be an instance of nmr_sims.spin_system.SpinSystem."


def check_nucleus(obj: Any) -> Optional[str]:
    if isinstance(obj, Nucleus):
        return
    elif isinstance(obj, str):
        if obj in supported_nuclei:
            return
    return (
        "Should be an instance of nmr_sims.nuclei.Nucleus, or a key found in "
        "nmr_sims.nuclei.supported_nuclei."
    )


def check_convertible_list(obj: Any, dim: int) -> Optional[str]:
    isnum = lambda x: isinstance(x, (int, float))
    if not isiter(obj):
        return "Should be a tuple or list."
    if len(obj) != dim:
        return f"Should be of length {dim}."

    msg = (
        "Each element should be one of:\n"
        "A numerical type\n"
        "A tuple or list of numerical types\n"
        "None"
    )

    for elem in obj:
        if isnum(elem) or elem is None:
            pass
        elif isiter(elem):
            if not all([isnum(x) for x in elem]):
                return msg
        else:
            return msg


def check_frequency_conversion(obj: Any, ppm_valid: bool) -> Optional[str]:
    pattern = r"^(idx|ppm|hz)->(idx|ppm|hz)$"
    if not bool(re.match(pattern, obj)):
        return (
            "Should be a str of the form \"{from}->{to}\", "
            "where {from} and {to} are each one of \"idx\", \"hz\", or \"ppm\""
        )
    if (not ppm_valid) and ("ppm" in obj):
        return "Cannot convert to/from ppm when sfo has not been specified."


def check_start_time(obj: Any, dim: int) -> Optional[str]:
    if not isiter(obj):
        return "Should be a list or tuple"
    if not all([(isnum(x) or bool(re.match(r"^-?\d+dt$", x))) for x in obj]):
        return "At least one invalid start time specifier."
