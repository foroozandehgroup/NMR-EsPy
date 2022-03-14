# funcs.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 14 Mar 2022 15:32:11 GMT

from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

import numpy as np
from nmr_sims.nuclei import Nucleus, supported_nuclei
from nmr_sims.spin_system import SpinSystem

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


def check_list_with_ints_less_than_n(obj: Any, n: int) -> Optional[str]:
    if not isinstance(obj, (list, tuple)):
        "Should be a list or a tuple."
    if not all([0 <= i < n for i in obj]):
        return f"All elements should be be between 0 and {n-1}."


def check_frequency_unit(obj: Any) -> Optional[str]:
    if obj not in ["hz", "ppm"]:
        return "Should be one of \"hz\" and \"ppm\""


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
