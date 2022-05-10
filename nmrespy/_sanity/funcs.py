# funcs.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 10 May 2022 16:54:54 BST

from pathlib import Path
import re
from typing import Any, Iterable, Optional, Tuple, Union

import matplotlib as mpl
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

import numpy as np
from nmr_sims.nuclei import Nucleus, supported_nuclei
from nmr_sims.spin_system import SpinSystem


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


def check_float(
    obj: Any,
    greater_than_zero: bool = False,
    greater_than_one: bool = False,
) -> Optional[str]:
    if not isinstance(obj, float):
        return "Should be a float."
    if greater_than_zero and obj < 0.0:
        return "Should be greater than 0.0"
    if greater_than_one and obj < 1.0:
        return "Should be greater than 1.0"


def check_int(
    obj: Any,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> Optional[str]:
    if not isint(obj):
        return "Should be an int."
    if isint(min_value) and obj < min_value:
        return f"Should be greater than or equal to {min_value}."
    if isint(max_value) and obj > max_value:
        return f"Should be less than or equal to {max_value}."


def check_index(obj: Any, length: int) -> Optional[str]:
    intcheck = check_int(obj)
    if isinstance(intcheck, str):
        return intcheck
    l = length * [0]
    try:
        l[obj]
        return
    except IndexError:
        return f"Invalid index {obj} for Estimator with {length} saved results."


def check_float_list(
    obj: Any,
    length: Optional[int] = None,
    len_one_can_be_listless: bool = False,
    must_be_positive: bool = False,
    allow_none: bool = False,
) -> Optional[str]:
    if length == 1 and len_one_can_be_listless:
        if isfloat(obj):
            return
    if not isiter(obj):
        return "Should be a tuple or list."
    if (length is not None) and (len(obj) != length):
        return f"Should be of length {length}."
    if not all([isfloat(x) for x in obj]) and not allow_none:
        return "All elements should be floats."
    if not all([(isfloat(x) or x is None) for x in obj]) and allow_none:
        return "All elements should be floats or Nones."
    if (
        not all([x > 0.0 for x in filter(lambda y: y is not None, obj)]) and
        must_be_positive
    ):
        return "All elements should be positive."
    if not any([isfloat(x) for x in obj]):
        return "At least one element must be a float, all Nones not allowed."


def check_int_list(
    obj: Any,
    length: Optional[int] = None,
    len_one_can_be_listless: bool = False,
    must_be_positive: bool = False,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    allow_none: bool = False,
) -> Optional[str]:
    if (length == 1 or length is None) and len_one_can_be_listless and isint(obj):
        return
    if not isiter(obj):
        return "Should be a tuple or list."
    if (length is not None) and (len(obj) != length):
        return f"Should be of length {length}."
    if not all([isint(x) for x in obj]) and not allow_none:
        return "All elements should be int."
    if not all([(isint(x) or x is None) for x in obj]) and allow_none:
        return "All elements should be ints or Nones."
    if (
        not all([x >= 0 for x in filter(lambda y: y is not None, obj)]) and
        must_be_positive
    ):
        return "All elements should be positive."
    if (
        isint(max_value) and
        not all([x <= max_value for x in filter(lambda y: y is not None, obj)])
    ):
        return f"All elements must be less than or equal to {max_value}."
    if (
        isint(min_value) and
        not all([x >= min_value for x in filter(lambda y: y is not None, obj)])
    ):
        return f"All elements must be greater than or equal to {min_value}."
    if not any([isint(x) for x in obj]):
        return "At least one element must be an int, all Nones not allowed."


def check_str_list(obj: Any, length: Optional[int] = None) -> Optional[str]:
    if isiter(obj):
        if not all([isinstance(x, str) for x in obj]):
            return "Each element should be a str."
        if length is not None and len(obj) != length:
            return f"Should be of length {length}."

    elif not ((length == 1 or length is None) and isinstance(obj, str)):
        return "Should be an iterable of strs."


def check_positive_float(obj: Any, allow_zero: bool = False) -> Optional[str]:
    if not (isfloat(obj) and obj > 0):
        return "Should be a positive float."


def check_positive_int(obj: Any, zero_allowed: bool = False) -> Optional[str]:
    msg = "Should be a positive int."
    if not isint(obj):
        return msg
    if zero_allowed and obj < 0:
        return msg
    elif not zero_allowed and obj <= 0:
        return msg


def check_parameter_array(obj: Any, dim: int) -> Optional[str]:
    if not isinstance(obj, np.ndarray):
        return "Should be a numpy array."
    p = 2 * (dim + 1)
    if not (obj.ndim == 2 and obj.shape[1] == p):
        return f"Should be a 2-dimensional array with shape (M, {p})."


# TODO: deprecate
def check_positive_float_list(
    obj: Any,
    length: Optional[int] = None,
    allow_none: bool = False,
) -> Optional[str]:
    if not isiter(obj):
        return "Should be a tuple or list."
    if (length is not None) and (len(obj) != length):
        return f"Should be of length {length}."
    if not all([isfloat(x) for x in obj]) and not allow_none:
        return "All elements should be floats."
    if not all([(isfloat(x) or x is None) for x in obj]) and allow_none:
        return "All elements should be floats or None."
    if not all([x > 0 for x in filter(lambda y: y is not None, obj)]):
        return "All elements should be positive."
    if not any([isfloat(x) for x in obj]):
        return "At least one element must be a float, all Nones not allowed."


# TODO: deprecate
def check_positive_int_list(
    obj: Any,
    length: Optional[int] = None,
    allow_zero: bool = False
) -> Optional[str]:
    if not isiter(obj):
        return "Should be a tuple or list."
    if (length is not None) and (len(obj) != length):
        return f"Should be of length {length}."
    if not all([isint(x) for x in obj]):
        return "All elements should be ints."
    if allow_zero and (not all([x >= 0 for x in obj])):
        return "All elements should be positive or 0."
    elif (not allow_zero) and (not all([x > 0 for x in obj])):
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


def check_ndarray_list(
    obj: Any,
    dim: Optional[int] = None,
    shapes: Optional[Iterable[Iterable[Tuple[int, int]]]] = None,
) -> Optional[str]:
    if isiter(obj):
        for i, (item, shape) in enumerate(zip(obj, shapes)):
            outcome = check_ndarray(item, dim, shape)
            if isinstance(outcome, str):
                return f"Issue with element {i}: {outcome}"
        return
    check_for_array = check_ndarray(obj, dim, shapes[0])
    if isinstance(check_for_array, str):
        return check_for_array


def check_expinfo(obj: Any, dim: Optional[int] = None) -> Optional[str]:
    if not type(obj).__name__ == "ExpInfo":
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


def check_initial_guess(obj: Any, dim: int) -> Optional[str]:
    if isinstance(obj, int):
        return check_positive_int(obj)
    elif isinstance(obj, np.ndarray):
        return check_parameter_array(obj, dim)
    else:
        return "Should be an int, a NumPy array, or None."


def check_frequency_unit(obj: Any, ppm_valid: bool) -> Optional[str]:
    if obj not in ["hz", "ppm"]:
        return "Must be one of \"hz\" and \"ppm\""
    if obj == "ppm" and not ppm_valid:
        return "Cannot generate ppm values without sfo specification."


# --- Frequency regions ---
def _check_region(
    obj: Any,
    full_region: Iterable[Tuple[float, float]],
    type_: type,
) -> Optional[str]:
    if not isiter(obj):
        return "Should be a list or tuple."

    dim = len(full_region)
    # 1D signals: user has given (left, right) instead of ((left, right),)
    if dim == 1 and len(obj) == 2:
        return _check_region_dim(obj, full_region[0], type_, 0)
    elif len(obj) != dim:
        return f"Should be of length {dim}."

    for i, (axis, full) in enumerate(zip(obj, full_region)):
        result = _check_region_dim(axis, full, type_, i)
        if isinstance(result, str):
            return result

    if all([axis is None for axis in obj]):
        return "Should not have all elements as None."


def _check_region_dim(
    axis: Any,
    full: Tuple[Union[float, int], Union[float, int]],
    type_: type,
    i: int,
) -> Optional[str]:
    msg = (
        f"Issue with element {i}: should be a list or tuple of 2 {type_.__name__}s "
        "within the spectral window or None."
    )
    if axis is None:
        return
    if not isiter(axis):
        return msg
    if len(axis) != 2:
        return msg
    if not all([isinstance(x, type_) for x in axis]):
        return msg
    if not all([min(full) <= x <= max(full) for x in axis]):
        return msg


def check_region(
    obj: Any,
    sw: Iterable[float],
    offset: Iterable[float],
) -> Optional[str]:
    full_region = [
        [sw_ / 2 + off_, -sw_ / 2 + off_]
        for sw_, off_ in zip(sw, offset)
    ]
    return _check_region(obj, full_region, float)


def check_region_idx(obj: Any, pts: Iterable[int]) -> Optional[str]:
    full_region = [[0, p - 1] for p in pts]
    return _check_region(obj, full_region, int)


def _check_jres_region(
    obj: Any,
    full_region: Tuple[float, float],
    type_: type,
) -> Optional[str]:
    if isinstance(_check_region_dim(obj, full_region, type_, 0), str):
        return (
            f"Should be a list or tuple of 2 {type_.__name__}s within the F2 spectral "
            "window or None."
        )


def check_jres_region_hz(obj: Any, expinfo) -> Optional[str]:
    sw = expinfo._sw[1]
    offset = expinfo._offset[1]
    full_region = [sw / 2 + offset, -sw / 2 + offset]
    return _check_jres_region(obj, full_region, float)


def check_jres_region_ppm(obj: Any, expinfo) -> Optional[str]:
    if expinfo._sfo is None:
        return "Cannot specify region in ppm. sfo is not defined in expinfo."
    sw = expinfo._sw[1]
    offset = expinfo._offset[1]
    sfo = expinfo._sfo[1]
    full_region = [(sw / 2 + offset) / sfo, (-sw / 2 + offset) / sfo]
    return _check_jres_region(obj, full_region, float)


def check_jres_region_idx(obj: Any, pts: int) -> Optional[str]:
    full_region = [0, pts - 1]
    return _check_jres_region(obj, full_region, int)


def check_ints_less_than_n(obj: Any, n: int) -> Optional[str]:
    if not isinstance(obj, (int, list, tuple)):
        "Should be an int, list or tuple."
    if isinstance(obj, int) and not 0 <= obj < n:
        return f"Should be between 0 and {n - 1}."
    if isinstance(obj, (list, tuple)) and not all([0 <= i < n for i in obj]):
        return f"All elements should be between 0 and {n - 1}."


def check_file_format(obj: Any) -> Optional[str]:
    if obj not in ["txt", "pdf", "csv"]:
        return "Should be one of \"txt\" and \"pdf\", \"csv\""


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


def check_nmrsims_nucleus(obj: Any) -> Optional[str]:
    if isinstance(obj, Nucleus):
        return
    elif isinstance(obj, str):
        if obj in supported_nuclei:
            return
    return (
        "Should be an instance of nmr_sims.nuclei.Nucleus, or a key found in "
        "nmr_sims.nuclei.supported_nuclei."
    )


def isnuc(obj: Any) -> bool:
    return isinstance(obj, str) and bool(re.fullmatch(r"\d+[a-zA-Z]+", obj))


def check_nucleus(obj: Any) -> Optional[str]:
    if not isnuc(obj):
        return "Should be a string satisfying the regex r\"^\\d+[a-zA-Z]+$\""


def check_nucleus_list(
    obj: Any,
    length: Optional[int] = None,
    len_one_can_be_listless: bool = False,
    none_allowed: bool = False,
) -> Optional[str]:
    if length == 1 and len_one_can_be_listless:
        if isnuc(obj):
            return
    if not isiter(obj):
        return "Should be a list."
    if (length is not None) and len(obj) != length:
        return f"Should be of length {length}."
    if not all([isnuc(x) for x in obj]) and not none_allowed:
        return "Each element should be a str satifying r\"^\\d+[a-zA-Z]+$\""
    if not all([isnuc(x) or x is None for x in obj]) and none_allowed:
        return "Each element should be a str satifying r\"^\\d+[a-zA-Z]+$\" or None."
    if none_allowed:
        if not any([isnuc(x) for x in obj]):
            return "At least one element must not be None."


def check_start_time(
    obj: Any,
    dim: int,
    len_one_can_be_listless: bool = False,
) -> Optional[str]:
    valid_time = lambda x: (
        isnum(x) or
        isinstance(x, str) and bool(re.match(r"^-?\d+dt$", x))
    )
    if dim == 1 and len_one_can_be_listless and valid_time(obj):
        return
    if not isiter(obj):
        return "Should be a list or tuple"
    if len(obj) != dim:
        return f"Should be of length {dim}"
    if not all([valid_time(x) for x in obj]):
        return "At least one invalid start time specifier."


def check_sci_lims(obj: Any) -> Optional[str]:
    if not isiter(obj):
        return "Should be a list or tuple."
    if len(obj) != 2:
        return "Should be of length 2."
    if not all([isint(x) for x in obj]):
        return "Elements should be ints."
    if not obj[0] < 0:
        return "First element should be less than 0."
    if not obj[1] > 0:
        return "Second element should be greater than 0."


def check_nmrespyplot(obj: Any) -> Optional[str]:
    if type(obj).__name__ != "NmrespyPlot":
        return "Should be a `nmrespy.plot.NmrespyPlot` object."


def check_stylesheet(obj: Any) -> Optional[str]:
    if not isinstance(obj, str):
        return "Should be a str."

    # Check two possible paths.
    # First one is simply the user input:
    # This will be valid if a full path to a stylesheet has been given.
    # Second one is to check whether the user has given a name for one of
    # the provided stylesheets that ship with matplotlib.
    paths = [
        Path(obj).resolve(),
        Path(mpl.__file__).resolve().parent / f"mpl-data/stylelib/{obj}.mplstyle",
    ]

    for path in paths:
        if path.is_file():
            rc = str(
                mpl.rc_params_from_file(
                    path, fail_on_error=True, use_default_template=False
                )
            )
            # If the file exists, but no lines can be parsed, an empty
            # string is returned.
            if rc:
                return

    return (
        "Error in finding/reading the stylesheet. Check you gave a valid path or name"
        "for the stylesheet, and that the stylesheet is formatted correctly."
    )


def check_sort_by(obj: Any, dim: int) -> Optional[str]:
    if not isinstance(obj, str):
        return "Should be a str."

    valids = (
        ["a", "p", "f-1", "d-1"] +
        ([f"f{i}" for i in range(1, dim + 1)] if dim > 1 else ["f", "f1"]) +
        ([f"d{i}" for i in range(1, dim + 1)] if dim > 1 else ["d", "d1"])
    )

    if obj not in valids:
        valid_list = ", ".join(valids)
        return f"Invalid value. Should be one of: {valid_list}."


def check_fn_mode(obj: Any) -> Optional[str]:
    valids = ["QF", "QSED", "TPPI", "States", "States-TPPI", "Echo-Anitecho"]
    if obj not in valids:
        return "Should be one of " + ", ".join([f"\"{x}\"" for x in valids])
