# funcs.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 12 May 2023 00:33:11 BST

from pathlib import Path
import re
from typing import Any, Iterable, Optional, Tuple, Union

import matplotlib as mpl
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


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
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Optional[str]:
    if not isinstance(obj, float):
        return "Should be a float."
    if greater_than_zero and obj < 0.0:
        return "Should be greater than 0.0"
    if greater_than_one and obj < 1.0:
        return "Should be greater than 1.0"
    if isfloat(min_value) and obj < min_value:
        return f"Should be greater than or equal to {min_value}."
    if isfloat(max_value) and obj > max_value:
        return f"Should be less than or equal to {max_value}."


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


def check_list_with_elements_in(obj: Any, allowed: Iterable[Any]) -> Optional[str]:
    if not isiter(obj):
        return "Should be a list of tuple."
    if any([x not in allowed for x in obj]):
        return (
            "All elements should be one of the following values:\n" +
            ", ".join(allowed)
        )


def check_index(obj: Any, length: int) -> Optional[str]:
    intcheck = check_int(obj)
    if isinstance(intcheck, str):
        return intcheck
    l = length * [0]
    try:
        l[obj]
        return
    except IndexError:
        return f"Invalid index ({obj}) for Estimator with {length} saved results."


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


def check_int_list_list(
    obj: Any,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> Optional[str]:
    if not isiter(obj):
        return "Should be a tuple or list."
    for i, elem in enumerate(obj):
        msg = f"Issue with element {i}:\n"
        if not isiter(elem):
            return f"{msg}Each element should be a tuple or list of ints."
        if not all([isint(x) for x in elem]):
            return f"{msg}Each element should be a tuple or list of ints."
        if (isint(max_value) and not all([x <= max_value for x in elem])):
            return f"{msg}All values must be less than or equal to {max_value}."
        if (isint(min_value) and not all([x >= min_value for x in elem])):
            return f"{msg}All values must be greater than or equal to {min_value}."


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
        print(dim)
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


def check_optimiser_mode(obj: Any) -> Optional[str]:
    if not isinstance(obj, str):
        return "Should be a str."
    # check if mode is empty or contains and invalid character
    if any(c not in "apfd" for c in obj) or obj == "":
        return "Invalid character present, or string is empty."
    # check if mode contains a repeated character
    count = {}
    for c in obj:
        if c in count.keys():
            count[c] += 1
        else:
            count[c] = 1
    if not all(map(lambda x: x == 1, count.values())):
        return "Repeated character present."


def check_spinach_couplings(obj: Any, nspins: int) -> Optional[str]:
    if not isiter(obj):
        return "Should be a list or a tuple."
    for i, elem in enumerate(obj):
        if not (
            isiter(elem) and
            len(elem) == 3 and
            isinstance(elem[0], int) and
            isinstance(elem[1], int) and
            isinstance(elem[2], float)
        ):
            return (
                f"Issue with element {i}: Each element should be a tuple of 3"
                "elements of the form (int, int, float)."
            )

        if not all([1 <= x <= nspins for x in elem[:2]]):
            return (
                f"Issue with element {i}: The first two elements should be between "
                f"1 and {nspins}."
            )

        if elem[0] == elem[1]:
            return (
                f"Issue with element {i}: The first two elements should be different."
            )

    # Check for duplicate specifications
    pairs = []
    for elem in obj:
        pair = sorted(elem[:2])
        if pair in pairs:
            return (
                f"Coupling between spins {pair[0]} and {pair[1]} specified multiple "
                "times! Ensure each pair is only given once to prevent ambiguity."
            )
        pairs.append(pair)


def check_xticks(obj: Any, regions: Iterable[Tuple[float, float]]) -> Optional[str]:
    n_regions = len(regions)
    if not isiter(obj):
        return "Should be a list or tuple."
    for i, elem in enumerate(obj):
        msg = f"Issue with entry {i}:\n"
        if not (isiter(elem) and len(elem) == 2):
            return f"{msg}Each entry should be a list or tuple of length 2."
        if not (isinstance(elem[0], int) and 0 <= elem[0] < n_regions):
            return (
                f"{msg}The first element of each entry should be an int between (and "
                f"including) 0 and {n_regions - 1}."
            )
        if not (isiter(elem[1]) and all([isfloat(x) for x in elem[1]])):
            return (
                f"{msg}The second element of each entry should be a list or tuple of "
                "floats."
            )
        index = elem[0]
        ticks = elem[1]
        region = regions[index]
        if not all([region[1] <= tick <= region[0] for tick in ticks]):
            return (
                f"{msg}All ticks should lie with in the region "
                f"({region[0]} - {region[1]})"
            )


def check_split_oscs(obj: Any, dim: int, n: int) -> Optional[str]:
    if not isinstance(obj, dict):
        return "Should be a dict."
    valid_keys = ("separation", "number", "amp_ratio")
    valid_keys_str = ", ".join([f"\"{x}\"" for x in valid_keys])
    for key, value in obj.items():
        if not (isint(key) and 0 <= key <= n):
            return f"Each key should be an int between 0 and {n}."
        msg = f"Issue with element with key {key}:\n"

        if value is None:
            continue
        if not (isinstance(value, dict)):
            return f"{msg}Should be a dict or None."
        if not all([k in valid_keys for k in value.keys()]):
            return f"{msg}Only valid elements are: {valid_keys_str}."

        msg = f"{msg}Issue with \"<KEY>\":\n<RESULT>"
        if "separation" in value:
            result = check_float_list(
                value["separation"], length=dim, len_one_can_be_listless=True,
                must_be_positive=True,
            )
            if isinstance(result, str):
                return msg.replace("<KEY>", "separation").replace("<RESULT>", result)

        if "number" in value:
            result = check_int(value["number"], min_value=2)
            if isinstance(result, str):
                return msg.replace("<KEY>", "number").replace("<RESULT>", result)

        if "amp_ratio" in value:
            length = value["number"] if "number" in value else None
            result = check_float_list(
                value["amp_ratio"], length=length, must_be_positive=True
            )
            if isinstance(result, str):
                return msg.replace("<KEY>", "amp_ratio").replace("<RESULT>", result)
            # Wouldn;t be picked up in case "number" isn't given, and a
            # single-element list is given
            if len(value["amp_ratio"]) == 1:
                return msg.replace("<KEY>", "amp_ratio").replace(
                    "<RESULT>", "Should be at least 2 elements long.",
                )


def check_xaxis_ticks(
    obj: Any,
    regions: Iterable[Tuple[float, float]],
) -> Optional[str]:
    msg = (
        "Should be a list or tuple with each element being of the form "
        "`[int, [float, float, ...]]`"
    )

    if not isiter(obj):
        return msg
    n_regions = len(regions)
    already_found = []
    for i, elem in enumerate(obj):
        if not isiter(elem) or len(elem) != 2 or not isint(elem[0]):
            return msg

        msg_prefix = f"Issue with element {i}:\n"
        if elem[0] < 0 or elem[0] >= n_regions:
            return f"{msg_prefix}Region index should be between 0-{n_regions - 1}."
        if elem[0] in already_found:
            return f"Duplicated region index: {elem[0]}."
        already_found.append(elem[0])
        region = regions[elem[0]]
        if not isiter(elem[1]):
            return f"{msg_prefix}Tick specification should be a list or tuple."
        for tick in elem[1]:
            if not isfloat(tick):
                return "{msg_prefix}Tick specifications should be floats."
            mn, mx = min(region), max(region)
            if mn > tick or mx < tick:
                return (
                    f"{msg_prefix}At least one tick is out of bounds "
                    f"(should be within the range {mn}-{mx})."
                )
