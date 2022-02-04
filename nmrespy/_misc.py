# _misc.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 04 Feb 2022 14:03:37 GMT

"""Various miscellaneous functions/classes for internal nmrespy use."""

from collections.abc import Callable
import functools
from pathlib import Path
import re
from typing import Any, Iterable, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np

from nmrespy import RED, ORA, GRE, END, USE_COLORAMA, ExpInfo

if USE_COLORAMA:
    import colorama

    colorama.init()


class ArgumentChecker:
    """Checks that user-given arguments are of an appropriate type.

    Parameters
    ----------
    components: list of 3-tuples
        Each tuple should contain the following elements:

            * The object to check.
            * A string to identify the object in any error messages`
            * A string specifying whaty type the object should be. Valid
              options are:

              + `'ndarray'`
              + `'parameter'`
              + `'int_iter'`
              + `'float_iter'`
              + `'str_iter'`
              + `'array_iter'`
              + `'region_int'`
              + `'region_float'`
              + `'bool'`
              + `'int'`
              + `'float'`
              + `'str'`
              + `'list'`
              + `'positive_int'`
              + `'positive_int_or_zero'`
              + `'positive_float'`
              + `'optimiser_mode'`
              + `'optimiser_algorithm'`
              + `'zero_to_one'`
              + `'greater_than_one'`
              + `'negative_amplidue'`
              + `'file_fmt'`
              + `'pos_neg_tuple'`
              + `'mpl_color'`
              + `'osc_cols'`
              + `'displacement'`
              + `'start_time'`

    dim : 1, 2 or None, default: None
        Dimension of the data. Only needs to be specified as `1` or `2`
        if one or more of the arguments to check have a structure that
        depends on the data dimension.
    """

    def __init__(self, dim: int = None) -> None:
        """Create an ArgumentChecker instance."""
        self.dim = dim

    def check(self) -> None:
        """Check that each object satisfies its criterion."""
        assert self.components

        for (obj, name, typ, none_allowed) in self.components:
            if none_allowed and obj is None:
                test = True
            elif typ == "ndarray":
                test = isinstance(obj, np.ndarray)
            elif typ == "parameter":
                test = self.check_parameter_array(obj)
            elif typ == "int_iter":
                test = self.check_iter(obj, int)
            elif typ == "float_iter":
                test = self.check_iter(obj, float)
            elif typ == "str_iter":
                test = self.check_iter(obj, str)
            elif typ == "array_iter":
                test = self.check_iter(obj, np.ndarray)
            elif typ == "region_int":
                test = self.check_region(obj, int)
            elif typ == "region_float":
                test = self.check_region(obj, float)
            elif typ == "bool":
                test = isinstance(obj, bool)
            elif typ == "int":
                test = isinstance(obj, int)
            elif typ == "float":
                test = isinstance(obj, float)
            elif typ == "str":
                test = isinstance(obj, str)
            elif typ == "list":
                test = isinstance(obj, list)
            elif typ == "positive_int":
                test = isinstance(obj, int) and obj > 0
            elif typ == "positive_int_or_zero":
                test = isinstance(obj, int) and obj >= 0
            elif typ == "positive_float":
                test = isinstance(obj, float) and obj > 0
            elif typ == "optimiser_mode":
                test = self.check_optimiser_mode(obj)
            elif typ == "optimiser_algorithm":
                test = obj in ["trust_region", "lbfgs"]
            elif typ == "zero_to_one":
                test = isinstance(obj, float) and 0.0 <= obj < 1.0
            elif typ == "greater_than_one":
                test = isinstance(obj, float) and obj > 1.0
            elif typ == "negative_amplidue":
                test = obj in ["remove", "flip_phase"]
            elif typ == "file_fmt":
                test = obj in ["txt", "pdf", "csv"]
            elif typ == "pos_neg_tuple":
                test = self.check_pos_neg_tuple(obj)
            elif typ == "mpl_color":
                test = self.check_mpl_color(obj)
            elif typ == "osc_cols":
                test = self.check_oscillator_colors(obj)
            elif typ == "generic_int_iter":
                test = isinstance(obj, list) and all(
                    isinstance(item, int) for item in obj
                )
            elif typ == "displacement":
                test = self.check_displacement(obj)
            elif typ == "modulation":
                test = obj in ["none", "amp", "phase"]
            elif typ == "start_time":
                test = self.check_start_time(obj)
            elif typ == "hessian":
                test = obj in ["exact", "gauss-newton"]

            # Error message to be shown if invalid arguments are found
            if test is False:
                try:
                    # If at least one previous fail has already been found,
                    # append the new fail to the pre-existing errmsg variable
                    errmsg += f"--> {name}\n"
                except NameError:
                    # First fail: errmsg doesn't exist yet, so initialise
                    errmsg = (
                        f"{RED}The following arguments are invalid:\n" f"--> {name}\n"
                    )

        try:
            # If errmsg exists, it implies that at least one test failed.
            # Add a final remark to the message and raise a TypeError.
            errmsg += f"Have a look at the documentation for more info." f"{END}"
            raise TypeError(errmsg)

        except NameError:
            # errmsg doesn't exist, implying no failed tests occurred.
            pass

    def stage(self, *args: tuple) -> None:
        """Add tuples of components for processing prior to testing.

        Each argument should be a tuple, with as a minimum
        """
        try:
            self.components
        except AttributeError:
            self.components = []

        for arg in args:
            if len(arg) == 3:
                self.components.append(tuple(a for a in arg) + (False,))
            elif len(arg) == 4:
                self.components.append((arg))

    def check_dim(f: Callable) -> Callable:
        """Check that the `dim` attribute is an `int`."""

        def wrapper(*args, **kwargs):
            if not isinstance(args[0].dim, int):
                raise ValueError(f"{RED}---BUG---{END} dim needs to be specified.")
            return f(*args, **kwargs)

        return wrapper

    @check_dim
    def check_parameter_array(self, obj: Any) -> bool:
        """Check for numpy array of shape ``(M, 4)`` or ``(M, 6)``."""
        if not isinstance(obj, np.ndarray):
            return False

        # Check (M x 4) or (M x 6) array
        p = 2 * (self.dim + 1)
        if obj.ndim == 2 and obj.shape[1] == p:
            return True

        return False

    @check_dim
    def check_iter(self, obj: Any, typ: Any) -> bool:
        """Check object is an iterable of the correct length."""
        try:
            # Check the object is iterable
            iter(obj)
            # Check length is known and = dim
            assert len(obj) == self.dim
            # Check all elements match the desired type
            assert all([isinstance(elem, typ) for elem in obj])
            return True
        except Exception:
            return False

    @check_dim
    def check_region(self, obj: Any, typ: Union[int, float]) -> bool:
        """Check for a valid region specifier.

        Should be one of the following:

        * ``Tuple[Tuple[typ, typ]]``
        * ``Tuple[Tuple[typ, typ], Tuple[typ, typ]]``
        """
        # Check for a list of the correct shape
        if not isinstance(obj, (list, tuple)) or len(obj) != self.dim:
            return False
        # Check that every element in the list is a list of length 2
        for sublist in obj:
            if not isinstance(sublist, (list, tuple)) and len(sublist) == 2:
                return False
            # Check that each bound is of the correct type
            for element in sublist:
                if not isinstance(element, typ):
                    return False
        return True

    @staticmethod
    def check_optimiser_mode(obj: Any) -> bool:
        """Ensure that the optimisation mode is valid.

        This should be a string containing only the characters ``'a'``,
        ``'p'``, ``'f'``, and ``'d'``, without any repetition.
        """
        if not isinstance(obj, str):
            return False

        # check if mode is empty or contains and invalid character
        if any(c not in "apfd" for c in obj) or obj == "":
            return False

        # check if mode contains a repeated character
        count = {}
        for c in obj:
            if c in count.keys():
                count[c] += 1
            else:
                count[c] = 1

        return all(map(lambda x: x == 1, count.values()))

    @staticmethod
    def check_pos_neg_tuple(obj: Any) -> bool:
        """Check for tuple with one negative and one positive int."""
        return (
            isinstance(obj, tuple)
            and len(obj) == 2
            and isinstance(obj[0], int)
            and obj[0] < 0
            and isinstance(obj[1], int)
            and obj[1] > 1
        )

    @staticmethod
    def check_mpl_color(obj: Any) -> bool:
        """Check for a valid matplotlib color."""
        try:
            mcolors.to_hex(obj)
            return True
        except ValueError:
            return False

    def check_oscillator_colors(self, obj: Any) -> bool:
        """Check for valid oscillator colorcycle."""
        # Check for single matplotlib color
        if self.check_mpl_color(obj):
            return True

        # Check for matplotlib colormap
        if obj in plt.colormaps():
            return True

        # Check if a list or numpy array with valid mpl colors
        if isinstance(obj, (list, np.ndarray)):
            for elem in obj:
                if not self.check_mpl_color(elem):
                    # At least one colour in the iterable is not valid.
                    # All options are exhausted not, so return False
                    return False
            # If we get to here, all list elements were valid colours.
            return True
        # No hits from the above conditionals...
        return False

    @staticmethod
    def check_displacement(obj: Any) -> bool:
        """Check for a valid mpl label displacement tuple."""
        if isinstance(obj, tuple) and len(obj) == 2:
            return all(isinstance(x, float) and abs(x) < 1.0 for x in obj)
        return False

    @check_dim
    def check_start_time(self, obj: Any) -> bool:
        """Check whether object is a valid start time specifier for signal."""
        if isinstance(obj, (list, tuple)) and (len(obj) == self.dim):
            return all(
                isinstance(item, float) or re.match(r"^-?\d+dt$", item) for item in obj
            )
        return False


class FrequencyConverter:
    """Handle frequency unit conversion."""

    def __init__(self, expinfo: ExpInfo, pts: Iterable[int]) -> None:
        """Initialise the converter.

        Parameters
        ----------
        expinfo
            Information on experiemnt parameters. From this, the number of
            experiment points, sweep width, transmitter offset, and
            transmitter offset frequency (optional) are obtained.

        Notes
        -----
        It is necessary for `expinfo` to have `sfo` not set to `None` if
        conversion from/tp ppm is desired.
        """
        try:
            sw, offset, sfo, dim = expinfo.unpack(
                "sw", "offset", "sfo", "dim"
            )
        except Exception:
            raise TypeError(f"{RED}Check `expinfo` is valid.{END}")

        checker = ArgumentChecker(dim=dim)
        checker.stage(
            (pts, "pts", "int_iter"),
            (sw, "sw", "float_iter"),
            (offset, "offset", "float_iter"),
            (sfo, "sfo", "float_iter", True),
        )
        checker.check()
        self.pts = pts
        self.sw = sw
        self.offset = offset
        self.sfo = sfo
        self.dim = dim

    def __len__(self):
        """Return number of experiment dimensions."""
        return self.dim

    def convert(
        self, lst: Iterable[Union[int, float]], conversion: str
    ) -> Iterable[Union[int, float]]:
        """Convert quantities contained within an iterable.

        Parameters
        ----------
        lst
            An iterable of numerical values, with the same length as
            ``len(self)``.

        conversion
            A string denoting the coversion to be applied. The form of
            `conversion` should be ``f'{from}->{to}'``, where ``from`` and
            ``to`` are each one of the following:

            * `'idx'`: array index
            * `'hz'`: Hertz
            * `'ppm'`: parts per million

        Returns
        -------
        converted_lst
            An iterable of the same dimensions as `lst`, with converted values.
        """
        if not self._check_valid_conversion(conversion):
            raise ValueError(f"{RED}convert is not valid.{END}")

        if len(self) != len(lst):
            raise ValueError(f"{RED}lst should be of length {len(self)}.{END}")

        # List for storing final converted contents
        converted_lst = []
        for dim, elem in enumerate(lst):
            # try/except block enables code to work with both lists and
            # lists of lists
            try:
                # Test whether element is an iterable
                iterable = iter(elem)
                # Create sublist
                converted_sublst = []

                while True:
                    try:
                        converted_sublst.append(
                            self._convert_value(
                                next(iterable),
                                dim,
                                conversion,
                            )
                        )
                    except StopIteration:
                        break

                converted_lst.append(type(elem)((converted_sublst)))

            except TypeError:
                # elem is a float/int...
                converted_lst.append(self._convert_value(elem, dim, conversion))

        return type(lst)(converted_lst)

    def _check_valid_conversion(self, conversion: str) -> bool:
        """Check that conversion is a valid value."""
        pattern = r"^(idx|ppm|hz)->(idx|ppm|hz)$"
        return bool(re.match(pattern, conversion))

    def _convert_value(
        self, value: Union[int, float], dim: int, conversion: str
    ) -> Union[int, float]:
        """Convert frequency."""
        pts = self.pts[dim]
        sw = self.sw[dim]
        off = self.offset[dim]
        if (self.sfo is None) and ("ppm" in conversion):
            raise ValueError(
                f"{RED}WARNING tried to convert to/from ppm, when sfo"
                f" has not been specified!{END}"
            )
        elif self.sfo is None:
            pass
        else:
            sfo = self.sfo[dim]

        if conversion == "idx->hz":
            return off + sw * (0.5 - (float(value) / (pts - 1)))

        elif conversion == "idx->ppm":
            return (off + sw * (0.5 - (float(value) / (pts - 1)))) / sfo

        elif conversion == "ppm->idx":
            return int(round(((pts - 1)) / (2 * sw) * (sw + 2 * (off - (value * sfo)))))

        elif conversion == "ppm->hz":
            return value * sfo

        elif conversion == "hz->idx":
            return int(round((pts - 1) / (2 * sw) * (sw + 2 * (off - value))))

        elif conversion == "hz->ppm":
            return value / sfo

        else:
            # conversion will be one of 'hz->hz', 'ppm->ppm', 'idx->idx'
            return value


class PathManager:
    """Class for performing checks on paths."""

    def __init__(self, fname: str, dir: str) -> None:
        """Initialise the path manager.

        Parameters
        ----------
        fname
            Filename.
        dir
            Directory.
        """
        self.fname = Path(fname)
        self.dir = Path(dir)
        self.path = (self.dir / self.fname).resolve()

    def check_file(self, force_overwrite: bool = False) -> int:
        """Perform checks on the path.

        Parameters
        ----------
        force_overwrite
            Specifies whether to ask the user if they are happy for the file
            `self.path` to be overwritten if it already exists in their
            filesystem.

        Returns
        -------
        return_code
            See notes for details

        Notes
        -----
        This method first checks whether the path directory exists. If it
        does, it checks whether the full path exists. If it does, the user
        is asked for permission to overwrite, if `force_overwrite` is `False`.
        The following codes can be returned:

        * ``0``: `dir` exists, `path` doesn't exist/can be overwritten.
        * ``1``: `path` already exists and the user does not give permission
          to overwrite.
        * ``2``: `dir` does not exist.
        """
        if not self.dir.is_dir():
            return 2

        if self.path.is_file() and not force_overwrite:
            overwrite = self.ask_overwrite()
            if not overwrite:
                print(f"{ORA}Overwrite denied{END}")
                return 1

        return 0

    def ask_overwrite(self) -> bool:
        """Ask the user if the are happy to overwrite `path`."""
        prompt = f"{ORA}The file {str(self.path)} already exists. " f"Overwrite?{END}"
        return get_yes_no(prompt)


def get_yes_no(prompt: str) -> bool:
    """Ask user to input 'yes' or 'no'.

    User should provide one of the following: ``'y'``, ``'Y'``, ``'n'``,
    ``'N'``. If an invalid response is given, the user is asked again.
    """
    print(f"{prompt}\nEnter [y] or [n]")
    while True:
        response = input().lower()
        if response == "y":
            return True
        elif response == "n":
            return False
        else:
            print(f"{RED}Invalid input. Please enter [y] or [n]:{END}")


def start_end_wrapper(start_text: str, end_text: str) -> Callable:
    """Print a message prior to and after a method call."""

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def inner(*args, **kwargs) -> Any:

            inst = args[0]
            if inst.fprint is False:
                return f(*args, **kwargs)

            print(
                f"{GRE}{len(start_text) * '='}\n"
                f"{start_text}\n"
                f"{len(start_text) * '='}{END}"
            )

            result = f(*args, **kwargs)

            print(
                f"{GRE}{len(end_text) * '='}\n"
                f"{end_text}\n"
                f"{len(end_text) * '='}{END}"
            )

            return result

        return inner

    return decorator


def latex_nucleus(nucleus: str) -> str:
    r"""Create a isotope symbol string for processing by LaTeX.

    Parameters
    ----------
    nucleus
        Of the form ``f'{mass}{sym}'``, where ``mass`` is the nuceleus'
        mass number and ``sym`` is its chemical symbol. I.e. for
        lead-207, `nucleus` should be ``'207Pb'``.

    Returns
    -------
    latex_nucleus
        Of the form ``f'$^{mass}${sym}'`` i.e. given ``'207Pb'``, the
        return value would be ``'$^{207}$Pb'``

    Raises
    ------
    ValueError
        If `nucleus` does not match the regex ``r'^\d+[a-zA-Z]+$'``
    """
    if re.match(r"\d+[a-zA-Z]+", nucleus):
        mass = re.search(r"\d+", nucleus).group()
        sym = re.search(r"[a-zA-Z]+", nucleus).group()
        return f"$^{{{mass}}}${sym}"

    else:
        raise ValueError(
            f"{RED}`nucleus` is invalid. Should match the regex " f"\\d+[a-zA-Z]+{END}"
        )
