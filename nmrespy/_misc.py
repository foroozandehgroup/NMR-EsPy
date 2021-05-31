# _misc.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Various miscellaneous functions/classes for internal nmrespy use."""

import functools
import itertools
from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np

import nmrespy._cols as cols
if cols.USE_COLORAMA:
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
              + `'int_list'`
              + `'float_list'`
              + `'str_list'`
              + `'array_list'`
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

    dim : 1, 2 or None, default: None
        Dimension of the data. Only needs to be specified as `1` or `2`
        if one or more of the arguments to check have a structure that
        depends on the data dimension.
    """

    def __init__(self, components, dim=None, n=None):

        self.dim = dim

        for obj, name, typ in components:
            if typ == 'ndarray':
                test = isinstance(obj, np.ndarray)
            elif typ == 'parameter':
                test = self.check_parameter_array(obj)
            elif typ == 'int_list':
                test = self.check_list(obj, int)
            elif typ == 'float_list':
                test = self.check_list(obj, float)
            elif typ == 'str_list':
                test = self.check_list(obj, str)
            elif typ == 'array_list':
                test = self.check_list(obj, np.ndarray)
            elif typ == 'region_int':
                test = self.check_region(obj, int)
            elif typ == 'region_float':
                test = self.check_region(obj, float)
            elif typ == 'bool':
                test = isinstance(obj, bool)
            elif typ == 'int':
                test = isinstance(obj, int)
            elif typ == 'float':
                test = isinstance(obj, float)
            elif typ == 'str':
                test = isinstance(obj, str)
            elif typ == 'list':
                test = isinstance(obj, list)
            elif typ == 'positive_int':
                test = isinstance(obj, int) and obj > 0
            elif typ == 'positive_int_or_zero':
                test = isinstance(obj, int) and obj >= 0
            elif typ == 'positive_float':
                test = isinstance(obj, float) and obj > 0
            elif typ == 'optimiser_mode':
                test = self.check_optimiser_mode(obj)
            elif typ == 'optimiser_algorithm':
                test = obj in ['trust_region', 'lbfgs']
            elif typ == 'zero_to_one':
                test = isinstance(obj, float) and 0. <= obj < 1.
            elif typ == 'greater_than_one':
                test = isinstance(obj, float) and obj > 1.0
            elif typ == 'negative_amplidue':
                test = obj in ['remove', 'flip_phase']
            elif typ == 'file_fmt':
                test = obj in ['txt', 'pdf', 'csv']
            elif typ == 'pos_neg_tuple':
                test = self.check_pos_neg_tuple(obj)
            elif typ == 'mpl_color':
                test = self.check_mpl_color(obj)
            elif typ == 'osc_cols':
                test = self.check_oscillator_colors(obj)
            elif typ == 'generic_int_list':
                test = (isinstance(obj, list) and
                        all(isinstance(item, int) for item in obj))
            elif typ == 'displacement':
                test = self.check_displacement(obj)
            elif typ == 'modulation':
                test = obj in ['none', 'amp', 'phase']

            # Error message to be shown if invalid arguments are found
            if test is False:
                try:
                    # If at least one previous fail has already been found,
                    # append the new fail to the pre-existing errmsg variable
                    errmsg += f'--> {name}\n'
                except NameError:
                    # First fail: errmsg doesn't exist yet, so initialise
                    errmsg = (
                        f'{cols.R}The following arguments are invalid:\n'
                        f'--> {name}\n'
                    )

        try:
            # If errmsg exists, it implies that at least one test failed.
            # Add a final remark to the message and raise a TypeError.
            errmsg += (
                f'Have a look at the documentation for more info.'
                f'{cols.END}'
            )
            raise TypeError(errmsg)

        except NameError:
            # errmsg doesn't exist, implying no failed tests occurred.
            pass

    def check_dim(f):
        def wrapper(*args, **kwargs):
            if args[0].dim is None:
                raise ValueError(
                    f'{cols.R}---BUG--- dim needs to be specified{cols.END}'
                )
            return f(*args, **kwargs)
        return wrapper

    @check_dim
    def check_parameter_array(self, obj):
        """Checks for numpy array of shape (M, 4) or (M, 6)"""

        if not isinstance(obj, np.ndarray):
            return False

        # Check (M x 4) or (M x 6) array
        p = 2 * (self.dim + 1)
        if obj.ndim == 2 and obj.shape[1] == p:
            return True

        return False

    @check_dim
    def check_list(self, obj, typ):
        """Checks for `[int]`, `[int, int]`, `[float]`, `[float, float]`"""
        # Check for a list of the correct shape
        if not isinstance(obj, list) or len(obj) != self.dim:
            return False
        # Check that every element in the list is of the correct type
        for element in obj:
            if not isinstance(element, typ):
                return False

        return True

    @check_dim
    def check_region(self, obj, typ):
        """Checks for `[[int, int]]`, `[[int, int], [int, int]]`,
        `[[float, float]]`, and `[[float, float], [float, float]]`"""
        # Check for a list of the correct shape
        if not isinstance(obj, list) or len(obj) != self.dim:
            return False
        # Check that every element in the list is a list of length 2
        for sublist in obj:
            if not isinstance(sublist, list) and len(sublist) == 2:
                return False
            # Check that each bound is of the correct type
            for element in sublist:
                if not isinstance(element, typ):
                    return False
        return True

    @staticmethod
    def check_optimiser_mode(obj):
        """Ensures that the optimisation mode is valid. This should be a
        string containing only the characters 'a', 'p', 'f', and 'd', without
        any repetition.
        """
        if not isinstance(obj, str):
            return False

        # check if mode is empty or contains and invalid character
        if any(c not in 'apfd' for c in obj) or obj == '':
            return False

        # check if mode contains a repeated character
        count = {}
        for c in obj:
            if c in count.keys():
                count[c] += 1
            else:
                count[c] = 1

        for key in count:
            if count[key] > 1:
                return False

        return True

    @staticmethod
    def check_pos_neg_tuple(obj):
        """Check for object of the form ``(-x, y)`` where `x` and `y` are
        positive ints.
        """
        return (isinstance(obj, tuple) and len(obj) == 2
                and isinstance(obj[0], int) and obj[0] < 0
                and isinstance(obj[1], int) and obj[1] > 1)

    @staticmethod
    def check_mpl_color(obj):
        """Check for a valid matplotlib color."""
        try:
            mcolors.to_hex(obj)
            return True
        except ValueError:
            return False

    def check_oscillator_colors(self, obj):
        """Check for valid oscillator colorcycle"""
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
    def check_displacement(obj):
        """Check for a valid mpl label displacement tuple."""
        if isinstance(obj, tuple) and len(obj) == 2:
            for element in obj:
                if isinstance(element, float) and abs(element) < 1.:
                    pass
                else:
                    return False
            return True
        return False


class FrequencyConverter:
    """Handles converting objects with frequency values between units

    Parameters
    ----------
    n : [int] or [int, int]
        Number of points in each dimension.

    sw : [float] or [float, float]
        Experiment sweep width in each dimension (Hz)

    offset : [float] or [float, float]
        Transmitter offset in each dimension (Hz)

    sfo : [float] or [float, float] or None, default: None
        Transmitter frequency in each dimension (MHz). If set to `None`, only
        conversion between Hz and array indices will be possible.
    """
    def __init__(self, n, sw, offset, sfo=None):

        try:
            dim = len(n)
        except Exception:
            raise TypeError(f'{cols.R}n should be iterable.{cols.END}')

        components = [
            (n, 'n', 'int_list'),
            (sw, 'sw', 'float_list'),
            (offset, 'offset', 'float_list'),
        ]

        if sfo is not None:
            components.append((sfo, 'sfo', 'float_list'))

        ArgumentChecker(components, dim)

        self.n = n
        self.sw = sw
        self.offset = offset
        self.sfo = sfo

    def __len__(self):
        return len(self.n)

    def convert(self, lst, conversion):
        """Convert quantities contained within a list

        Parameters
        ----------
        lst : list
            A list of numerical values, with the same length as ``len(self)``.

        conversion : str
            A string denoting the coversion to be applied. The form of
            `conversion` should be ``'from->to'``, where ``from`` and ``to``
            are not matching, and are one of the following:

            * `'idx'`: array index
            * `'hz'`: Hertz
            * `'ppm'`: parts per million

        Returns
        -------
        converted_lst : lst
            A list of the same dimensions as `lst`, with converted values.
        """

        if not self._check_valid_conversion(conversion):
            raise ValueError(f'{cols.R}convert is not valid.{cols.END}')

        if len(self) != len(lst):
            raise ValueError(
                f'{cols.R}lst should be of length {len(self)}.{cols.END}'
            )

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
                                next(iterable), dim, conversion,
                            )
                        )
                    except StopIteration:
                        break

                converted_lst.append(converted_sublst)

            except TypeError:
                # elem is a float/int...
                converted_lst.append(
                    self._convert_value(elem, dim, conversion)
                )

        return converted_lst

    def _check_valid_conversion(self, conversion):
        """check that conversion is a valid value"""
        units = ['idx', 'ppm', 'hz']
        for pair in itertools.permutations(units, r=2):
            pair = iter(pair)
            if f'{next(pair)}->{next(pair)}' == conversion:
                return True
        return False

    def _convert_value(self, value, dim, conversion):
        n = self.n[dim]
        sw = self.sw[dim]
        off = self.offset[dim]
        if self.sfo is not None:
            sfo = self.sfo[dim]
        else:
            if 'ppm' in conversion:
                raise ValueError(
                    f'{cols.R}WARNING tried to convert to/from ppm, when sfo'
                    f' has not been specified!{cols.END}'
                )

        if conversion == 'idx->hz':
            return off + sw * (0.5 - (float(value) / (n - 1)))

        elif conversion == 'idx->ppm':
            return (off + sw * (0.5 - (float(value) / (n - 1)))) / sfo

        elif conversion == 'ppm->idx':
            return int(
                round(
                    ((n - 1)) / (2 * sw) * (sw + 2 * (off - (value * sfo)))
                )
            )

        elif conversion == 'ppm->hz':
            return value * sfo

        elif conversion == 'hz->idx':
            return int(round((n - 1) / (2 * sw) * (sw + 2 * (off - value))))

        elif conversion == 'hz->ppm':
            return value / sfo


class PathManager:
    """Class for performing checks on paths.

    Parameters
    ----------
    fname : str
        Filename.
    dir : str
        Directory.
    """
    def __init__(self, fname, dir):

        self.fname = Path(fname)
        self.dir = Path(dir)
        self.path = self.dir / self.fname

    def check_file(self, force_overwrite=False):
        """Performs checks on the path file dir/fname

        Parameters
        ----------
        force_overwrite : bool, default: False
            Specifies whether to ask the user if they are happy for the file
            `self.path` to be overwritten if it already exists in their
            filesystem.

        Returns
        -------
        return_code : int
            See notes for details

        Notes
        -----
        This method first checks whether dir exists. If it does, it checks
        whether the file `dir/fname` exists. If it does, the user is asked for
        permission to overwrite, if `force_overwrite` is `False`. The following
        codes can be returned:

        * ``0`` `dir/fname` doesn't exist/can be overwritten, and `dir` exists.
        * ``1`` `dir/fname` already exists and the user does not give
          permission to overwrite.
        * ``2`` `dir` does not exist.
        """

        if not self.dir.is_dir():
            return 2

        if self.path.is_file() and not force_overwrite:
            overwrite = self.ask_overwrite()
            if not overwrite:
                print(f'{cols.OR}Overwrite denied{cols.END}')
                return 1

        return 0

    def ask_overwrite(self):
        """Asks the user if the are happy to overwrite the file given
        by the path `self.path`"""

        prompt = (
            f'{cols.OR}The file {str(self.path)} already exists. Overwrite?\n'
            f'Enter [y] or [n]:{cols.END}'
        )

        return get_yes_no(prompt)


def get_yes_no(prompt):
    """Ask user to input 'yes' or 'no' (Y/y or N/n). Repeatedly does this
    until a valid response is recieved"""

    print(prompt)
    response = input().lower()
    while True:
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            print(f'{cols.R}Invalid input. Please enter [y] or [n]:{cols.END}')
            response = input().lower()


def start_end_wrapper(start_text, end_text):
    """Decorator which prints a message prior to and after a method.

    Messages are sandwiched between double-lines.
    """
    def decorator(f):

        @functools.wraps(f)
        def inner(*args, **kwargs):

            inst = args[0]
            if inst.fprint is False:
                return f(*args, **kwargs)

            print(f"{cols.G}{len(start_text) * '='}\n"
                  f"{start_text}\n"
                  f"{len(start_text) * '='}{cols.END}")

            result = f(*args, **kwargs)

            print(f"{cols.G}{len(end_text) * '='}\n"
                  f"{end_text}\n"
                  f"{len(end_text) * '='}{cols.END}")

            return result

        return inner

    return decorator


def latex_nucleus(nucleus):
    """Creates a isotope symbol string for processing by LaTeX.

    Parameters
    ----------
    nucleus : str
        Of the form `'<mass><sym>'`, where `'<mass>'` is the nuceleus'
        mass number and `'<sym>'` is its chemical symbol. I.e. for
        lead-207, `nucleus` would be `'207Pb'`.

    Returns
    -------
    latex_nucleus : str
        Of the form ``$^{<mass>}$<sym>`` i.e. given `'207Pb'`, the
        return value would be ``$^{207}$Pb``

    Raises
    ------
    ValueError
        If `nucleus` does not match the regex ``^[0-9]+[a-zA-Z]+$``
    """
    if re.match(r'\d+[a-zA-Z]+', nucleus):
        mass = re.search(r'\d+', nucleus).group()
        sym = re.search(r'[a-zA-Z]+', nucleus).group()
        return f'$^{{{mass}}}${sym}'

    else:
        raise ValueError(
            f'{cols.R}`nucleus` is invalid. Should match the regex'
            f' \\d+[a-zA-Z]+{cols.END}'
        )


def significant_figures(value, s):
    """Rounds `value` to `s` significant figures."""
    if value != 0:
        value = round(value, s - int(np.floor(np.log10(abs(value)))) - 1)
        # If value of form 123456.0, convert to 123456
        if float(value).is_integer():
            value = int(value)
    return value
