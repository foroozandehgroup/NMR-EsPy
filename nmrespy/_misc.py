#!/usr/bin/python3
# _misc.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

import copy
import functools

import numpy as np

import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama

        
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

    sfo : [float] or [float, float]
        Transmitter frequency in each dimension (MHz)
    """
    def __init__(self, n, sw, offset, sfo):

        # ensure all inputs are same length
        for obj in (n, sw, offset, sfo):
            if not isinstance(obj, list):
                raise TypeError(
                    f'{cols.R}n, sw, offset and sfo should all be'
                    f'  lists{cols.END}'
                )
            for value in obj:
                if not isinstance(value, (float, int)):
                    raise TypeError(
                        f'{cols.R}The elements of n, sw, offset and sfo'
                        f' should be numerical types (int, float){cols.END}'
                    )

        if not len(n) == len(sw) == len(offset) == len(sfo):
            raise ValueError(
                f'{cols.R}n, sw, offset and sfo should all be of the same'
                f' length{cols.END}'
            )

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

            * ``'idx'``: array index
            * ``'hz'``: Hertz
            * ``'ppm'``: parts per million

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



        # list for storing final converted contents (will be returned as tuple)
        converted_lst = []
        for dim, elem in enumerate(lst):

            # try/except block enables code to work with both lists and
            # lists of lists
            try:
                # test whether element is an iterable (i.e. tuple)
                iterable = iter(elem)

                converted_sublst = []

                while True:
                    try:
                        converted_sublst.append(
                            self._convert_value(next(iterable), dim, conversion)
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
        valid = False
        units = ['idx', 'ppm', 'hz']
        for pair in itertools.permutations(units, r=2):
            pair = iter(pair)
            if f'{next(pair)}->{next(pair)}' == conversion:
                valid = True
                break

        return valid


    def _convert_value(self, value, dim, conversion):
        n = self.n[dim]
        sw = self.sw[dim]
        off = self.offset[dim]
        sfo = self.sfo[dim]

        if conversion == 'idx->hz':
            return float(off + sw * (0.5 - (value / (n - 1))))

        elif conversion == 'idx->ppm':
            return float((off + sw * (0.5 - (value / (n - 1)))) / sfo)

        elif conversion == 'ppm->idx':
            return int(round(((n - 1)) / (2 * sw) * (sw + 2 * (off - (value * sfo)))))

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
        permission to overwrite, `force_overwrite` is `False`. The following
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
                print(f'{cols.O}Overwrite denied{cols.END}')
                return 1

        return 0

    def ask_overwrite(self):
        """Asks the user if the are happy to overwrite the file given
        by the path `self.path`"""

        prompt = (
            f'{cols.O}The file {str(self.path)} already exists. Overwrite?\n'
            f'Enter [y] or [n]:{cols.END} '
        )

        return get_yes_no(prompt)


def get_yes_no(prompt):
    """Ask user to input 'yes' or 'no' (Y/y or N/n). Repeatedly does this
    until a valid response is recieved"""

    response = input(prompt).lower()
    if response == 'y':
        return True
    elif response == 'n':
        return False
    else:
        get_yes_no(
            f'{cols.R}Invalid input. Please enter [y] or [n]:{cols.END} '
        )


def start_end_wrapper(start_text, end_text):

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


def aligned_tabular(columns, titles=None):
    """Tabularises a list of lists, with the option of including titles.

    Parameters
    ----------
    columns : list
        A list of lists, representing the columns of the table. Each list
        must be of the same length.

    titles : None or list, default: None
        Titles for the table. If desired, the ``titles`` should be of the same
        length as all of the lists in ``columns``.

    Returns
    -------
    msg : str
        A string with the contents of ``columns`` tabularised.
    """

    if titles:
        sep = ' │'
        for i,(title, column) in enumerate(zip(titles, columns)):
            columns[i] = [title] + column

    else:
        sep = ' '

    pads = []
    for column in columns:
        pads.append(max(len(element) for element in column))

    msg = ''
    for i, row in enumerate(zip(*columns)):
        for j, (pad, e1, e2) in enumerate(zip(pads, row, row[1:])):
            p = pad - len(e1)
            if j == 0:
                msg += f"{e1}{p*' '}{sep}{e2}"
            else:
                msg += f"{p*' '}{sep}{e2}"
        if titles and i == 0:
            for i, pad in enumerate(pads):
                if i == 0:
                    msg += f"\n{(pad+1)*'─'}┼"
                else:
                    msg += f"{(pad+1)*'─'}┼"
            msg = msg[:-1]
        msg += '\n'

    return msg
