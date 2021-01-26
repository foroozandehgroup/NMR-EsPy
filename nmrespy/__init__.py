import itertools
import os

import nmrespy._cols as cols
from ._version import __version__


NMRESPYPATH = os.path.dirname(__file__)
MFLOGOPATH = os.path.join(NMRESPYPATH, 'pics/mf_logo.png')
NMRESPYLOGOPATH = os.path.join(NMRESPYPATH, 'pics/nmrespy_full.png')

GITHUBPATH = 'https://github.com/foroozandehgroup/NMR-EsPy'
MFGROUPPATH = 'http://foroozandeh.chem.ox.ac.uk/home'


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
            print(dim)

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
        print(repr(dim))
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
