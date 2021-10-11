# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 07 Oct 2021 12:09:33 BST

"""NMR-EsPy: Nuclear Magnetic Resonance Estimation in Python."""

from importlib.util import find_spec
from numbers import Number
from pathlib import Path
from platform import system
from typing import Any, Iterable, Tuple, Type, Union
from ._version import __version__  # noqa: F401


# Paths and links
NMRESPYPATH = Path(__file__).parent
IMAGESPATH = NMRESPYPATH / 'images'
MFLOGOPATH = IMAGESPATH / 'mf_logo.png'
NMRESPYLOGOPATH = IMAGESPATH / 'nmrespy_full.png'
BOOKICONPATH = IMAGESPATH / 'book_icon.png'
GITHUBLOGOPATH = IMAGESPATH / 'github.png'
EMAILICONPATH = IMAGESPATH / 'email_icon.png'
TOPSPINPATH = NMRESPYPATH / 'app/_topspin.py'
GITHUBLINK = 'https://github.com/foroozandehgroup/NMR-EsPy'
MFGROUPLINK = 'http://foroozandeh.chem.ox.ac.uk/home'
DOCSLINK = 'https://foroozandehgroup.github.io/NMR-EsPy'
MAILTOLINK = r'mailto:simon.hulse@chem.ox.ac.uk?subject=NMR-EsPy query'

# Coloured terminal output
END = '\033[0m'  # end editing
RED = '\033[31m'  # red
GRE = '\033[32m'  # green
ORA = '\033[33m'  # orange
BLU = '\033[34m'  # blue
MAG = '\033[35m'  # magenta
CYA = '\033[96m'  # cyan

USE_COLORAMA = False

# If on windows, enable ANSI colour escape sequences if colorama
# is installed
if system() == 'Windows':
    if find_spec("colorama"):
        USE_COLORAMA = True
    # If colorama not installed, make color attributes empty to prevent
    # bizzare outputs
    else:
        END = ''
        RED = ''
        GRE = ''
        ORA = ''
        BLU = ''
        MAG = ''
        CYA = ''


class ExpInfo:
    """Stores general information about experiments."""

    def __init__(
        self,
        pts: Union[int, Iterable[int]],
        sw: Union[int, float, Iterable[Union[int, float]]],
        offset: Union[int, float, Iterable[Union[int, float]], None] = None,
        sfo: Union[int, float, Iterable[Union[int, float]], None] = None,
        nuclei: Union[str, Iterable[str], None] = None,
        dim: Union[int, None] = None,
        **kwargs
    ) -> None:
        """Create an ExpInfo instance.

        Parameters
        ----------
        pts
            The number of points the signal is composed of.

        sw
            The sweep width (spectral window) (Hz).

        offset
            The transmitter offset (Hz).

        sfo
            The transmitter frequency (MHz).

        nuclei
            The identity of each channel.

        dim
            The number of dimensions associated with the experiment.

        kwargs
            Any extra parameters to be included

        Notes
        -----
        If `dim` is not specified, you must be explicit about the identites
        of inputs in each dimensions. If the arguments `sw`, `offset`,
        `sfo` and `nuclei` do not all have the same number of associated
        parameters, and error will be thrown, as the dimension of the
        experiment is ambiguous. For example, even if you have a 2D experiment
        where the sweep width, offset, nucleus, etc are identical in both
        dimensions, you must specify the values in each dimension, i.e.

        .. code:: python

            >>> expinfo = ExpInfo(pts=(1024, 128), sw=(5000, 5000),
            ...                   offset=(1000, 1000))
            >>> expinfo.__dict__
            {'pts': (1024, 128), 'sw': (5000.0, 5000.0),
             'offset': (1000.0, 1000.0), 'sfo': None, 'nuclei': None,
             'dim': 2, 'kwargs': {},
             'self': <nmrespy.ExpInfo object at 0x7f788ea22310>}

        Alternatively, you can set ``dim`` manually, and then any
        underspecified parameters will be implicitly filled in:

        .. code:: python

            >>> expinfo = ExpInfo(pts=(1024, 128), sw=5000, offset=1000,
            ...                   dim=2)
            >>> expinfo.__dict__
            {'pts': (1024, 128), 'sw': (5000.0, 5000.0),
             'offset': (1000.0, 1000.0), 'sfo': None, 'nuclei': None,
             'dim': 2, 'kwargs': {},
             'self': <nmrespy.ExpInfo object at 0x7f788ebc2cd0>}

        """
        # Be leinient with parameter specfiication.
        # Most of nmrespy expects parameters to be lists with either
        # floats or ints for rigour.
        # If dim is specified, will be strict with ensuring each
        # parameter has the correct number of values. If not, will
        # duplicate values to match correct dim.
        self._dim = dim
        for kwkey, kwvalue in kwargs.items():
            self.__dict__.update({kwkey: kwvalue})

        names = ['_pts', '_sw', '_offset', '_sfo', '_nuclei']
        locs = locals()
        values = [locs[name.replace('_', '')] for name in names]
        test_types = [int, Number, Number, Number, str]

        # Filter out any optional arguments that are None
        rm_idx = [i for i, value in enumerate(values[2:], start=2)
                  if value is None]
        names = [x for i, x in enumerate(names) if i not in rm_idx]
        values = [x for i, x in enumerate(values) if i not in rm_idx]
        test_types = [x for i, x in enumerate(test_types) if i not in rm_idx]

        for name, value, test_type in zip(names, values, test_types):
            errmsg = ("f{RED}Unable to process input{END}")
            # If single value (not in list/tuple/etc.) is given, pack into
            # a list (values will be converted to tuples at the end)
            if isinstance(value, test_type):
                value = [value]
            if isinstance(value, Iterable):
                if not all([isinstance(v, test_type) for v in value]):
                    raise ValueError(errmsg)
                if test_type == Number:
                    value = [float(v) for v in value]
            else:
                raise ValueError(errmsg)

            if isinstance(self._dim, int):
                diff = self._dim - len(value)
                if diff == 0:
                    pass
                elif diff > 0:
                    value += diff * [value[-1]]
                else:
                    raise ValueError(errmsg)

            self.__dict__[name] = value

        if self._dim is None:
            # Check all lists are of the same length
            length_set = {len(self.__dict__[name]) for name in names}
            if len(length_set) == 1:
                self._dim = len(self._pts)
            else:
                raise ValueError(errmsg)

        if not isinstance(self._dim, int):
            raise ValueError(f'{RED}Invalid value for `dim`{END}')
        if locs['offset'] is None:
            self._offset = tuple([0.] * self._dim)
        if locs['sfo'] is None:
            self._sfo = None
        if locs['nuclei'] is None:
            self._nuclei = None
        for name in names:
            self.__dict__[name] = tuple(self.__dict__[name])

    @property
    def pts(self) -> Iterable[int]:
        """Get number of points in the data."""
        return self._pts

    @pts.setter
    def pts(self, new_value: Any) -> None:
        pts = self._validate('pts', new_value, int)
        # Error will have been raised if new_value is invalid
        self._pts = pts

    @property
    def sw(self) -> Iterable[float]:
        """Get sweep width (Hz)."""
        return self._sw

    @sw.setter
    def sw(self, new_value: Any) -> None:
        sw = self._validate('sw', new_value, Number, float)
        # Error will have been raised if new_value is invalid
        self._sw = sw

    @property
    def offset(self) -> Iterable[float]:
        """Get transmitter offset frequency (Hz)."""
        return self._offset

    @offset.setter
    def offset(self, new_value: Any) -> None:
        offset = self._validate('offset', new_value, Number, float)
        # Error will have been raised if new_value is invalid
        self._offset = offset

    @property
    def sfo(self) -> Iterable[float]:
        """Get transmitter frequency (MHz)."""
        return self._sfo

    @sfo.setter
    def sfo(self, new_value: Any) -> None:
        sfo = self._validate('sfo', new_value, Number, float)
        # Error will have been raised if new_value is invalid
        self._sfo = sfo

    @property
    def nuclei(self) -> Iterable[str]:
        """Get nuclei associated with each channel."""
        return self._nuclei

    @nuclei.setter
    def nuclei(self, new_value: Any) -> None:
        nuclei = self._validate('nuclei', new_value, str)
        # Error will have been raised if new_value is invalid
        self._nuclei = nuclei

    @property
    def dim(self) -> int:
        """Get number of dimensions in the expeirment."""
        return self._dim

    @dim.setter
    def dim(self, new_value):
        raise ValueError(f'{RED}`dim` cannot be mutated.{END}')

    def _validate(
        self, name: str, value: Any, test_type: Type[Any],
        final_type: Union[Type[Any], None] = None
    ) -> None:
        errmsg = f'{RED}Invalid value supplied to {name}: {repr(value)}{END}'

        if isinstance(value, test_type):
            if self._dim == 1:
                value = [value]
            else:
                value = self._dim * [value]

        elif (isinstance(value, Iterable) and
              all(isinstance(v, test_type) for v in value)):
            value = list(value)
            diff = self._dim - len(value)
            if diff < 0:
                raise ValueError(errmsg)
            elif diff > 0:
                value += diff * [value[-1]]

        else:
            raise ValueError(errmsg)

        if not final_type:
            final_type = test_type
        return tuple([final_type(v) for v in value])

    def unpack(self, *args) -> Tuple[Any]:
        """Unpack attributes.

        `args` should be strings with names that match attribute names.
        """
        to_underscore = ['pts', 'sw', 'offset', 'sfo', 'nuclei', 'dim']
        ud_args = [f'_{a}' if a in to_underscore else a for a in args]
        if len(args) == 1:
            return self.__dict__[ud_args[0]]
        else:
            return tuple([self.__dict__[arg] for arg in ud_args])
