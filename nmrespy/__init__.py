from pathlib import Path
from ._version import __version__

NMRESPYPATH = Path(__file__).parent
IMAGESPATH = NMRESPYPATH / 'images'
MFLOGOPATH = IMAGESPATH / 'mf_logo.png'
NMRESPYLOGOPATH = IMAGESPATH / 'nmrespy_full.png'
BOOKICONPATH = IMAGESPATH / 'book_icon.png'
GITHUBLOGOPATH = IMAGESPATH / 'github.png'
EMAILICONPATH = IMAGESPATH / 'email_icon.png'

# To assist users with manual install of TopSpin GUI loader
TOPSPINPATH = NMRESPYPATH / 'app/_topspin.py'

GITHUBLINK = 'https://github.com/foroozandehgroup/NMR-EsPy'
MFGROUPLINK = 'http://foroozandeh.chem.ox.ac.uk/home'
DOCSLINK = 'https://foroozandehgroup.github.io/NMR-EsPy'
MAILTOLINK = r'mailto:simon.hulse@chem.ox.ac.uk?subject=NMR-EsPy query'

# Coloured terminal output
import importlib
import platform

END = '\033[0m'  # end editing
RED = '\033[31m'  # red
GRE = '\033[32m'  # green
ORA = '\033[33m'  # orange
BLU = '\033[34m'  # blue
MAG = '\033[35m'  # magenta (M is reserved for no. of oscillators)
CYA = '\033[96m'  # cyan

USE_COLORAMA = False

# If on windows, enable ANSI colour escape sequences if colorama
# is installed
if platform.system() == 'Windows':
    colorama_spec = importlib.util.find_spec("colorama")
    if colorama_spec is not None:
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
    """Stores general information about experiments.

    Parameters
    ----------
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
    If ``dim`` is not specified, you must be explicit about the identites
    of inputs in each dimensions. If the arguments ``sw``, ``offset``,
    ``sfo`` and ``nuclei`` do not all have the same number of associated
    parameters, and error will be thrown, as the dimension of the experiment
    is ambiguous. For example, even if you have a 2D experiment
    where the sweep width, offset, nucleus, etc are identical, you must
    specify the values in each dimension, i.e.

    .. code:: python3

        expinfo = ExpInfo(sw=(5000, 5000), offset=(1000, 1000))

    Alternatively, you can set ``dim`` manually, and then any underspecified
    parameters will be implicitly filled in:

    .. code:: python3

        expinfo = ExpInfo(sw=5000, offset=1000, dim=2)
    """
    def __init__(
        self,
        sw: Union[int, float, Iterable[Union[int, float]]],
        offset: Union[int, float, Iterable[Union[int, float]], None] = None,
        sfo: Union[int, float, Iterable[Union[int, float]], None] = None,
        nuclei: Union[str, Iterable[str], None] = None,
        dim: Union[int, None] = None,
        **kwargs
    ) -> None:
        # Be leinient with parameter specfiication.
        # Mopst of nmrespy expects parameters to be lists with either
        # floats or ints for rigour.

        # If dim is specified, will be strict with ensuring each
        # parameter has the correct number of values. If not, will
        # duplicate values to match correct dim.

        self.__dict__.update(locals())
        names = ['sw']
        instances = [Number]
        for name, inst in zip(('offset', 'sfo', 'nuclei'),
                              (Number, Number, str)):
            if self.__dict__[name] is not None:
                names.append(name)
                instances.append(inst)

        errmsg = ("f{RED}Unable to process input{END}")

        for name, inst in zip(names, instances):
            value = self.__dict__[name]
            # If single value (not in list/tuple/etc.) is given, pack into
            # a list (values will be converted to tuples at the end)
            if isinstance(value, inst):
                if inst == Number:
                    # Convert numerical value to float
                    self.__dict__[name] = [float(value)]
                else:
                    # Case for nuclei, which should be a string
                    self.__dict__[name] = [value]

            elif isinstance(value, Iterable):
                if not all([isinstance(v, inst) for v in value]):
                    raise ValueError(errmsg)

                if inst == Number:
                    self.__dict__[name] = [float(v) for v in value]
                else:
                    self.__dict__[name] = list(value)

            else:
                raise ValueError(errmsg)

        if isinstance(dim, int):
            for name in names:
                diff = dim - len(self.__dict__[name])
                if diff == 0:
                    pass
                elif diff > 0:
                    self.__dict__[name] += diff * [self.__dict__[name][-1]]
                else:
                    raise ValueError(errmsg)

        else:
            lengths = [len(self.__dict__[name]) for name in names]
            # Check all lists are of the same length
            if len(set(lengths)) > 1:
                raise ValueError(errmsg)
            else:
                self.dim = lengths[0]

        if self.offset is None:
            self.offset = [0.] * self.dim

        for name in names:
            self.__dict__[name] = tuple(self.__dict__[name])

    def unpack(self, *args):
        if len(args) == 1:
            return self.__dict__[args[0]]
        else:
            return tuple([self.__dict__[arg] for arg in args])
