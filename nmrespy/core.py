import copy
import datetime
import functools
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from nmrespy import *
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
    colorama.init()
import nmrespy._errors as errors
from nmrespy._misc import *
from nmrespy.load import load_bruker
from nmrespy.freqfilter import FrequencyFilter
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp.nlp import NonlinearProgramming
from nmrespy.plot import plot_result
from nmrespy.write import write_result
from nmrespy import sig


class Estimator:
    """Estimation class

    .. note::
       The methods :py:meth:`new_bruker`, :py:meth:`new_synthetic_from_data`
       and :py:meth:`new_synthetic_from_parameters` generate instances
       of the class. The method :py:meth:`from_pickle` loads an estimator
       instance that was previously saved using :py:meth:`to_pickle`.
       While you can manually input the listed parameters
       as arguments to initialise the class, it is more straightforward
       to use one of these.

    Parameters
    ----------
    source : {'bruker_fid', 'bruker_pdata', 'synthetic'}
        The type of data imported. `'bruker_fid'` indicates the data is
        derived from a FID file (`fid` for 1D data, `ser` for 2D data).
        `'bruker_pdata'` indicates the data is derived from files found
        in a `pdata` directory (`1r` for 1D data; `2rr` for 2D data).
        `'synthetic'` indicates that the data is synthetic.

    data : numpy.ndarray
        The data associated with the binary file in `path`.

    path : pathlib.Path or None
        The path to the directory contaioning the NMR data.

    sw : [float] or [float, float]
        The experiment sweep width in each dimension (Hz).

    offset : [float] or [float, float]
        The transmitter's offset frequency in each dimension (Hz).

    sfo : [float] or [float, float] or None
        The transmitter frequency in each dimension (MHz)

    nuc : [str] or [str, str] or None
        The nucleus in each dimension. Elements will be of the form
        `'<mass><element>'`, where `'<mass>'` is the mass number of the
        isotope and `'<element>'` is the chemical symbol of the element.

    fmt : str or None
        The format of the binary file from which the data was obtained.
        Of the form `'<endian><unitsize>'`, where `'<endian>'` is either
        `'<'` (little endian) or `'>'` (big endian), and `'<unitsize>'`
        is either `'i4'` (32-bit integer) or `'f8'` (64-bit float).

    _origin : dict or None, default None
        For internal use. Specifies how the instance was initalised. If `None`,
        implies that the instance was initialised manually, rather than using
        one of :py:meth:`new_bruker`, :py:meth:`new_synthetic_from_data`
        and :py:meth:`new_synthetic_from_parameters`.
    """

    def __init__(
        self, source, data, path, sw, off, sfo, nuc, fmt, _origin=None
    ):
        self.source = source
        self.data = data
        self.dim = self.data.ndim
        self.n = [int(n) for n in self.data.shape]
        self.path = path
        self.sw = sw
        self.offset = off
        self.sfo = sfo
        self.nuc = nuc
        self.fmt = fmt

        # Attributes that will be assigned to after the user runs
        # the folowing methods:
        # * frequency_filter (filter_info)
        # * matrix_pencil or nonlinear_programming (result)
        # * nonlinear_programming (errors)
        self.filter_info = None
        self.result = None
        self.errors = None
        # Specifies whether the last time self.result was changed
        # was when nonlinear_prograaming was called.
        self._saveable = False
        # Create a converter object, enabling conversion idx, hz (and ppm,
        # if sfo is not None)
        self._converter = FrequencyConverter(
            list(data.shape), self.sw, self.offset, self.sfo
        )

        # --- Create attribute for logging method calls ------------------
        now = datetime.datetime.now()
        self._log = (
            "==============================\n"
            "Logfile for Estimator instance\n"
            "==============================\n"
            f"--> Instance created @ {now.strftime('%d-%m-%y %H:%M:%S')}"
        )

        if _origin is not None:
            self._log += (f" from `{_origin['method']}` with args "
                          f"{_origin['args']}")

        self._log += "\n"

    def __repr__(self):
        msg = (
            f'nmrespy.core.Estimator('
            f'{self.source}, '
            f'{self.data}, '
            f'{self.path}, '
            f'{self.sw}, '
            f'{self.offset}, '
            f'{self.n}, '
            f'{self.sfo}, '
            f'{self.nuc}, '
            f'{self.fmt})'
        )

        return msg

    def __str__(self):
        """A formatted list of class attributes"""

        msg = (
            f"{cols.MA}<{__class__.__module__}.{__class__.__qualname__} at "
            f"{hex(id(self))}>{cols.END}\n"
        )

        dic = self.__dict__
        keys, vals = dic.keys(), dic.values()
        items = [f'{cols.MA}{k}{cols.END} : {v}'
                 for k, v in zip(keys, vals) if k[0] != '_']
        msg += '\n'.join(items)

        return msg

    def logger(f):
        """Decorator for logging :py:class:`Estimator` method calls"""
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # The first arg is the class instance.
            # Append to the log text.
            args[0]._log += f'--> `{f.__name__}` {args[1:]} {kwargs}\n'

            # Run the method...
            return f(*args, **kwargs)

        return wrapper

    @classmethod
    def new_bruker(cls, dir, ask_convdta=True):
        """Generate an instance of :py:class:`Estimator` from a
        Bruker-formatted data directory.

        Parameters
        ----------
        dir : str
            The path to the data containing the data of interest.

        ask_convdta : bool
            See :py:meth:`nmrespy.load_bruker`

        Returns
        -------
        estimator : :py:class:`Estimator`

        Notes
        -----
        For a more detailed specification of the directory requirements,
        see :py:meth:`nmrespy.load_bruker`."""

        origin = {'method': 'new_bruker', 'args': locals()}
        info = load_bruker(dir, ask_convdta=ask_convdta)

        return cls(
            info['source'], info['data'], info['directory'],
            info['sweep_width'], info['offset'], info['transmitter_frequency'],
            info['nuclei'], info['binary_format'], _origin=origin
        )

    @classmethod
    def from_pickle(cls, path):
        """Loads an intance of :py:class:`Estimator`, which was saved
        previously using :py:meth:`to_pickle`.

        Parameters
        ----------
        path : str
            The path to the pickle file. **DO NOT INCLUDE THE FILE
            EXTENSION.**

        Returns
        -------
        estimator : :py:class:`Estimator`

        Notes
        -----
        .. warning::
           `From the Python docs:`

           *"The pickle module is not secure. Only unpickle data you trust.
           It is possible to construct malicious pickle data which will
           execute arbitrary code during unpickling. Never unpickle data
           that could have come from an untrusted source, or that could have
           been tampered with."*

           You should only use :py:meth:`from_pickle` on files that
           you are 100% certain were generated using
           :py:meth:`to_pickle`. If you load pickled data from a .pkl file,
           and the resulting output is not an instance of
           :py:class:`Estimator`, an error will be raised.
        """

        path = Path(path).with_suffix('.pkl')
        if path.is_file():
            with open(path, 'rb') as fh:
                obj = pickle.load(fh)
            if isinstance(obj, __class__):
                return obj
            else:
                raise TypeError(
                    f'{cols.R}It is expected that the object opened by'
                    ' `from_pickle` is an instance of'
                    f' {__class__.__module__}.{__class__.__qualname__}.'
                    f' What was loaded didn\'t satisfy this!{cols.END}'
                )

        else:
            raise ValueError(
                f'{cols.R}Invalid path specified.{cols.END}'
            )

    @logger
    def to_pickle(
        self, path='./estimator', force_overwrite=False, fprint=True,
    ):
        """Converts the class instance to a byte stream using Python's
        "Pickling" protocol, and saves it to a .pkl file.

        Parameters
        ----------
        path : str, default: './estimator'
            Path of file to save the byte stream to. **DO NOT INCLUDE A
            `'.pkl'` EXTENSION!** `'.pkl'` is added to the end of the path
            automatically.

        force_overwrite : bool, default: False
            Defines behaviour if ``f'{path}.pkl'`` already exists:

            * If `force_overwrite` is set to `False`, the user will be prompted
              if they are happy overwriting the current file.
            * If `force_overwrite` is set to `True`, the current file will be
              overwritten without prompt.

        fprint : bool, default: True
            Specifies whether or not to print infomation to the terminal.

        Notes
        -----
        This method complements :py:meth:`from_pickle`, in that
        an instance saved using :py:meth:`to_pickle` can be recovered by
        :py:func:`~nmrespy.load.pickle_load`.
        """

        ArgumentChecker(
            [
                (path, 'path', 'str'),
                (force_overwrite, 'force_overwrite', 'bool'),
                (fprint, 'fprint', 'bool'),
            ]
        )

        # Get full path
        path = Path(path).resolve()
        # Append extension to file path
        path = path.parent / (path.name + '.pkl')
        # Check path is valid (check directory exists, ask user if they are
        # happy overwriting if file already exists).
        pathres = PathManager(
            path.name, path.parent
        ).check_file(force_overwrite)
        # Valid path, we are good to proceed
        if pathres == 0:
            pass
        # Overwrite denied by the user. Exit the program
        elif pathres == 1:
            exit()
        # pathres == 2: Directory specified doesn't exist
        else:
            raise ValueError(
                f'{cols.R}The directory implied by path does not'
                f'exist{cols.END}'
            )

        with open(path, 'wb') as fh:
            pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)

        if fprint:
            print(f'{cols.G}Saved instance of Estimator to {path}{cols.END}')

    def get_datapath(self, type_='Path', kill=True):
        """Return path of the data directory.

        Parameters
        ----------
        type_ : 'Path' or 'str', default: 'Path'
            The type of the returned path. If `'Path'`, the returned object
            is an instance of `pathlib.Path <https://docs.python.org/3/\
            library/pathlib.html#pathlib.Path>`_. If `'str'`, the returned
            object is an instance of str.

        kill : bool, default: True
            If the path is `None`, `kill` specifies how the method will act:

            * If `True`, an AttributeIsNoneError is raised.
            * If `False`, `None` is returned.

        Returns
        -------
        path : str or pathlib.Path
        """

        path = self._check_if_none('path', kill)

        if type_ == 'Path':
            return path
        elif type_ == 'str':
            return str(path) if path is not None else None
        else:
            raise ValueError(f'{R}type_ should be \'Path\' or \'str\'')

    def get_data(self):
        """Return the original data.

        Returns
        -------
        data : numpy.ndarray
        """
        return self.data

    def get_dim(self):
        """Return the data dimension.

        Returns
        -------
        dim : 1 or 2
        """
        return self.dim

    def get_n(self):
        """Return the number of datapoints in each dimension

        Returns
        -------
        n : [int] or [int, int]
        """
        return self.n

    def get_sw(self, unit='hz', kill=True):
        """Return the experiment sweep width in each dimension.

        Parameters
        ----------
        unit : 'hz' or 'ppm', default: 'hz'

        kill : bool, default: True
            If `unit` is `'ppm'`, but `self.sfo` is `None`, `kill` specifies
            how the method will act:

            * If `True`, an AttributeIsNoneError is raised.
            * If `False`, `None` is returned.

        Returns
        -------
        sw : [float] or [float, float]

        Raises
        ------
        InvalidUnitError
            If `unit` is not `'hz'` or `'ppm'`

        Notes
        -----
        If `unit` is set to `'ppm'` and `self.sfo` is not specified
        (`None`), there is no way of retreiving the sweep width in ppm.
        `None` will be returned.
        """

        sw = copy.deepcopy(self.sw)
        if unit == 'hz':
            return sw
        elif unit == 'ppm':
            sfo = self.get_sfo(kill)
            if sfo is None:
                return None
            return self._converter.convert(sw, 'hz->ppm')
        else:
            raise errors.InvalidUnitError('hz', 'ppm')

    def get_offset(self, unit='hz', kill=True):
        """Return the transmitter's offset frequency in each dimesnion.

        Parameters
        ----------
        unit : 'hz' or 'ppm', default: 'hz'

        Returns
        -------
        offset : [float] or [float, float]

        kill : bool, default: True
            If `unit` is `'ppm'`, but `self.sfo` is `None`, `kill` specifies
            how the method will act:

            * If `True`, an AttributeIsNoneError is raised.
            * If `False`, `None` is returned.

        Raises
        ------
        InvalidUnitError
            If `unit` is not `'hz'` or `'ppm'`

        Notes
        -----
        If `unit` is set to `'ppm'` and `self.sfo` is not specified
        (`None`), there is no way of retreiving the offset in ppm.
        `None` will be returned.
        """

        offset = copy.deepcopy(self.offset)
        if unit == 'hz':
            return offset
        elif unit == 'ppm':
            sfo = self.get_sfo(kill)
            if sfo is None:
                return None
            return self._converter.convert(offset, 'hz->ppm')
        else:
            raise errors.InvalidUnitError('hz', 'ppm')

    def get_sfo(self, kill=True):
        """Return transmitter frequency for each channel (MHz).

        Parameters
        ----------
        kill : bool, default: True
            If the path is `None`, `kill` specifies how the method will act:

            * If `True`, an AttributeIsNoneError is raised.
            * If `False`, `None` is returned.

        Returns
        -------
        sfo : [float] or [float, float]
        """

        return self._check_if_none('sfo', kill)

    def get_bf(self, kill=True):
        """Return the transmitter's basic frequency for each channel (MHz).

        Parameters
        ----------
        kill : bool, default: True
            If the path is `None`, `kill` specifies how the method will act:

            * If `True`, an AttributeIsNoneError is raised.
            * If `False`, `None` is returned.

        Returns
        -------
        bf : [float] or [float, float]
        """

        sfo = self._check_if_none('sfo', kill)
        if sfo is None:
            return None

        off = self.get_offset()
        return [s - (o / 1E6) for s, o in zip(sfo, off)]

    def get_nucleus(self, kill=True):
        """Return the target nucleus of each channel.

        Parameters
        ----------
        kill : bool, default: True
            If the path is `None`, `kill` specifies how the method will act:

            * If `True`, an AttributeIsNoneError is raised.
            * If `False`, `None` is returned.

        Returns
        -------
        nuc : [str] or [str, str]
        """

        return self._check_if_none('nuc', kill)

    def get_shifts(self, unit='hz', meshgrid=False, kill=True):
        """Return the sampled frequencies consistent with experiment's
        parameters (sweep width, transmitter offset, number of points).

        Parameters
        ----------
        unit : 'ppm' or 'hz', default: 'ppm'
            The unit of the value(s).

        meshgrid : bool
            Only appicable for 2D data. If set to `True`, the shifts in
            each dimension will be fed into
            `numpy.meshgrid <https://numpy.org/doc/stable/reference/\
            generated/numpy.meshgrid.html>`_

        kill : bool
            If `self.sfo` (need to get shifts in ppm) is `None`, `kill`
            specifies how the method will act:

            * If `True`, an AttributeIsNoneError is raised.
            * If `False`, `None` is returned.

        Returns
        -------
        shifts : [numpy.ndarray] or [numpy.ndarray, numpy.ndarray]
            The frequencies sampled along each dimension.

        Raises
        ------
        InvalidUnitError
            If `unit` is not `'hz'` or `'ppm'`

        Notes
        -----
        The shifts are returned in ascending order.
        """

        if unit not in ['ppm', 'hz']:
            raise errors.InvalidUnitError('ppm', 'hz')

        if unit == 'ppm' and kill is False:
            if self._check_if_none('sfo', kill) is None:
                return None

        shifts = sig.get_shifts(
            self.get_n(), self.get_sw(unit=unit),
            self.get_offset(unit=unit, kill=kill), flip=True,
        )

        if self.get_dim() == 2 and meshgrid:
            return list(np.meshgrid(shifts[0], shifts[1]))

        return shifts

    def get_timepoints(self, meshgrid=False):
        """Return the sampled times consistent with experiment's
        parameters (sweep width, number of points).

        Parameters
        ----------
        meshgrid : bool
            Only appicable for 2D data. If set to `True`, the time-points in
            each dimension will be fed into
            `numpy.meshgrid <https://numpy.org/doc/stable/reference/\
            generated/numpy.meshgrid.html>`_

        Returns
        -------
        tp : [numpy.ndarray] or [numpy.ndarray, numpy.ndarray]
            The times sampled along each dimension (seconds).
        """

        tp = sig.get_timepoints(self.get_n(), self.get_sw())

        if meshgrid and self.get_dim() == 2:
            return list(np.meshgrid(tp[0], tp[1], indexing='ij'))

        return tp

    def get_result(self, kill=True, freq_unit='hz'):
        """Returns the estimation result

        Parameters
        ----------
        kill : bool, default: True
            If `self.result` is `None`, `kill` specifies how the method will
            act:

            * If `True`, an AttributeIsNoneError is raised.
            * If `False`, `None` is returned.

        freq_unit : 'hz' or 'ppm', default: 'hz'
        """
        return self._get_array('result', kill, freq_unit)

    def get_errors(self, kill=True, freq_unit='hz'):
        """Returns the errors of the estimation result derived from
        :py:meth:`nonlinear_programming`

        Parameters
        ----------
        kill : bool, default: True
            If `self.errors` is `None`, `kill` specifies how the method will
            act:

            * If `True`, an AttributeIsNoneError is raised.
            * If `False`, `None` is returned.

        freq_unit : 'hz' or 'ppm', default: 'hz'
        """
        return self._get_array('errors', kill, freq_unit)

    def _get_array(self, name, kill, freq_unit):
        """Returns an array (result or errors), wioth frequencies in either
        Hz or ppm"""

        if name == "result":
            errmsg = "matrix_pencil and/or nonlinear_programming"
        else:
            errmsg = "nonlinear_programming"

        array = copy.deepcopy(self._check_if_none(name, kill, errmsg))

        if freq_unit == 'hz':
            return array

        elif freq_unit == 'ppm':
            # Get frequencies in Hz, and format to enable input into
            # the frequency converter.
            # Then convert values to ppm and reconvert back to NumPy array
            try:
                ppm = np.array(
                    self._converter.convert(
                        [list(array[:, 2])], conversion='hz->ppm',
                    )
                )
                array[:, 2] = ppm
                return array

            except Exception:
                raise TypeError(
                    f'{cols.R}Error in trying to convert frequencies to '
                    f'ppm. Perhaps you didn\'t specify sfo when you made '
                    f'the Estimator instance?{cols.END}'
                )

        else:
            raise InvalidUnitError('hz', 'ppm')

    def _check_if_none(self, name, kill, method=None):
        """Retrieve attributes that may be assigned the value `None`. Return
        None/raise error depending on the value of ``kill``

        Parameters
        ----------
        name : str
            The name of the attribute requested.

        kill : bool
            Whether or not to raise an error if the desired attribute is
            `None`.

        method : str or None, default: None
            The name of the method that needs to be run to obtain the
            desired attribute. If `None`, it implies that the attribute
            requested was never given to the class in the first place.

        Returns
        -------
        attribute : any
            The attribute requested.
        """

        attribute = self.__dict__[name]
        if attribute is None:
            if kill is True:
                raise errors.AttributeIsNoneError(name, method)
            else:
                return None

        else:
            return attribute

    def view_data(self, domain='frequency', freq_xunit='ppm',
                  component='real'):
        """Generate a simple, interactive plot of the data using matplotlib.

        Parameters
        ----------
        domain : 'frequency' or 'time', default: 'frequency'
            The domain of the sig.

        freq_xunit : 'ppm' or 'hz', default: 'ppm'
            The unit of the x-axis, if `domain` is set as `'frequency'`. If
            `domain` is set as `'time'`, the x-axis unit will the seconds.

        component : 'real', 'imag' or 'both', default: 'real'
            The component of the data to display. `'both'` displays both
            the real and imaginary components
        """
        # TODO: 2D equivalent
        dim = self.get_dim()

        if domain == 'time':
            if dim == 1:
                xlabel = '$t\\ (s)$'
                ydata = self.get_data()
                xdata = self.get_timepoints()[0]

            elif dim == 2:
                raise errors.TwoDimUnsupportedError()

        elif domain == 'frequency':
            # frequency domain treatment
            if dim == 1:
                ydata = sig.ft(self.get_data())

                if freq_xunit == 'hz':
                    xlabel = '$\\omega\\ (Hz)$'
                    xdata = self.get_shifts(unit='hz')[0]
                elif freq_xunit == 'ppm':
                    xlabel = '$\\omega\\ (ppm)$'
                    xdata = self.get_shifts(unit='ppm')[0]
                else:
                    msg = f'{R}freq_xunit was not given a valid value' \
                          + f' (should be \'ppm\' or \'hz\').{END}'
                    raise ValueError(msg)

            elif dim == 2:
                raise errors.TwoDimUnsupportedError()

        else:
            msg = f'{R}domain was not given a valid value' \
                  + f' (should be \'frequency\' or \'time\').{END}'
            raise ValueError(msg)

        if component == 'real':
            plt.plot(xdata, np.real(ydata), color='k')
        elif component == 'imag':
            plt.plot(xdata, np.imag(ydata), color='k')
        elif component == 'both':
            plt.plot(xdata, np.real(ydata), color='k', label='Re')
            plt.plot(xdata, np.imag(ydata), color='#808080', label='Im')
            plt.legend()
        else:
            msg = f'{R}component was not given a valid value' \
                  + f' (should be \'real\', \'imag\' or \'both\').{END}'
            raise ValueError(msg)

        if domain == 'frequency':
            plt.xlim(xdata[0], xdata[-1])
        plt.xlabel(xlabel)
        plt.show()

# TODO
# make_fid:
# include functionality to write to Bruker files, Varian files,
# JEOL files etc

    def make_fid(self, n=None, oscillators=None, kill=True):
        """Constructs a synthetic FID using a parameter estimate and
        experiment parameters.

        Parameters
        ----------
        n : [int], or [int, int], or None default: None
            The number of points to construct the FID with in each dimesnion.
            If `None`, :py:meth:`get_n` will be used, meaning the signal will
            have the same number of points as the original data.

        oscillators : None or list, default: None
            Which oscillators to include in result. If `None`, all
            oscillators will be included. If a list of ints, the subset of
            oscillators corresponding to these indices will be used. Note
            that all elements should be in ``range(self.result.shape[0])``.

        kill : bool, default: True
            If `self.result` is `None`, `kill` specifies how the method will
            act:

            * If `True`, an AttributeIsNoneError is raised.
            * If `False`, `None` is returned.

        Returns
        -------
        fid : numpy.ndarray
            The generated FID.

        tp : [numpy.ndarray] or [numpy.ndarray, numpy.ndarray]
            The time-points at which the signal is sampled, in each dimension.

        See Also
        --------
        :py:func:`nmrespy.sig.make_fid`
        """

        result = self.get_result(kill=kill)

        if oscillators is None:
            oscillators = list(range(result.shape[0]))

        if n is None:
            n = self.get_n()

        ArgumentChecker(
            [
                (n, 'n', 'int_list'),
                (oscillators, 'oscillators', 'int_list'),
            ],
            dim=self.get_dim(),
        )

        return sig.make_fid(result[[oscillators]], n, self.get_sw(),
                            offset=self.get_offset())

    @logger
    def phase_data(self, p0=None, p1=None):
        """Phase `self.data`

        Parameters
        ----------
        p0 : [float], [float, float], or None default: None
            Zero-order phase correction in each dimension in radians.
            If `None`, the phase will be set to `0.0` in each dimension.

        p1 : [float], [float, float], or None default: None
            First-order phase correction in each dimension in radians.
            If `None`, the phase will be set to `0.0` in each dimension.
        """

        if p0 is None:
            p0 = self.get_dim() * [0.0]
        if p1 is None:
            p1 = self.get_dim() * [0.0]

        self.data = sig.ift(
            sig.phase(sig.ft(self.data), p0, p1)
        )

    def manual_phase_data(self, max_p1=None):
        """Perform manual phase correction of `self.data`.

        Zero- and first-order phase pharameters are determined via
        interaction with a Tkinter- and matplotlib-based graphical user
        interface.

        Parameters
        ----------
        max_p1 : float or None, default: None
            Specifies the range of first-order phases permitted. For each
            dimension, the user will be allowed to choose a value of `p1`
            within [`-max_p1`, `max_p1`]. By default, `max_p1` will be
            ``10 * numpy.pi``.
        """
        p0, p1 = sig.manual_phase_spectrum(sig.ft(self.data), max_p1)
        if not (p0 is None and p1 is None):
            self.phase_data(p0=[p0], p1=[p1])

    @logger
    def frequency_filter(
        self, region, noise_region, cut=True, cut_ratio=3.0, region_unit='ppm',
    ):
        """Generates frequency-filtered data from `self.data`.

        Parameters
        ----------
        region: [[int, int]], [[int, int], [int, int]], [[float, float]] or\
        [[float, float], [float, float]]
            Cut-off points of the spectral region to consider.
            If the signal is 1D, this should be of the form `[[a,b]]`
            where `a` and `b` are the boundaries.
            If the signal is 2D, this should be of the form
            `[[a,b], [c,d]]` where `a` and `b` are the boundaries in
            dimension 1, and `c` and `d` are the boundaries in
            dimension 2. The ordering of the bounds in each dimension is
            not important.

        noise_region: [[int, int]], [[int, int], [int, int]],\
        [[float, float]] or [[float, float], [float, float]]
            Cut-off points of the spectral region to extract the spectrum's
            noise variance. This should have the same structure as `region`.

        cut : bool, default: True
            If `False`, the filtered signal will comprise the same number of
            data points as the original data. If `True`, prior to inverse
            FT, the data will be sliced, with points not in the region
            specified by `cut_ratio` being removed.

        cut_ratio : float, default: 2.5
            If cut is `True`, defines the ratio between the cut signal's sweep
            width, and the region width, in each dimesnion.
            It is reccommended that this is comfortably larger than `1.0`.
            `2.0` or higher should be appropriate.

        region_unit : 'ppm', 'hz' or 'idx', default: 'ppm'
            The unit the elements of `region` and `noise_region` are
            expressed in.

        Notes
        -----
        This method assigns the attribute `filter_info` to an instance of
        :py:class:`nmrespy.freqfilter.FrequencyFilter`. To obtain information
        on the filtration, use :py:meth:`get_filter_info`.
        """

        self.filter_info = FrequencyFilter(
            self.get_data(), region, noise_region, region_unit=region_unit,
            sw=self.get_sw(), offset=self.get_offset(),
            sfo=self.get_sfo(kill=True), cut=cut, cut_ratio=cut_ratio,
        )

    def get_filter_info(self, kill=True):
        """Returns information relating to frequency filtration.

        Parameters
        ----------
        kill : bool, default: True
            If `filter_info` is `None`, and `kill` is `True`, an error will
            be raised. If `kill` is False, `None` will be returned.

        Returns
        -------
        filter_info : nmrespy.freqfilter.FrequencyFilter

        Notes
        -----
        There are numerous methods associated with `filter_info` for
        obtaining relavent infomation about the filtration. See
        :py:class:`nmrespy.freqfilter.FrequencyFilter` for details.
        """

        return self._check_if_none(
            'filter_info', kill, method='frequency_filter'
        )

    def _get_data_sw_offset(self):
        """Retrieve data, sweep width and offset, based on whether
        frequency filtration have been applied.

        Returns
        -------
        data : numpy.ndarray

        sw : [float] or [float, float]
            Sweep width (Hz).

        offset : [float] or [float, float]
            Transmitter offset (Hz).

        Notes
        -----
        * If `self.filter_info` is equal to `None`, `self.data` will be
          analysed
        * If `self.filter_info` is an instance of
          :py:class:`nmrespy.freqfilter.FrequencyFilter`,
          `self.filter_info.filtered_signal` will be analysed.
        """
        if self.filter_info is not None:
            data = self.filter_info.get_fid()
            sw = self.filter_info.get_sw()
            offset = self.filter_info.get_offset()

        else:
            data = self.get_data()
            sw = self.get_sw()
            offset = self.get_offset()

        return data, sw, offset

    @logger
    def matrix_pencil(self, M=0, trim=None, fprint=True):
        """Implementation of the 1D Matrix Pencil Method [#]_ [#]_ or 2D
        Modified Matrix Enchancement and Matrix Pencil (MMEMP) method [#]_
        [#]_ with the option of Model Order Selection using the Minumum
        Descrition Length (MDL) [#]_.

        Parameters
        ----------
        M : int, default: 0
            The number of oscillators to use in generating a parameter
            estimate. If `M` is set to `0`, the number of oscillators will be
            estimated using the MDL.

        trim : [int], [int, int], or None, default: None
            If `trim` is a list, the analysed data will be sliced such that
            its shape matches `trim`, with the initial points in the signal
            being retained. If `trim` is `None`, the data will not be
            sliced. Consider using this in cases where the full signal is
            large, such that the method takes a very long time, or your PC
            has insufficient memory to process it.

        fprint : bool, default: True
            If `True` (default), the method provides information on
            progress to the terminal as it runs. If `False`, the method
            will run silently.

        Notes
        -----
        The data analysed will be the following:

        * If `self.filter_info` is equal to `None`, `self.data` will be
          analysed
        * If `self.filter_info` is an instance of
          :py:class:`nmrespy.freqfilter.FrequencyFilter`,
          `self.filter_info.filtered_signal` will be analysed.

        **For developers:** See :py:meth:`_get_data_sw_offset`

        Upon successful completion is this method, `self.mpm_info` will
        be updated with an instance of :py:class:`nmrespy.mpm.MatrixPencil`.

        References
        ----------
        .. [#] Yingbo Hua and Tapan K Sarkar. “Matrix pencil method for
           estimating parameters of exponentially damped/undamped sinusoids
           in noise”. In: IEEE Trans. Acoust., Speech, Signal Process. 38.5
           (1990), pp. 814–824.

        .. [#] Yung-Ya Lin et al. “A novel detection–estimation scheme for
           noisy NMR signals: applications to delayed acquisition data”.
           In: J. Magn. Reson. 128.1 (1997), pp. 30–41.

        .. [#] Yingbo Hua. “Estimating two-dimensional frequencies by matrix
           enhancement and matrix pencil”. In: [Proceedings] ICASSP 91: 1991
           International Conference on Acoustics, Speech, and Signal
           Processing. IEEE. 1991, pp. 3073–3076.

        .. [#] Fang-Jiong Chen et al. “Estimation of two-dimensional
           frequencies using modified matrix pencil method”. In: IEEE Trans.
           Signal Process. 55.2 (2007), pp. 718–724.

        .. [#] M. Wax, T. Kailath, Detection of signals by information
           theoretic criteria, IEEE Transactions on Acoustics, Speech, and
           Signal Processing 33 (2) (1985) 387–392.
        """

        data, sw, offset = self._get_data_sw_offset()

        if trim is None:
            trim = [s for s in data.shape]

        ArgumentChecker([(trim, 'trim', 'int_list')], dim=self.dim)

        trim = tuple(np.s_[0:t] for t in trim)
        # Slice data
        data = data[trim]

        mpm_info = MatrixPencil(
            data, sw, offset, self.sfo, M, fprint
        )

        self.result = mpm_info.get_result()
        self.errors = None
        self._saveable = False

    # TODO: support for mode
    # Also look at nlp.nlp.NonlinearProgramming

    @logger
    def nonlinear_programming(self, trim=None, **kwargs):
        """Estimation of signal parameters using nonlinear programming, given
        an inital guess.

        Parameters
        ----------
        trim : None, [int], or [int, int], default: None
            If `trim` is a list, the analysed data will be sliced such that
            its shape matches `trim`, with the initial points in the signal
            being retained. If `trim` is `None`, the data will not be
            sliced. Consider using this in cases where the full signal is
            large, such that the method takes a very long time, or your PC
            has insufficient memory to process it.

        **kwargs
            Properties of :py:class:`nmrespy.nlp.nlp.NonlinearProgramming`.
            Valid arguments:

            * `phase_variance`
            * `method`
            * `bound`
            * `max_iterations`
            * `amp_thold`
            * `freq_thold`
            * `negative_amps`
            * `fprint`

            Other keyword arguments that are valid in
            :py:func:`nmrespy.nlp.nlp.NonlinearProgramming` will be ignored
            (these are generated internally by the class instance).

        Raises
        ------
        PhaseVarianceAmbiguityError
            Raised when ``phase_variance`` is set to ``True``, but the user
            has specified that they do not wish to optimise phases using the
            ``mode`` argument.

        Notes
        -----
        The data analysed will be the following:

        * If `self.filter_info` is equal to `None`, `self.data` will be
          analysed
        * If `self.filter_info` is an instance of
          :py:class:`nmrespy.freqfilter.FrequencyFilter`,
          `self.filter_info.filtered_signal` will be analysed.

        Upon successful completion is this method, `self.result` and
        `self.errors` will be updated.

        See Also
        --------
        :py:class:`nmrespy.nlp.nlp.NonlinearProgramming`
        """

        # TODO: include freq threshold

        data, sw, offset = self._get_data_sw_offset()

        if trim is None:
            trim = list(data.shape)

        ArgumentChecker([(trim, 'trim', 'int_list')], dim=self.dim)

        # Slice data
        data = data[tuple(np.s_[0:t] for t in trim)]

        x0 = self.get_result()

        kwargs['sfo'] = self.get_sfo()
        kwargs['offset'] = offset

        # nlp_info = NonlinearProgramming(data, x0, sw, **kwargs)
        nlp_info = NonlinearProgramming(data, x0, sw, **kwargs)

        self.result = nlp_info.get_result()
        self.errors = nlp_info.get_errors()
        self._saveable = True

    @logger
    def write_result(self, **kwargs):
        """Saves an estimation result to a file in a human-readable format
        (text, PDF, CSV).

        Parameters
        ----------
        kwargs : Properties of :py:func:`nmrespy.write.write_result`.
            Valid arguments are:

            * `path`
            * `description`
            * `sig_figs`
            * `sci_lims`
            * `fmt`
            * `force_overwrite`
            * `fprint`

            Other keyword arguments that are valid in
            :py:func:`nmrespy.write.write_result` will be ignored (these are
            generated internally by the class instance).

        Raises
        ------
        AttributeIsNoneError
            If no parameter estimate derived from nonlinear programming
            is found (see :py:meth:`nonlinear_programming`).

        See Also
        --------
        :py:func:`nmrespy.write.write_result`
        """

        # Retrieve result
        # If self.result is None, an error will be raised inside
        # _check_if_none
        result = self.get_result()

        if not self._saveable:
            raise ValueError(
                f'{cols.OR}The last action to be applied to the estimation '
                f'result was not `nonlinear_programming`. You should ensure '
                f'this is so before saving the result.{cols.END}'
            )

        # Remove any invalid arguments from kwargs (avoid repetition
        # in call to nmrespy.write.write_result)
        for key in ['sfo', 'integrals', 'info', 'info_headings']:
            try:
                kwargs.pop(key)
            except KeyError:
                pass

        errors = self.get_errors()

        # Information for experiment info
        sw_h = self.get_sw()
        sw_p = self.get_sw(unit='ppm')
        off_h = self.get_offset()
        off_p = self.get_offset(unit='ppm')
        sfo = self.get_sfo(kill=False)
        bf = self.get_bf(kill=False)
        nuc = self.get_nucleus(kill=False)
        filter = self.get_filter_info(kill=False)

        if filter is not None:
            region_h = filter.get_region(unit='hz')
            region_p = filter.get_region(unit='ppm')

        # Peak integrals
        integrals = [
            sig.oscillator_integral(osc, self.get_n(), sw_h, offset=off_h)
            for osc in result
        ]

        # --- Package experiment information ----------------------------
        info_headings = []
        info = []
        sigfig = 6
        if self.get_dim() == 1:
            # Sweep width
            info_headings.append('Sweep Width (Hz)')
            info.append(str(significant_figures(sw_h[0], sigfig)))
            if sw_p is not None:
                info_headings.append('Sweep Width (ppm)')
                info.append(str(significant_figures(sw_p[0], sigfig)))

            # Offset
            info_headings.append('Transmitter Offset (Hz)')
            info.append(str(significant_figures(off_h[0], sigfig)))
            if off_p is not None:
                info_headings.append('Transmitter Offset (ppm)')
                info.append(str(significant_figures(off_p[0], sigfig)))

            # Transmitter frequency
            if sfo is not None:
                info_headings.append('Transmitter Frequency (MHz)')
                info.append(str(significant_figures(sfo[0], sigfig)))

            # Basic frequency
            if bf is not None:
                info_headings.append('Basic Frequency (MHz)')
                info.append(str(significant_figures(bf[0], sigfig)))

            # Nuclei
            if nuc is not None:
                info_headings.append('Nucleus')
                if 'fmt' in kwargs.keys() and kwargs['fmt'] == 'pdf':
                    info.append(latex_nucleus(nuc[0]))
                else:
                    info.append(nuc[0])

            # Region
            try:
                info.append(
                    f'{significant_figures(region_h[0][0], sigfig)} -'
                    f' {significant_figures(region_h[0][1], sigfig)}'
                )
                info.append(
                    f'{significant_figures(region_p[0][0], sigfig)} -'
                    f' {significant_figures(region_p[0][1], sigfig)}'
                )
                info_headings.append('Filter region (Hz):')
                info_headings.append('Filter region (ppm):')

            except NameError:
                pass

        # TODO
        elif self.get_dim() == 2:
            raise TwoDimUnsupportedError()

        write_result(result, errors=errors, integrals=integrals,
                     info_headings=info_headings, info=info, sfo=sfo,
                     **kwargs)

    @logger
    def plot_result(self, **kwargs):
        """Produces a figure of an estimation result.

        The figure consists of the original data, in the Fourier domain,
        along with each oscillator.

        Parameters
        ----------
        kwargs : Properties of :py:func:`nmrespy.write.write_result`.
            Valid arguments are:

            * `shifts_unit`
            * `plot_residual`
            * `plot_model`
            * `residual_shift`
            * `model_shift`
            * `data_color`
            * `oscillator_colors`
            * `residual_color`
            * `model_color`
            * `labels`
            * `stylesheet`

            Other keyword arguments that are valid in
            :py:func:`nmrespy.plot.plot_result` will be ignored (these are
            generated internally by the class instance).

        Raises
        ------
        AttributeIsNoneError
            If no parameter estimate derived from nonlinear programming
            is found (see :py:meth:`nonlinear_programming`).

        See Also
        --------
        :py:func:`nmrespy.plot.plot_result`
        """

        result = self.get_result()

        if not self._saveable:
            raise ValueError(
                f'{cols.OR}The last action to be applied to the estimation '
                f'result was not `nonlinear_programming`. You should ensure '
                f'this is so before plotting the result.{cols.END}'
            )

        dim = self.get_dim()
        # Check dim is valid (only 1D data supported so far)
        if dim == 2:
            raise TwoDimUnsupportedError()

        for key in ['sfo', 'nuceleus', 'region']:
            try:
                kwargs.pop(key)
            except KeyError:
                pass

        try:
            unit = kwargs['shifts_unit']
        except KeyError:
            kwargs['shifts_unit'] = unit = 'ppm'

        return plot_result(
            self.get_data(), result, self.get_sw(),
            self.get_offset(), sfo=self.get_sfo(kill=False),
            nucleus=self.get_nucleus(kill=False),
            region=self.get_filter_info().get_region(unit=unit), **kwargs,
        )

    @logger
    def add_oscillators(self, oscillators):
        """Adds new oscillators an estimation result.

        Parameters
        ----------
        oscillators : numpy.ndarray
            An array of the new oscillator(s) to add to the array.
            *NB* `oscillators` should always be a two-dimensional array,
            even if only one oscillator is being added:

            .. code:: python3

               >>> oscillators = np.array([[a, φ, f, η]]) # 1D
               >>> oscillators = np.array([[a, φ, f1, f2, η1, η2]]) # 2D
               >>> # or, equivalently:
               >>> oscillators = np.insert_axis(
               ...      np.array([a, φ, f, η]), axis=1
               ... ) # 1D
               >>> oscillators = np.insert_axis(
               ...      np.array([a, φ, f1, f2, η1, η2]), axis=1
               ... ) # 2D
        """

        ArgumentChecker(
            [(oscillators, 'oscillators', 'parameter')], self.get_dim(),
        )

        result = self.get_result()
        new_result = np.vstack((result, oscillators))
        # Order according to frequency
        new_result = new_result[np.argsort(new_result[:, 2])]
        # Assign new result to nlp_info
        self.result = new_result
        # User has manually edited the result after estimation.
        self._saveable = False
        self.errors = None

    @logger
    def remove_oscillators(self, indices):
        """Removes the oscillators corresponding to ``indices``.

        Parameters
        ----------
        indices : list
            A list of indices corresponding to the oscillators to be
            removed. The elements of `indices` should be ints that
            are in ``range(result.shape[0])``, where `result` is the
            current estimation result.
        """

        ArgumentChecker([(indices, 'indices', 'list')])

        result = self.get_result()
        for i in indices:
            if i not in list(range(result.shape[0])):
                raise ValueError(
                    f'{cols.R}Invalid index in `indices`{cols.END}'
                )

        self.result = np.delete(result, indices, axis=0)
        # User has manually edited the result after estimation.
        self._saveable = False
        self.errors = None

    @logger
    def merge_oscillators(self, indices):
        """Merges the oscillators corresponding to `indices`.

        Removes the osccilators specified, and constructs a single new
        oscillator with a cumulative amplitude, and averaged phase,
        frequency and damping.

        Parameters
        ----------
        indices : list, tuple or numpy.ndarray
            A list of indices corresponding to the oscillators to be
            merged. The elements of `indices` should be ints that
            are in ``range(result.shape[0])``, where `result` is the
            current estimation result.

        Notes
        -----
        Assuming that an estimation result contains a subset of oscillators
        denoted by indices :math:`\\{m_1, m_2, \\cdots, m_J\\}`, where
        :math:`J \\leq M`, the new oscillator formed by the merging of the
        oscillator subset will possess the following parameters:

            * :math:`a_{\\mathrm{new}} = \\sum_{i=1}^J a_{m_i}`
            * :math:`\\phi_{\\mathrm{new}} = \\frac{1}{J} \\sum_{i=1}^J
              \\phi_{m_i}`
            * :math:`f_{\\mathrm{new}} = \\frac{1}{J} \\sum_{i=1}^J f_{m_i}`
            * :math:`\\eta_{\\mathrm{new}} = \\frac{1}{J} \\sum_{i=1}^J
              \\eta_{m_i}`
        """

        ArgumentChecker([(indices, 'indices', 'list')])

        result = self.get_result()

        if len(indices) < 2:
            print(
                f'\n{cols.OR}`indices` should contain at least two elements.'
                f'No merging will happen.{cols.END}'
            )
            return

        for i in indices:
            if i not in list(range(result.shape[0])):
                raise ValueError(
                    f'{cols.R}Invalid index in `indices`{cols.END}'
                )

        to_merge = result[indices]
        # Sum amps, phases, freqs and damping over the oscillators
        # to be merged.
        # keepdims ensures that the final array is [[a, φ, f, η]]
        # rather than [a, φ, f, η]
        new_osc = np.sum(to_merge, axis=0, keepdims=True)

        # Get mean for phase, frequency and damping
        new_osc[:, 1:] = new_osc[:, 1:] / float(len(indices))
        # wrap phase
        new_osc[:, 1] = (new_osc[:, 1] + np.pi) % (2 * np.pi) - np.pi

        result = np.delete(result, indices, axis=0)
        result = np.vstack((result, new_osc))
        self.result = result[np.argsort(result[..., 2])]

        # User has manually edited the result after estimation.
        self._saveable = False
        self.errors = None

    # TODO make 2D compatible
    @logger
    def split_oscillator(
        self, index, separation_frequency=None, unit='hz', split_number=2,
        amp_ratio=None,
    ):
        """Splits the oscillator corresponding to `index`.

        Removes an oscillator, and incorporates two or more oscillators
        whose cumulative amplitudes match that of the removed oscillator.

        Parameters
        ----------
        index : int
            Array index of the oscilator to be split.

        separation_frequency : float, or None default: None
            The frequency separation given to adjacent oscillators formed
            from the splitting. If `None`, the splitting will be set to
            ``sw / n`` where `sw` is the sweep width and `n` is the number
            of points in the data.

        unit : 'hz' or 'ppm', default: 'hz'
            The unit of `separation_frequency`.

        split_number: int, default: 2
            The number of peaks to split the oscillator into.

        amp_ratio: list or None, default: None
            The ratio of amplitudes to be fulfilled by the newly formed
            peaks. If a list, ``len(amp_ratio) == split_number`` must be
            satisfied. The first element will relate to the highest
            frequency oscillator constructed, and the last element will
            relate to the lowest frequency oscillator constructed. If `None`,
            all oscillators will be given equal amplitudes.
        """

        # get separation_frequency in correct units
        if unit not in ['hz', 'ppm']:
            raise errors.InvalidUnitError('hz', 'ppm')

        result = self.get_result(freq_unit=unit)

        try:
            # Of form: [a, φ, f, η] (i.e. 1D array)
            osc = result[index]
        except Exception:
            raise ValueError(
                f'{cols.R}index should be an int in range('
                f'{result.shape[0]}){cols.END}'
            )

        # --- Determine frequencies --------------------------------------
        if separation_frequency is None:
            separation_frequency = self.get_sw()[0] / self.get_n()[0]
            if unit == 'ppm':
                separation_frequency = separation_frequency / self.get_sfo()[0]

        # Highest frequency of all the new oscillators
        max_freq = osc[2] + ((split_number - 1) * separation_frequency / 2)
        # Array of all frequencies (lowest to highest)
        freqs = [max_freq - i * separation_frequency
                 for i in range(split_number)]

        # --- Determine amplitudes ---------------------------------------
        if amp_ratio is None:
            amp_ratio = [1.] * split_number

        if not isinstance(amp_ratio, list):
            raise TypeError(
                f'{cols.R}`amp_ratio` should be None or a list{cols.END}'
            )

        if len(amp_ratio) != split_number:
            raise ValueError(
                f'{cols.R}The length of `amp_ratio` should equal'
                f' `split_number`{cols.END}'
            )

        # Scale amplitude ratio values such that their sum is 1
        amp_ratio = np.array(amp_ratio)
        amp_ratio = amp_ratio / np.sum(amp_ratio)

        # Obtain amplitude values
        amps = osc[0] * amp_ratio

        # --- Generate array of new oscillators --------------------------
        new_oscs = np.zeros((split_number, 2 * (self.get_dim() + 1)))
        new_oscs[:, 0] = amps
        new_oscs[:, 2] = freqs
        new_oscs[:, 1] = [osc[1]] * split_number
        new_oscs[:, 3] = [osc[3]] * split_number

        result = np.delete(result, index, axis=0)
        result = np.append(result, new_oscs, axis=0)

        if unit == 'ppm':
            result[:, 2] = result[:, 2] * self.get_sfo()[0]

        self.result = result[np.argsort(result[..., 2])]

        # User has manually edited the result after estimation.
        self._saveable = False
        self.errors = None

    def save_logfile(self, path='./nmrespy_log', force_overwrite=False):
        """Saves log file of class instance usage to a specified path.

        Parameters
        ----------
        path : str, default: './nmrespy_log'
            The path to save the file to. DO NOT INCLUDE A FILE EXTENSION.
            `.log` will be added automatically.

        force_overwrite : bool. default: False
            Defines behaviour if ``f'{path}.log'`` already exists:

            * If `force_overwrite` is set to `False`, the user will be prompted
              if they are happy overwriting the current file.
            * If `force_overwrite` is set to `True`, the current file will be
              overwritten without prompt.
        """

        ArgumentChecker(
            [
                (path, 'path', 'str'),
                (force_overwrite, 'force_overwrite', 'bool'),
            ]
        )

        # Get full path and extend .log extension
        path = Path(path).resolve().with_suffix('.log')
        # Check path is valid (check directory exists, ask user if they are
        # happy overwriting if file already exists).
        pathres = PathManager(
            path.name, path.parent
        ).check_file(force_overwrite)
        # Valid path, we are good to proceed
        if pathres == 0:
            pass
        # Overwrite denied by the user. Exit the program
        elif pathres == 1:
            exit()
        # pathres == 2: Directory specified doesn't exist
        else:
            raise ValueError(
                f'{cols.R}The directory implied by path does not'
                f' exist{cols.END}'
            )

        try:
            with open(path, "w") as fh:
                fh.write(self._log)
            print(
                f'{cols.G}Log file succesfully saved to'
                f' {str(path)}{cols.END}'
            )

        # trouble writing to file...
        except Exception as e:
            raise e
