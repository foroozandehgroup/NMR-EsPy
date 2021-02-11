from copy import deepcopy
import datetime
import functools
import inspect
import itertools
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
from scipy.integrate import simps

from nmrespy import *
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
import nmrespy._errors as errors

from nmrespy._misc import ArgumentChecker, FrequencyConverter
from nmrespy.filter import FrequencyFilter
import nmrespy.load as load
from . import signal, load, filter, mpm, nlp, _plot, _write

# Wrapper for logging method calls
# A file is generated and placed in
def logger(f):
    """Decorator for logging :py:class:`Estimator` method calls"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        # The first arg is the class instance. Get the path to the logfile.
        path = args[0]._logpath
        with open(path, 'a') as fh:
            # Append the method call to the log file in the following format:
            # --> method_name (args) {kwargs}
            fh.write(f'--> {f.__name__} {args[1:]} {kwargs}\n')

        # Run the method...
        return f(*args, **kwargs)

    return wrapper


class Estimator:
    """Estimation class

    .. note::
       The methods :py:meth:`new_bruker`, :py:meth:`new_synthetic_from_data`
       and :py:meth:`new_synthetic_from_parameters` generate instances
       of the class. While you can manually input the listed parameters
       as arguments to initialise the class. It is more straightforward
       to use one of these.nmrespy

    Parameters
    ----------
    source : 'bruker_fid', 'bruker_pdata', 'synthetic'
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

    off : [float] or [float, float]
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
    """

    def new_bruker(dir, ask_convdta=True):
        """Generate an instance of :py:class:`Estimator` from a
        Bruker-formatted data directory.

        Parameters
        ----------
        dir : str
            The path to the data containing the data of interest.

        ask_convdta : bool
            See :py:meth:`nmrespy.load.import_bruker`

        Returns
        -------
        estimator : :py:meth:`nmrespy.core.Estimator`

        Notes
        -----
        For a more detailed specification of the directory requirements,
        see :py:meth:`nmrespy.load.import_bruker`
        """

        info = load.import_bruker(dir, ask_convdta=ask_convdta)

        return Estimator(
            info['source'], info['data'], info['directory'],
            info['sweep_width'], info['offset'], info['transmitter_frequency'],
            info['nuclei'], info['binary_format'],
        )

    def new_synthetic_from_data(data, sw, offset=None, sfo=None):

        # --- Check validity of parameters -------------------------------
        # Check data is a numpy array
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f'{cols.R}data should be a numpy ndarray{cols.END}'
            )
        # Determine data dimension. If greater than 2, return error.
        if data.ndim >= 3:
            raise errors.MoreThanTwoDimError()

        dim = data.ndim

        # If offset is None, set it to zero in each dimension
        if offset == None:
            offset = [0.0] * dim

        components = [
            (sw, 'sw', 'float_list'),
            (offset, 'offset', 'float_list'),
        ]

        if sfo != None:
            components.append((sfo, 'sfo', 'float_list'))

        ArgumentChecker(components, dim)

        return Estimator(
            'synthetic', data, None, sw, offset, sfo, None, None,
        )


    def new_synthetic_from_parameters(
        parameters, sw, points, snr=None, offset=None, sfo=None
    ):
        """
        .. todo::
        """
        print(f'{cols.O}new_synthetic_from_parameters is not yet'
              f'implemented!{cols.END}')


    def __init__(self, source, data, path, sw, off, sfo, nuc, fmt):

        self.source = source
        self.data = data
        self.dim = self.data.ndim
        self.n = [int(n) for n in self.data.shape]
        self.path = path
        self.sw = sw
        self.off = off
        self.sfo = sfo
        self.nuc = nuc
        self.fmt = fmt

        # Attributes that info will be assigned to after the user runs
        # the folowing methods:
        # 1. frequency_filter (filter_info)
        # 2. matrix_pencil (mpm_info)
        # 3. nonlinear_programming (nlp_info)
        self._filter_info = None
        self._mpm_info = None
        self._nlp_info = None

        if self.sfo != None:
            self._converter = FrequencyConverter(
                list(data.shape), self.sw, self.off, self.sfo
            )

        # --- Create file for logging method calls -----------------------
        # Set path of file to be inside the nmrespy/logs directory
        # File name is a timestamp: yyyymmddHHMMSS.log
        now = datetime.datetime.now()
        self._logpath = Path(NMRESPYPATH) / \
                       f"logs/{now.strftime('%y%m%d%H%M%S')}.log"

        # Add a header to the log file
        header = (
            '==============================\n'
            'Logfile for Estimator instance\n'
            '==============================\n'
           f"--> Instance created @ {now.strftime('%d-%m-%y %H:%M:%S')}\n"
        )
        with open(self._logpath, 'w') as fh:
            fh.write(header)


    def __repr__(self):
        msg = (
            f'nmrespy.core.Estimator('
            f'{self.source}, '
            f'{self.data}, '
            f'{self.path}, '
            f'{self.sw}, '
            f'{self.off}, '
            f'{self.n}, '
            f'{self.sfo}, '
            f'{self.nuc}, '
            f'{self.fmt})'
        )

        return msg

    def __str__(self):
        """A formatted list of class attributes"""
        name = str(self.__class__).replace('<class \'', '').replace('\'>', '')
        msg = msg = f"{MA}<{name} at {hex(id(self))}>{END}\n"
        dic = self.__dict__
        for key, value in zip(dic.keys(), dic.values()):
            msg += f'{cols.MA}{key}{cols.END} : {value}\n'

        return msg

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
            return str(path) if path != None else None
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
        dim : 1, 2
        """
        return self.dim

    def get_n(self):
        """Return the number of datapoints in each dimension"""
        return self.n

    def get_sw(self, unit='hz'):
        """Return the experiment sweep width in each dimension.

        Parameters
        ----------
        unit : 'hz' or 'ppm', default: 'hz'

        Returns
        -------
        sw : [float] or [float, float]

        Raises
        ------
        InvalidUnitError
            If `unit` is not `'hz'` or `'ppm'`
        """

        if unit == 'hz':
            return self.sw
        elif unit == 'ppm':
            return self._unit_convert(self.sw, convert='hz->ppm')
        else:
            raise errors.InvalidUnitError('hz', 'ppm')

    def get_offset(self, unit='hz'):
        """Return the transmitter's offset frequency in each dimesnion.

        Parameters
        ----------
        unit : 'hz' or 'ppm', default: 'hz'

        Returns
        -------
        offset : [float] or [float, float]

        Raises
        ------
        InvalidUnitError
            If `unit` is not `'hz'` or `'ppm'`
        """

        if unit == 'hz':
            return self.off
        elif unit == 'ppm':
            return self._unit_convert(self.off, convert='hz->ppm')
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
        if sfo == None:
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
            each dimension will be fed into `numpy.meshgrid <https://numpy.org/\
            doc/stable/reference/generated/numpy.meshgrid.html>`_

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

        shifts = signal.get_shifts(
            self.get_n(), self.get_sw(), self.get_offset()
        )

        if unit == 'ppm':
            sfo = self.get_sfo(kill)
            if sfo == None:
                return None
            shifts = [shifts_ / sfo for shifts_ in shifts]

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
            each dimension will be fed into `numpy.meshgrid <https://numpy.org/\
            doc/stable/reference/generated/numpy.meshgrid.html>`_

        Returns
        -------
        tp : [numpy.ndarray] or [numpy.ndarray, numpy.ndarray]
            The times sampled along each dimension (seconds).
        """

        tp = signal.get_timepoints(self.get_n(), self.get_sw())
        if meshgrid and self.get_dim() == 2:
            return list(np.meshgrid(tp[0], tp[1]))
        return tp


    def get_region(self, unit='idx', kill=True):
        """Return the region of interest.

        Parameters
        ----------
        unit : 'idx', 'hz', 'ppm', default: 'idx'
            The unit of the elements.

        kill : bool, default: True
            If ``self.region`` is ``None``, ``kill`` specifies
            how to act:

            * If ``True``, an AttributeIsNoneError is raised.
            * If ``False``, ``None`` is returned.

        Returns
        -------
        region : [[float,float]], [[int,int]],
                 [[float, float],[float, float]], [[int, int],[int, int]],
                 or None

        Raises
        ------
        AttributeIsNoneError
            If ``self.region`` is ``None``, and ``kill`` is ``True``.
        InvalidUnitError
            If ``unit`` is not ``'idx'``, ``'hz'``, or ``'ppm'``.

        Notes
        -----
        If ``self.region`` is ``None``, it is likely that
        :py:meth:`frequency_filter` is yet to be called on the class instance.
        """

        return self._get_region('region', unit, kill)


    def get_noise_region(self, unit='idx', kill=True):
        """Return the noise region.

        Parameters
        ----------
        unit : 'idx', 'hz', 'ppm', default: 'idx'
            The unit of the elements.

        kill : bool, default: True
            If ``self.region`` is ``None``, ``kill`` specifies
            how to act:

            * If ``kill`` is ``True``, an error is raised.
            * If ``kill`` is ``False``, ``None`` is returned.

        Returns
        -------
        noise_region : [[float,float]], [[int,int]],
                 [[float, float],[float, float]], [[int, int],[int, int]],
                 or None

        Raises
        ------
        AttributeIsNoneError
            If ``self.noise_region`` is ``None``, and ``kill`` is ``True``.
        InvalidUnitError
            If ``unit`` is not ``'idx'``, ``'hz'``, or ``'ppm'``.

        Notes
        -----
        If ``self.noise_region`` is ``None``, it is likely that
        :py:meth:`frequency_filter` is yet to be called on the class instance.
        """

        return self._get_region('noise_region', unit, kill)


    def _get_region(self, name, unit, kill):
        """Called within :py:meth:`get_region` and :py:meth:`get_noise_region`.
        Retrieves the desired region in the desired unit.
        """

        region = self._get_nondefault_param(
            name, 'frequency_filter()', kill
        )

        if unit == 'idx':
            return region

        elif unit in ['hz', 'ppm']:
            return self._unit_convert(region, convert=f'idx->{unit}')

        else:
            raise errors.InvalidUnitError('idx', 'hz', 'ppm')


    def get_p0(self, unit='radians', kill=True):
        """Return the zero-order phase correction specified using
        :py:meth:`frequency_filter`.

        Parameters
        ----------
        unit : 'radians', 'degrees', default: 'rad'
            The unit the phase is expressed in.

        kill : bool, default: True
            If ``self.p0`` is ``None``, ``kill`` specifies how to act:

            * If ``kill`` is ``True``, an error is raised.
            * If ``kill`` is ``False``, ``None`` is returned.

        Returns
        -------
        p0 : float or None

        Raises
        ------
        AttributeIsNoneError
            Raised if ``p0`` is ``None``, and ``kill`` is ``True``
        InvalidUnitError
            If ``unit`` is not ``'radians'`` or ``'degrees'``

        Notes
        -----
        If ``self.p0`` is ``None``, it is likely that :py:meth:`frequency_filter`
        is yet to be called on the class instance.
        """

        return self._get_phase('p0', unit, kill)


    def get_p1(self, unit='radians', kill=True):
        """Return the first-order phase correction specified using
        :py:meth:`frequency_filter`.

        Parameters
        ----------
        unit : 'radians', 'degrees', default: 'radians'
            The unit the phase is expressed in.

        kill : bool, default: True
            If ``self.p1`` is ``None``, `kill` specifies how to act:

            * If ``kill`` is ``True``, an error is raised.
            * If ``kill`` is ``False``, ``None`` is returned.

        Returns
        -------
        p1 : float or None

        Raises
        ------
        AttributeIsNoneError
            Raised if ``self.p1`` is ``None``, and ``kill`` is ``True``.
        InvalidUnitError
            If ``unit`` is not ``'radians'`` or ``'degrees'``

        Notes
        -----
        If ``self.p1`` is ``None``, it is likely that :py:meth:`frequency_filter`
        is yet to be called on the class instance.
        """

        return self._get_phase('p1', unit, kill)


    def _get_phase(self, name, unit, kill):
        """Called within :py:meth:`get_p0` and :py:meth:`get_p1`.
        Retrieves the desired region in the desired unit.
        """

        if unit == 'radians':
            return self._get_nondefault_param(name, 'frequency_filter()', kill)

        elif unit == 'degrees':
            return self._get_nondefault_param(name, 'frequency_filter()', kill) \
                   * (180 / np.pi)

        else:
            raise errors.InvalidUnitError('radians', 'degrees')


    def get_filtered_spectrum(self, kill=True):
        """Return the filtered spectral data generated using
        :py:meth:`frequency_filter`.

        Parameters
        ----------
        kill : bool, default: True
            If ``self.filtered_spectrum`` is ``None``, `kill` specifies how to act:

            * If ``kill`` is ``True``, an error is raised.
            * If ``kill`` is ``False``, ``None`` is returned.

        Returns
        -------
        filtered_spectrum : numpy.ndarray or None

        Raises
        ------
        AttributeIsNoneError
            Raised if ``self.filtered_spectrum`` is ``None``, and ``kill`` is ``True``.

        Notes
        -----
        If ``self.filtered_spectrum`` is ``None``, it is likely that
        :py:meth:`frequency_filter` is yet to be called on the class instance.
        """

        return self._get_nondefault_param(
            'filtered_spectrum', 'frequency_filter()', kill
        )


    def get_filtered_signal(self, kill=True):
        """Return the filtered time-domain data generated using
        :py:meth:`frequency_filter`.

        Parameters
        ----------
        kill : bool, default: True
            If ``self.filtered_signal`` is ``None``, `kill` specifies how to act:

            * If ``kill`` is ``True``, an error is raised.
            * If ``kill`` is ``False``, ``None`` is returned.

        Returns
        -------
        filtered_spectrum : numpy.ndarray or None

        Raises
        ------
        AttributeIsNoneError
            Raised if ``self.filtered_signal`` is ``None``, and ``kill`` is ``True``.

        Notes
        -----
        If ``self.filtered_signal`` is ``None``, it is likely that
        :py:meth:`frequency_filter` is yet to be called on the class instance.
        """

        return self._get_nondefault_param(
            'filtered_signal', 'frequency_filter()', kill
        )


    def get_filtered_n(self, kill=True):
        """Return the number of points of the signal generated using
        :py:meth:`frequency_filter` with ``cut = True``, in each dimesnion.

        Parameters
        ----------
        kill : bool, default: True
            If ``self.n`` is ``None``, `kill` specifies how to act:

            * If ``kill`` is ``True``, an error is raised.
            * If ``kill`` is ``False``, ``None`` is returned.

        Returns
        -------
        filtered_n : (int,), (int, int), or None

        Raises
        ------
        AttributeIsNoneError
            Raised if ``self.n`` is ``None``, and ``kill`` is ``True``.

        Notes
        -----
        If ``self.n`` is ``None``, it is likely that
        :py:meth:`frequency_filter`, with ``cut = True`` has not been called on
        the class instance.
        """

        return self._get_nondefault_param(
            'filtered_n', 'frequency_filter() with cut=True', kill
        )


    def get_filtered_sw(self, unit='hz', kill=True):
        """Return the sweep width of the signal generated using
        :py:meth:`frequency_filter` with ``cut = True``, in each dimesnion.

        Parameters
        ----------
        unit : 'hz', 'ppm', default: 'hz'
            The unit of the value(s).

        kill : bool, default: True
            If ``self.filtered_sw`` is ``None``, `kill` specifies how to act:

            * If ``kill`` is ``True``, an error is raised.
            * If ``kill`` is ``False``, ``None`` is returned.

        Returns
        -------
        filtered_sw : (float,), (float, float), or None

        Raises
        ------
        AttributeIsNoneError
            Raised if ``self.filtered_sw`` is ``None``, and ``kill`` is ``True``.
        InvalidUnitError
            If ``unit`` is not ``'hz'`` or ``'ppm'``.

        Notes
        -----
        If ``self.filtered_sw`` is ``None``, it is likely that
        :py:meth:`frequency_filter`, with ``cut = True`` has not been called on the
        class instance.
        """

        filtered_sw =  self._get_nondefault_param(
            'filtered_sw', 'frequency_filter() with cut=True', kill
        )

        if unit == 'hz':
            return filtered_sw

        elif unit == 'ppm':
            return self._unit_convert(filtered_sw, convert='hz->ppm')

        else:
            raise errors.InvalidUnitError('hz', 'ppm')


    def get_filtered_offset(self, unit='hz', kill=True):
        """Return the transmitter's offest frequency of the signal generated
        using :py:meth:`frequency_filter` with ``cut = True``, in each
        dimesnion.

        Parameters
        ----------
        unit : 'hz', 'ppm', default: 'hz'
            The unit of the value(s).

        kill : bool, default: True
            If ``self.filtered_offset`` is ``None``, `kill` specifies how to act:

            * If ``kill`` is ``True``, an error is raised.
            * If ``kill`` is ``False``, ``None`` is returned.

        Returns
        -------
        filtered_offset : (float,), (float, float), or None

        Raises
        ------
        AttributeIsNoneError
            Raised if ``self.filtered_offset`` is ``None``, and ``kill`` is ``True``.
        InvalidUnitError
            If `unit` is not 'hz' or 'ppm'

        Notes
        -----
        If ``self.filtered_offset`` is ``None``, it is likely that
        :py:meth:`frequency_filter`, with ``cut = True`` has not been called on the
        class instance.
        """

        filtered_offset =  self._get_nondefault_param(
            'filtered_offset', 'frequency_filter() with cut=True', kill
        )

        if unit == 'hz':
            return filtered_offset

        elif unit == 'ppm':
            return self._unit_convert(filtered_offset, convert='hz->ppm')

        else:
            raise errors.InvalidUnitError('hz', 'ppm')


    def get_theta0(self, kill=True):
        """Return the parameter estimate derived using
        :py:meth:`matrix_pencil`

        Parameters
        ----------
        kill : bool, default: True
            If ``self.theta0`` is ``None``, `kill` specifies how to act:

            * If ``kill`` is ``True``, an error is raised.
            * If ``kill`` is ``False``, ``None`` is returned.

        Returns
        -------
        theta0 : numpy.ndarray or None
            An array with ``theta0.shape[1] = 4`` (1D signal) or
            ``theta0.shape[1] = 6`` (2D signal).

        Raises
        ------
        AttributeIsNoneError
            Raised if ``self.theta0`` is ``None``, and ``kill`` is ``True``.

        Notes
        -----
        If ``self.theta0`` is ``None``, it is likely that
        :py:meth:`matrix_pencil` has not been called on the class instance.
        """


        return self._get_nondefault_param('theta0', 'matrix_pencil()', kill)


    def get_theta(self, frequency_unit='hz', kill=True):
        """Return the parameter estimate derived using
        :py:meth:`nonlinear_programming`

        Parameters
        ----------
        kill : bool, default: True
            If ``self.theta`` is ``None``, `kill` specifies how to act:

            * If ``kill`` is ``True``, an error is raised.
            * If ``kill`` is ``False``, ``None`` is returned.

        Returns
        -------
        theta : numpy.ndarray or None
            An array with ``theta.shape[1] = 4`` (1D signal) or
            ``theta.shape[1] = 6`` (2D signal).

        Raises
        ------
        AttributeIsNoneError
            Raised if ``self.theta`` is ``None``, and ``kill`` is ``True``.

        Notes
        -----
        If ``self.theta`` is ``None``, it is likely that
        :py:meth:`nonlinear_programming` has not been called on the class
        instance.
        """

        if frequency_unit == 'hz':
            return self._get_nondefault_param(
                'theta', 'nonlinear_programming()', kill,
            )

        elif frequency_unit == 'ppm':
            theta = deepcopy(self._get_nondefault_param(
                'theta', 'nonlinear_programming()', kill,
            ))

            for osc in theta:
                for column in range(2, 2 + self.get_dim()):
                    osc[column] = self._unit_convert(
                        (osc[column],), convert=f'hz->ppm',
                    )[0]
            return theta

        else:
            raise errors.InvalidUnitError('hz', 'ppm')


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



    def view_data(self, domain='frequency', freq_xunit='ppm', component='real'):
        """Generate a simple, interactive plot of the data using matplotlib.

        Parameters
        ----------
        domain : 'frequency' or 'time', default: 'frequency'
            The domain of the signal.

        freq_xunit : 'ppm' or 'hz', default: 'ppm'
            The unit of the x-axis, if `domain` is set as `'frequency'`. If
            `domain` is set as `'time'`, the x-axis unit will the seconds.
        """
        # TODO: 2D equivalent
        dim = self.get_dim()

        if domain == 'time':
            if dim == 1:
                ydata = self.get_data()
                xdata = self.get_tp()[0]

            elif dim == 2:
                print('TODO')
                return

        elif domain == 'frequency':
            # frequency domain treatment
            if dim == 1:
                ydata = fftshift(fft(self.get_data()))

                if freq_xunit == 'hz':
                    xdata = self.get_shifts(unit='hz')[0]
                elif freq_xunit == 'ppm':
                    xdata = self.get_shifts(unit='ppm')[0]
                else:
                    msg = f'{R}freq_xunit was not given a valid value' \
                          + f' (should be \'ppm\' or \'hz\').{END}'
                    raise ValueError(msg)

            elif dim == 2:
                print('TODO')
                return

        else:
            msg = f'{R}domain was not given a valid value' \
                  + f' (should be \'frequency\' or \'time\').{END}'
            raise ValueError(msg)

        if component == 'real':
            plt.plot(xdata, np.real(ydata), color='k')
        elif component == 'imag':
            plt.plot(xdata, np.imag(ydata), color='k')
        elif component == 'both':
            plt.plot(xdata, np.real(ydata), color='k')
            plt.plot(xdata, np.imag(ydata), color='#808080')
        else:
            msg = f'{R}component was not given a valid value' \
                  + f' (should be \'real\', \'imag\' or \'both\').{END}'
            raise ValueError(msg)

        if domain == 'frequency':
            plt.xlim(xdata[-1], xdata[0])
        plt.show()

# TODO ========================================================
# make_fid:
# include functionality to write to Bruker files, Varian files,
# JEOL files etc
# =============================================================

    def make_fid(self, result_name=None, oscillators=None, n=None):
        """Constructs a synthetic FID using a parameter estimate and
        experiment parameters.

        Parameters
        ----------
        result_name : None, 'theta', or 'theta0', default: None
            The parameter array to use. If ``None``, the parameter estimate
            to use will be determined in the following order of priority:

            1. ``self.theta`` will be used if it is not ``None``.
            2. ``self.theta0`` will be used if it is not ``None``.
            3. Otherwise, an error will be raised.

        oscillators : None or list, default: None
            Which oscillators to include in result. If ``None``, all
            oscillators will be included. If a list of ints, the subset of
            oscillators corresponding to these indices will be used.

        n : None, (int,), or (int, int) default: None
            Determines the number of points to construct the FID with. If
            ``None``, the value of :py:meth:`get_n` will be used.

        Returns
        -------
        fid : numpy.ndarray
            The generated FID.
        """

        result, _ = self._check_result(result_name)

        if oscillators:
            # slice desired oscillators
            result = result[[oscillators],:]

        dim = self.get_dim()
        n = self._check_int_float(n)

        if not n:
            n = self.get_n()

        elif isinstance(n, tuple) and len(n) == dim:
            pass

        else:
            raise TypeError(f'{R}n should be None or a tuple of ints{END}')

        return _misc.mkfid(
            result, n, self.get_sw(), self.get_offset(), dim
        )


    @logger
    def frequency_filter(
        self, region, noise_region, p0=None, p1=None, cut=True, cut_ratio=3.0,
        region_unit='ppm',
    ):
        """Generates phased, frequency-filtered data from the original data
        supplied.

        Parameters
        ----------
        region: [[int, int]], [[int, int], [int, int]], [[float, float]] or [[float, float], [float, float]]
            Cut-off points of the spectral region to consider.
            If the signal is 1D, this should be of the form `[[a,b]]`
            where `a` and `b` are the boundaries.
            If the signal is 2D, this should be of the form
            `[[a,b], [c,d]]` where `a` and `b` are the boundaries in
            dimension 1, and `c` and `d` are the boundaries in
            dimension 2. The ordering of the bounds in each dimension is
            not important.

        noise_region: [[int, int]], [[int, int], [int, int]], [[float, float]] or [[float, float], [float, float]]
            Cut-off points of the spectral region to extract the spectrum's
            noise variance. This should have the same structure as `region`.

        p0 : [float], [float, float], or None, default: None
            Zero order phase correction to apply to the data (radians).
            If set to `None`, this will be set to zero in each dimension.

        p1 : [float], [float, float], or None, default: None
            First order phase correction to apply to the data (radians).
            If set to `None`, this will be set to zero in each dimension.

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
        :py:class:`nmrespy.filter.FrequencyFilter`. To obtain information
        on the filtration, use :py:meth:`get_filter_info`.
        """

        self.filter_info = FrequencyFilter(
            self.get_data(), region, noise_region, region_unit=region_unit,
            sw=self.get_sw(), offset=self.get_offset(),
            sfo=self.get_sfo(kill=False), p0=p0, p1=p1, cut=cut,
            cut_ratio=cut_ratio,
        )

    def get_filter_info(self, kill=True):
        """Returns information relating to frequency filtration.

        Parameters
        ----------
        kill : bool, default: True
            If `filter_info` is `None`, and `kill` is `True`, an error will
            be raised. If `kill` is False, `None will be returned`.

        Returns
        -------
        filter_info : nmrespy.filter.FrequencyFilter

        Notes
        -----
        There are numerous methods associated with `filter_info` for
        obtaining relavent infomation about the filtration. See
        :py:class:`nmrespy.filter.FrequencyFilter` for details.
        """

        return self._check_if_none(
            'filter_info', kill, method='frequency_filter'
        )


    @logger
    def matrix_pencil(self, M=0, trim=None, fprint=True):
        """Implementation of the 1D Matrix Pencil Method [1]_ [2]_ or 2D
        Modified Matrix Enchancement and Matrix Pencil (MMEMP) method [3]_
        [4]_ with the option of Model Order Selection using the Minumum
        Descrition Length (MDL).

        Parameters
        ----------
        M : int, default: 0
            The number of oscillators to use in generating a parameter
            estimate. If M is set to 0, the number of oscillators will be
            estimated using the MDL.

        trim : None, (int,), or (int, int), default: None
            If ``trim`` is a tuple, the analysed data will be sliced such that
            its shape matches trim, with the initial points in the signal
            being retained. If ``trim`` is ``None``, the data will not be
            sliced. Consider using this in cases where the full signal is
            large, such that the method takes a very long time, or your PC
            has insufficient memory to process it.

        fprint : bool, deafult: True
            If ``True`` (default), the method provides information on
            progress to the terminal as it runs. If ``False``, the method
            will run silently.

        Notes
        -----
        The method requires appropriate time-domain data to run. If
        frequency-filtered data has been generated by :py:meth:`frequency_filter`
        (stored in the attribute ``filtered_signal``), prior to calling this method,
        this will be analysed. If no such data is found, but the original data
        is a raw FID (i.e. ``self.get_dtype()`` is ``'fid'``), that will
        analysed. If the original data is processed data (i.e.
        ``self.get_dtype()`` is ``'pdata'``), and no signal has been generated
        using :py:meth:`frequency_filter`, an error will be raised.

        The class attribute ``theta0`` will be updated upon successful running
        of this method. If the  data is 1D, ``self.theta0.shape[1] = 4``,
        whilst if the data is 2D, ``self.theta0.shape[1] = 6``. Elements along
        axis 0 of ``theta0`` contain the parameters associated with each
        individual oscillator, with parameters ordered as follows:

        * :math:`[a_m, \phi_m, f_m, \eta_m]` (1D)
        * :math:`[a_m, \phi_m, f_{1,m}, f_{2,m}, \eta_{1,m}, \eta_{2,m}]` (2D)

        References
        ----------
        .. [1] Yingbo Hua and Tapan K Sarkar. “Matrix pencil method for
           estimating parameters of exponentially damped/undamped sinusoids
           in noise”. In: IEEE Trans. Acoust., Speech, Signal Process. 38.5
           (1990), pp. 814–824.

        .. [2] Yung-Ya Lin et al. “A novel detection–estimation scheme for
           noisy NMR signals: applications to delayed acquisition data”.
           In: J. Magn. Reson. 128.1 (1997), pp. 30–41.

        .. [3] Yingbo Hua. “Estimating two-dimensional frequencies by matrix
           enhancement and matrix pencil”. In: [Proceedings] ICASSP 91: 1991
           International Conference on Acoustics, Speech, and Signal
           Processing. IEEE. 1991, pp. 3073–3076.

        .. [4] Fang-Jiong Chen et al. “Estimation of two-dimensional
           frequencies using modified matrix pencil method”. In: IEEE Trans.
           Signal Process. 55.2 (2007), pp. 718–724.
        """

        data, sw, offset = self._check_data_sw_offset()

        # slice data if user provided a tuple
        trim = self._check_trim(trim, data)
        data = data[tuple([np.s_[0:int(t)] for t in trim])]

        self.matrix_pencil_info = mpm.MatrixPencil(
            data, sw, offset, self.sfo, M, fprint
        )

        self.theta0 = self.matrix_pencil_info.get_parameters()


    @logger
    def nonlinear_programming(
        self, trim=None, method='trust_region', mode=None, bound=False,
        phase_variance=False, maxit=None, amp_thold=None, freq_thold=None,
        negative_amps='remove', fprint=True
    ):
        """Estimation of signal parameters using nonlinear programming, given
        an inital guess.

        Parameters
        ----------
        trim : None, (int,), or (int, int), default: None
            If ``trim`` is a tuple, the analysed data will be sliced such that
            its shape matches ``trim``, with the initial points in the signal
            being retained. If ``trim`` is ``None``, the data will not be
            sliced. Consider using this in cases where the full signal is
            large, such that the method takes a very long time, or your PC
            has insufficient memory to process it.

        method : 'trust_region' or 'lbfgs', default: 'trust_region'
            The optimization method to be used. Both options use
            the ``scipy.optimize.minimize`` function [5]_

            * ``'trust_region'`` sets ``method='trust-constr'``
            * ``'lbfgs'`` sets ``method='L-BFGS-B'``.

            See the Notes below for advice on choosing ``method``.

        mode : None or str, default: None
            Specifies which parameters to optimize. If ``None``,
            all parameters in the initial guess are subjected to the
            optimization. If a string containing a combination of the letters
            ``'a'``, ``'p'``, ``'f'``, and ``'d'`` is used, then only the
            parameters specified will be considered. For example ``mode='af'``
            would make the routine optimise amplitudes and frequencies,
            while leaving phases and damping factors fixed.

        bound : bool, default: False
            Specifies whether or not to carry out an optimisation where
            the parameters are bounded. If ``False``, the optimsation
            will be unconstrained. If ``True``, the following bounds
            are set on the parameters:

            * amplitudes: :math:`0 < a_m < \infty`
            * phases: :math:`- \pi < \phi_m \leq \pi`
            * frequencies: :math:`f_{\\mathrm{off}} -
              \\frac{f_{\\mathrm{sw}}}{2} \\leq f_m \\leq f_{\\mathrm{off}} +
              \\frac{f_{\\mathrm{sw}}}{2}`
            * damping: :math:`0 < \eta_m < \infty`

        phase_variance : bool, default: False
            Specifies whether or not to include the variance of phase in
            the cost function under consideration.

        maxit : int or None, default: None
            The maximum number of iterations the routine will carry out
            before being forced to terminate. If ``None``, the default number
            of maximum iterations is set (``100`` if ``method='trust_region'``,
            and ``500`` if ``method='lbfgs'``).

        amp_thold : float or None, default: None
            If ``None``, does nothing. If a float, oscillators with
            amplitudes satisfying :math:`a_m < a_{\\mathrm{thold}}
            \\lVert \\boldsymbol{a} \\rVert`` will be removed from the
            parameter array, where :math:`\\lVert \\boldsymbol{a} \\rVert`
            is the norm of the vector of all the oscillator amplitudes. It is
            advised to set ``amp_thold`` at least a couple of orders of
            magnitude below 1.

        freq_thold : float or None, default: None
            .. warning::

               NOT IMPLEMENTED YET

            If ``None``, does nothing. If a float, oscillator pairs with
            frequencies satisfying
            :math:`\\lvert f_m - f_p \\rvert < f_{\\mathrm{thold}}` will be
            removed from the parameter array. A new oscillator will be included
            in the array, with parameters:

            * amplitude: :math:`a = a_m + a_p`
            * phase: :math:`\phi = \\frac{\phi_m + \phi_p}{2}`
            * frequency: :math:`f = \\frac{f_m + f_p}{2}`
            * damping: :math:`\eta = \\frac{\eta_m + \eta_p}{2}`

        negative_amps : 'remove' or 'flip_phase', default: 'remove'
            Indicates how to treat oscillators which have gained negative
            amplitudes during the optimisation. ``'remove'`` will result
            in such oscillators being purged from the parameter estimate.
            The optimisation routine will the be re-run recursively until
            no oscillators have a negative amplitude. ``'flip_phase'`` will
            retain oscillators with negative amplitudes, but the the amplitudes
            will be turned positive, and a π radians phase shift will be
            applied to the oscillator.

        fprint : bool, default: True
            If ``True``, the method provides information on progress to
            the terminal as it runs. If ``False``, the method will run silently.

        Raises
        ------
        NoSuitableDataError
            Raisd when this method is called on an class instance that does
            not possess appropriate time-domain data for analysis (see Notes).

        PhaseVarianceAmbiguityError
            Raised when ``phase_variance`` is set to ``True``, but the user
            has specified that they do not wish to optimise phases using the
            ``mode`` argument.

        NoParameterEstimateError
            Raised when the attribute ``theta0`` is ``None``.

        Notes
        -----
        The method requires appropriate time-domain
        data to run. If frequency-filtered data has been
        generated by :py:meth:`frequency_filter` (stored in the attribute
        ``filtered_signal``) prior to calling this method, this will be analysed.
        If no such data is found, but the original data is a raw
        FID (i.e. :py:meth:`get_dtype` returns ``'fid'``), the original FID will
        analysed. If the original data is processed data (i.e.
        :py:meth:`get_dtype` returns ``'pdata'``), and no signal has been
        generated using :py:meth:`frequency_filter`, an error will be raised.

        The method also requires an initial guess, stored in the attribute
        ``theta0``. To generate this initial guess, you first need to apply
        the :py:meth:`matrix_pencil` method.

        The class attribute ``theta`` will be updated upon successful running
        of this method. If the  data is 1D,
        ``self.theta.shape[1] = 4``, whilst if the data is 2D,
        ``self.theta.shape[1] = 6``. Elements along axis 0 of `theta`
        contain the parameters associated with each individual oscillator,
        with parameters ordered as follows:

        * :math:`[a_m, \phi_m, f_m, \eta_m]` (1D)
        * :math:`[a_m, \phi_m, f_{1,m}, f_{2,m}, \eta_{1,m}, \eta_{2,m}]` (2D)

        The two optimisation algorithms primarily differ in how they treat
        the calculation of the matrix of cost function second derivatives
        (called the Hessian). ``'trust_region'`` will calculate the
        Hessian explicitly at every iteration, whilst ``'lbfgs'`` uses an
        update formula based on gradient information to estimate the Hessian.
        The upshot of this is that the convergence rate (the number of
        iterations needed to reach convergence) is typically better for
        ``'trust_region'``, though each iteration typically takes longer to
        generate. By default, it is advised to use ``'trust_region'``, however
        if your guess has a large number of signals (as a rule of thumb,
        ``theta0.shape[0] > 50``), you may find ``'lbfgs'`` performs more
        effectively.

        References
        ----------
        .. [5] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        """

        # TODO: include freq threshold

        # unpack parameters
        dim = self.get_dim()

        # initial guess: should have come from self.matrix_pencil()
        theta0 = self.get_theta0()

        # get sweep width and offset
        # which set is the correct set will depend on whether the user
        # called self.frequency with cut=True or False
        if self.filtered_sw is None:
            sw = self.get_sw()
            off = self.get_offset()

        else:
            sw = self.get_filtered_sw()
            off = self.get_filtered_offset()

        # check inputs are valid

        # types of parameters to be optimised
        # (amplitudes, phases, frequencies, damping factors)
        mode = self._check_mode(mode, phase_variance)
        # retrieve data to be analysed
        data = self._check_data()
        # trimmed data tuple
        trim = self._check_trim(trim, data)
        # trim data
        data = data[tuple([np.s_[0:int(t)] for t in trim])]

        # nonlinear programming method
        if method not in ['trust_region', 'lbfgs']:
            raise ValueError(f'\n{R}method should be \'trust_region\''
                             f' or \'lbfgs\'.{END}')

        # maximum iterations
        if maxit is None:
            if method == 'trust_region':
                maxit = 100
            else: # lbfgs: more iters as much faster but poorer convergence
                maxit = 500
        elif isinstance(maxit, int):
            pass
        else:
            raise TypeError(f'\n{R}maxit should be an int or None.{END}')

        # treatment of negative amplitudes
        if negative_amps not in ['remove', 'flip_phase']:
            raise ValueError(f'{R}negative_amps should be \'remove\' or'
                             f' \'flip_phase\'{END}')


        self.theta, self.errors = _nlp.nlp(
            data, dim, theta0, sw, off, phase_variance, method, mode, bound,
            maxit, amp_thold, freq_thold, negative_amps, fprint, True, None
        )


    def pickle_save(
        self, fname='NMREsPy_result.pkl', dir='.', force_overwrite=False
    ):
        """Converts the class instance to a byte stream using Python's
        "Pickling" protocol, and saves it to a .pkl file.

        Parameters
        ----------
        fname : str, default: 'NMREsPy_result.pkl'
            Name of file to save the byte stream to. Arguments without
            an extension or with a '.pkl' extension are permitted.

        dir : str, default: '.'
            Path of the desried directory to save the file to. Default
            is the current working directory.

        force_overwrite : bool, default: False
            If ``False``, if a file with the desired path already
            exists, the user will be prompted to confirm whether they wish
            to overwrite the file. If ``True``, the file will be overwritten
            without prompt.

        Notes
        -----
        This method complements :py:func:`~nmrespy.load.pickle_load`, in that
        an instance saved using :py:func:`pickle_save` can be recovered by
        :py:func:`~nmrespy.load.pickle_load`.
        """

        if os.path.isdir(dir):
            pass
        else:
            raise IOError(f'{R}directory {dir} doesn\'t exist{END}')

        if fname[-4:] == '.pkl':
            pass
        elif '.' in fname:
            raise ValueError(f'{R}fname: {fname} - Unexpected file'
                             f' extension.{END}')
        else:
            fname += '.pkl'

        path = os.path.join(dir, fname)

        if os.path.isfile(path):
            if force_overwrite:
                os.remove(path)
            else:
                prompt = f'{O}The file {path} already exists.' \
                         + f' Overwrite? [y/n]:{END} '
                _misc.get_yn(prompt)
                os.remove(path)

        with open(path, 'wb') as fh:
            pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)
        print(f'{G}Saved instance of NMREsPyBruker to {path}{END}')


    def write_result(
        self, description=None, fname='NMREsPy_result', dir='.',
        result_name=None, sf=5, sci_lims=(-2,3), fmt='txt',
        inc_pdf_figure=False, figure_for_pdf=None, force_overwrite=False,
        **kwargs
    ):
        """Saves an estimation result to a file in a human-readable format
        (either a textfile or a PDF).

        Parameters
        ----------
        description : str or None, default: None
            A description of the result, which is appended at the top of the
            file. If `None`, no description is added.

        fname : str, default: 'NMREsPy_result'
            The name of the result file.

            * If ``fmt`` is ``'txt'``, either a name with no extension or
              the extension '.txt' will be accepted.
            * If `fmt` is 'pdf', either a name with no extension or the
              extension '.pdf' will be accepted.

        dir : str, default: '.'
            Path to the desried directory to save the file to.

        result_name : None, 'theta', or 'theta0', default: None
            The parameter array to use. If ``None``, the parameter estimate
            to use will be determined in the following order of priority:

            1. ``self.theta`` will be used if it is not ``None``.
            2. ``self.theta0`` will be used if it is not ``None``.
            3. Otherwise, an error will be raised.

        sf : int, default: 5
            The number of significant figures used.

        sci_lims : (int, int), default: (-2, 3)
            Specifies the smallest magnitdue negative and positive orders of
            magnitude which values need to possess in order to be presented
            using scientific notaiton. A certain value will be displayed
            using scientific notation if if satisfies one of the following:

            * ``abs(value) <= 10 ** sci_lims[0]``
            * ``abs(value) >= 10 ** sci_lims[1]``

        fmt : 'txt' or 'pdf', default: 'txt'
            Specifies the format of the file. To produce a pdf, a LaTeX
            installation is required. See the Notes below for details.

        inc_pdf_figure : bool, default: False
            If ``True``, :meth:`plot_result` will be use to create a figure
            of the result, which will be appended to the file. This is
            valid only when ``fmt`` is set to ``'pdf'``.

        figure_for_pdf : matplotlib.figure.Figure or None, default: None
            A figure object to add to a PDF result file. Make sure
            ``include_pdf_figure`` is set to ``True``.

        force_overwrite : bool, default: False
            If ``False``, if a file with the desired path already
            exists, the user will be prompted to confirm whether they wish
            to overwrite the file. If ``True``, the file will be overwritten
            without any prompt.

        kwargs : :py:meth:`plot_result` properties
            If ``include_pdf_figure`` is ``True`` and ``figure_for_pdf`` is
            ``None``, the figure is generated by calling :py:meth:`plot_result`.
            ``kwargs`` are used to specify the properties of the figure.

        Raises
        ------
        LaTeXFailedError
            With ``fmt`` set to ``'pdf'``, this will be raised if an error
            was encountered in running ``pdflatex``.

        Notes
        -----
        To generate pdf files of NMR-EsPy results, it is necessary to have a
        LaTeX installation set up on your system.
        For a simple to set up implementation that is supported on all
        major operating systems, consider
        `TexLive <https://www.tug.org/texlive/>`_. To ensure that
        you have a functioning LaTeX installation, open a command
        prompt/terminal and type ``pdflatex``.

        The following is a full list of packages that your LaTeX installation
        will need to successfully compile the .tex file generated by
        :py:meth:`write_result`:

        * `amsmath <https://ctan.org/pkg/amsmath?lang=en>`_
        * `array <https://ctan.org/pkg/array?lang=en>`_
        * `booktabs <https://ctan.org/pkg/booktabs?lang=en>`_
        * `cmbright <https://ctan.org/pkg/cmbright>`_
        * `geometry <https://ctan.org/pkg/geometry>`_
        * `hyperref <https://ctan.org/pkg/hyperref?lang=en>`_
        * `longtable <https://ctan.org/pkg/longtable>`_
        * `siunitx <https://ctan.org/pkg/siunitx?lang=en>`_
        * `tcolorbox <https://ctan.org/pkg/tcolorbox?lang=en>`_
        * `xcolor <https://ctan.org/pkg/xcolor?lang=en>`_

        Most of these are pretty ubiquitous and are likely to be installed
        even with lightweight LaTeX installations. If you wish to check the
        packages are available, run::
            $ kpsewhich <package-name>.sty
        If a pathname appears, the package is installed to that path.
        """

        # retrieve result
        result, _ = self._check_result(result_name)

        # check format is sensible
        if fmt in ['txt', 'pdf']:
            pass
        else:
            raise ValueError(f'{R}fmt should be \'txt\' or \'pdf\'{END}')

        # basic info
        info = []
        info.append(result)
        info.append(self.get_dim()) # signal dimension
        info.append(self.get_datapath()) # data path
        info.append(self.get_sw()) # sweep width (Hz)
        info.append(self.get_sw(unit='ppm')) # sweep width (Hz)
        info.append(self.get_offset()) # offset (Hz)
        info.append(self.get_offset(unit='ppm')) # offset (Hz)
        info.append(self.get_sfo()) # transmitter frequency
        info.append(self.get_bf()) # basic frequency
        info.append(self.get_nucleus()) # nuclei

        # peak integrals
        integrals = ()
        # dx in each dimension (gap between successive points in Hz)
        delta = [sw / n for sw, n in zip(self.get_sw(), self.get_n())]

        # integrate each oscillator numerically
        # constructs absolute real spectrum for each oscillator and
        # uses Simpson's rule
        # TODO: Perhaps this could be done analytically?
        for m, osc in enumerate(result):
            # make fid corresponding to each individual oscillator
            f = self.make_fid(result_name, oscillators=[m])

            # absolute real spectrum
            s = np.absolute(np.real(fftshift(fft(f))))

            # inegrate successively over each dimension
            if self.get_dim() == 1:
                integrals += simps(s, dx=delta[0]),
            elif self.get_dim() == 2:
                integrals += simps(simps(s, dx=delta[1]), dx=delta[0]),

        info.append(integrals)

        # frequency filter region
        info.append(self.get_region(unit='hz', kill=False))
        info.append(self.get_region(unit='ppm', kill=False))

        if fmt == 'pdf':
            if inc_pdf_figure:
                if isinstance(figure_for_pdf, matplotlib.figure.Figure):
                    info.append(figure_for_pdf)
                elif figure_for_pdf is None:
                    fig, *_ = self.plot_result(**kwargs)
                    info.append(fig)
                else:
                    raise TypeError(
                        f'{R}figure_for_pdf should be of type'
                        f'matplotlib.figure.Figure or None.{END}'
                    )

        _write.write_file(info, description, fname, dir, sf, sci_lims, fmt,
                          force_overwrite)



    def plot_result(self, result_name=None, datacol=None, osccols=None,
                    labels=True, stylesheet=None):
        """Generates a figure with the result of an estimation routine.
        A spectrum of the original data is plotted, along with the
        Fourier transform of each individual oscillator that makes up the
        estimation result.

        .. note::
            Currently, this method is only compatible with 1-dimesnional
            data.

        Parameters
        ----------
        result_name : None, 'theta', or 'theta0', default: None
            The parameter array to use. If ``None``, the parameter estimate
            to use will be determined in the following order of priority:

            1. ``self.theta`` will be used if it is not ``None``.
            2. ``self.theta0`` will be used if it is not ``None``.
            3. Otherwise, an error will be raised.

        datacol : matplotlib color or None, default: None
            The color used to plot the original data. Any value that is
            recognised by matplotlib as a color is permitted. See:

            https://matplotlib.org/3.1.0/tutorials/colors/colors.html

            If ``None``, the default color :grey:`#808080` will be used.

        osccols : matplotlib color, matplotlib colormap, list, or None,\
        default: None
            Describes how to color individual oscillators. The following
            is a complete list of options:

            * If the value denotes a matplotlib color, all oscillators will
              be given this color.
            * If a string corresponding to a matplotlib colormap is given,
              the oscillators will be consecutively shaded by linear increments
              of this colormap.
              For all valid colormaps, see:

              https://matplotlib.org/3.3.1/tutorials/colors/colormaps.html
            * If a list or NumPy array containing valid matplotlib colors is
              given, these colors will be cycled.
              For example, if ``osccols=['r', 'g', 'b']``:

              + Oscillators 1, 4, 7, ... would be :red:`red (#FF0000)`
              + Oscillators 2, 5, 8, ... would be :green:`green (#008000)`
              + Oscillators 3, 6, 9, ... would be :blue:`blue (#0000FF)`

            * If ``None``, the default colouring method will be applied,
              which involves cycling through the following colors:

              + :oscblue:`#1063E0`
              + :oscorange:`#EB9310`
              + :oscgreen:`#2BB539`
              + :oscred:`#D4200C`

        labels : Bool, default: True
            If ``True``, each oscillator will be given a numerical label
            in the plot, if ``False``, no labels will be produced.

        stylesheet : None or str, default: None
            The name of/path to a matplotlib stylesheet for further
            customisation of the plot. Note that all the features of the
            stylesheet will be adhered to, except for the colors, which are
            overwritten by whatever is specified by ``datacol`` and
            ``osccols``. If ``None``, a custom stylesheet, found in the
            follwing path is used:

            ``/path/to/NMR-EsPy/nmrespy/config/nmrespy_custom.mplstyle``

            To see built-in stylesheets that are available, enter
            the following into a python interpreter: ::
                >>> import matplotlib.pyplot as plt
                >>> print(plt.style.available)
            Alternatively, enter the full path to your own custom stylesheet.

        Returns
        -------
        fig : `matplotlib.figure.Figure <https://matplotlib.org/3.3.1/\
        api/_as_gen/matplotlib.figure.Figure.html>`_
            The resulting figure.

        ax : `matplotlib.axes._subplots.AxesSubplot <https://matplotlib.org/\
        3.3.1/api/axes_api.html#the-axes-class>`_
            The resulting set of axes.

        lines : dict
            A dictionary containing a series of
            `matplotlib.lines.Line2D <https://matplotlib.org/3.3.1/\
            api/_as_gen/matplotlib.lines.Line2D.html>`_
            instances. The data plot is given the key ``'data'``, and the
            individual oscillator plots are given the keys ``'osc1'``,
            ``'osc2'``, ``'osc3'``, ..., ``'osc<M>'`` where ``<M>`` is the
            number of oscillators in the parameter estimate.

        labs : dict
            If ``labels`` is True, this dictionary will contain a series
            of `matplotlib.text.Text <https://matplotlib.org/3.1.1/\
            api/text_api.html#matplotlib.text.Text>`_ instances, with the
            keys ``'osc1'``, ``'osc2'``, etc. as is the case with the
            ``lines`` dictionary. If ``labels`` is False, ``labs`` will be
            and empty dictionary.

        Raises
        ------
        TwoDimUnsupportedError
            If ``self.dim`` is 2.

        NoParameterEstimateError
            If ``result_name`` is ``None``, and both ``self.theta0`` and
            ``self.theta`` are ``None``.

        Notes
        -----
        The ``fig``, ``ax``, ``lines`` and ``labels`` objects that are returned
        provide the ability to customise virtually any feature of the plot. If
        you wish to edit a particular line or label, simply use the following
        syntax: ::
            >>> fig, ax, lines, labs = example.plot_result()
            >>> lines['osc7'].set_lw(1.6) # set oscillator 7's linewith to 2
            >>> labs['osc2'].set_x(4) # move the x co-ordinate of oscillator 2's label to 4ppm

        To save the figure, simply use `savefig: <https://matplotlib.org/\
        3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html>`_
        ::
            >>> fig.savefig('example_figure.pdf', format='pdf')
        """

        dim = self.get_dim()
        # check dim is valid (only 1D data supported so far)
        if dim == 2:
            raise TwoDimUnsupportedError()

        result, result_name = self._check_result(result_name)

        data = fftshift(fft(self.data))[::-1]

        # phase data
        p0 = self.get_p0(kill=False)
        p1 = self.get_p1(kill=False)

        if p0 is None:
            pass
        else:
            data = _ve.phase(data, p0, p1)

        # FTs of FIDs generated from individual oscillators in result array
        peaks = []
        for m, osc in enumerate(result):
            f = self.make_fid(result_name, oscillators=[m])
            peaks.append(np.real(fftshift(fft(f))))

        # left and right boundaries of filter region
        region = self.get_region(kill=False)

        nuc = self.get_nucleus()
        shifts = self.get_shifts(unit='ppm')

        return _plot.plotres_1d(
            data, peaks, shifts, region, nuc, datacol, osccols, labels,
            stylesheet
        )


    @logger
    def add_oscillators(self, oscillators, result_name=None):
        """Adds new oscillators to a parameter array.

        Parameters
        ----------
        oscillators : numpy.ndarray
            An array of the new oscillator(s) to add to the array. The array
            should be of shape (I, 4) or (I, 6), where I is greater than 0.

        result_name : None, 'theta', or 'theta0', default: None
            The parameter array to use. If ``None``, the parameter estimate
            to use will be determined in the following order of priority:

            1. ``self.theta`` will be used if it is not ``None``.
            2. ``self.theta0`` will be used if it is not ``None``.
            3. Otherwise, an error will be raised.
        """

        self._log_method()

        if isinstance(oscillators, np.ndarray):
            # if oscillators is 1D, convert to 2D array
            if oscillators.ndim == 1:
                oscillators = oscillators.reshape(-1, oscillators.shape[0])

            # number of parameters per oscillator (should be 4 for 1D, 6 for 2D)
            num = oscillators.shape[1]
            # expected number of parameters per oscillator
            exp = 2 * self.get_dim() + 2

            if num == exp:
                pass
            else:
                msg = f'\n{R}oscillators should have a size of {exp} along' \
                      + f' axis-1{END}'
                raise ValueError(msg)

        else:
            raise TypeError(f'\n{R}oscillators should be a numpy array{END}')

        result, result_name = self._check_result(result_name)

        result = np.append(result, oscillators, axis=0)
        self.__dict__[result_name] = result[np.argsort(result[..., 2])]


    @logger
    def remove_oscillators(self, indices, result_name=None):
        """Removes the oscillators corresponding to ``indices``.

        Parameters
        ----------
        indices : list, tuple or numpy.ndarray
            A list of indices corresponding to the oscillators to be removed.

        result_name : None, 'theta', or 'theta0', default: None
            The parameter array to use. If ``None``, the parameter estimate
            to use will be determined in the following order of priority:

            1. ``self.theta`` will be used if it is not ``None``.
            2. ``self.theta0`` will be used if it is not ``None``.
            3. Otherwise, an error will be raised.
        """

        self._log_method()

        if isinstance(indices, (tuple, list, np.ndarray)):
            pass
        else:
            raise TypeError(f'\n{R}indices does not have a suitable type{END}')

        result, result_name = self._check_result(result_name)

        try:
            result = np.delete(result, indices, axis=0)
        except:
            msg = f'{R}oscillator removal failed. Check the values in' \
                  + f' indices are valid.{END}'
            raise ValueError(msg)

        self.__dict__[result_name] = result[np.argsort(result[..., 2])]
        print(self.__dict__[result_name])


    @logger
    def merge_oscillators(self, indices, result_name=None):
        """Removes the oscillators corresponding to ``indices``, and
        constructs a single new oscillator with a cumulative amplitude, and
        averaged phase, frequency and damping factor.

        .. note::
            Currently, this method is only compatible with 1-dimesnional
            data.

        Parameters
        ----------
        indices : list, tuple or numpy.ndarray
            A list of indices corresponding to the oscillators to be merged.

        result_name : None, 'theta', or 'theta0', default: None
            The parameter array to use. If ``None``, the parameter estimate
            to use will be determined in the following order of priority:

            1. ``self.theta`` will be used if it is not ``None``.
            2. ``self.theta0`` will be used if it is not ``None``.
            3. Otherwise, an error will be raised.

        Notes
        -----
        Assuming that an estimation result contains a subset of oscillators
        denoted by indices :math:`\{m_1, m_2, \cdots, m_J\}`, where
        :math:`J \leq M`, the new oscillator formed by the merging of the
        oscillator subset will possess the follwing parameters:

            * :math:`a_{\\mathrm{new}} = \\sum_{i=1}^J a_{m_i}`
            * :math:`\\phi_{\\mathrm{new}} = \\frac{1}{J} \\sum_{i=1}^J \\phi_{m_i}`
            * :math:`f_{\\mathrm{new}} = \\frac{1}{J} \\sum_{i=1}^J f_{m_i}`
            * :math:`\\eta_{\mathrm{new}} = \\frac{1}{J} \\sum_{i=1}^J \\eta_{m_i}`
        """

        self._log_method()

        # determine number of elements in indices
        # if fewer than 2 elements, return without doing anything
        if isinstance(indices, (tuple, list, np.ndarray)):
            number = len(indices)
            if number < 2:
                msg = f'\n{O}indices should contain at least two elements.' \
                      + f'No merging will happen.{END}'
                return
        else:
            raise TypeError(f'\n{R}indices does not have a suitable type{END}')

        result, result_name = self._check_result(result_name)

        to_merge = result[indices]
        new_osc = np.sum(to_merge, axis=0, keepdims=True)

        # get mean for phase, frequency and damping
        new_osc[:, 1:] = new_osc[:, 1:] / number

        result = np.delete(result, indices, axis=0)
        result = np.append(result, new_osc, axis=0)
        self.__dict__[result_name] = result[np.argsort(result[..., 2])]


    @logger
    def split_oscillator(self, index, result_name=None, frequency_sep=2.,
                         unit='hz', split_number=2, amp_ratio='same'):
        """Removes the oscillator corresponding to ``index``. Incorporates two
        or more oscillators whose cumulative amplitudes match that of the
        removed oscillator.

        .. note::
            Currently, this method is only compatible with 1-dimesnional
            data.

        Parameters
        ----------
        index : int
            Array index of the oscilator to be split.

        result_name : None, 'theta', or 'theta0', default: None
            The parameter array to use. If ``None``, the parameter estimate
            to use will be determined in the following order of priority:

            1. ``self.theta`` will be used if it is not ``None``.
            2. ``self.theta0`` will be used if it is not ``None``.
            3. Otherwise, an error will be raised.

        frequency_sep : float, default: 2.
            The frequency separation given to adjacent oscillators formed from
            splitting.

        unit : 'hz' or 'ppm', default: 'hz'
            The unit of ``frequency_sep``.

        split_number: int, default: 2
            The number of peaks to split the oscillator into.

        amp_ratio: list or 'same', default: 'same'
            The ratio of amplitudes to be fulfilled by the newly formed peaks.
            If a list, ``len(amp_ratio) == split_number`` must be
            ``True``. The first element will relate to the highest frequency
            oscillator constructed (furthest to the left in a conventional
            spectrum), and the last element will relate to the lowest
            frequency oscillator constructed. If ``'same'``, all oscillators
            will be given equal amplitudes.
        """

        # get frequency_Sep in correct units
        if unit == 'hz':
            pass
        elif unit == 'ppm':
            frequency_sep = frequency_sep * self.get_sfo()
        else:
            raise errors.InvalidUnitError('hz', 'ppm')

        result, result_name = self._check_result(result_name)

        try:
            osc = result[index]
        except:
            raise ValueError(f'{R}index should be an int in'
                             f' range({result.shape[0]}){END}')

        # lowest frequency of all the new oscillators
        max_freq = osc[2] + ((split_number - 1) * frequency_sep / 2)
        # array of all frequencies (lowest to highest)
        freqs = [max_freq - (i*frequency_sep) for i in range(split_number)]

        # determine amplitudes of new oscillators
        if amp_ratio == 'same':
            amp_ratio = [1] * split_number

        if isinstance(amp_ratio, list):
            pass
        else:
            raise TypeError(f'{R}amp_ratio should be \'same\' or a list{END}')

        if len(amp_ratio) == split_number:
            pass
        else:
            raise ValueError(f'{R}len(amp_ratio) should equal'
                             f' split_number{END}')

        # scale amplitude ratio values such that their sum is 1
        amp_ratio = np.array(amp_ratio)
        amp_ratio = amp_ratio / np.sum(amp_ratio)

        # obtain amplitude values
        amps = osc[0] * amp_ratio

        new_oscs = np.zeros((split_number, 4))
        new_oscs[:, 0] = amps
        new_oscs[:, 2] = freqs
        new_oscs[:, 1] = [osc[1]] * split_number
        new_oscs[:, 3] = [osc[3]] * split_number

        result = np.delete(result, index, axis=0)
        result = np.append(result, new_oscs, axis=0)
        self.__dict__[result_name] = result[np.argsort(result[..., 2])]


    def save_logfile(self, fname=None, dir='.', force_overwrite=False):
        """Saves log file of class instance usage to a specified path.

        Parameters
        ----------
        fname : str or None, default: None
            Name of log file. If `None`, the default filename, specified
            by a timestamp, is used. This default filename is given by
            ``str(self.logpath)``.

        dir : str, default: '.'
            The path to the directory to save the file to. Deault if the
            current working directory.
        """

        if fname is None:
            fname = str(self.logpath.name)
        elif not isinstance(fname, str):
            raise TypeError(f'{R}fname should be a string or None{END}')

        if not isinstance(dir, str):
            raise TypeError(f'{R}dir should be a string or None{END}')

        path_manager = PathManager(fname, dir)

        result = path_manager.check_file(force_overwrite)

        if result == 0:
            pass
        elif result == 1:
            print(f'{O}Log file will not be saved to the specified path{END}')
            return
        elif result == 2:
            raise ValueError(f'{R}dir ({dir}) does not exist{END}')

        try:
            shutil.copyfile(self.logpath, path_manager.path)
            print(
                f'{G}Log file succesfully saved to'
                f' {str(path_manager.path)}{END}'
            )

        # trouble copying file...
        except Exception as e:
            raise e



    @staticmethod
    def _check_int_float(param):
        """Check if param is a float or int. If it is, convert to a tuple"""

        if isinstance(param, (float, int)):
            return param,

        return param

    def _check_data_sw_offset(self):
        """Returns time-domain data, sweep width and offset.

        Returns
        -------
        data : np.ndarray
            The time-domain signal.

        sw : [float] or [float, float]
            The sweep width in each dimension.

        offset : [float] or [float, float]
            The offset in each dimension.

        Notes
        -----
        The following hierarchical order is used to determine `data`, `sw`
        and `offset`:

        * If a filtered signal is present, this is returned as `data`.
        * Otherwise, the originally imported data is returned.

        The appropriate sweep width and offset values are returned, based
        on whether the the filtered signal was cut (see
        :py:meth:`frequency_filter` for more info).
        """

        # check whether filtered_signal signal is None or not
        filtered_signal = self.get_filtered_signal(kill=False)
        if isinstance(filtered_signal, np.ndarray):
            print('yep')
            # not None -> return filtred signal
            data = filtered_signal

            # check whether filtered_sw is None or not
            if self.get_filtered_sw(kill=False):
                return data, self.get_filtered_sw(), self.get_filtered_offset()

            return data, self.get_sw(), self.get_offset()

        # None -> return original data, sw, and offset
        return self.get_data(), self.get_sw(), self.get_offset()


    def _check_result(self, result_name):

        if result_name is None:
            for attr in ['theta', 'theta0']:
                result = getattr(self, attr)
                if isinstance(result, np.ndarray):
                    return result, attr

            raise NoParameterEstimateError()

        elif result_name in ['theta', 'theta0']:
            result = getattr(self, result_name)
            if isinstance(result, np.ndarray):
                return result, result_name
            else:
                raise ValueError(f'{R}{result_name} does not correspond to a valid'
                                 f' estimation result (it should be a numpy'
                                 f' array). Perhaps you have forgotten a step'
                                 f' in generating the parameter estimate?{END}')
        else:
            raise ValueError(f'{R}result_name should be None, \'theta\', or'
                             f' \'theta0\'{END}')

    def _check_trim(self, trim, data):
        if trim != None:
            trim = self._check_int_float(trim)
            # check trim is not larger than full signal size in any dim
            for i, (t, n) in enumerate(zip(trim, data.shape)):
                if t <= n:
                    pass
                else:
                    msg = f'\n{R}trim value is invalid. Ensure that each' \
                          + f' element in trim in smaller than or equal to' \
                          + f' the corresponding data shape in that' \
                          + f' dimension.\n' \
                          + f'trim: {trim}   data shape: {data.shape}{END}\n'
                    raise ValueError(msg)
            return trim

        else:
            return data.shape


    def _unit_convert(self, lst, convert):
        """Converts unit of a list of values
        '|a|->|b|', where |a| and |b| are not the same, and in
        ['idx', 'ppm', 'hz']"""

        # flag to determine whether value of convert is valid
        valid = False

        # check that convert is a valid value
        valid_units = ['idx', 'ppm', 'hz']
        for pair in itertools.permutations(valid_units, r=2):
            if f'{pair[0]}->{pair[1]}' == convert:
                valid = True
                break

        if valid:
            pass
        else:
            raise ValueError(f'{R}convert is not valid.')

        # list for storing final converted contents (will be returned as tuple)
        lst_conv = []
        for dimension, element in enumerate(lst):

            # try/except block enables code to work with both tuples and
            # tuples of tuples
            try:
                # test whether element is an iterable (i.e. tuple)
                iterable = iter(element)

                # elem is a tuple...
                sublst_conv = []
                while True:
                    try:
                        sublst_conv.append(
                            self._convert(next(iterable), convert, dimension)
                        )
                    except StopIteration:
                        break

                lst_conv.append(sublst_conv)

            except TypeError:
                # elem is a float/int...
                lst_conv.append(self._convert(element, convert, dimension))

        return lst_conv


    def _convert(self, value, conv, dimension):

        sw = self.get_sw()[dimension]
        off = self.get_offset()[dimension]
        n = self.get_n()[dimension]
        sfo = self.get_sfo()[dimension]

        if conv == 'idx->hz':
            return float(off + (sw / 2) - ((value * sw) / n))

        elif conv == 'idx->ppm':
            return float((off + sw / 2 - value * sw / n) / sfo)

        elif conv == 'ppm->idx':
            return int(round((off + (sw / 2) - sfo * value) * (n / sw)))

        elif conv == 'ppm->hz':
            return value * sfo

        elif conv == 'hz->idx':
            return int((n / sw) * (off + (sw / 2) - value))

        elif conv == 'hz->ppm':
            return value / sfo



# Some descriptions of attributes
    # filtered_spectrum : numpy.ndarray or None, default: `None`
    #     Spectral data which has been filtered using :py:meth:`frequency_filter`
    #
    # virtual_echo : numpy.ndarray or None, default: `None`
    #     Time-domain virtual echo derived using :py:meth:`frequency_filter`
    #
    # filtered_signal : numpy.ndarray or None, default: `None`
    #     First half of ``virtual_echo``, derived using :py:meth:`frequency_filter`
    #
    # filtered_n : [int] or [int, int] or None, default: `None`
    #     The size of the virtual echo generated using :py:meth:`frequency_filter`
    #     if ``cut=True``
    #
    # filtered_sw : [float] or [float, float] or None, default: `None`
    #     The sweep width (Hz) of the virtual echo generated using
    #     :py:meth:`frequency_filter` if ``cut=True``, in each dimension.
    #
    # filtered_offset : [float] or [float, float] or None, default: `None`
    #     The transmitter offset (Hz) of the virtual echo generated using
    #     :py:meth:`frequency_filter` if ``cut=True``, in each dimension.
    #
    # region : [float, float], [[float, float], [float, float]] or None, default: `None`
    #     The region of interest specified with :py:meth:`frequency_filter`,
    #     in units of array indices.
    #
    # noise_region : [float, float], [[float, float], [float, float]] or None, default: `None`
    #     The noise region specified with :py:meth:`frequency_filter`, in units
    #     of array indices.
    #
    # p0 : float or None, default: `None`
    #     The zero order phase correction applied to the frequency domain data
    #     during :py:meth:`frequency_filter`.
    #
    # p1 : float or None, default: `None`
    #     The first order phase correction applied to the frequency domain data
    #     during :py:meth:`frequency_filter`.
    #
    # theta0 : numpy.ndarray or None, default: `None`
    #     The parameter estimate derived using :py:meth:`matrix_pencil`
    #
    # theta : numpy.ndarray or None, default: `None`
    #     The parameter estimate derived using :py:meth:`nonlinear_programming`
