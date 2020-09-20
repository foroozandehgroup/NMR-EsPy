from copy import deepcopy
import inspect
import json
import os
import pickle
import re
import sys

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
from scipy.integrate import simps

from ._cols import *
if USE_COLORAMA:
    import colorama
from ._errors import *
from . import _misc, _mpm, _nlp, _plot, _ve, _write


np.set_printoptions(precision=5, threshold=32)

class NMREsPyBruker:
    """A class for consideration of Bruker data.

    .. note::
        An instance of this class should not be invoked directly. You should
        use one of the following functions to generate/de-serialise an
        instance of `NMREsPyBruker`:

        * :py:func:`~nmrespy.load.import_bruker_fid`
        * :py:func:`~nmrespy.load.import_bruker_pdata`
        * :py:func:`~nmrespy.load.pickle_load`


    Parameters
    ----------
    dtype : 'raw', 'pdata'
        The type of data imported. 'raw' indicates the data is derived
        from a raw FID file (fid for 1D data; ser for 2D data). 'pdata'
        indicates the data is derived from files found in a pdata directory
        (1r and 1i for 1D data; 2rr, 2ri, 2ir, 2ii for 2D data).

    data : numpy.ndarray or dict
        The data associated with binary files in `path`. If `dtype` is
        `'raw'`, this will be a NumPy array of the raw FID. If `dtype` is
        `'pdata'`, this will be a dictionary with each component of the
        processed data referenced by a key with the same name as the file
        it was derived from.

    path : str
        The path to the directory contaioning the NMR data.

    sw : (float,) or (float, float)
        The experiment sweep width in each dimension (Hz).

    off : (float,) or (float, float)
        The transmitter's offset frequency in each dimension (Hz).

    n : (int,) or (int, int)
        The number of data-points in each dimension.

    sfo : (float,) or (float, float)
        The transmitter frequency in each dimension (MHz)

    nuc : (str,) or (str, str)
        The nucleus in each dimension. Elements will be of the form ``'1H'``,
        ``'13C'``, ``'15N'``, etc.

    endian : 0, 1
        The endianess of the binary files. This is equivalent to either
        BYTORDA or BYTORDP, depending on the data type.

    intfloat : 0, 2
        The numeric type used to store the data in the binary files.
        This is equivalent to DTYPA or DTYPP, depending on the data type.
        0 indicates the data is stored as 4-bit integers. 2 indicates the
        data is stored as 8-bit floats.

    dim : 1, 2
        The dimension of the data.

    filt_spec : numpy.ndarray or None, default: `None`
        Spectral data which has been filtered using :py:meth:`virtual_echo`

    virt_echo : numpy.ndarray or None, default: `None`
        Time-domain virtual echo derived using :py:meth:`virtual_echo`

    half_echo : numpy.ndarray or None, default: `None`
        First half of ``virt_echo``, derived using :py:meth:`virtual_echo`

    ve_n : (int,), (int, int) or None, default: `None`
        The size of the virtual echo generated using :py:meth:`virtual_echo`
        if ``cut=True``

    ve_sw : (float,), (float, float) or None, default: `None`
        The sweep width (Hz) of the virtual echo generated using
        :py:meth:`virtual_echo` if ``cut=True``, in each dimension.

    ve_off : (float,), (float, float) or None, default: `None`
        The transmitter offset (Hz) of the virtual echo generated using
        :py:meth:`virtual_echo` if ``cut=True``, in each dimension.

    highs : (int,), (int, int) or None, default: `None`
        The index of the point corresponding to the highest ppm value that
        is contained within the filter region specified using
        :py:meth:`virtual_echo`, in each dimension.

    lows : (int,), (int, int) or None, default: `None`
        The index of the point corresponding to the lowest ppm value that
        is contained within the filter region specified using
        :py:meth:`virtual_echo`, in each dimension.

    p0 : float or None, default: `None`
        The zero order phase correction applied to the frequency domain data
        during :py:meth:`virtual_echo`.

    p1 : float or None, default: `None`
        The first order phase correction applied to the frequency domain data 
        during :py:meth:`virtual_echo`.

    theta0 : numpy.ndarray or None, default: `None`
        The parameter estimate derived using :py:meth:`matrix_pencil`

    theta : numpy.ndarray or None, default: `None`
        The parameter estimate derived using :py:meth:`nonlinear_programming`
    """

    def __init__(self, dtype, data, path, sw, off, n, sfo, nuc, endian,
                 intfloat, dim, filt_spec=None, virt_echo=None,
                 half_echo=None, ve_n=None, ve_sw=None, ve_off=None,
                 highs=None, lows=None, p0=None, p1=None, theta0=None,
                 theta=None):

        self.dtype = dtype # type of data ('raw' or 'pdata')
        self.data = data
        self.path = path # path to data directory
        self.sw = sw # sweep width (Hz)
        self.off = off # transmitter offset (Hz)
        self.n = n # number of points in data
        self.sfo = sfo # transmitter frequency (MHz)
        self.nuc = nuc # nucleus label
        self.endian = endian # endianness of binary file(s)
        self.intfloat = intfloat # data type in binary file(s)
        self.dim = dim # data dimesnion
        self.filt_spec = filt_spec # filtered spectrum
        self.virt_echo = virt_echo # virtual echo
        self.half_echo = half_echo # first half of virt_echo
        self.ve_n = ve_n # number of points in virtual echo if cut
        self.ve_sw = ve_sw # sw width of virtual echo if cut
        self.ve_off = ve_off # offset of virtual echo if cut
        self.highs = highs # idx values corresponding to highest ppms of region
        self.lows = lows # idx values corresponding to lowest ppms of region
        self.p0 = p0 # zero-order phase correction for virtual echo
        self.p1 = p1 # first-order phase correction for virtual echo
        self.theta0 = theta0 # mpm result
        self.theta = theta # nlp result

        # TODO add checks here


    def __repr__(self):
        msg = f'nmrespy.core.NMREsPyBruker('
        msg += f'{self.dtype}, '
        msg += f'{self.data}, '
        msg += f'{self.path}, '
        msg += f'{self.sw}, '
        msg += f'{self.off}, '
        msg += f'{self.n}, '
        msg += f'{self.sfo}, '
        msg += f'{self.nuc}, '
        msg += f'{self.endian}, '
        msg += f'{self.intfloat}, '
        msg += f'{self.dim}, '
        msg += f'{self.filt_spec}, '
        msg += f'{self.virt_echo}, '
        msg += f'{self.half_echo}, '
        msg += f'{self.ve_n}, '
        msg += f'{self.ve_sw}, '
        msg += f'{self.ve_off}, '
        msg += f'{self.highs}, '
        msg += f'{self.lows}, '
        msg += f'{self.theta0}, '
        msg += f'{self.theta})'

        return msg


    def __str__(self):
        cats = [] # categories
        vals = [] # values
        msg = ''
        cats.append(f'\n{MA}<NMREsPyBruker object at {hex(id(self))}>{END}\n')
        vals.append('')
        cats.append(f'{MA}───Basic Info───{END}')
        vals.append('')
        cats.append('Path:')
        vals.append(self.get_datapath())
        cats.append('Type:')
        vals.append(self.get_dtype())
        cats.append('Dimension:')
        vals.append(str(self.get_dim()))

        data = self.get_data()
        dtype = self.get_dtype()
        if  type == 'raw':
            cats.append('Data:')
            vals.append(f'numpy.ndarray of shape {data.shape}\n')

        elif dtype == 'pdata':
            for i, (k, v) in enumerate(data.items()):
                if i == 0:
                    cats.append('Data:')
                    vals.append(f'{k}, numpy.ndarray of shape {v.shape}')
                else:
                    cats.append('')
                    vals.append(f'{k}, numpy.ndarray of shape {v.shape}')

        sw_h = self.get_sw()
        sw_p = self.get_sw(unit='ppm')

        for i, (sh, sp) in enumerate(zip(sw_h, sw_p)):
            if i == 0:
                cats.append('Sweep Width:')
            else:
                cats.append('')

            s = f'{sh:.4f}Hz ({sp:.4f}ppm) (F{i+1})'
            vals.append(s)


        off_h = self.get_offset()
        off_p = self.get_offset(unit='ppm')

        for i, (oh, op) in enumerate(zip(off_h, off_p)):
            if i == 0:
                cats.append('Transmitter Offset:')
            else:
                cats.append('')

            s = f'{oh:.4f}Hz ({op:.4f}ppm) (F{i+1})'
            vals.append(s)

        bf = self.get_bf()
        for i, b in enumerate(bf):
            if i == 0:
                cats.append('Basic Frequency:')
            else:
                cats.append('')

            s = f'{b:.4f}MHz (F{i+1})'
            vals.append(s)

        sfo = self.get_sfo()
        for i, sf in enumerate(sfo):
            if i == 0:
                cats.append('Transmitter Frequency:')
            else:
                cats.append('')

            s = f'{sf:.4f}MHz (F{i+1})'
            vals.append(s)

        nuc = self.get_nuc()
        for i, n in enumerate(nuc):
            if i == 0:
                cats.append('Nuclei:')
            else:
                cats.append('')

            vals.append(f'{n} (F{i+1})')

        # Virtual echo information
        virt_echo = self.get_virt_echo(kill=False)
        if virt_echo is None:
            pass
        else:
            cats.append(f'\n{MA}───Virtual Echo Info───{END}')
            vals.append('')
            half_echo = self.get_half_echo()
            filt_spec = self.get_filt_spec()
            bounds = zip(self.get_highs(unit='Hz'),
                         self.get_highs(unit='ppm'),
                         self.get_lows(unit='Hz'),
                         self.get_lows(unit='ppm'))

            cats.append('Virtual Echo:')
            vals.append(f'numpy.ndarray of shape {virt_echo.shape}')
            cats.append('Half Echo:')
            vals.append(f'numpy.ndarray of shape {half_echo.shape}')
            cats.append('Filtered Spectrum:')
            vals.append(f'numpy.ndarray of shape {filt_spec.shape}')

            for i, (hi, hi_p, lo, lo_p) in enumerate(bounds):
                if i == 0:
                    cats.append('Region:')

                else:
                    cats.append('')

                s = f'{hi:.4f} - {lo:.4f}Hz' + f' ({hi_p:.4f} - ' \
                    + f'{lo_p:.4f}ppm) (F{i+1})\n'

                vals.append(s)

        theta0 = self.get_theta0(kill=False)
        if theta0 is None:
            pass
        else:
            cats.append(f'\n{MA}───Matrix Pencil Info───{END}')
            vals.append('')
            cats.append('Result (theta0):')
            vals.append(f'numpy.ndarray with shape {theta0.shape}')

        theta = self.get_theta(kill=False)
        if theta is None:
            pass
        else:
            cats.append(f'\n{MA}───Nonlinear Programming Info───{END}')
            vals.append('')
            cats.append('Result (theta):')
            vals.append(f'numpy.ndarray with shape {theta.shape}')

        # string with consistent padding
        # elements in cats with magenta coloring are titles, do not
        # involve in padding considerations
        pad = max(len(c) for c in cats if f'{MA}' not in c)
        for c, v in zip(cats, vals):
            p = (pad - len(c) + 1) + len(v)
            msg += c + v.rjust(p) + '\n'

        return msg


    # get parameters from class
    def get_datapath(self):
        """Return path of the data directory.

        Returns
        -------
        datapath : str
        """
        return self.path

    def get_dtype(self):
        """Return data type.

        Returns
        -------
        dtype : 'raw', 'pdata'
        """
        return self.dtype

    def get_data(self, pdata_key=None):
        """Return the original data.

        Parameters
        ----------
        pdata_key : None or str, default: None
            If `self.dtype` is 'pdata', this specifices which component of
            data you wish to obtain. If `None`, the entire dictionary will
            be returned. If it is a string that matches a key in the
            dictionary, the matching data will be returned.

        Returns
        -------
        data : numpy.ndarray or dict

        Raises
        ------
        ValueError
            If pdata_key is not `None`, and does not match a key in
            `self.data`
        """
        if self.dtype == 'raw':
            return self.data
        elif self.dtype == 'pdata':
            if pdata_key is None:
                return self.data
            else:
                try:
                    return self.data[pdata_key]
                except KeyError:
                    msg = f'\n{R}pdata_key does not match a dataset. The' \
                          + f' available datasets are: '
                    msg += ', '.join([repr(k) for k in self.data.keys()]) + END
                    raise ValueError(msg)

    def get_n(self):
        """Return number of points the data is composed of in each
        dimesnion.

        Returns
        -------
        n : (int,) or (int, int)
        """
        return self.n

    def get_dim(self):
        """Return the data dimension.

        Returns
        -------
        dim : 1, 2
        """
        return self.dim

    def get_sw(self, unit='Hz'):
        """Return the experiment sweep width in each dimension.

        Parameters
        ----------
        unit : 'Hz' or 'ppm', default: 'Hz'
            The unit of the value(s).

        Returns
        -------
        sw : (float,) or (float, float)

        Raises
        ------
        InvalidUnitError
            If `unit` is not 'Hz' or 'ppm'
        """
        if unit == 'Hz':
            return self.sw
        elif unit == 'ppm':
            sw_p = ()
            for sw, sfo in zip(self.sw, self.sfo):
                sw_p += (sw / sfo),
            return sw_p
        else:
            raise InvalidUnitError('Hz', 'ppm')

    def get_offset(self, unit='Hz'):
        """Return the transmitter's offset frequency in each dimesnion.

        Parameters
        ----------
        unit : 'Hz' or 'ppm', default: 'Hz'
            The unit of the value(s).

        Returns
        -------
        offset : (float,) or (float, float)

        Raises
        ------
        InvalidUnitError
            If `unit` is not 'Hz' or 'ppm'
        """
        if unit == 'Hz':
            return self.off
        elif unit == 'ppm':
            off_p = ()
            for off, sfo in zip(self.off, self.sfo):
                off_p += (off / sfo),
            return off_p
        else:
            raise InvalidUnitError('Hz', 'ppm')

    def get_sfo(self):
        """Return transmitter frequency in each dimension (MHz).

        Returns
        -------
        sfo : (float,) or (float, float)
        """
        return self.sfo

    def get_bf(self):
        """Return the transmitter's basic frequency in each dimesnion (MHz).

        Returns
        -------
        bf : (float,) or (float, float)
        """
        bf = ()
        for sfo, off in zip(self.sfo, self.off):
            bf += (sfo - (off / 1E6)),
        return bf

    def get_nuc(self):
        """Return the nucelus of each channel.

        Returns
        -------
        nuc : (str,) or (str, str)
        """
        return self.nuc

    def get_shifts(self, unit='ppm'):
        """Return the sampled frequencies consistent with experiment's
        parameters (sweep width, transmitter offset, number of points).

        Parameters
        ----------
        unit : 'ppm' or 'Hz', default: 'ppm'
            The unit of the value(s).

        Returns
        -------
        shifts : (numpy.ndarray,) or (numpy.ndarray, numpy.ndarray)
            The frequencies sampled along each dimension.

        Raises
        ------
        InvalidUnitError
            If `unit` is not 'Hz' or 'ppm'
        """
        if unit == 'ppm':
            sw = self.get_sw(unit='ppm')
            off = self.get_offset(unit='ppm')

        elif unit == 'Hz':
            sw = self.get_sw()
            off = self.get_offset()

        else:
            raise InvalidUnitError('ppm', 'Hz')

        n = self.get_n()
        shifts = []
        for s, o, n_ in zip(sw, off, n):
            shifts.append(np.linspace(o+(s/2), o-(s/2), n_))
        return tuple(shifts)

    def get_tp(self):
        """Return the sampled times consistent with experiment's
        parameters (sweep width, number of points).

        Returns
        -------
        tp : (numpy.ndarray,) or (numpy.ndarray, numpy.ndarray)
            The times sampled along each dimension (seconds).
        """
        sw = self.get_sw()
        n = self.get_n()
        tp = []
        for s, n_ in zip(sw, n):
            tp.append(np.linspace(0, float(n_-1)/s, n_))
        return tuple(tp)

    def get_highs(self, unit='idx', kill=True):
        """Return the value of the point with the highest ppm value that
        is contained within the filter region specified using
        :py:meth:`virtual_echo`, in each dimension.

        Parameters
        ----------
        unit : 'idx', 'Hz', 'ppm', default: 'idx'
            The unit of the value(s). 'idx' corresponds to array index,
            'ppm' corresonds to parts per million, 'Hz' corresponds to
            Hertz.

        kill : bool, default: True
            If no attribute called `highs` exists, `kill` specifies
            how to act. If `True`, an error is raised. If `False`, None
            is returned.

        Returns
        -------
        highs : (float,), (int,), (float, float), (int, int), or None

        Raises
        ------
        AttributeNotFoundError
            If no attribute called `highs` exists, and `kill` is `True`.
        InvalidUnitError
            If `unit` is not 'idx', 'Hz', or 'ppm'
        """
        highs = self._get_nondefault_param('highs', 'virtual_echo()', kill)
        if unit == 'idx':
            return highs
        elif unit == 'Hz':
            return self._indices_to_hz(highs)
        elif unit == 'ppm':
            return self._indices_to_ppm(highs)
        else:
            raise InvalidUnitError('idx', 'Hz', 'ppm')


    def get_lows(self, unit='idx', kill=True):
        """Return the value of the point with the lowest ppm value that
        is contained within the filter region specified using
        :py:meth:`virtual_echo`, in each dimension.

        Parameters
        ----------
        unit : 'idx', 'Hz', 'ppm', default: 'idx'
            The unit of the value(s). 'idx' corresponds to array index,
            'ppm' corresonds to parts per million, 'Hz' corresponds to
            Hertz.

        kill : bool, default: True
            If no attribute called `highs` exists, `kill` specifies
            how to act. If `True`, an error is raised. If `False`, None
            is returned.

        Returns
        -------
        highs : (float,), (int,), (float, float), (int, int), or None

        Raises
        ------
        AttributeNotFoundError
            If no attribute called `highs` exists, and `kill` is `True`.
        InvalidUnitError
            If `unit` is not 'idx', 'Hz', or 'ppm'
        """
        lows = self._get_nondefault_param('lows', 'virtual_echo()', kill)
        if unit == 'idx':
            return lows
        elif unit == 'Hz':
            return self._indices_to_hz(lows)
        elif unit == 'ppm':
            return self._indices_to_ppm(lows)
        else:
            raise InvalidUnitError('idx', 'Hz', 'ppm')

    def get_p0(self, unit='rad', kill=True):
        """Return the zero-order phase correction specified using
        :py:meth:`virtual_echo`.

        Parameters
        ----------
        unit : 'rad', 'deg', default: 'rad'
            The unit to set p0 to. ``'rad'`` corresponds to radians,
            ``'deg'`` corresonds to degress.

        kill : bool, default: True
            If p0 is `None` exists, `kill` specifies how to act. If `True`,
            an error is raised. If `False`, None is returned.

        Returns
        -------
        p0 : float or None

        Raises
        ------
        AttributeIsNoneError
            Raised if ``p0`` is ``None``, and ``kill = True``
        InvalidUnitError
            If `unit` is not ``'rad'`` or ``'deg'``
        """
        p0 = self._get_nondefault_param('p0', 'virtual_echo()', kill)
        if unit == 'rad':
            return p0
        elif unit == 'deg':
            return p0 * (180 / np/pi)
        else:
            raise InvalidUnitError('rad', 'deg')


    def get_p1(self, unit='rad', kill=True):
        """Return the first-order phase correction specified using
        :py:meth:`virtual_echo`.

        Parameters
        ----------
        unit : 'rad', 'deg', default: 'rad'
            The unit to set p1 to. ``'rad'`` corresponds to radians,
            ``'deg'`` corresonds to degress.

        kill : bool, default: True
            If p1 is ``None``, `kill` specifies how to act. If ``True``,
            an error is raised. If ``False``, None is returned.

        Returns
        -------
        p1 : float or None

        Raises
        ------
        AttributeIsNoneError
            Raised if ``p1`` is ``None``, and ``kill = True``
        InvalidUnitError
            If `unit` is not ``'rad'`` or ``'deg'``
        """
        p1 = self._get_nondefault_param('p1', 'virtual_echo()', kill)
        if unit == 'rad':
            return p1
        elif unit == 'deg':
            return p1 * (180 / np/pi)
        else:
            raise InvalidUnitError('rad', 'deg')

    def get_filt_spec(self, kill=True):
        """Return the filtered spectral data generated using
        :py:meth:`virtual_echo`

        Parameters
        ----------
        kill : bool, default: True
            If no attribute called `filt_spec` exists, `kill` specifies
            how to act. If `True`, an error is raised. If `False`, None
            is returned.

        Returns
        -------
        filt_spec : numpy.ndarray or None

        Raises
        ------
        AttributeNotFoundError
            If no attribute called `filt_spec` exists, and `kill` is `True`.
        """
        return self._get_nondefault_param('filt_spec', 'virtual_echo()', kill)

    def get_virt_echo(self, kill=True):
        """Return the virtual echo data generated using
        :py:meth:`virtual_echo`

        Parameters
        ----------
        kill : bool, default: True
            If no attribute called `virt_echo` exists, `kill` specifies
            how to act. If `True`, an error is raised. If `False`, None
            is returned.

        Returns
        -------
        filt_spec : numpy.ndarray or None

        Raises
        ------
        AttributeNotFoundError
            If no attribute called `virt_echo` exists, and `kill` is `True`.
        """
        return self._get_nondefault_param('virt_echo', 'virtual_echo()', kill)

    def get_half_echo(self, kill=True):
        """Return the halved virtual echo data generated using
        :py:meth:`virtual_echo`

        Parameters
        ----------
        kill : bool, default: True
            If no attribute called `half_echo` exists, `kill` specifies
            how to act. If `True`, an error is raised. If `False`, None
            is returned.

        Returns
        -------
        filt_spec : numpy.ndarray or None

        Raises
        ------
        AttributeNotFoundError
            If no attribute called `virt_echo` exists, and `kill` is `True`.
        """
        return self._get_nondefault_param('half_echo', 'virtual_echo()', kill)

    def get_ve_sw(self, unit='Hz', kill=True):
        """Return the sweep width of the signal generated using
        :py:meth:`virtual_echo` with ``cut = True``, in each dimesnion.

        Parameters
        ----------
        unit : 'Hz', 'ppm', default: 'Hz'
            The unit of the value(s).

        kill : bool, default: True
            If no attribute called `ve_sw` exists, `kill` specifies
            how to act. If `True`, an error is raised. If `False`, None
            is returned.

        Returns
        -------
        ve_sw : (float,), (float, float), or None

        Raises
        ------
        AttributeNotFoundError
            If no attribute called `ve_sw` exists, and `kill` is `True`.
        InvalidUnitError
            If `unit` is not 'Hz' or 'ppm'
        """
        ve_sw =  self._get_nondefault_param('ve_sw',
                                            'virtual_echo() with cut=True',
                                            kill)
        if unit == 'Hz':
            return ve_sw
        elif unit == 'ppm':
            ve_sw_p = ()
            for sw, sfo in zip(ve_sw, self.sfo):
                ve_sw_p += (sw / sfo),
            return ve_sw_p
        else:
            raise InvalidUnitError('Hz', 'ppm')

    def get_ve_offset(self, unit='Hz', kill=True):
        """Return the transmitter's offest frequency of the signal generated
        using :py:meth:`virtual_echo` with ``cut = True``, in each
        dimesnion.

        Parameters
        ----------
        unit : 'Hz', 'ppm', default: 'Hz'
            The unit of the value(s).

        kill : bool, default: True
            If no attribute called `ve_off` exists, `kill` specifies
            how to act. If `True`, an error is raised. If `False`, None
            is returned.

        Returns
        -------
        ve_off : (float,), (float, float), or None

        Raises
        ------
        AttributeNotFoundError
            If no attribute called `ve_off` exists, and `kill` is `True`.
        InvalidUnitError
            If `unit` is not 'Hz' or 'ppm'
        """
        ve_off =  self._get_nondefault_param('ve_off',
                                             'virtual_echo() with cut=True',
                                             kill)
        if unit == 'Hz':
            return ve_off
        elif unit == 'ppm':
            ve_off_p = ()
            for off, sfo in zip(ve_off, self.sfo):
                ve_off_p += (off / sfo),
            return ve_off_p
        else:
            raise InvalidUnitError('Hz', 'ppm')

    def get_theta0(self, kill=True):
        """Return the parameter estimate derived using
        :py:meth:`matrix_pencil`

        Parameters
        ----------
        kill : bool, default: True
            If no attribute called `theta0` exists, `kill` specifies
            how to act. If `True`, an error is raised. If `False`, None
            is returned.

        Returns
        -------
        theta0 : numpy.ndarray or None
            An array with ``theta0.shape[1] = 4`` (1D signal) or
            ``theta0.shape[1] = 6`` (2D signal).

        Raises
        ------
        AttributeNotFoundError
            If no attribute called `theta0` exists, and `kill` is `True`.
        """
        return self._get_nondefault_param('theta0', 'matrix_pencil()', kill)

    def get_theta(self, kill=True):
        """Return the parameter estimate derived using
        :py:meth:`nonlinear_programming`

        Parameters
        ----------
        kill : bool, default: True
            If no attribute called `theta` exists, `kill` specifies
            how to act. If `True`, an error is raised. If `False`, None
            is returned.

        Returns
        -------
        theta : numpy.ndarray or None
            An array with ``theta.shape[1] = 4`` (1D signal) or
            ``theta.shape[1] = 6`` (2D signal).

        Raises
        ------
        AttributeNotFoundError
            If no attribute called `theta` exists, and `kill` is `True`.
        """
        return self._get_nondefault_param('theta', 'nonlinear_programming()',
                                          kill)

    def _get_nondefault_param(self, name, method, kill):
        """Retrieve attributes that may be assigned the value None. Warn user/
        raise error depending on the value of ``kill``"""
        if self.__dict__[name] is not None: # determine if attribute is not None
            return self.__dict__[name]
        else:
            if kill is True:
                raise AttributeIsNoneError(name, method)
            else:
                return None

    def make_fid(self, resname=None, oscs=None):
        """Constructs a synthetic FID using a parameter estimate.

        Parameters
        ----------
        resname : None, 'theta', or 'theta0', default: None
            The parameter array to use. If `None`, `self.theta` will be used
            if it exists. If not, `self.theta0` will be used if it exists.
            Otherwise, an error will be raised.

        oscs : None or list of ints, default: None
            Which oscillators to include in result. If None, all oscillators
            will be included. If a list of ints, the subset of oscillators
            corresponding to these indices will be used.

        Returns
        -------
        fid : numpy.ndarray
            The generated FID.
        """
        res = self._check_res(resname)

        if oscs:
            res = res[oscs]

        n = self.get_n()
        sw = self.get_sw()
        off = self.get_offset()
        dim = self.get_dim()

        return _misc.mkfid(res, n, sw, off, dim)

    def virtual_echo(self, highs, lows, highs_n, lows_n, p0=0.0, p1=0.0,
                     cut=False):
        """Generates phased, frequency-filtered data from the original data
        supplied.

        Parameters
        ----------
        highs : (float,) or (float, float,)
            The highest ppm value of the spectral region to consider, for
            each dimension.

        lows : (float,) or (float, float,)
            The lowest ppm value of the spectral region to consider, for
            each dimension.

        highs_n : (float,), (float, float,) or None
            The highest ppm value of a region of the signal that doesn't
            contain any noticeable signals, in each dimesion. If `cut`
            is set to `True`, it will not be considered. Set `highs_n`
            to `None` in this case.

        lows_n : (float,), (float, float,) or None
            The lowest ppm value of a region of the signal that doesn't
            contain any noticeable signals, in each dimesion. If `cut`
            is set to `True`, it will not be considered. Set `lows_n` to
            `None` in this case.

        p0 : float, default: 0.0
            Zero order phase correction to apply to the data (radians).

        p1 : float, default: 0.0
            First order phase correction to apply to the data (radians).

        cut : bool, default: False
            If `False`, the final virtual echo will comprise the
            same number of data points as the original data. Noise will
            be added to regions that are set to zero via super-Gaussian
            filtration. If `True`, the signal will be sliced, so that
            only the spectral region specified by high and low will be
            retained.

        Notes
        -----
        The class gains the following attributes following successful
        running of this method:

        * `filt_spec` - The frequency-filtered spectrum used to generate the
          resultant time domin data
        * `virt_echo` - Virtual echo, the inverse FT of `filt_spec`. This
          signal is conjugate symmetric.
        * `half_echo` - The first half of `virt_echo`. This is the signal that
          is subsequently analysed by default by :py:meth:`matrix_pencil`
          and :py:meth:`nonlinear_programming`.
        * `highs` - The value of `highs` input to the method, converted to
          the unit of array indices.
        * `lows` - The value of `lows` input to the method, converted to the
          unit of array indices.
        * `p0` - The value of `p0` input to the method.
        * `p1` - The value of `p1` input to the method.

        If `cut` is set to `True`, the following are additional attributes
        that are gained by the class:

        * `ve_sw` - The sweep width of the sliced signal, in Hz
        * `ve_off` - The transmitter offset frequency of the sliced signal,
          in Hz
        """

        n = self.n
        dim = self.dim
        sfo = self.sfo
        # check dim is valid (only 1D data supported so far)
        if dim == 2:
            raise TwoDimUnsupportedError()

        if self.dtype == 'raw':
            dtype = 'raw'
            data = np.flip(fftshift(fft(self.data)))
        elif self.dtype == 'pdata':
            dtype = 'pdata'
            data = self.get_data(pdata_key='1i') + 1j * self.get_data(pdata_key='1i')

        # convert floats to tuples if user didn't follow my documentation...
        # (1D case)
        highs = self._check_int_float(highs)
        lows = self._check_int_float(lows)
        highs_n = self._check_int_float(highs_n)
        lows_n = self._check_int_float(lows_n)

        # convert bounds from ppm values to indices
        highs_idx = self._ppm_to_indices(highs)
        lows_idx = self._ppm_to_indices(lows)
        if cut is False:
            highs_n_idx = self._ppm_to_indices(highs_n)
            lows_n_idx = self._ppm_to_indices(lows_n)

        # phase data
        data = np.real(_ve.phase(data, p0, p1))
        # generate super gaussian filter
        superg = _ve.super_gaussian(n, highs_idx, lows_idx)

        if cut is True:
            # remove spectral region outside bounds
            slice_ = tuple(np.s_[hi:lo] for hi, lo in zip(highs_idx, lows_idx))
            self.filt_spec = (np.real(data) * superg)[slice_]
            self.ve_n = self.filt_spec.shape
            # get sweep width and offset of cut signal
            ve_sw, ve_off = [], []
            for hi, lo, s in zip(highs, lows, sfo):
                ve_sw.append((hi - lo) * s)
                ve_off.append((lo + hi) * (s / 2))
            self.ve_sw, self.ve_off = tuple(ve_sw), tuple(ve_off)
        else:
            # keep regions outside the bounds, but add synthetic noise
            slice_ = tuple(np.s_[hi:lo] for hi, lo in zip(highs_n_idx, lows_n_idx))
            noise_region = np.real(data[slice_])
            var = np.var(noise_region)
            noise = _ve.sg_noise(superg, var)
            self.filt_spec = (np.real(data) * superg) + noise

        self.virt_echo = ifft(ifftshift(self.filt_spec))
        half = tuple([np.s_[0:int(np.floor(n_/2))] for n_ in self.virt_echo.shape])
        self.half_echo = 2 * self.virt_echo[half]
        self.highs = highs_idx
        self.lows = lows_idx
        self.p0 = p0
        self.p1 = p1

    def matrix_pencil(self, M_in=0, trim=None, func_print=True):
        """Implementation of the 1D Matrix Pencil Method [1]_ [2]_ or 2D
        Modified Matrix Enchancement and Matrix Pencil (MMEMP) method [3]_
        [4]_ with the option of Model Order Selection using Minumum
        Descrition Length (MDL).

        Parameters
        ----------
        M_in : int, default: 0
            The number of oscillators to use in generating a parameter
            estimate. If M is set to 0, the number of oscillators will be
            estimated using the MDL.

        trim : None, (int,), or (int, int), default: None
            If trim is a tuple, the analysed data will be sliced such that
            its shape matches trim, with the initial points in the signal
            being retained. If trim is `None`, the data will not be
            sliced. Consider using this in cases where the full signal is
            large, such that the method takes a very long time, or your PC
            has insufficient memory to process it.

        func_print : bool, deafult: True
            If True (default), the method provides information on progress
            to the terminal as it runs. If False, the method will run
            silently.

        Notes
        -----
        The method requires - and will search for - appropriate time-domain
        data to run. If frequency-filtered data has been
        generated by :py:meth:`virtual_echo` (stored in the attribute
        `half_echo`), prior to calling this method, this will be analysed.
        If no such data is found, but the original data is a raw
        FID (i.e. ``self.get_dtype() = 'raw'``), that will analysed. If the
        original data is processed data (i.e. ``self.get_dtype() = 'pdata'``),
        and no signal has been generated using :py:meth:`virtual_echo`, an
        error will be raised.

        The class gains an attribute called `theta0`; the parameter estimate
        determined by nonlinear programming. If the  data is 1D,
        ``self.theta0.shape[1] = 4``, whilst if the data is 2D,
        ``self.theta0.shape[1] = 6``. Elements along axis 0 of `theta0`
        contain the parameters associated with each individual oscillator,
        with parameters ordered as follows:

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
        # unpack required parameters
        dtype = self.get_dtype()
        dim = self.get_dim()
        sw = self.get_sw()
        off = self.get_offset()

        # look for appropriate data to analyse (see Notes in docstring)
        data = self._check_data()
        # slice data if user provided a tuple
        trim = self._check_trim(trim, data)
        data = data[tuple([np.s_[0:int(t)] for t in trim])]

        # 1D ITMPM
        if dim == 1:
            self.theta0 = _mpm.mpm_1d(data, M_in, sw[0], off[0], func_print)

        # 2D ITMEMPM
        elif dim == 2:
            self.theta0 = _mpm.mpm_2d(data, M_in, sw, off, func_print)


    def nonlinear_programming(self, trim=None, method='trust_region',
                              mode=None, bound=False, phase_variance=False,
                              maxit=None, amp_thold=None, freq_thold=None,
                              fprint=True):
        """Estimation of signal parameters using nonlinear programming, given
        an inital guess generated using the :py:meth:`matrix_pencil`.

        Parameters
        ----------
        trim : None, (int,), or (int, int), default: None
            If trim is a tuple, the analysed data will be sliced such that
            its shape matches trim, with the initial points in the signal
            being retained. If trim is `None`, the data will not be
            sliced. Consider using this in cases where the full signal is
            large, such that the method takes a very long time, or your PC
            has insufficient memory to process it.

        method : 'trust_region' or 'lbfgs', default: 'trust_region'
            The optimization method to be used. Both options use
            the ``scipy.optimize.minimize`` function [5]_

            * `'trust_region'` sets ``method='trust-constr'``
            * `'lbfgs'`, sets ``method='L-BFGS-B'``.

            See the Notes below for advice on choosing `method`.

        mode : None or str, default: None
            Sprecifies which parameters to optimize. If `None`,
            all parameters in the initial guess are subjected to the
            optimization. If a string containing a combination of the letters
            `'a'`, `'p'`, `'f'`, and `'d'` is used, then only the parameters
            specified will be considered. For example ``mode='af'`` would make
            the routine only consider amplitudes and frequencies.

        bound : bool, default: False
            Specifies whether or not to carry out an optimisation where
            the parameters are bounded. If `False`, the optimsation
            will be unconstrained. If `True`, the following bounds
            are set:

            * amplitudes: :math:`0 < a_m < \infty`
            * phases: :math:`- \pi < \phi_m \leq \pi`
            * frequencies: :math:`f_{\\mathrm{off}} - \\frac{f_{\\mathrm{sw}}}{2} \\leq f_m \\leq f_{\\mathrm{off}} + \\frac{f_{\\mathrm{sw}}}{2}`
            * damping: :math:`0 < \eta_m < \infty`

        phase_variance : bool, default: False
            Specifies whether or not to include the variance of phase in
            the cost function under consideration. It is advised this is
            set to `True` if the data is well phased.

        maxit : int or None, default: None
            The maximum number of iterations the routine will carry out
            before being formced to terminate. If None, the default number of
            maximum iterations is set (100 if ``method='trust_region'``, and
            500 if ``method='lbfgs'``).

        amp_thold : float or None, default: None
            If `None`, does nothing. If a float, oscillators with
            amplitudes satisfying :math:`a_m < a_{\\mathrm{thold}} \\times \\lVert \\boldsymbol{a} \\rVert`` will be
            removed from the parameter array, where
            :math:`\\lVert \\boldsymbol{a} \\rVert`
            is the norm of all the oscillator amplitudes. This should be
            comfortably lower than 1.

        freq_thold : float or None, default: None
            .. warning::

               NOT IMPLEMENTED YET

            If `None`, does nothing. If a float, oscillator pairs with
            frequencies satisfying
            :math:`\\lvert f_m - f_p \\rvert < f_{\\mathrm{thold}}` will be
            removed from the parameter array. A new oscillator will be included
            to the array, with parameters:

            * amplitude: :math:`a = a_m + a_p`
            * phase: :math:`\phi = \\frac{\phi_m + \phi_p}{2}`
            * frequency: :math:`f = \\frac{f_m + f_p}{2}`
            * damping: :math:`\eta = \\frac{\eta_m + \eta_p}{2}`

        fprint : bool, default: True
            If `True`, the method provides information on progress to
            the terminal as it runs. If `False`, the method will run silently.

        Raises
        ------
        NoSuitableDataError
            Raisd when this method is called on an instance that does not
            possess appropriate time-domain data for analysis (see Notes).

        PhaseVarianceAmbiguityError
            Raised when `phase_variance` is set to `True`, but the user
            has specified that they do not wish to optimise phases using the
            `mode` argument.

        NoParameterEstimateError
            Raised if no attribute called `theta0` exitst in the class
            instance.

        References
        ----------
        .. [5] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        .. [6] My paper when* published

        *if....

        Notes
        -----
        The method requires - and will search for - appropriate time-domain
        data to run. If frequency-filtered data has been
        generated by :py:meth:`virtual_echo` (stored in the attribute
        `half_echo`), prior to calling this method, this will be analysed.
        If no such data is found, but the original data is a raw
        FID (i.e. ``self.get_dtype() = 'raw'``), that will analysed. If the
        original data is processed data (i.e. ``self.get_dtype() = 'pdata'``),
        and no signal has been generated using :py:meth:`virtual_echo`, an
        error will be raised.

        The method also requires an initial guess, stored in the attribute
        `theta0`. To generate this initial guess, you first need to apply
        the :py:meth:`matrix_pencil` method.

        The class gains an attribute called `theta`; the parameter estimate
        determined by nonlinear programming. If the  data is 1D,
        ``self.theta.shape[1] = 4``, whilst if the data is 2D,
        ``self.theta.shape[1] = 6``. Elements along axis 0 of `theta`
        contain the parameters associated with each individual oscillator,
        with parameters ordered as follows:

        * :math:`[a_m, \phi_m, f_m, \eta_m]` (1D)
        * :math:`[a_m, \phi_m, f_{1,m}, f_{2,m}, \eta_{1,m}, \eta_{2,m}]` (2D)

        The two optimisation algorithms primarily differ in how they treat
        calculation of the Hessian. `'trust_region'` will calculate the
        Hessian explicitely at every iteration, whilst `'lbfgs'` uses an
        update formula based on gradient information to estimate the Hessian.
        The upshot of this is that the convergence rate (the number of
        iterations needed to reach convergence) is typically better for
        `'trust_region'`, though each iteration typically takes longer to
        generate. By default, it is advised to use `'trust_region'`, however
        if your guess has a large number of signals (as a rule of thmub,
        > 50), you may find `'lbfgs'` to performs more effectively.
        """

        # TODO: include freq threshold

        dim = self.get_dim()
        theta0 = self.get_theta0()
        sw = self.get_sw()
        off = self.get_offset()

        mode = self._check_mode(mode, phase_variance)
        self._check_method(method)
        maxit = self._check_maxit(maxit, method)
        data = self._check_data()
        trim = self._check_trim(trim, data)
        data = data[tuple([np.s_[0:int(t)] for t in trim])]

        self.theta = _nlp.nlp(data, dim, theta0, sw, off, phase_variance, method,
                         mode, bound, maxit, amp_thold, freq_thold, fprint,
                         True, None)


    def pickle_save(self, fname='NMREsPy_result.pkl', dir='.',
                    force_overwrite=False):
        """Converts the class instance of to a byte stream using Python's
        "Pickling" protocol, and saves it to a .pkl file.

        Parameters
        ----------
        fname : str, default: 'NMREsPy_result.pkl'
            Name of file to save the byte stream to. Arguments without
            an extension or with a '.pkl' extension are permitted.

        dir : str, default: '.'
            Path of the desried directory to save the .pkl file to. Default
            is the current working directory.

        force_overwrite : bool, default: False
            If `False`, if a file with the desired path already
            exists, the user will be prompted to confirm whether they wish
            to overwrite the file. If `True`, the file will be overwritten
            without prompt.

        Raises
        ------
        IOError
            When the path specified by `dir` doesn't exist.
        ValueError
            When `fname` is invalid (contains an extension that isn't '.pkl')

        Notes
        -----
        This method complements :py:func:`~nmrespy.load.pickle_load`, in that
        an instance saved using :py:func:`pickle_save` can be recovered by
        :py:func:`~nmrespy.load.pickle_load`.
        """

        if os.path.isdir(dir):
            pass
        else:
            raise IOError(f'\n{R}directory {dir} doesn\'t exist')

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

        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        print(f'{G}Saved instance of NMREsPyBruker to {path}{END}')


    def write_result(self, descrip=None, fname='NMREsPy_result', dir='.',
                     resname=None, sf=5, sci_lims=(-2,3), format='txt',
                     force_overwrite=False):
        """Saves an estimation result to a file in a human-readable format
        (either a textfile or a PDF).

        Parameters
        ----------
        descrip : str or None, default: None
            A description of the result, which is appended at the top of the
            file. If `None`, no description is added.

        fname : str, default: 'NMREsPy_result'
            The name of the result file. If `format` is 'txt', either a name
            with no extension or the extension '.txt' will be accepted. If
            `format` is 'pdf', either a name with no extension or the
            extension '.pdf' will be accepted. Default is 'NMREsPy_result.txt',
            or 'NMREsPy_result.pdf', depending on `format`.

        dir : str, default: '.'
            Path to the desried directory to save the file to. Default
            is the current working directory.

        resname : None, 'theta', or 'theta0', default: None
            The name of the attribute corresponding to the estimation result
            to be presented. The feasible string options are:

            * `'theta'` - The result derived using
              :py:meth:`nonlinear_programming`
            * `'theta0'` - The result derived using :py:meth:`matrix_pencil`

            If `resname` is `None`, `self.theta` will be used if it exists.
            If it doesn't exist, `self.theta0` will be used. If neither exist,
            an error will be thrown.

        sf : int, default: 5
            The number of significant figures used.

        sci_lims : (int, int), default: (-2, 3)
            Specifies the smallest negative and positive powers which values
            need to possess in order to be presented using scientific notaiton.
            If ``abs(value) <= 10 ** sci_lims[0]``
            or ``abs(value) >= 10 ** sci_lims[1]``,
            then the value will be displayed using scientific notation. For
            example, ``1234.56`` would be displayed as ``1.23456E3``

        format : 'txt' or 'pdf', default: 'txt'
            Specifies the format of the file. To produce a pdf, a LaTeX
            installation is required. See the Notes below for details.

        force_overwrite : bool, default: False
            If `False`, if a file with the desired path already
            exists, the user will be prompted to confirm whether they wish
            to overwrite the file. If True, the file will be overwritten
            without any prompt.

        Raises
        ------
        LaTeXFailedError
            With `format` set to `'pdf'`, this will be raised if an error
            was encountered in trying to run ``pdflatex``.

        Notes
        -----
        The be able to generate pdf files of NMR-EsPy results, it is
        necessary to have a LaTeX installation set up on your system.
        For a simple to set up implementation that is supported on most
        major operating systems, consider
        `TexLive <https://www.tug.org/texlive/>`_. To ensure that
        you have a functioning LaTeX installation, open a command prompt/
        terminal type ``pdflatex``.

        The following is a full list of packages that your LaTeX installation
        will need to successfully compile the .tex file generated by
        :py:meth:`write_result`:

        * amsmath
        * geometry
        * cmbright
        * booktabs
        * enumitem
        * graphicx
        * hyperref
        * longtable
        * siunitx
        * xcolor

        Most of these are pretty ubiquitous and are likely to be installed
        even with lightweight LaTeX installations. If you wish to check the
        packages are available, run::
            $ kpsewhich <package-name>.sty
        If a pathname appears, the package is installed.
        """

        res = self._check_res(resname)

        # check format is sensible
        if format in ['txt', 'pdf']:
            pass
        else:
            raise ValueError(f'{R}format should be \'txt\' or \'pdf\'{END}')

        # basic info
        info = []
        info.append(res)
        info.append(self.get_datapath()) # data path
        info.append(self.get_dim()) # signal dimension
        info.append(self.get_sw()) # sweep width (Hz)
        info.append(self.get_sw(unit='ppm')) # sweep width (Hz)
        info.append(self.get_offset()) # offset (Hz)
        info.append(self.get_offset(unit='ppm')) # offset (Hz)
        info.append(self.get_sfo()) # transmitter frequency
        info.append(self.get_bf()) # basic frequency
        info.append(self.get_nuc()) # channels

        # ==============
        # peak integrals
        # ==============
        integrals = []
        # dx in each dimension (gap between successive points in Hz)
        delta = []
        for sw, n_ in zip(self.get_sw(), self.get_n()):
            delta.append(sw / n_)

        for m, osc in enumerate(res):
            f = self.make_fid(resname, oscs=[m])
            s = np.absolute(np.real(fftshift(fft(f))))
            # inegrate successively over each dimension
            if self.get_dim() == 1:
                integrals.append(simps(s, dx=delta[0]))
            elif self.get_dim() == 2:
                integrals.append(simps(simps(s, dx=delta[1]), dx=delta[0]))
        info.append(integrals)

        # ===================
        # virtual echo region
        # ===================
        region_h, region_p = None, None # will contain region in Hz and ppm
        if 'highs' in self.__dict__ and 'lows' in self.__dict__:
            highs_h = self.get_highs(unit='Hz')
            highs_p = self.get_highs(unit='ppm')
            lows_h = self.get_lows(unit='Hz')
            lows_p = self.get_lows(unit='ppm')
            region_h = []
            region_p = []
            for hi_h, lo_h, hi_p, lo_p in zip(highs_h, lows_h, highs_p, lows_p):
                region_h.append((hi_h, lo_h))
                region_p.append((hi_p, lo_p))
        info.append(region_h)
        info.append(region_p)

        _write.write_file(info, descrip, fname, dir, sf, sci_lims, format,
                         force_overwrite)



    def plot_result(self, resname=None, datacol=None, osccols=None,
                    labels=True, stylesheet=None):
        """ Generates a figure with the result of an estimation routine.
        A spectrum of the original data is plotted, along with each
        individual oscillator that makes up the estimation result.

        Parameters
        ----------
        resname : None, 'theta', or 'theta0', default: None
            The name of the attribute corresponding to the estimation result
            to be presented. The feasible string options are:

            * `'theta'` - The result derived using
              :py:meth:`nonlinear_programming`
            * `'theta0'` - The result derived using :py:meth:`matrix_pencil`

            If `resname` is `None`, `self.theta` will be used if it exists.
            If it doesn't exist, `self.theta0` will be used. If neither exist,
            an error will be thrown.

        datacol : matplotlib color or None, default: None
            The color used to plot the original data. Any value that is
            recognised by matplotlib as a color is permitted. See:

            https://matplotlib.org/3.1.0/tutorials/colors/colors.html

            If `None`, the default color :grey:`#808080` will be used
            (exactly halfway between black and white).

        osccols : matplotlib color, matplotlib colormap, list, or None,\
        default: None
            Describes how to colour individual oscillators. The following
            is a complete list of options:

            * If the value denotes a matplotlib color, all oscillators will
              be given this color.
            * If a string corresponding to a matplotlib colormap is given,
              the oscillators will be consecutively shaded by linear increments
              of this colormap.
              For all valid colormaps, see:

              https://matplotlib.org/3.3.1/tutorials/colors/colormaps.html
            * If a list/numpy array containing valid matplotlib colors is
              given, these colours will be cycled.
              For example, if ``osccols=['r', 'g', 'b']``:

              + Oscillators 1, 4, 7, ... would be :red:`red` (#FF0000)
              + Oscillators 2, 5, 8, ... would be :green:`green` (#008000)
              + Oscillators 3, 6, 9, ... would be :blue:`blue` (#0000FF)

            * If `None`, the default colouring method will be applied,
              which involves cycling through the following colors:

              + :oscblue:`#1063E0`
              + :oscorange:`#EB9310`
              + :oscgreen:`#2BB539`
              + :oscred:`#D4200C`

        labels : bool, default: True
            If `True`, each oscillator will be given a numerical label
            in the plot, if `False`, no labels will be produced.

        stylesheet : None or str, default: None
            The name of/path to a matplotlib stylesheet for further
            customaisation of the plot. Note that all the features of the
            styleshet will be adhered to, except for the colors, which are
            overwritten by datacol and osccols. If `None`, a custom
            stylesheet, found in
            `/path/to/NMR-EsPy/nmrespy/config/nmrespy_custom.mplstyle`
            is used. To see built-in stylesheets that are available, enter
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
            instances. The data plot is given the key `'data'`, and the
            individual oscillator plots are given the keys `'osc1'`,
            `'osc2'`, `'osc3'`, ..., `'osc<M>'` where `<M>` is the number of
            oscillators in the parameter estimate.

        labs : dict
            If `labels` is True, this dictionary will contain a series
            of `matplotlib.text.Text <https://matplotlib.org/3.1.1/\
            api/text_api.html#matplotlib.text.Text>`_ instances, with the
            keys `'osc1'`, `'osc2'`, etc. as is the case with the `lines`
            dictionary. If `labels` is False, the dict will be empty.

        Raises
        ------
        TwoDimUnsupportedError
            If `self.dim` is 2.

        NoParameterEstimateError
            If `resname` is `None`, and neither `theta0` nor `theta` are
            attributes of `self`.

        Notes
        -----
        The `fig`, `ax`, `lines` and `labels` objects that are returned give
        you the ability to customise virtually anything you wish about the plot
        after running :py:meth:`plot_result`. If you wish to edit a particular
        line or label after running plot_result, simply use the following
        syntax: ::
            >>> fig, ax, lines, labs = example.plot_result()
            >>> lines['osc7'].set_lw(1.6) # set oscillator 7's linewith to 2
            >>> labs['osc2'].set_x(4) # move the x co-ordinate of oscillator 2's label to 4ppm

        To save the figure, simply use `savefig: <https://matplotlib.org/\
        3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html>`_
        ::
            >>> fig.savefig('example_figure.pdf', format='pdf')
        """

        dim = self.dim
        # check dim is valid (only 1D data supported so far)
        if dim == 2:
            raise TwoDimUnsupportedError()

        res = self._check_res(resname)

        if self.dtype == 'raw':
            data = np.flip(fftshift(fft(self.data)))
        elif self.dtype == 'pdata':
            data = self.data['1r']

        p0 = self.get_p0(kill=False)
        p1 = self.get_p1(kill=False)

        if p0 is None:
            pass
        else:
            data = _ve.phase(data, p0, p1)

        # FTs of FIDs generated from individual oscillators in result array
        peaks = []
        for m, osc in enumerate(res):
            f = self.make_fid(resname, oscs=[m])
            peaks.append(np.real(fftshift(fft(f))))

        if 'highs' in self.__dict__ and 'lows' in self.__dict__:
            region = [self.get_highs(), self.get_lows()]
        else:
            region = None

        nuc = self.get_nuc()
        shifts = self.get_shifts(unit='ppm')

        if dim == 1:
            return _plot.plotres_1d(data, peaks, shifts, region, nuc, datacol,
                                   osccols, labels, stylesheet)


    # ---Internal use methods---
    @staticmethod
    def _check_int_float(param):
        """Check if param is a float or int. If it is, convert to a tuple"""
        if isinstance(param, float) or isinstance(param, int):
            return param,
        return param

    def _check_data(self):
        if self.half_echo is not None:
            return self.get_half_echo()
        elif self.dtype == 'raw':
            return self.get_data()
        else:
            raise NoSuitableDataError()

    @staticmethod
    def _check_mode(mode, phase_var):
        if mode is None:
            return 'apfd'
        elif isinstance(mode, str):
            # could use regex here, but I haven't looked into it enough
            # to know what I'm doing...
            if any(c not in 'apfd' for c in mode):
                raise ValueError(f'\n{R}mode should only contain'
                                 f' the characters \'a\', \'p\', \'f\', and'
                                 f' \'d\'.{END}')
            else:
                # check each character doesn't appear more than once
                count = {}
                for c in mode:
                    if c in count:
                        count[c] += 1
                    else:
                        count[c] = 1

                for key in count:
                    if count[key] > 1:
                        raise ValueError(f'\n{R}mode should not contain any'
                                         f' repeated characters!{END}')

                # gets upset when phase variance is switched on, but phases
                # are not to be optimised (the user is being unclear about
                # their purpose)
                if phase_var is True:
                    if 'p' not in mode:
                        raise PhaseVarianceAmbiguityError(mode)

                return mode

            raise TypeError(f'\n{R}mode should be None or a string containing'
                            f' only the characters \'a\', \'p\', \'f\', and'
                            f' \'d\'.{END}')


    @staticmethod
    def _check_method(method):
        if method == 'trust_region':
            pass
        elif method == 'lbfgs':
            pass
        else:
            raise ValueError(f'\n{R}method should be \'trust_region\''
                             f' or \'lbfgs\'.{END}')

    @staticmethod
    def _check_maxit(maxit, method):
        if maxit is None:
            if method == 'trust_region':
                return 100
            elif method == 'lbfgs':
                return 500
        elif type(maxit) is int:
            return maxit
        else:
            raise TypeError(f'\n{R}maxit should be an int.{END}')

    def _check_res(self, resname):
        if resname is None:
            res = None
            for att in ['theta', 'theta0']:
                try:
                    res = getattr(self, att)
                    return res
                except AttributeError:
                    pass
            if res is None:
                raise NoParameterEstimateError()

        if resname in self.__dict__:
            if resname in ['theta', 'theta0']:
                return getattr(self, resname)
            else:
                raise ValueError(f'{R}resname should be \'theta\' or'
                                 f' \'x0\'. \'{resname}\' does not'
                                 f' correspond to an estimation result{END}')
        else:
            raise ValueError(f'{R}resname dones not refer to an attribute'
                             f'in the class!{END}')

    def _check_trim(self, trim, data):
        if trim is not None:
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

    def _ppm_to_indices(self, tup_ppm):
        """Converts tuple of values from ppm to index"""

        sw_p = self.get_sw(unit='ppm')
        off_p = self.get_offset(unit='ppm')
        n = self.get_n()

        tup_idx = ()
        for elem, sw, off, n_ in zip(tup_ppm, sw_p, off_p, n):
            tup_idx += (int(round((off + (sw / 2) - elem) * (n_ / sw)))),
        return tup_idx

    def _indices_to_ppm(self, tup_idx):
        """Converts tuple of values from indices to ppm"""

        sw_p = self.get_sw(unit='ppm')
        off_p = self.get_offset(unit='ppm')
        n = self.get_n()

        tup_ppm = ()
        for elem, sw, off, n_ in zip(tup_idx, sw_p, off_p, n):
            tup_ppm += float(off + (sw / 2) - ((elem * sw) / n_)),
        return tup_ppm

    def _hz_to_indices(self, tup_hz):
        """Converts tuple of values from Hz to index"""

        sw_h = self.get_sw()
        off_h = self.get_offset()
        n = self.get_n()

        tup_idx = ()
        for elem, sw, off, n_ in zip(tup_hz, sw_h, off_h, n):
            tup_idx += (int(round((off + (sw / 2) - elem) * (n_ / sw)))),
        return tup_idx

    def _indices_to_hz(self, tup_idx):
        """Converts tuple of values from indices to Hz"""

        sw_h = self.get_sw()
        off_h = self.get_offset()
        n = self.get_n()

        tup_hz = ()
        for elem, sw, off, n_ in zip(tup_idx, sw_h, off_h, n):
            tup_hz += float(off + (sw / 2) - ((elem * sw) / n_)),
        return tup_hz
