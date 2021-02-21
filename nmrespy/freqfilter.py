# filter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Frequecy filtration of NMR data using super-Gaussian band-pass filters"""

import copy
import itertools

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
import numpy.random as nrandom
import scipy.linalg as slinalg

from nmrespy._misc import ArgumentChecker, FrequencyConverter
import nmrespy.signal as signal


def super_gaussian(region, shape, p=40.0):
    """
    Generates a super-Gaussian for filtration of frequency-domian data.

    .. math::

      g\\left[n_1, \\cdots, n_D\\right] =
      \\exp \\left[ \\sum\\limits_{d=1}^D -2^{p+1}
      \\left(\\frac{n_d - c_d}{b_d}\\right)^p\\right]

    Parameters
    ----------
    region : [[int, int]] or [[int, int], [int, int]]
        The region for the filter to span. For each dimension, a list
        of 2 entries should exist, with the first element specifying the
        low boundary of the region, and the second element specifying the
        high boundary of the region (in array indices). Note that for a given
        dimension :math:`d`,

    shape : [int] or [int, int]
        The number of elements along each axis.

    p : float, default: 40.0
        Power of the super-Gaussian. The greater the value, the more box-like
        the filter.

    Returns
    -------
    super_gaussian : numpy.ndarray
        Super-Gaussian filter.

    center : [int] or [int, int]
        Index of the center of the filter in each dimension.

    bw : [int] or [int, int]
        Bandwidth of the filter in each dimension, in terms of the number
        of points spanned.
    """

    # Determine center and bandwidth of super gaussian in each dimension
    center = []
    bw = []
    for bounds in region:
        center.append(int((bounds[0] + bounds[1]) // 2))
        bw.append(bounds[1] - bounds[0])

    # Construct super gaussian
    for n, c, b in zip(shape, center, bw):
        # 1D array of super-Gaussian for particular dimension
        sg = np.exp(-2**(p+1) * ((np.arange(1, n+1) - c) / b) ** p)
        try:
            super_gaussian = super_gaussian[..., None] * sg
        except NameError:
            super_gaussian = sg

    return super_gaussian, center, bw


class FrequencyFilter:
    """Fequency filter class.

    Parameters
    ----------
    data : numpy.ndarray
        The time-domain signal.

    region : [[int, int]], [[float, float]], [[int, int], [int, int]] or [[float, float], [float, float]]
        Boundaries specifying the region to apply the filter to.

    noise_region : (Same type as `region`)
        Boundaries specifying a region which does not contain any noticable
        signals (i.e. just containing experiemntal noise).

    region_unit : 'idx', 'ppm' or 'hz', default: 'idx'
        The units which the boundaries in `region` and `noise_region` are
        given in.

    sw : [float], [float, float] or None, default: None
        The sweep width of the signal in each dimension. Required as float list
        if `region_unit` is `'ppm'` or `'hz'`.

    offset : [float], [float, float] or None, default: None
        The transmitter offset in each dimension (Hz). Required as float
        list if `region_unit` is `'ppm'` or `'hz'`.

    sfo : [float], [float, float] or None, default: None
        The tansmitter frequency in each dimnesion (MHz). Required as float
        list if `region_unit` is `'ppm'` or `'hz'`.

    p0 : [float], [float, float], or None default: None
        Zero-order phase correction in each dimension in radians. If `None`,
        the phase will be set to `0.0` in each dimension.

    p1 : [float] or [float, float], default: [0.0, 0.0]
        First-order phase correction in each dimension in radians. If `None`,
        the phase will be set to `0.0` in each dimension.

    cut : bool, default: True
        If `True`, the filtered frequency-domain data will be trancated
        prior to inverse Fourier Transformation, reducing the number
        of signal points. If False, the data is not truncated after FT.

    cut_ratio : float, default: 3.0
        If `cut` is set to `True`, this gives the ratio of the cut signal's
        bandwidth and the filter bandwidth. This should be greater than 1.0.

    Notes
    -----
    .. todo::
       Write me!
    """

    def __init__(
        self, data, region, noise_region, region_unit='idx', sw=None,
        offset=None, sfo=None, cut=True, cut_ratio=3.0,
    ):

        # --- Check validity of parameters -------------------------------
        # Data should be a NumPy array.
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f'{cols.R}data should be a numpy ndarray{cols.END}'
            )
        self.data = data

        # Number of points in each dimension
        self.n = list(data.shape)

        # Determine data dimension. If greater than 2, return error.
        self.dim = self.data.ndim
        if self.dim >= 3:
            raise errors.MoreThanTwoDimError()

        # Components to give to ArgumentChecker
        components = [
            (cut, 'cut', 'bool'),
            (cut_ratio, 'cut_ratio', 'greater_than_one'),
        ]

        # Type of object that region and noise region should be.
        # If the unit is idx, the values inside these lists should
        # be ints. Otherwise, they should be floats
        if region_unit == 'idx':
            components.append((region, 'region', 'region_int'))
            components.append((noise_region, 'noise_region', 'region_int'))

        elif region_unit in ['ppm', 'hz']:
            components.append((region, 'region', 'region_float'))
            components.append((noise_region, 'noise_region', 'region_float'))
            # If region boundaries are specified in ppm or hz, require sweep
            # width, offset and sfo in order to convert to array indices.
            components.append((sw, 'sw', 'float_list'))
            components.append((offset, 'offset', 'float_list'))
            components.append((sfo, 'sfo', 'float_list'))

        else:
            raise TypeError(
                f'{cols.R}region_unit is invalid. Should be one of: \'idx\','
                f' \'ppm\' or \'hz\'{cols.END}'
            )

        # Check arguments are valid!
        ArgumentChecker(components, self.dim)

        self.cut = cut
        self.cut_ratio = cut_ratio

        # Generate FrequencyConverter instance, which will carry out
        # conversion of region bounds to unit of array indices.
        if sw != None and offset != None and sfo != None:
            self.converter = FrequencyConverter(self.n, sw, offset, sfo)

            if region_unit in ['ppm', 'hz']:
                # Convert region and noise_region to array indices
                self.region = self.converter.convert(
                    region, f'{region_unit}->idx',
                )
                self.noise_region = self.converter.convert(
                    noise_region, f'{region_unit}->idx',
                )

            else:
                # region and noise_region already in unit of array indices
                self.region = region
                self.noise_region = noise_region

        # Ensure region bounds are in correct order, i.e. [low, high],
        # and double values (to reflect zero-filling)
        for i, (bd, nbd) in enumerate(zip(self.region, self.noise_region)):
            if bd[0] < bd[1]:
                self.region[i] = [2 * bd[0], 2 * bd[1]]
            else:
                self.region[i] = [2 * bd[1], 2 * bd[0]]
            if nbd[0] < nbd[1]:
                self.noise_region[i] = [2 * nbd[0], 2 * nbd[1]]
            else:
                self.noise_region[i] = [2 * nbd[1], 2 * nbd[0]]

        # --- Generate frequency-domain data -----------------------------
        # Zero fill data to double its size in each dimension
        for axis in range(self.dim):
            try:
                self.zf_data = np.concatenate(
                (self.zf_data, np.zeros_like(self.zf_data)), axis=axis,
                dtype='complex',
            )
            except AttributeError:
                self.zf_data = np.concatenate(
                (self.data, np.zeros_like(self.data)), axis=axis,
                dtype='complex'
            )
        # Fourier transform
        self.init_spectrum = np.real(signal.ft(self.zf_data))

        # --- Generate super-Gaussian filter and assocaited noise --------
        # Shape of full spectrum
        shape = self.init_spectrum.shape
        # Generate super-Guassian, and retrieve the central index and bandwidth
        # of the filter in each dimension.
        self.super_gaussian, self.center, self.bw = \
            super_gaussian(self.region, shape)
        # Extract noise
        noise_slice = tuple(np.s_[bds[0]:bds[1]] for bds in self.noise_region)
        noise = self.init_spectrum[noise_slice]
        # Determine noise mean and variance
        mean = np.mean(noise)
        variance = np.var(noise)
        # Generate noise array
        self.noise = nrandom.normal(0, np.sqrt(variance), size=shape)
        # Scale noise elements according to corresponding value of the
        # super-Gaussian filter
        self.noise = self.noise * (1 - self.super_gaussian)
        # Correct for veritcal baseline shift
        # TODO consult Ali - should this be done?
        # self.noise += mean * (1 - self.super_gaussian)
        # Filter the spectrum!
        self.filtered_spectrum = \
            self.init_spectrum * self.super_gaussian + self.noise

        # --- Generate filtered FID --------------------------------------
        # This FID will have the same shape as the original data
        # If cut is False, this will be the final signal.
        # If cut is True, the norm of this signal will be utilised to
        # correctly scale the final signal derived from a cut spectrum
        uncut_ve = 2 * signal.ift(self.filtered_spectrum)
        half_slice = tuple(np.s_[0:int(s // 2)] for s in uncut_ve.shape)
        uncut_fid = uncut_ve[half_slice]

        if cut:
            cut_slice = []
            for n, c, b in zip(shape, self.center, self.bw):
                _min = int(np.floor(c - (b / 2 * self.cut_ratio)))
                _max = int(np.ceil(c + (b / 2 * self.cut_ratio)))
                # Ensure the cut region remains within the valid span of
                # values (0 -> N-1)
                if _min < 0:
                    _min = 0
                if _max > n:
                    _max = n
                cut_slice.append(np.s_[_min:_max])

            # Cut the filtered spectrum
            self.filtered_spectrum = self.filtered_spectrum[tuple(cut_slice)]
            # Generate time-domain signal from spectrum
            cut_ve = signal.ift(self.filtered_spectrum)
            half_slice = tuple(np.s_[0:int(s // 2)] for s in cut_ve.shape)
            cut_fid = cut_ve[half_slice]

            # Get norms of cut and uncut signals
            uncut_norm = slinalg.norm(uncut_fid)
            cut_norm = slinalg.norm(cut_fid)
            self.filtered_signal = cut_norm / uncut_norm * cut_fid
            self.virtual_echo = cut_norm / uncut_norm * cut_ve

            # Determine sweep width and transmitter offset of cut signal
            # NB division by 2 is to correct for doubling the region bounds
            # on account of zero filling
            min_hz, max_hz = \
                [self.converter.convert([int(x / 2)], 'idx->hz') for x in [_min, _max]]
            self.sw = [abs(mx - mn) for mx, mn in zip(max_hz, min_hz)]
            self.offset = [(mx + mn) / 2 for mx, mn in zip(max_hz, min_hz)]

        else:
            self.sw = sw
            self.offset = offset
            self.filtered_signal = uncut_fid
            self.virtual_echo = uncut_ve

        # Need to halve region indices to correct for removal of half the
        # signal
        for i, (bd, nbd) in enumerate(zip(self.region, self.noise_region)):
            self.region[i] = [int(bd[0] / 2), int(bd[1] / 2)]
            self.noise_region[i] = [int(nbd[0] / 2), int(nbd[1] / 2)]

    def get_filtered_signal(self):
        """Returns frequency-filtered time domain data."""
        return self.filtered_signal

    def get_filtered_spectrum(self):
        """Returns frequency-filtered spectral data."""
        return self.filtered_spectrum

    def get_super_gaussian(self):
        """Returns super-Gaussian filter used."""
        return self.super_gaussian

    def get_synthetic_noise(self):
        """Returns synthetic Gaussian noise added to the filtered
        frequency-domain data."""
        return self.noise

    def get_virtual_echo(self):
        """Returns the virtual echo of the filtered spectrum (this signal
        will be conjugate symmetric)."""
        return self.virtual_echo

    def get_sw(self, unit='hz'):
        """Returns the sweep width of the cut signal

        Parameters
        ----------
        unit : 'hz', 'ppm', default: 'hz'
            Unit to express the sweep width in.
        """
        if unit == 'hz':
            return self.sw
        elif unit == 'ppm':
            return self.converter.convert(self.sw, f'hz->ppm')
        else:
            raise ValueError(
                f'{cols.R}unit should be \'ppm\' or \'hz\'{cols.END}'
            )

    def get_offset(self, unit='hz'):
        """Returns the transmitter offset of the cut signal

        Parameters
        ----------
        unit : 'hz', 'ppm', default: 'hz'
            Unit to express the sweep width in.
        """
        if unit == 'hz':
            return self.offset
        elif unit == 'ppm':
            return self.converter.convert(self.offset, f'hz->ppm')
        else:
            raise ValueError(
                f'{cols.R}unit should be \'ppm\' or \'hz\'{cols.END}'
            )

    def get_region(self, unit='idx'):
        """Returns the spectral region selected

        Parameters
        ----------
        unit : 'idx', 'hz', 'ppm', default: 'idx'
            Unit to express the region bounds in.
        """
        return self._get_region('region', unit)

    def get_noise_region(self, unit='idx'):
        """Returns the spectral noise region selected

        Parameters
        ----------
        unit : 'idx', 'hz', 'ppm', default: 'idx'
            Unit to express the region bounds in.
        """
        return self._get_region('noise_region', unit)


    def _get_region(self, name, unit):
        """Return either `region` or `noise_region`, based on `name`

        Parameters
        ----------
        name : 'region' or 'noise_region'
            Name of attribute to obtain.

        unit : 'idx', 'hz', 'ppm'
            Unit to express the region bounds in.
        """
        if unit == 'idx':
            return self.__dict__[name]
        elif unit in ['hz', 'ppm']:
            return self.converter.convert(self.__dict__[name], f'idx->{unit}')
        else:
            raise ValueError(
                f'{cols.R}unit should be \'idx\', \'ppm\' or \'hz\'{cols.END}'
            )
