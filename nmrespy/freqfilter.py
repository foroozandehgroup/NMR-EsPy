# filter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Frequecy filtration of NMR data using super-Gaussian band-pass filters"""

from dataclasses import dataclass
import functools
from typing import Union

import numpy as np
import numpy.random as nrandom

from nmrespy._misc import ArgumentChecker, FrequencyConverter
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
    colorama.init()
import nmrespy._errors as errors
from nmrespy import sig


@dataclass
class FilterInfo:
    _spectrum: np.ndarray
    _sg: np.ndarray
    _sg_noise: np.ndarray
    _region: list[int]
    _noise_region: list[int]
    _cut_region: Union[list[int], None]
    _converter: FrequencyConverter

    @property
    def spectrum(self):
        """Unfiltered spectrum."""
        return self._spectrum

    @property
    def sg(self):
        """Super-Gaussian filter."""
        return self._sg

    @property
    def sg_noise(self):
        """Additive noise vector."""
        return self._sg_noise

    @property
    def shape(self):
        """Shape of :py:meth:`spectrum`."""
        return list(self.spectrum.shape)

    @property
    def cut_shape(self):
        """Shape of :py:meth:`cut_spectrum`."""
        return [r[1] - r[0] + 1 for r in self.get_cut_region(unit='idx')]

    @property
    def filtered_spectrum(self):
        """Filtered spectrum (uncut).

        Returns
        -------
        numpy.ndarray
        """
        return (self.spectrum * self.sg) + self.sg_noise

    @property
    def filtered_fid(self):
        """Filtered time-domain signal (uncut)

        Returns
        -------
        numpy.ndarray
        """
        return self._ift_and_slice(self.filtered_spectrum)

    @property
    def cut_spectrum(self):
        """Filtered, cut spectrum. If the user set ``cut=False`` when calling
        :py:func:`filter_spectrum`, :py:meth:`filtered_spectrum` will be
        returned.

        Returns
        -------
        numpy.ndarray
        """
        if self._cut_region:
            cut_slice = tuple(np.s_[r[0]:r[1] + 1]
                              for r in self.get_cut_region(unit='idx'))
            return self.filtered_spectrum[cut_slice]
        else:
            return self.filtered_spectrum

    @property
    def cut_fid(self):
        """Filtered, cut time-domain signal. If the user set ``cut=False``
        when calling :py:func:`filter_spectrum`, :py:meth:`filtered_fid` will
        be returned.

        Returns
        -------
        numpy.ndarray
        """
        if self._cut_region:
            ratios = [unct / ct
                      for unct, ct in zip(self.shape, self.cut_shape)]
            factor = np.prod(ratios)
            return self._ift_and_slice(self.cut_spectrum) / factor
        else:
            return self.filtered_fid

    def check_unit(valid_units):
        """Decorator which checks that the `unit` argument is valid"""
        def decorator(f):
            @functools.wraps(f)
            def checker(*args, **kwargs):
                if (not kwargs) or (kwargs['unit'] in valid_units):
                    return f(*args, **kwargs)
                else:
                    raise ValueError(
                        f'{cols.R}`unit` should be one of: {{'
                        + ', '.join(['\'' + v + '\'' for v in valid_units])
                        + f'}}{cols.END}'
                    )
            return checker
        return decorator

    @check_unit(['idx', 'hz', 'ppm'])
    def get_center(self, unit='hz'):
        """Get the center of the super-Gaussian filter.

        Parameters
        ----------
        unit : {'idx', 'hz', 'ppm'}
            Unit specifier.

        Returns
        center : Union[[int], [float]]
        """
        region_idx = self.get_region(unit='idx')
        center_idx = [int((r[0] + r[1]) // 2) for r in region_idx]
        return self._converter.convert(center_idx, f'idx->{unit}')

    @check_unit(['idx', 'hz', 'ppm'])
    def get_bw(self, unit='hz'):
        """Get the bandwidth of the super-Gaussian filter.

        Parameters
        ----------
        unit : {'idx', 'hz', 'ppm'}
            Unit specifier.

        Returns
        bw : Union[[int], [float]]
        """
        region = self.get_region(unit=unit)
        return [abs(r[1] - r[0]) for r in region]

    @property
    def sfo(self):
        """Transmitter frequency, in MHz.

        Returns
        -------
        sfo : [float]
        """
        return self._converter.sfo

    @check_unit(['hz', 'ppm'])
    def get_sw(self, unit='hz'):
        """Sweep width of the original spectrum.

        Parameters
        ----------
        unit : {'hz', 'ppm'}
            Unit specifier.

        Returns
        sw : [float]
        """
        return self._converter.convert(self._converter.sw, f'hz->{unit}')

    @check_unit(['hz', 'ppm'])
    def get_offset(self, unit='hz'):
        """Transmitter offset of the original spectrum.

        Parameters
        ----------
        unit : {'hz', 'ppm'}
            Unit specifier.

        Returns
        offset : [float]
        """
        return self._converter.convert(self._converter.offset, f'hz->{unit}')

    @check_unit(['idx', 'hz', 'ppm'])
    def get_region(self, unit='hz'):
        """Selected spectral region for filtration.

        Parameters
        ----------
        unit : {'idx', 'hz', 'ppm'}
            Unit specifier.

        Returns
        region : Union[[float], [int]]
        """
        return self._converter.convert(self._region, f'idx->{unit}')

    @check_unit(['idx', 'hz', 'ppm'])
    def get_noise_region(self, unit='hz'):
        """Selected spectral noise region for filtration.

        Parameters
        ----------
        unit : {'idx', 'hz', 'ppm'}
            Unit specifier.

        Returns
        noise_region : Union[[float], [int]]
        """
        return self._converter.convert(self._noise_region, f'idx->{unit}')

    @check_unit(['idx', 'hz', 'ppm'])
    def get_cut_region(self, unit='hz'):
        """Bounds of the cut spectral data.

        Parameters
        ----------
        unit : {'idx', 'hz', 'ppm'}
            Unit specifier.

        Returns
        cut_region : Union[[float], [int]]
        """
        if self._cut_region:
            cut_region = self._cut_region
        else:
            cut_region = [[0, s - 1] for s in self.shape]

        return self._converter.convert(cut_region, f'idx->{unit}')

    @check_unit(['hz', 'ppm'])
    def get_cut_sw(self, unit='hz'):
        """Sweep width of cut spectrum

        Parameters
        ----------
        unit : {'idx', 'hz', 'ppm'}
            Unit specifier.

        Returns
        cut_sw : [float]
        """
        region = self.get_cut_region(unit=unit)
        return [abs(r[1] - r[0]) for r in region]

    @check_unit(['hz', 'ppm'])
    def get_cut_offset(self, unit='hz'):
        region = self.get_cut_region(unit=unit)
        return [(r[1] + r[0]) / 2 for r in region]

    @staticmethod
    def _ift_and_slice(spectrum, slice_axes=None):

        dim = spectrum.ndim
        if slice_axes is None:
            slice_axes = list(range(dim))

        slice = []
        for i in range(dim):
            if i in slice_axes:
                slice.append(np.s_[0:int(spectrum.shape[i] // 2)])
            else:
                slice.append(np.s_[0:spectrum.shape[i]])

        return sig.ift(spectrum)[tuple(slice)]


def superg(region, shape, p=40.0):
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
    sg : numpy.ndarray
        Super-Gaussian filter.

    center : [int] or [int, int]
        Index of the center of the filter in each dimension.

    bw : [int] or [int, int]
        Bandwidth of the filter in each dimension, in terms of the number
        of points spanned.
    """

    # Determine center and bandwidth of super gaussian in each dimension
    center = [int((r[0] + r[1]) // 2) for r in region]
    bw = [abs(r[1] - r[0]) for r in region]

    # Construct super gaussian
    for i, (n, c, b) in enumerate(zip(shape, center, bw)):
        # 1D array of super-Gaussian for particular dimension
        s = np.exp(-2 ** (p + 1) * ((np.arange(1, n + 1) - c) / b) ** p)
        if i == 0:
            sg = s
        else:
            sg = sg[..., None] * s

    return sg + 1j * np.zeros(sg.shape)


def superg_noise(spectrum, noise_region, sg):
    """Given a spectrum, a region to sample the noise fron, and a
    super-Gaussian filter, construct a synthetic noise sequence.

    Parameters
    ----------
    spectrum : numpy.ndarray
        The spectrum.

    noise_region : [[int, int]] or [[int, int], [int, int]]
        The start and end indices in each dimension to slice the spectrum
        in order to sample the noise variance.

    sg : numpy.ndarray
        The super-Gaussian filter being applied to the spectrum.

    Returns
    -------
    sg_noise : numpy.ndarray
        The synthetic noise signal.
    """
    noise_slice = tuple(np.s_[n[0]:n[1] + 1] for n in noise_region)
    noise_variance = np.var(spectrum[noise_slice])
    sg_noise = nrandom.normal(0, np.sqrt(noise_variance), size=spectrum.shape)
    # Scale noise elements according to corresponding value of the
    # super-Gaussian filter
    return sg_noise * (1 - sg)


def filter_spectrum(spectrum, region, noise_region, sw, offset, sfo=None,
                    region_unit='hz', sg_power=40., cut=True, cut_ratio=3.0):
    """Applies a super-Gaussian filter to a spectrum.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Frequency-domian data.

    region : [[float, float]], or [[float, float], [float, float]]
        Boundaries specifying the region to apply the filter to.

    noise_region : Same type as `region`
        Boundaries specifying a region which does not contain any noticable
        signals (i.e. just containing noise).

    sw : [float] or [float, float]
        The sweep width of the signal in each dimension.

    offset : [float] or [float, float]
        The transmitter offset in each dimension.

    sfo : [float], [float, float] or None, default: None
        The tansmitter frequency in each dimnesion (MHz). Required as float
        list if `region_unit` is `'ppm'`.

    region_unit : {'ppm', 'hz'} default: 'hz'
        The units which the boundaries in `region` and `noise_region` are
        given in.

    sg_power : float, default: 40.
        Power of the super-Gaussian. The greater the value, the more box-like
        the filter.

    cut : bool, default: True
        If `True`, the filtered frequency-domain data will be trancated
        prior to inverse Fourier Transformation, reducing the number
        of signal points. If False, the data is not truncated after FT.

    cut_ratio : float, default: 3.0
        If `cut` is set to `True`, this gives the ratio of the cut signal's
        bandwidth and the filter bandwidth. This should be greater than 1.0.

    Returns
    -------
    filter : freqfilter.FilterInfo
    """

    # --- Check validity of parameters -------------------------------
    if not isinstance(spectrum, np.ndarray):
        raise TypeError(
            f'{cols.R}`spectrum` should be a numpy array{cols.END}'
        )

    # Determine data dimension. If greater than 2, return error.
    dim = spectrum.ndim
    if dim >= 3:
        raise errors.MoreThanTwoDimError()

    components = [
        (sw, 'sw', 'float_list'),
        (offset, 'offset', 'float_list'),
        (sg_power, 'sg_power', 'float'),
        (cut, 'cut', 'bool'),
        (cut_ratio, 'cut_ratio', 'greater_than_one'),
    ]

    if sfo:
        components.append((sfo, 'sfo', 'float_list'))

    if (region_unit == 'ppm') and (sfo is None):
        raise ValueError(
            f'{cols.R}`sfo` needs cannot be None when `region_unit` is set '
            f'to \'ppm\'{cols.END}'
        )
    elif region_unit in ['hz', 'ppm']:
        components.append((region, 'region', 'region_float'))
        components.append((noise_region, 'noise_region', 'region_float'))
    elif region_unit == 'idx':
        components.append((region, 'region', 'region_int'))
        components.append((noise_region, 'noise_region', 'region_int'))
    else:
        raise ValueError(
            f'{cols.R}`region_unit` is invalid. Should be one of {{\'hz\', '
            f'\'idx\' and \'ppm\'}}{cols.END}'
        )

    ArgumentChecker(components, dim)

    shape = list(spectrum.shape)
    # Convert region from hz or ppm to array indices
    converter = FrequencyConverter(shape, sw, offset, sfo)
    region_idx = \
        converter.convert(region, f'{region_unit}->idx')
    noise_region_idx = \
        converter.convert(noise_region, f'{region_unit}->idx')
    region_idx = [sorted(r) for r in region_idx]
    noise_region_idx = [sorted(r) for r in noise_region_idx]

    sg = superg(region_idx, shape, p=sg_power)
    noise = superg_noise(spectrum, noise_region_idx, sg)

    center = [int((r[0] + r[1]) // 2) for r in region_idx]
    bw = [abs(r[1] - r[0]) for r in region_idx]

    if cut:
        cut_idx = []
        for n, c, b in zip(shape, center, bw):
            mn = int(np.floor(c - (b / 2 * cut_ratio)))
            mx = int(np.ceil(c + (b / 2 * cut_ratio)))
            # Ensure the cut region remains within the valid span of
            # values (0 -> n-1)
            if mn < 0:
                mn = 0
            if mx >= n:
                mx = n - 1
            cut_idx.append([mn, mx])

    else:
        cut_idx = None

    return FilterInfo(
        spectrum, sg, noise, region_idx, noise_region_idx, cut_idx, converter
    )
