# filter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Frequecy filtration of NMR data using super-Gaussian band-pass filters"""

import numpy as np
import numpy.random as nrandom

from nmrespy._misc import ArgumentChecker, FrequencyConverter
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
    colorama.init()
import nmrespy._errors as errors
from nmrespy import sig


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
    sg : numpy.ndarray
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
    for i, (n, c, b) in enumerate(zip(shape, center, bw)):
        # 1D array of super-Gaussian for particular dimension
        s = np.exp(-2 ** (p + 1) * ((np.arange(1, n + 1) - c) / b) ** p)
        if i == 0:
            sg = s
        else:
            sg = sg[..., None] * s

    return sg, center, bw


class FrequencyFilter:
    """Fequency filter class.

    Parameters
    ----------
    data : numpy.ndarray
        The time-domain signal.

    region : [[int, int]], [[float, float]], [[int, int], [int, int]] or\
    [[float, float], [float, float]]
        Boundaries specifying the region to apply the filter to.

    noise_region : (Same type as `region`)
        Boundaries specifying a region which does not contain any noticable
        signals (i.e. just containing experiemntal noise).

    region_unit : 'idx', 'ppm' or 'hz', default: 'idx'
        The units which the boundaries in `region` and `noise_region` are
        given in.

    sw : [float], [float, float] or None, default: None
        The sweep width of the signal in each dimension. Required as float
        list if `region_unit` is `'ppm'` or `'hz'`.

    offset : [float], [float, float] or None, default: None
        The transmitter offset in each dimension. Required as float
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
        self, data, region, noise_region, sw, offset, sfo, region_unit='idx',
        cut=True, cut_ratio=3.0,
    ):

        # --- Check validity of parameters -------------------------------
        # Data should be a NumPy array.
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f'{cols.R}`data` should be a numpy ndarray{cols.END}'
            )

        # Determine data dimension. If greater than 2, return error.
        dim = data.ndim
        if dim >= 3:
            raise errors.MoreThanTwoDimError()

        # Components to give to ArgumentChecker
        components = [
            (cut, 'cut', 'bool'),
            (cut_ratio, 'cut_ratio', 'greater_than_one'),
            (sw, 'sw', 'float_list'),
            (offset, 'offset', 'float_list'),
            (sfo, 'sfo', 'float_list'),
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

        else:
            raise TypeError(
                f'{cols.R}`region_unit` is invalid. Should be one of: \'idx\','
                f' \'ppm\' or \'hz\'{cols.END}'
            )

        # Check arguments are valid!
        ArgumentChecker(components, dim)

        # Generate frequency-domain data
        ve = sig.make_virtual_echo([data])
        init_spec = sig.ft(ve) + ve[0]

        data_shape = list(data.shape)
        ve_shape = list(ve.shape)
        # Generate FrequencyConverter instance, which will carry out
        # conversion of region bounds to unit of array indices.
        self.converter = FrequencyConverter(data_shape, sw, offset, sfo)

        if region_unit in ['ppm', 'hz']:
            # Convert region and noise_region to array indices
            region = \
                self.converter.convert(region, f'{region_unit}->idx')
            noise_region = \
                self.converter.convert(noise_region, f'{region_unit}->idx')

        # Ensure region bounds are in correct order, i.e. [low, high],
        # and double to reflect increased size of virtual echo
        for i, (bd, nbd) in enumerate(zip(region, noise_region)):
            region[i] = [2 * min(bd), 2 * max(bd)]
            noise_region[i] = [2 * min(nbd), 2 * max(nbd)]

        self.converter.n = ve_shape

        # Generate super-Guassian, and retrieve the central index and bandwidth
        # of the filter in each dimension.
        sg, center, bw = super_gaussian(region, ve_shape)
        # Extract noise
        noise_slice = tuple(np.s_[nbd[0]:nbd[1] + 1] for nbd in noise_region)
        variance = np.var(init_spec[noise_slice])
        # Generate noise array
        noise = nrandom.normal(0, np.sqrt(variance), size=ve_shape)
        # Scale noise elements according to corresponding value of the
        # super-Gaussian filter
        noise *= (1 - sg)
        # Filter the spectrum!
        filt_spec = init_spec * sg + noise

        # --- Generate filtered FID --------------------------------------
        # This FID will have the same shape as the original data
        # If cut is False, this will be the final signal.
        # If cut is True, the norm of this signal will be utilised to
        # correctly scale the final signal derived from a cut spectrum
        half_slice = tuple(np.s_[0:int(n // 2)] for n in ve_shape)
        uncut_fid = sig.ift(filt_spec)[half_slice]

        if cut:
            mini = []
            maxi = []
            cut_slice = []
            for n, c, b in zip(ve_shape, center, bw):
                mn = int(np.floor(c - (b / 2 * cut_ratio)))
                mx = int(np.ceil(c + (b / 2 * cut_ratio)))
                # Ensure the cut region remains within the valid span of
                # values (0 -> N-1)
                if mn < 0:
                    mn = 0
                if mx >= n:
                    mx = n - 1
                mini.append(mn)
                maxi.append(mx)
                cut_slice.append(np.s_[mn:mx + 1])

            # Cut the filtered spectrum
            cut_filt_spec = filt_spec[tuple(cut_slice)]
            # Generate time-domain signal from spectrum
            cut_half_slice = \
                tuple(np.s_[0:int(s // 2)] for s in cut_filt_spec.shape)
            cut_fid = sig.ift(cut_filt_spec)[cut_half_slice]
            # Scale signals
            cut_fid = cut_fid * cut_fid.size / uncut_fid.size

            # Determine sweep width and transmitter offset of cut signal
            # NB division by 2 is to correct for doubling the region bounds
            # on account of zero filling
            cut_sw = []
            for s, cut_n, uncut_n in zip(
                sw, cut_filt_spec.shape, filt_spec.shape,
            ):
                cut_sw.append((cut_n - 1) / (uncut_n - 1) * s)

            cut_offset = self.converter.convert(
                [(mx + mn) / 2 for mx, mn in zip(mini, maxi)],
                'idx->hz',
            )

        self.fid = {'uncut': uncut_fid, 'cut': None}
        self.filtered_spectrum = {'uncut': filt_spec, 'cut': None}
        self.sw = {'uncut': sw, 'cut': None}
        self.offset = {'uncut': offset, 'cut': None}

        try:
            self.fid['cut'] = cut_fid
            self.filtered_spectrum['cut'] = cut_filt_spec
            self.sw['cut'] = cut_sw
            self.offset['cut'] = cut_offset
        except NameError:
            pass

        self.sg = sg
        self.region = region
        self.noise_region = noise_region

    def _get_cut_uncut(self, name, cut):
        if cut and self.__dict__[name]['cut'] is not None:
            return self.__dict__[name]['cut']
        else:
            return self.__dict__[name]['uncut']

    def get_fid(self, cut=True):
        """Returns frequency-filtered time domain data.

        Parameters
        ----------
        cut : bool, default: True
            If `True`, and `cut` was set to `True` when the class was
            initialised, the FID derived from the cut, filtered spectrum is
            returned. Otherwise, the FID of the uncut spectrum is returned.

        Returns
        -------
        fid : numpy.ndarray"""

        return self._get_cut_uncut('fid', cut)

    def get_filtered_spectrum(self, cut=True):
        """Returns frequency-filtered spectral data.

        Parameters
        ----------
        cut : bool, default: True
            If `True`, and `cut` was set to `True` when the class was
            initialised, the cut, filtered spectrum is returned. Otherwise,
            the uncut spectrum is returned.

        Returns
        -------
        filtered_spectrum : numpy.ndarray"""

        return self._get_cut_uncut('filtered_spectrum', cut)

    def get_fs(self, cut=True):
        """Shorthand for :py:meth:`get_filtered_spectrum`."""
        return self.get_filtered_spectrum(cut)

    def get_super_gaussian(self):
        """Returns the super-Gaussian filter used.

        Returns
        -------
        super_gaussian : numpy.ndarray
        """
        return self.sg

    def get_sg(self):
        """Shorthand for :py:meth:`get_super_gaussian`."""
        return self.sg

    def get_sw(self, unit='hz', cut=True):
        """Returns the sweep width of the cut signal

        Parameters
        ----------
        unit : {'hz', 'ppm'}, default: 'hz'
            Unit to express the sweep width in.

        cut : If `True`, and `cut` was set to `True` when the class was
            initialised, the sweep width of the cut, filtered signal is
            returned. Otherwise, the sweep width of the uncut signal is
            returned.
        """

        if unit == 'hz':
            return self._get_cut_uncut('sw', cut)
        elif unit == 'ppm':
            return self.converter.convert(
                self._get_cut_uncut('sw', cut), 'hz->ppm',
            )
        else:
            raise ValueError(
                f'{cols.R}unit should be \'ppm\' or \'hz\'{cols.END}'
            )

    def get_offset(self, unit='hz', cut=True):
        """Returns the offset of the cut signal

        Parameters
        ----------
        unit : {'hz', 'ppm'}, default: 'hz'
            Unit to express the sweep width in.

        cut : If `True`, and `cut` was set to `True` when the class was
            initialised, the offset of the cut, filtered signal is
            returned. Otherwise, the offset of the uncut signal is
            returned.
        """

        if unit == 'hz':
            return self._get_cut_uncut('offset', cut)
        elif unit == 'ppm':
            return self.converter.convert(
                self._get_cut_uncut('offset', cut), 'hz->ppm',
            )
        else:
            raise ValueError(
                f'{cols.R}unit should be \'ppm\' or \'hz\'{cols.END}'
            )

    def _get_region(self, name, unit):
        """Return either `region` or `noise_region`, based on `name`

        Parameters
        ----------
        name : 'region' or 'noise_region'
            Name of attribute to obtain.

        unit : 'idx', 'hz', 'ppm'
            Unit to express the region bounds in.

        Returns
        -------
        region : [[int, int]], [[int, int], [int, int]], [[float, float]],\
        or [[float, float], [float, float]]
        """

        if unit == 'idx':
            return self.__dict__[name]
        elif unit in ['hz', 'ppm']:
            return self.converter.convert(self.__dict__[name], f'idx->{unit}')
        else:
            raise ValueError(
                f'{cols.R}unit should be \'idx\', \'ppm\' or \'hz\'{cols.END}'
            )

    def get_region(self, unit='idx'):
        """Returns the spectral region selected

        Parameters
        ----------
        unit : 'idx', 'hz', 'ppm', default: 'idx'
            Unit to express the region bounds in.

        Returns
        -------
        region : [[int, int]], [[int, int], [int, int]], [[float, float]],\
        or [[float, float], [float, float]]
        """

        return self._get_region('region', unit)

    def get_noise_region(self, unit='idx'):
        """Returns the spectral noise region selected

        Parameters
        ----------
        unit : 'idx', 'hz', 'ppm', default: 'idx'
            Unit to express the region bounds in.

        Returns
        -------
        noise_region : [[int, int]], [[int, int], [int, int]],\
        [[float, float]], or [[float, float], [float, float]]
        """

        return self._get_region('noise_region', unit)
