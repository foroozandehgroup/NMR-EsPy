# filter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Frequecy filtration of NMR data using super-Gaussian band-pass filters."""
import functools
from typing import Iterable, NewType, Tuple, Union

import numpy as np
import numpy.random as nrandom

from nmrespy import RED, END, USE_COLORAMA, ExpInfo
if USE_COLORAMA:
    import colorama
    colorama.init()
from nmrespy._misc import ArgumentChecker, FrequencyConverter
import nmrespy._errors as errors
from nmrespy import sig


RegionIntType = NewType(
    'RegionIntType',
    Union[
        Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]],
        None
    ]
)

RegionIntFloatType = NewType(
    'RegionIntFloatType',
    Union[
        Union[
            Tuple[Union[int, float], Union[int, float]],
            Tuple[Tuple[Union[int, float], Union[int, float]],
                  Tuple[Union[int, float], Union[int, float]]],
        ],
        None
    ]
)


# Used to have this as a dataclass but as the chemistry Linux machines
# cannot have Python > 3.6, I have resorted back to a vanilla class.
class FilterInfo:
    """Object describing filtration proceedure.

    .. note::
        This should not be invoked directly, but is instead created by calling
        the :py:func:`filter_spectrum` function.

    Parameters
    ----------
    _spectrum
        Spectral data to be filtered.

    _sg
        Super-Gaussian filter applied to the spectrum.

    _sg_noise
        Additive Gaussian noise added to the spectrum.

    _region
        Region (in array indices) of the spectrum selected for filtration.

    _noise_region
        Region (in array indices) of the spectrum selected for determining
        the noise variance.

    _cut_region
        Region (in array indices) of the spectral data that was sliced away
        from the original afer filtering.

    _converter
        Used for converting between Hz, ppm, and array indices.
    """

    def __init__(
        self, _spectrum: np.ndarray, _sg: np.ndarray, _sg_noise: np.ndarray,
        _region: RegionIntType, _noise_region: RegionIntType,
        _cut_region: RegionIntType, _converter: FrequencyConverter
    ) -> None:
        self.__dict__.update(locals())

    @property
    def cut_expinfo(self) -> ExpInfo:
        """Get :py:class:`nmrespy.ExpInfo` for the cut signal."""
        pts = self.cut_shape
        sw = self.get_cut_sw()
        offset = self.get_cut_offset()
        sfo = self.sfo
        return ExpInfo(pts=pts, sw=sw, offset=offset, sfo=sfo)

    @property
    def uncut_expinfo(self) -> ExpInfo:
        """Get :py:class:`nmrespy.ExpInfo` for the uncut signal."""
        pts = self.shape
        sw = self.get_sw()
        offset = self.get_offset()
        sfo = self.sfo
        return ExpInfo(pts=pts, sw=sw, offset=offset, sfo=sfo)

    @property
    def expinfo(self) -> ExpInfo:
        """Get :py:class:`nmrespy.ExpInfo` for the filtered signal.

        If a cut region has been specified, returns :py:meth:`cut_expinfo`.
        Otherwise, returns :py:meth:`uncut_expinfo`.
        """
        if self._cut_region is not None:
            return self.cut_expinfo
        else:
            return self.uncut_expinfo

    @property
    def spectrum(self) -> np.ndarray:
        """Get unfiltered spectrum."""
        return self._spectrum

    @property
    def sg(self) -> np.ndarray:
        """Get super-Gaussian filter."""
        return self._sg

    @property
    def sg_noise(self) -> np.ndarray:
        """Get additive noise vector."""
        return self._sg_noise

    @property
    def shape(self) -> Iterable[int]:
        """Get shape of :py:meth:`spectrum`."""
        return self.spectrum.shape

    @property
    def cut_shape(self) -> Iterable[int]:
        """Get shape of :py:meth:`cut_spectrum`."""
        return tuple(
            [r[1] - r[0] + 1 for r in self.get_cut_region(unit='idx')]
        )

    @property
    def filtered_spectrum(self) -> np.ndarray:
        """Get filtered spectrum (uncut)."""
        return (self.spectrum * self.sg) + self.sg_noise

    @property
    def filtered_fid(self) -> np.ndarray:
        """Get filtered time-domain signal (uncut)."""
        return self._ift_and_slice(self.filtered_spectrum)

    @property
    def cut_spectrum(self) -> np.ndarray:
        """Get filtered, cut spectrum.

        If the user set ``cut=False`` when calling :py:func:`filter_spectrum`,
        :py:meth:`filtered_spectrum` will be returned.
        """
        if self._cut_region:
            cut_slice = tuple(np.s_[r[0]:r[1] + 1]
                              for r in self.get_cut_region(unit='idx'))
            return self.filtered_spectrum[cut_slice]
        else:
            return self.filtered_spectrum

    @property
    def cut_fid(self) -> np.ndarray:
        """Get filtered, cut time-domain signal.

        If the user set ``cut=False`` when calling :py:func:`filter_spectrum`,
        :py:meth:`filtered_fid` will be returned.
        """
        if self._cut_region:
            ratios = [unct / ct
                      for unct, ct in zip(self.shape, self.cut_shape)]
            factor = np.prod(ratios)
            return self._ift_and_slice(self.cut_spectrum) / factor
        else:
            return self.filtered_fid

    def _check_unit(valid_units):
        """Check that the `unit` argument is valid (decorator)."""
        def decorator(f):
            @functools.wraps(f)
            def checker(*args, **kwargs):
                if (not kwargs) or (kwargs['unit'] in valid_units):
                    return f(*args, **kwargs)
                else:
                    raise ValueError(
                        f'{RED}`unit` should be one of: {{'
                        + ', '.join(['\'' + v + '\'' for v in valid_units])
                        + f'}}{END}'
                    )
            return checker
        return decorator

    @_check_unit(['idx', 'hz', 'ppm'])
    def get_center(self, unit: str = 'hz') -> Iterable[Union[int, float]]:
        """Get the center of the super-Gaussian filter.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'idx'``, ``'hz'``, ``'ppm'``.
        """
        region_idx = self.get_region(unit='idx')
        center_idx = tuple([int((r[0] + r[1]) // 2) for r in region_idx])
        return self._converter.convert(center_idx, f'idx->{unit}')

    @_check_unit(['idx', 'hz', 'ppm'])
    def get_bw(self, unit: str = 'hz') -> Iterable[Union[int, float]]:
        """Get the bandwidth of the super-Gaussian filter.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'idx'``, ``'hz'``, ``'ppm'``.
        """
        region = self.get_region(unit=unit)
        return tuple([abs(r[1] - r[0]) for r in region])

    @property
    def sfo(self) -> Iterable[float]:
        """Transmitter frequency, in MHz."""
        return self._converter.sfo

    @_check_unit(['hz', 'ppm'])
    def get_sw(self, unit: str = 'hz') -> Iterable[float]:
        """Sweep width of the original spectrum.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'hz'``, ``'ppm'``.
        """
        return self._converter.convert(self._converter.sw, f'hz->{unit}')

    @_check_unit(['hz', 'ppm'])
    def get_offset(self, unit: str = 'hz') -> Iterable[float]:
        """Transmitter offset of the original spectrum.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'hz'``, ``'ppm'``.
        """
        return self._converter.convert(self._converter.offset, f'hz->{unit}')

    @_check_unit(['idx', 'hz', 'ppm'])
    def get_region(
        self, unit: str = 'hz'
    ) -> RegionIntFloatType:
        """Get selected spectral region for filtration.

        Parameters
        ----------
        unit : {'idx', 'hz', 'ppm'}
            Unit specifier. Should be one of ``'hz'``, ``'ppm'``, ``'idx'``.
        """
        return self._converter.convert(self._region, f'idx->{unit}')

    @_check_unit(['idx', 'hz', 'ppm'])
    def get_noise_region(
        self, unit: str = 'hz'
    ) -> RegionIntFloatType:
        """Get selected spectral noise region for filtration.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'hz'``, ``'ppm'``, ``'idx'``.
        """
        return self._converter.convert(self._noise_region, f'idx->{unit}')

    @_check_unit(['idx', 'hz', 'ppm'])
    def get_cut_region(
        self, unit: str = 'hz'
    ) -> RegionIntFloatType:
        """Bounds of the cut spectral data.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'hz'``, ``'ppm'``, ``'idx'``.
        """
        if self._cut_region:
            cut_region = self._cut_region
        else:
            cut_region = tuple([[0, s - 1] for s in self.shape])

        return self._converter.convert(cut_region, f'idx->{unit}')

    @_check_unit(['hz', 'ppm'])
    def get_cut_sw(self, unit: str = 'hz') -> Iterable[float]:
        """Sweep width of cut spectrum.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'hz'``, ``'ppm'``.
        """
        region = self.get_cut_region(unit=unit)
        return tuple([abs(r[1] - r[0]) for r in region])

    @_check_unit(['hz', 'ppm'])
    def get_cut_offset(self, unit: str = 'hz') -> Iterable[float]:
        """Transmitter offset of cut spectrum.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'hz'``, ``'ppm'``.
        """
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


def superg(region: RegionIntType, shape: Iterable[int],
           p: float = 40.0) -> np.ndarray:
    r"""Generate a super-Gaussian for filtration of frequency-domian data.

    The super-Gaussian is described by the following expression:

    .. math::

      g\left[n_1, \cdots, n_D\right] =
      \exp \left[ \sum\limits_{d=1}^D -2^{p+1}
      \left(\frac{n_d - c_d}{b_d}\right)^p\right]

    Parameters
    ----------
    region
        The region for the filter to span. For each dimension, a list
        of 2 entries should exist, with the first element specifying the
        low boundary of the region, and the second element specifying the
        high boundary of the region (in array indices).

    shape
        The number of elements along each axis.

    p
        Power of the super-Gaussian. The greater the value, the more box-like
        the filter.

    Returns
    -------
    sg: numpy.ndarray
        Super-Gaussian filter.
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


def superg_noise(spectrum: np.ndarray, noise_region: RegionIntType,
                 sg: np.ndarray) -> np.ndarray:
    """Construct a synthetic noise sequence to add to the filtered spectrum.

    Parameters
    ----------
    spectrum
        The spectrum.

    noise_region
        The start and end indices in each dimension to slice the spectrum
        in order to sample the noise variance.

    sg
        The super-Gaussian filter being applied to the spectrum.

    Returns
    -------
    sg_noise: numpy.ndarray
        The synthetic noise signal.
    """
    noise_slice = tuple(np.s_[n[0]:n[1] + 1] for n in noise_region)
    noise_variance = np.var(spectrum[noise_slice])
    sg_noise = nrandom.normal(0, np.sqrt(noise_variance), size=spectrum.shape)
    # Scale noise elements according to corresponding value of the
    # super-Gaussian filter
    return sg_noise * (1 - sg)


def filter_spectrum(
    spectrum: np.ndarray, region: RegionIntFloatType,
    noise_region: RegionIntFloatType, expinfo: ExpInfo,
    region_unit: str = 'hz', sg_power: float = 40., cut: bool = True,
    cut_ratio: Union[float, None] = 3.0
) -> FilterInfo:
    """Frequency filtering via super-Gaussian filtration of spectral data.

    Parameters
    ----------
    spectrum
        Frequency-domian data.

    region
        Boundaries specifying the region to apply the filter to.

    noise_region
        Boundaries specifying a region which does not contain any noticable
        signals (i.e. just containing noise).

    expinfo
        Information on the experiment. Used to determine the sweep width,
        transmitter offset and transmitter frequency.

    region_unit
        The units which the boundaries in `region` and `noise_region` are
        given in. Should be one of ``'hz'`` or ``'ppm'``.

    sg_power
        Power of the super-Gaussian. The greater the value, the more box-like
        the filter.

    cut
        If ``True``, the filtered frequency-domain data will be trancated
        prior to inverse Fourier Transformation, reducing the number
        of signal points. If ``False``, the data is not truncated after FT.

    cut_ratio
        If ``cut`` is set to ``True``, this gives the ratio of the cut
        signal's bandwidth and the filter bandwidth. This should be greater
        than ``1.0``.

    Returns
    -------
    filterinfo: :py:class:`FilterInfo`
        Object with various attributes relating to the filtration process.

    Notes
    -----
    **Region specification**

    For a :math:`d`-dimensional experiment, the ``region`` and ``noise_region``
    arguments should be array-like objects (``list``, ``tuple``, etc.)
    containing :math:`d` length-2 array-like objects. Each of these specifies
    the boundaries of the region of interest in each dimension.

    As an example, for a 2-dimensional dataset, where the desired region
    is from 4 - 4.5ppm in dimension 1 and 1.2 - 1.6ppm in dimension 2,
    ``region`` would be specified as: ``((4., 4.5), (1.2, 1.6))``. Note that
    the order of values in each dimension is not important. Also,
    ``region_unit`` would have to be manually set as ``'ppm'`` in this
    example, as regions are expected in Hz by default.
    """
    if not isinstance(expinfo, ExpInfo):
        raise TypeError(f'{RED}Check `expinfo` is valid.{END}')
    dim = expinfo.unpack('dim')

    try:
        if dim != spectrum.ndim:
            raise ValueError(
                f'{RED}The dimension of `expinfo` does not agree with the '
                f'number of dimensions in `spectrum`.{END}'
            )
        elif dim == 2:
            raise errors.TwoDimUnsupportedError()
        elif dim >= 3:
            raise errors.MoreThanTwoDimError()
    except AttributeError:
        # spectrum.ndim raised an attribute error
        raise TypeError(
            f'{RED}`spectrum` should be a numpy array{END}'
        )

    checker = ArgumentChecker(dim=dim)
    checker.stage(
        (spectrum, 'spectrum', 'ndarray'),
        (sg_power, 'sg_power', 'float'),
        (cut, 'cut', 'bool'),
        (cut_ratio, 'cut_ratio', 'greater_than_one')
    )

    if (region_unit == 'ppm') and (expinfo.sfo is None):
        raise ValueError(
            f'{RED}`region_unit` is set to \'ppm\', but `sfo` has not been '
            f'specified in `expinfo`.{END}'
        )
    elif region_unit in ['hz', 'ppm']:
        checker.stage(
            (region, 'region', 'region_float'),
            (noise_region, 'noise_region', 'region_float')
        )
    elif region_unit == 'idx':
        checker.stage(
            (region, 'region', 'region_int'),
            (noise_region, 'noise_region', 'region_int')
        )
    else:
        raise ValueError(
            f'{RED}`region_unit` is invalid. Should be one of {{\'hz\', '
            f'\'idx\' and \'ppm\'}}{END}'
        )

    checker.check()

    expinfo.pts = spectrum.shape
    # Convert region from hz or ppm to array indices
    converter = FrequencyConverter(expinfo)
    region_idx = \
        converter.convert(region, f'{region_unit}->idx')
    noise_region_idx = \
        converter.convert(noise_region, f'{region_unit}->idx')
    region_idx = tuple([tuple(sorted(r)) for r in region_idx])
    noise_region_idx = tuple([tuple(sorted(r)) for r in noise_region_idx])

    sg = superg(region_idx, expinfo.pts, p=sg_power)
    noise = superg_noise(spectrum, noise_region_idx, sg)

    center = tuple([int((r[0] + r[1]) // 2) for r in region_idx])
    bw = tuple([abs(r[1] - r[0]) for r in region_idx])

    if cut:
        cut_idx = []
        for n, c, b in zip(expinfo.pts, center, bw):
            mn = int(np.floor(c - (b / 2 * cut_ratio)))
            mx = int(np.ceil(c + (b / 2 * cut_ratio)))
            # Ensure the cut region remains within the valid span of
            # values (0 -> n-1)
            if mn < 0:
                mn = 0
            if mx >= n:
                mx = n - 1
            cut_idx.append((mn, mx))
        cut_idx = tuple(cut_idx)
    else:
        cut_idx = None

    return FilterInfo(
        spectrum, sg, noise, region_idx, noise_region_idx, cut_idx, converter
    )
