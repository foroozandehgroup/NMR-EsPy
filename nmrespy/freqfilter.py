# freqfilter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 13 Oct 2021 17:43:02 BST

"""Frequecy filtration of NMR data using super-Gaussian band-pass filters."""
import functools
import operator
from typing import Iterable, NewType, Tuple, Union

import numpy as np
import numpy.random as nrandom
from scipy.optimize import minimize
from sklearn import linear_model

from nmrespy import RED, END, USE_COLORAMA, ExpInfo

if USE_COLORAMA:
    import colorama

    colorama.init()
from nmrespy._misc import ArgumentChecker, FrequencyConverter
import nmrespy._errors as errors
from nmrespy import sig


RegionIntType = NewType(
    "RegionIntType",
    Union[Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]], None],
)

RegionIntFloatType = NewType(
    "RegionIntFloatType",
    Union[
        Union[
            Tuple[Union[int, float], Union[int, float]],
            Tuple[
                Tuple[Union[int, float], Union[int, float]],
                Tuple[Union[int, float], Union[int, float]],
            ],
        ],
        None,
    ],
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

    _expinfo
        Experiment Information.

    _region
        Region (in array indices) of the spectrum selected for filtration.

    _noise_region
        Region (in array indices) of the spectrum selected for determining
        the noise variance.

    _sg_power
        Power of super-Gaussian filter

    _converter
        Used for converting between Hz, ppm, and array indices.
    """

    def __init__(
        self,
        _spectrum: np.ndarray,
        _expinfo: ExpInfo,
        _region: RegionIntType,
        _noise_region: RegionIntType,
        _sg_power: float,
        _converter: FrequencyConverter,
    ) -> None:
        self.__dict__.update(locals())
        self._sg_noise = None

    @property
    def spectrum(self) -> np.ndarray:
        """Get unfiltered spectrum."""
        return self._spectrum

    @property
    def sg(self) -> np.ndarray:
        """Get super-Gaussian filter."""
        return superg(self._region, self._spectrum.shape, self._sg_power)

    @property
    def sg_noise(self) -> np.ndarray:
        """Get additive noise vector."""
        if self._sg_noise is None:
            self._sg_noise = superg_noise(
                self.spectrum, self.get_noise_region(unit="idx"), self.sg
            )
        return self._sg_noise

    @property
    def shape(self) -> Iterable[int]:
        """Get shape of :py:meth:`spectrum`."""
        return self.spectrum.shape

    @property
    def sg_power(self) -> float:
        return self._sg_power

    @property
    def _filtered_unfixed_spectrum(self):
        """Filtered spectrum without any baseline fix."""
        return (self.spectrum * self.sg) + self.sg_noise

    def get_filtered_spectrum(
        self,
        *,
        cut_ratio: Union[float, None] = 1.1,
        fix_baseline: bool = False,
    ) -> Tuple[np.ndarray, ExpInfo]:
        """Get filtered spectrum."""
        if isinstance(cut_ratio, float):
            cut_idx = self._cut_indices(cut_ratio)
        elif cut_ratio is None:
            cut_idx = tuple([(0, s - 1) for s in self.shape])
        else:
            raise TypeError(f"{RED}`cut_ratio` should be a float or None{END}")

        cut_slice = tuple([slice(lft, rgt + 1) for lft, rgt in cut_idx])
        filtered_spectrum = self._filtered_unfixed_spectrum
        if fix_baseline:
            filtered_spectrum += self._baseline_fix(filtered_spectrum)
        filtered_spectrum = filtered_spectrum[cut_slice]
        if isinstance(cut_ratio, float):
            scaling_factor = 0.5 * self._cut_scaling_factor(cut_ratio)
            filtered_spectrum *= scaling_factor

        pts = filtered_spectrum.shape
        cut_hz = self._converter.convert(cut_idx, "idx->hz")
        sw = tuple([abs(lft - rgt) for lft, rgt in cut_hz])
        offset = tuple([(lft + rgt) / 2 for lft, rgt in cut_hz])
        sfo, nuclei = self._expinfo.unpack("sfo", "nuclei")
        expinfo = ExpInfo(
            pts=pts,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuceli=nuclei,
        )
        return filtered_spectrum, expinfo

    def get_filtered_fid(
        self, cut_ratio: Union[float, None] = 1.1, fix_baseline: bool = False,
    ) -> Tuple[np.ndarray, ExpInfo]:
        filtered_spectrum, expinfo = self.get_filtered_spectrum(
            cut_ratio=cut_ratio, fix_baseline=fix_baseline
        )
        filtered_fid = self._ift_and_slice(filtered_spectrum)
        return filtered_fid, expinfo

    def _cut_indices(self, cut_ratio: float):
        center = self.get_center(unit="idx")
        bw = self.get_bw(unit="idx")
        pts = self.spectrum.shape
        cut_idx = []
        for n, c, b in zip(pts, center, bw):
            mn = int(np.floor(c - (b / 2 * cut_ratio)))
            mx = int(np.ceil(c + (b / 2 * cut_ratio)))
            # ensure the cut region remains within the valid span of
            # values (0 -> n-1)
            if mn < 0:
                mn = 0
            if mx >= n:
                mx = n - 1
            cut_idx.append((mn, mx))
        return tuple(cut_idx)

    def _cut_scaling_factor(self, cut_ratio: float) -> float:
        cut_idx = self._cut_indices(cut_ratio)
        cut_shape = tuple([rgt - lft for lft, rgt in cut_idx])
        ratios = [cut / uncut for cut, uncut in zip(cut_shape, self.shape)]
        return functools.reduce(operator.mul, ratios)

    def _check_unit(valid_units):
        """Check that the `unit` argument is valid (decorator)."""

        def decorator(f):
            @functools.wraps(f)
            def checker(*args, **kwargs):
                if (not kwargs) or (kwargs["unit"] in valid_units):
                    return f(*args, **kwargs)
                else:
                    raise ValueError(
                        f"{RED}`unit` should be one of: {{"
                        + ", ".join(["'" + v + "'" for v in valid_units])
                        + f"}}{END}"
                    )

            return checker

        return decorator

    @_check_unit(["idx", "hz", "ppm"])
    def get_center(self, unit: str = "hz") -> Iterable[Union[int, float]]:
        """Get the center of the super-Gaussian filter.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'idx'``, ``'hz'``, ``'ppm'``.
        """
        region_idx = self.get_region(unit="idx")
        center_idx = tuple([int((r[0] + r[1]) // 2) for r in region_idx])
        return self._converter.convert(center_idx, f"idx->{unit}")

    @_check_unit(["idx", "hz", "ppm"])
    def get_bw(self, unit: str = "hz") -> Iterable[Union[int, float]]:
        """Get the bandwidth of the super-Gaussian filter.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'idx'``, ``'hz'``, ``'ppm'``.
        """
        region = self.get_region(unit=unit)
        return tuple([abs(r[1] - r[0]) for r in region])

    @_check_unit(["idx", "hz", "ppm"])
    def get_region(self, unit: str = "hz") -> RegionIntFloatType:
        """Get selected spectral region for filtration.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'hz'``, ``'ppm'``, ``'idx'``.
        """
        return self._converter.convert(self._region, f"idx->{unit}")

    @_check_unit(["idx", "hz", "ppm"])
    def get_noise_region(self, unit: str = "hz") -> RegionIntFloatType:
        """Get selected spectral noise region for filtration.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'hz'``, ``'ppm'``, ``'idx'``.
        """
        return self._converter.convert(self._noise_region, f"idx->{unit}")

    @staticmethod
    def _ift_and_slice(spectrum, slice_axes=None):
        dim = spectrum.ndim
        if slice_axes is None:
            slice_axes = list(range(dim))

        fid_slice = []
        factor = 1
        for i in range(dim):
            if i in slice_axes:
                factor *= 2
                fid_slice.append(slice(0, int(spectrum.shape[i] // 2)))
            else:
                fid_slice.append(slice(0, spectrum.shape[i]))
        return factor * sig.ift(spectrum)[tuple(fid_slice)]

    def _get_cf_boundaries(self) -> Iterable[Tuple[slice]]:
        def is_small(x):
            return x < 1e-6

        def is_large(x):
            return x > 1 - 1e-3

        ndim = len(self.shape)
        region = self.get_region(unit="idx")
        sg = np.real(self.sg)
        boundaries = []
        for dim, bounds in enumerate(region):
            for i, bound in enumerate(bounds):
                rcutoff = None
                lcutoff = None
                shift = 0
                while any([x is None for x in (lcutoff, rcutoff)]):
                    if i == 0:
                        if lcutoff is None and is_small(sg[bound - shift]):
                            lcutoff = bound - shift
                        if rcutoff is None and is_large(sg[bound + shift]):
                            rcutoff = bound + shift
                    if i == 1:
                        if lcutoff is None and is_large(sg[bound - shift]):
                            lcutoff = bound - shift
                        if rcutoff is None and is_small(sg[bound + shift]):
                            rcutoff = bound + shift
                    shift += 1
                boundaries.append(
                    tuple(
                        dim * [slice(None, None, None)]
                        + [slice(lcutoff, rcutoff)]
                        + (ndim - dim - 1) * [slice(None, None, None)]
                    )
                )

        return tuple(boundaries)

    def _fit_sg(
        self, filtered_spectrum: np.ndarray, boundaries: Iterable[Tuple[slice]]
    ) -> np.ndarray:
        amp = 0.0
        slices = len(boundaries)
        for i, bounds in enumerate(boundaries):
            if i % 2 == 0:
                amp -= filtered_spectrum[bounds[0].stop] / slices
            elif i % 2 == 1:
                amp -= filtered_spectrum[bounds[0].start] / slices
        sg = self.sg
        args = (filtered_spectrum, sg, boundaries)
        amp = minimize(self._sg_cost, amp, args=args, method="BFGS")["x"]
        return amp * sg

    def _fit_line(
        self, filtered_spectrum: np.ndarray, boundaries: Iterable[Tuple[slice]]
    ):
        x1 = boundaries[0][0].stop
        x2 = boundaries[1][0].start
        y1 = filtered_spectrum[x1]
        y2 = filtered_spectrum[x2]
        m = (y2 - y1) / (x2 - x1)
        c = (y2 + y1) - m * (x2 + x1) / 2
        x0 = (m, c)
        sg = self.sg
        args = (filtered_spectrum, sg, boundaries)
        m, c = minimize(self._linear_cost, x0, args=args, method="BFGS")["x"]
        line = -(m * np.arange(filtered_spectrum.size) + c)
        return line * sg

    def _fit_quadratic(
        self, filtered_spectrum: np.ndarray, boundaries: Iterable[Tuple[slice]]
    ):
        x1 = boundaries[0][0].stop
        x2 = boundaries[1][0].start
        y1 = filtered_spectrum[x1]
        y2 = filtered_spectrum[x2]
        a = 0.0
        b = (y2 - y1) / (x2 - x1)
        c = (y2 + y1) - b * (x2 + x1) / 2
        x0 = (a, b, c)
        sg = self.sg
        args = (filtered_spectrum, sg, boundaries)
        a, b, c = minimize(self._quadratic_cost, x0, args=args, method="BFGS")["x"]
        points = np.arange(sg.size)
        quadratic = -(a * (points ** 2) + b * points + c)
        return quadratic * sg

    def _baseline_fix(self, filtered_spectrum: np.ndarray) -> np.ndarray:
        boundaries = self._get_cf_boundaries()
        fix = self._fit_sg(filtered_spectrum, boundaries)
        fix += self._fit_quadratic(filtered_spectrum + fix, boundaries)
        return fix

    @staticmethod
    def _sg_cost(amp, *args):
        spectrum, sg, boundaries = args
        cf = 0.0
        for i, bounds in enumerate(boundaries):
            spectrum_slice = spectrum[bounds]
            sg_slice = sg[bounds]
            cf += np.sum((spectrum_slice + (amp * sg_slice)) ** 2)
        return cf

    @staticmethod
    def _linear_cost(coeffs, *args):
        m, c = coeffs
        spectrum, sg, boundaries = args
        cf = 0.0
        for i, bounds in enumerate(boundaries):
            spectrum_slice = spectrum[bounds]
            line = (-(m * np.arange(spectrum.size) + c) * sg)[bounds]
            cf += np.sum((spectrum_slice + line) ** 2)
        return cf

    @staticmethod
    def _quadratic_cost(coeffs, *args):
        a, b, c = coeffs
        spectrum, sg, boundaries = args
        cf = 0.0
        for i, bounds in enumerate(boundaries):
            spectrum_slice = spectrum[bounds]
            points = np.arange(spectrum.size)
            quad = (-(a * (points ** 2) + b * points + c) * sg)[bounds]
            cf += np.sum((spectrum_slice + quad) ** 2)
        return cf


def superg(region: RegionIntType, shape: Iterable[int], p: float = 40.0) -> np.ndarray:
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
        s = np.exp(-(2 ** (p + 1)) * ((np.arange(1, n + 1) - c) / b) ** p)
        if i == 0:
            sg = s
        else:
            sg = sg[..., None] * s

    return sg + 1j * np.zeros(sg.shape)


def superg_noise(
    spectrum: np.ndarray, noise_region: RegionIntType, sg: np.ndarray
) -> np.ndarray:
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
    # TODO generalise the linear regression to 2d and beyond
    noise_slice = tuple(np.s_[n[0] : n[1] + 1] for n in noise_region)
    noise = np.real(spectrum[noise_slice])
    # Remove linear component from baseline
    reg = linear_model.LinearRegression()
    reg.fit(np.arange(noise.size).reshape(-1, 1), noise.reshape(-1, 1))
    m = reg.coef_[0][0]
    c = reg.intercept_[0]
    noise -= m * np.arange(noise.size) + c
    noise_variance = np.var(spectrum[noise_slice])
    sg_noise = nrandom.normal(0, np.sqrt(noise_variance), size=spectrum.shape)
    # Scale noise elements according to corresponding value of the
    # super-Gaussian filter
    return sg_noise * (1 - sg)


def filter_spectrum(
    spectrum: np.ndarray,
    expinfo: ExpInfo,
    region: RegionIntFloatType,
    noise_region: RegionIntFloatType,
    *,
    region_unit: str = "hz",
    sg_power: float = 40.0,
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
        raise TypeError(f"{RED}Check `expinfo` is valid.{END}")
    dim = expinfo.unpack("dim")

    try:
        if dim != spectrum.ndim:
            raise ValueError(
                f"{RED}The dimension of `expinfo` does not agree with the "
                f"number of dimensions in `spectrum`.{END}"
            )
        elif dim == 2:
            raise errors.TwoDimUnsupportedError()
        elif dim >= 3:
            raise errors.MoreThanTwoDimError()
    except AttributeError:
        # spectrum.ndim raised an attribute error
        raise TypeError(f"{RED}`spectrum` should be a numpy array{END}")

    checker = ArgumentChecker(dim=dim)
    checker.stage(
        (spectrum, "spectrum", "ndarray"),
        (sg_power, "sg_power", "float"),
    )

    if (region_unit == "ppm") and (expinfo.sfo is None):
        raise ValueError(
            f"{RED}`region_unit` is set to 'ppm', but `sfo` has not been "
            f"specified in `expinfo`.{END}"
        )
    elif region_unit in ["hz", "ppm"]:
        checker.stage(
            (region, "region", "region_float"),
            (noise_region, "noise_region", "region_float"),
        )
    elif region_unit == "idx":
        checker.stage(
            (region, "region", "region_int"),
            (noise_region, "noise_region", "region_int"),
        )
    else:
        raise ValueError(
            f"{RED}`region_unit` is invalid. Should be one of {{'hz', "
            f"'idx' and 'ppm'}}{END}"
        )

    checker.check()

    expinfo.pts = spectrum.shape
    # Convert region from hz or ppm to array indices
    converter = FrequencyConverter(expinfo)
    region_idx = converter.convert(region, f"{region_unit}->idx")
    noise_region_idx = converter.convert(noise_region, f"{region_unit}->idx")
    region_idx = tuple([tuple(sorted(r)) for r in region_idx])
    noise_region_idx = tuple([tuple(sorted(r)) for r in noise_region_idx])

    return FilterInfo(
        spectrum, expinfo, region_idx, noise_region_idx, sg_power, converter
    )
