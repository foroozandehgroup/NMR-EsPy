# freqfilter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 10 May 2022 20:06:24 BST

"""Frequecy filtration of NMR data using super-Gaussian band-pass filters.

MWE
---

.. literalinclude:: examples/filter_example.py

.. image:: media/filter_example.png
"""

# TODO: I have commeneted out all code relating to baseline fixing.
# I may look into this in the future to see if I can achieve improvements
# in filtering performance.
# SH, 16-3-22

from __future__ import annotations
import copy
import functools
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import numpy.linalg as nlinalg
import numpy.random as nrandom
# from scipy.optimize import minimize

from nmrespy import ExpInfo
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy import sig


# Long-winded region types
RegionFloat = Union[Iterable[Optional[Tuple[float, float]]], Tuple[float, float]]
RegionInt = Union[Iterable[Optional[Tuple[int, int]]], Tuple[int, int]]
Region = Union[RegionInt, RegionFloat]


class Filter(ExpInfo):
    """Object with tools to generate frequency-filtered NMR data."""

    def __init__(
        self,
        fid: np.ndarray,
        expinfo: ExpInfo,
        region: Region,
        noise_region: Region,
        region_unit: str = "hz",
        sg_power: float = 40.0,
        twodim_dtype: Optional[str] = None,
    ) -> None:
        """Initialise an instance of the class.

        Parameters
        ----------
        fid
            Time-domain data to derive frequency-filtered data from.

        expinfo
            Experiment Information.

        region
            Region (in array indices) of the spectrum selected for filtration.

        noise_region
            Region (in array indices) of the spectrum selected for determining
            the noise variance.

        region_unit
            The units which the boundaries in `region` and `noise_region` are
            given in. Should be one of ``"hz"`` or ``"ppm"``.

        sg_power
            Power of the super-Gaussian. The greater the value, the more box-like
            the filter.

        Notes
        -----
        **Region specification**

        For a :math:`d`-dimensional experiment, the ``region`` and
        ``noise_region`` arguments should be array-like objects (``list``,
        ``tuple``, etc.) containing :math:`d` length-2 array-like objects. Each
        of these specifies the boundaries of the region of interest in each
        dimension. If no filtering is to be applied to a particular dimension
        (i.e. the F1 dimension of a pseudo-2D dataset), set the element for
        this dimension to ``None``.

        As an example, for a 2-dimensional dataset, where the desired region is
        from 4 - 4.5ppm in dimension 1 and 1.2 - 1.6ppm in dimension 2,
        ``region`` would be specified as: ``((4., 4.5), (1.2, 1.6))``. Note
        that the order of values in each dimension is not important. Also,
        ``region_unit`` would have to be manually set as ``'ppm'`` in this
        example, as regions are expected in Hz by default.
        """
        sanity_check(
            ("expinfo", expinfo, sfuncs.check_expinfo),
            ("sg_power", sg_power, sfuncs.check_float, (), {"greater_than_one": True}),
            ("fid", fid, sfuncs.check_ndarray, (expinfo.dim,)),
        )

        self._fid = fid
        self._sg_power = sg_power

        super().__init__(
            dim=expinfo.dim,
            sw=expinfo.sw("hz"),
            offset=expinfo.offset("hz"),
            sfo=expinfo.sfo,
            nuclei=expinfo.nuclei,
            default_pts=self._fid.shape,
            fn_mode=expinfo.fn_mode,
        )

        if self.dim == 2:
            sanity_check(
                ("twodim_dtype", twodim_dtype, sfuncs.check_one_of, ("jres",)),
            )
        elif self.dim == 3:
            sanity_check(
                ("twodim_dtype", twodim_dtype, sfuncs.check_one_of, ("amp", "phase")),
            )

        sanity_check(
            (
                "region_unit", region_unit, sfuncs.check_frequency_unit,
                (self.hz_ppm_valid,),
            ),
        )

        region_check_args = (
            self.sw(region_unit),
            self.offset(region_unit),
        )

        sanity_check(
            ("region", region, sfuncs.check_region, region_check_args),
            ("noise_region", noise_region, sfuncs.check_region, region_check_args),
        )

        ve = sig.make_virtual_echo(self._fid, twodim_dtype)
        # Need to set default points before using the frequency converter
        # to convert to array indices
        self.default_pts = ve.shape

        self._region = self._process_region(region, region_unit)
        self._noise_region = self._process_region(noise_region, region_unit)
        self._spectrum = sig.ft(ve, axes=self.axes)

    @property
    def axes(self):
        return list(
            filter(
                lambda x: x is not None,
                [i if r is not None else None for i, r in enumerate(self._region)],
            )
        )

    def _process_region(
        self,
        region: Union[Iterable[Optional[Tuple[float, float]]], Tuple[float, float]],
        region_unit: str,
    ) -> Iterable[Tuple[int, int]]:
        if self.dim == 1 and len(region) == 2:
            region = [region]
        return tuple(
            [tuple(sorted(r)) if r is not None else None
             for r in self.convert(region, f"{region_unit}->idx")]
        )

    @property
    def spectrum(self) -> np.ndarray:
        """Get unfiltered spectrum."""
        return self._spectrum

    @property
    def sg(self) -> np.ndarray:
        """Get super-Gaussian filter."""
        if getattr(self, "_sg", None) is None:
            self._sg = self._superg()
        return self._sg

    @property
    def sg_noise(self) -> np.ndarray:
        """Get additive noise vector."""
        if getattr(self, "_sg_noise", None) is None:
            self._sg_noise = self._superg_noise()
        return self._sg_noise

    @property
    def shape(self) -> Iterable[int]:
        """Get shape of the spectrum."""
        return self.spectrum.shape

    @property
    def sg_power(self) -> float:
        """Get the power of the Super-Gaussian."""
        return self._sg_power

    @property
    def _filtered_unfixed_spectrum(self):
        """Filtered spectrum without any baseline fix."""
        return (self.spectrum * self.sg) + self.sg_noise

    def get_filtered_spectrum(
        self,
        *,
        cut_ratio: Optional[float] = 1.1,
        # fix_baseline: bool = False,
    ) -> Tuple[np.ndarray, ExpInfo]:
        """Get filtered spectrum.

        Parameters
        ----------
        cut_ratio
            If a float, the filtered frequency-domain data will be trancated
            reducing the number of signal points. ``cut_ratio`` gives the ratio of
            the cut spectrum's bandwidth and the filter bandwidth. This should be
            greater than ``1.0``. If ``None``, no spectrum truncation will be carried
            out.

        Returns
        -------
        filtered_spectrum
            Filtered spectrum.

        expinfo
            Experiment information corresponding to the filtered signal.
        """
        sanity_check(
            (
                "cut_ratio", cut_ratio, sfuncs.check_float, (),
                {"greater_than_one": True}, True,
            )
            # ("fix_baseline", fix_baseline, sfuncs.check_bool),
        )

        filtered_spectrum = self._filtered_unfixed_spectrum
        # if fix_baseline:
        #     filtered_spectrum += self._baseline_fix(filtered_spectrum)

        if isinstance(cut_ratio, float):
            filtered_spectrum = filtered_spectrum[self._cut_slice(cut_ratio)]
            scaling_factor = self._cut_scaling_factor(cut_ratio)
            filtered_spectrum *= scaling_factor

            cut_hz = self.convert(
                self._cut_indices(cut_ratio), "idx->hz"
            )
            sw = tuple([abs(lft - rgt) for lft, rgt in cut_hz])
            offset = tuple([(lft + rgt) / 2 for lft, rgt in cut_hz])
            sfo, nuclei = self.sfo, self.nuclei

            expinfo = ExpInfo(
                dim=self.dim,
                sw=sw,
                offset=offset,
                sfo=sfo,
                nuceli=nuclei,
                default_pts=filtered_spectrum.shape,
            )

        else:
            expinfo = ExpInfo(
                dim=self.dim,
                sw=self.sw("hz"),
                offset=self.offset("hz"),
                sfo=self.sfo,
                nuceli=self.nuclei,
                default_pts=self.default_pts,
            )

        return filtered_spectrum, expinfo

    def get_filtered_fid(
        self,
        cut_ratio: Optional[float] = 1.1,
        # fix_baseline: bool = False,
    ) -> Tuple[np.ndarray, ExpInfo]:
        """Get filtered FID.

        Parameters
        ----------
        cut_ratio
            If a float, the filtered frequency-domain data will be trancated prior
            to IFT, reducing the number of signal points. ``cut_ratio`` gives the
            ratio of the cut spectrum's bandwidth and the filter bandwidth. This
            must be greater than ``1.0``. If ``None``, no spectrum truncation will
            be carried out before IFT.
        """
        filtered_spectrum, expinfo = self.get_filtered_spectrum(
            cut_ratio=cut_ratio,  # fix_baseline=fix_baseline
        )
        filtered_fid = self._ift_and_slice(filtered_spectrum)
        expinfo._default_pts = filtered_fid.shape
        return filtered_fid, expinfo

    def _superg(self) -> np.ndarray:
        r"""Super-Gaussian for filtration of frequency-domian data.

        The super-Gaussian is described by the following expression:

        .. math::

          g\left[n_1, \cdots, n_D\right] =
          \exp \left[ \sum\limits_{d=1}^D -2^{p+1}
          \left(\frac{n_d - c_d}{b_d}\right)^p\right]
        """
        center = self.get_center(unit="idx")
        bw = self.get_bw(unit="idx")
        p = self.sg_power

        # Construct super gaussian
        for i, (n, c, b) in enumerate(zip(self.shape, center, bw)):
            # 1D array of super-Gaussian for particular dimension
            if c is None:
                s = np.ones(n)
            else:
                s = np.exp(-(2 ** (p + 1)) * ((np.arange(1, n + 1) - c) / b) ** p)

            if i == 0:
                sg = s
            else:
                sg = sg[..., None] * s

        return sg + (1j * np.zeros(sg.shape))

    def _superg_noise(self) -> np.ndarray:
        """Construct a synthetic noise sequence to add to the filtered spectrum."""
        noise_slice = tuple(
            slice(r[0], r[1] + 1) if r is not None else slice(0, s)
            for r, s in zip(self.get_noise_region(unit="idx"), self.shape)
        )
        noise = copy.deepcopy(self._spectrum)[noise_slice].real
        # Remove linear term from noise. Goal is to remove non-flat baseline shape
        # from contributing to the noise variance determination.
        noise -= self._linear_correction(noise)

        variance = np.var(noise)
        sg_noise = nrandom.normal(0, np.sqrt(variance), size=self.shape)
        # Scale noise elements according to corresponding value of the
        # super-Gaussian filter
        sg_noise *= (1 - self.sg.real)

        return sg_noise

    @staticmethod
    def _linear_correction(noise: np.ndarray) -> np.ndarray:
        """Determine linear component in noise array.

        Utilised as a means to improve the noise variance computation (see
        :py:meth:`_superg_noise`).

        Parameters
        ----------
        noise
            Noise array to correct.

        Returns
        -------
        correction
            Linear term to remove from the noise.
        """
        X = np.ones((noise.size, noise.ndim + 1))
        X[:, 1:] = (np.indices(noise.shape).reshape(noise.ndim, noise.size)).T
        y = noise.reshape(-1, 1)
        return (X @ (nlinalg.pinv(X.T @ X) @ X.T @ y)).reshape(noise.shape)

    def _cut_indices(self, cut_ratio: float) -> Iterable[Tuple[int, int]]:
        """L and R bounds of cut spectrum in relation to the full spectrum."""
        center = self.get_center(unit="idx")
        bw = self.get_bw(unit="idx")
        pts = self.spectrum.shape
        cut_idx = []
        for n, c, b in zip(pts, center, bw):
            if c is None:
                mn = 0
                mx = n - 1
            else:
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

    def _cut_slice(self, cut_ratio: float) -> Iterable[slice]:
        """Slice to extract cut spectrum."""
        return tuple(
            [slice(lft, rgt + 1) for lft, rgt in self._cut_indices(cut_ratio)]
        )

    def _cut_shape(self, cut_ratio: float) -> Iterable[int]:
        """Shape of a cut spectrum."""
        return tuple([rgt - lft for lft, rgt in self._cut_indices(cut_ratio)])

    def _cut_scaling_factor(self, cut_ratio: float) -> float:
        """Ratio of size of cut spectrum and full spectrum in each dimension."""
        return functools.reduce(
            lambda x, y: x * y,
            [cut / uncut for cut, uncut in
             zip(self._cut_shape(cut_ratio), self.shape)],
        )

    def get_center(self, unit: str = "hz") -> Iterable[Optional[Union[int, float]]]:
        """Get the center of the super-Gaussian filter.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'idx'``, ``'hz'``, ``'ppm'``.
        """
        sanity_check(
            ("unit", unit, sfuncs.check_one_of, ("idx", "hz", "ppm")),
        )
        region_idx = self.get_region(unit="idx")
        center_idx = tuple(
            [int((r[0] + r[1]) // 2) if r is not None else None
             for r in region_idx]
        )
        return self.convert(center_idx, f"idx->{unit}")

    def get_bw(self, unit: str = "hz") -> Iterable[Optional[Union[int, float]]]:
        """Get the bandwidth of the super-Gaussian filter.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'idx'``, ``'hz'``, ``'ppm'``.
        """
        sanity_check(
            ("unit", unit, sfuncs.check_one_of, ("idx", "hz", "ppm")),
        )
        region = self.get_region(unit=unit)
        return tuple([abs(r[1] - r[0]) if r is not None else None for r in region])

    def get_region(self, unit: str = "hz") -> Region:
        """Get selected spectral region for filtration.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'hz'``, ``'ppm'``, ``'idx'``.
        """
        sanity_check(
            ("unit", unit, sfuncs.check_one_of, ("idx", "hz", "ppm")),
        )
        return self.convert(self._region, f"idx->{unit}")

    def get_noise_region(self, unit: str = "hz") -> Region:
        """Get selected spectral noise region for filtration.

        Parameters
        ----------
        unit
            Unit specifier. Should be one of ``'hz'``, ``'ppm'``, ``'idx'``.
        """
        sanity_check(
            ("unit", unit, sfuncs.check_one_of, ("idx", "hz", "ppm")),
        )
        return self.convert(self._noise_region, f"idx->{unit}")

    def _ift_and_slice(self, filtered_spectrum: np.ndarray) -> np.ndarray:
        """Inverse Fourier Transform and slice a filtered spectrum.

        Any dimension that has been filtered and IFTed is sliced in half to remove the
        second half of the virtual echo obtained from the IFT of a real spectrum.
        """
        fid_slice = []
        factor = 2 ** (len(self.axes) - 1)
        fid_slice = tuple(
            [slice(0, filtered_spectrum.shape[i] // 2) if i in self.axes
             else slice(0, filtered_spectrum.shape[i])
             for i in range(self.dim)]
        )

        return factor * sig.ift(filtered_spectrum, axes=self.axes)[fid_slice]

    # ================
    # Commented stuff below is related to baseline fixing.
    # May bring back to life at some point if I find it is actually useful.

    # def _get_cf_boundaries(self) -> Iterable[Tuple[slice]]:
    #     def is_small(x):
    #         return x < 1e-6

    #     def is_large(x):
    #         return x > 1 - 1e-3

    #     ndim = len(self.shape)
    #     region = self.get_region(unit="idx")
    #     sg = np.real(self.sg)
    #     boundaries = []
    #     for dim, bounds in enumerate(region):
    #         for i, bound in enumerate(bounds):
    #             rcutoff = None
    #             lcutoff = None
    #             shift = 0
    #             while any([x is None for x in (lcutoff, rcutoff)]):
    #                 if i == 0:
    #                     if lcutoff is None and is_small(sg[bound - shift]):
    #                         lcutoff = bound - shift
    #                     if rcutoff is None and is_large(sg[bound + shift]):
    #                         rcutoff = bound + shift
    #                 if i == 1:
    #                     if lcutoff is None and is_large(sg[bound - shift]):
    #                         lcutoff = bound - shift
    #                     if rcutoff is None and is_small(sg[bound + shift]):
    #                         rcutoff = bound + shift
    #                 shift += 1
    #             boundaries.append(
    #                 tuple(
    #                     dim * [slice(None, None, None)] +
    #                     [slice(lcutoff, rcutoff)] +
    #                     (ndim - dim - 1) * [slice(None, None, None)]
    #                 )
    #             )

    #     return tuple(boundaries)

    # def _fit_sg(
    #     self, filtered_spectrum: np.ndarray, boundaries: Iterable[Tuple[slice]]
    # ) -> np.ndarray:
    #     amp = 0.0
    #     slices = len(boundaries)
    #     for i, bounds in enumerate(boundaries):
    #         if i % 2 == 0:
    #             amp -= filtered_spectrum[bounds[0].stop] / slices
    #         elif i % 2 == 1:
    #             amp -= filtered_spectrum[bounds[0].start] / slices
    #     sg = self.sg
    #     args = (filtered_spectrum, sg, boundaries)
    #     amp = minimize(self._sg_cost, amp, args=args, method="BFGS")["x"]
    #     return amp * sg

    # def _fit_line(
    #     self, filtered_spectrum: np.ndarray, boundaries: Iterable[Tuple[slice]]
    # ):
    #     x1 = boundaries[0][0].stop
    #     x2 = boundaries[1][0].start
    #     y1 = filtered_spectrum[x1]
    #     y2 = filtered_spectrum[x2]
    #     m = (y2 - y1) / (x2 - x1)
    #     c = (y2 + y1) - m * (x2 + x1) / 2
    #     x0 = (m, c)
    #     sg = self.sg
    #     args = (filtered_spectrum, sg, boundaries)
    #     m, c = minimize(self._linear_cost, x0, args=args, method="BFGS")["x"]
    #     line = -(m * np.arange(filtered_spectrum.size) + c)
    #     return line * sg

    # def _fit_quadratic(
    #     self, filtered_spectrum: np.ndarray, boundaries: Iterable[Tuple[slice]]
    # ):
    #     x1 = boundaries[0][0].stop
    #     x2 = boundaries[1][0].start
    #     y1 = filtered_spectrum[x1]
    #     y2 = filtered_spectrum[x2]
    #     a = 0.0
    #     b = (y2 - y1) / (x2 - x1)
    #     c = (y2 + y1) - b * (x2 + x1) / 2
    #     x0 = (a, b, c)
    #     sg = self.sg
    #     args = (filtered_spectrum, sg, boundaries)
    #     a, b, c = minimize(self._quadratic_cost, x0, args=args, method="BFGS")["x"]
    #     points = np.arange(sg.size)
    #     quadratic = -(a * (points ** 2) + b * points + c)
    #     return quadratic * sg

    # def _baseline_fix(self, filtered_spectrum: np.ndarray) -> np.ndarray:
    #     boundaries = self._get_cf_boundaries()
    #     fix = self._fit_sg(filtered_spectrum, boundaries)
    #     fix += self._fit_quadratic(filtered_spectrum + fix, boundaries)
    #     return fix

    # @staticmethod
    # def _sg_cost(amp, *args):
    #     spectrum, sg, boundaries = args
    #     cf = 0.0
    #     for i, bounds in enumerate(boundaries):
    #         spectrum_slice = spectrum[bounds]
    #         sg_slice = sg[bounds]
    #         cf += np.sum((spectrum_slice + (amp * sg_slice)) ** 2)
    #     return cf

    # @staticmethod
    # def _linear_cost(coeffs, *args):
    #     m, c = coeffs
    #     spectrum, sg, boundaries = args
    #     cf = 0.0
    #     for i, bounds in enumerate(boundaries):
    #         spectrum_slice = spectrum[bounds]
    #         line = (-(m * np.arange(spectrum.size) + c) * sg)[bounds]
    #         cf += np.sum((spectrum_slice + line) ** 2)
    #     return cf

    # @staticmethod
    # def _quadratic_cost(coeffs, *args):
    #     a, b, c = coeffs
    #     spectrum, sg, boundaries = args
    #     cf = 0.0
    #     for i, bounds in enumerate(boundaries):
    #         spectrum_slice = spectrum[bounds]
    #         points = np.arange(spectrum.size)
    #         quad = (-(a * (points ** 2) + b * points + c) * sg)[bounds]
    #         cf += np.sum((spectrum_slice + quad) ** 2)
    #     return cf

    # =============
