# expinfo.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sun 15 May 2022 12:12:18 BST

import re
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.random as nrandom
import scipy.integrate as integrate

from nmrespy._freqconverter import FrequencyConverter
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy import sig


class ExpInfo(FrequencyConverter):
    """Stores information about NMR experiments."""

    def __init__(
        self,
        dim: int,
        sw: Iterable[float],
        offset: Optional[Iterable[float]] = None,
        sfo: Optional[Iterable[float]] = None,
        nuclei: Optional[Iterable[str]] = None,
        default_pts: Optional[Iterable[int]] = None,
        fn_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        dim
            The number of dimensions associated with the experiment.

        sw
            The sweep width (spectral window) (Hz).

        offset
            The transmitter offset (Hz).

        sfo
            The transmitter frequency (MHz).

        nuclei
            The identity of each channel.

        default_pts
            The default points included in methods that require points, if not
            explicitely stated.

        fn_mode
            Acquisition mode in indirect dimensions of mulit-dimensional experiments.
            If the data is not 1-dimensional, this should be one of
            ``QF``, ``QSED``, ``TPPI``, ``States``, ``States-TPPI``, ``Echo-Anitecho``,
            and if not explicitely given, it will be set as ``QF``.

        kwargs
            Any extra parameters to be included
        """
        sanity_check(("dim", dim, sfuncs.check_int, (), {"min_value": 1}))
        self._dim = dim
        sanity_check(
            (
                "sw", sw, sfuncs.check_float_list, (),
                {
                    "length": self.dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                },
            ),
            (
                "offset", offset, sfuncs.check_float_list, (),
                {
                    "length": self.dim,
                    "len_one_can_be_listless": True,
                },
                True,
            ),
            (
                "sfo", sfo, sfuncs.check_float_list, (),
                {
                    "length": self.dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                    "allow_none": True,
                },
                True,
            ),
            (
                "nuclei", nuclei, sfuncs.check_nucleus_list, (),
                {
                    "length": self.dim,
                    "len_one_can_be_listless": True,
                    "none_allowed": True
                },
                True,
            ),
            (
                "default_pts", default_pts, sfuncs.check_int_list, (),
                {
                    "length": self.dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                },
                True,
            ),
        )

        if offset is None:
            offset = tuple(dim * [0.0])

        self.__dict__.update(**kwargs)

        self._sw = self._totuple(sw)
        self._offset = self._totuple(offset)
        self._sfo = self._totuple(sfo)
        self._nuclei = self._totuple(nuclei)
        self._default_pts = self._totuple(default_pts)
        self._fn_mode = None
        if self._dim > 1:
            sanity_check(("fn_mode", fn_mode, sfuncs.check_fn_mode, (), {}, True))
            if fn_mode is None:
                self._fn_mode = "QF"
            else:
                self._fn_mode = fn_mode

        super().__init__(self._sfo, self._sw, self._offset, self._default_pts)

    def _totuple(self, obj: Any) -> Tuple:
        if self._dim == 1 and (not sfuncs.isiter(obj)) and (obj is not None):
            obj = (obj,)
        elif obj is not None:
            obj = tuple(obj)
        return obj

    def sw(self, unit: str = "hz") -> Iterable[float]:
        """Get the sweep width.

        Parameters
        ----------
        unit
            Must be ``"hz"`` or ``"ppm"``.
        """
        sanity_check(
            ("unit", unit, sfuncs.check_frequency_unit, (self.sfo is not None,)),
        )
        return self.convert(self._sw, f"hz->{unit}")

    def offset(self, unit: str = "hz") -> Iterable[float]:
        """Get the transmitter offset frequency.

        Parameters
        ----------
        unit
            Must be ``"hz"`` or ``"ppm"``.
        """
        sanity_check(
            ("unit", unit, sfuncs.check_frequency_unit, (self.sfo is not None,)),
        )
        return self.convert(self._offset, f"hz->{unit}")

    @property
    def sfo(self) -> Optional[Iterable[Optional[float]]]:
        "Get the transmitter frequency (MHz)."
        return self._sfo

    @property
    def nuclei(self) -> Optional[Iterable[Optional[str]]]:
        """Get the nuclei associated with each channel."""
        return self._nuclei

    @property
    def unicode_nuclei(self) -> Optional[Iterable[Optional[str]]]:
        """Get the nuclei associated with each channel with superscript numbers.

        .. code:: python3

           >>> expinfo = ExpInfo(..., nuclei=("1H", "15N"), ...)
           >>> expinfo.unicode_nuclei
           ('¹H', '¹⁵N')
        """
        if self._nuclei is None:
            return None

        return tuple([
            u''.join(dict(zip(u"0123456789", u"⁰¹²³⁴⁵⁶⁷⁸⁹")).get(c, c) for c in x)
            if x is not None else None
            for x in self._nuclei
        ])

    @property
    def latex_nuclei(self) -> Optional[Iterable[Optional[str]]]:
        """Get the nuclei associated with each channel with for use in LaTeX.

        .. code:: python3

           >>> expinfo = ExpInfo(..., nuclei=("1H", "15N"), ...)
           >>> expinfo.latex_nuclei
           ('\\textsuperscript{1}H', '\\textsuperscript{15}N')
        """
        if self._nuclei is None:
            return None

        components = [
            re.search(r"^(\d+)([A-Za-z]+)", nucleus).groups()
            if nucleus is not None else None
            for nucleus in self._nuclei
        ]

        return tuple([
            f"\\textsuperscript{{{c[0]}}}{{{c[1]}}}" if c is not None
            else None
            for c in components
        ])

    @property
    def default_pts(self) -> Iterable[int]:
        """Get default points associated with each dimension."""
        return self._default_pts

    @default_pts.setter
    def default_pts(self, value) -> None:
        sanity_check(
            (
                "default_pts", value, sfuncs.check_int_list, (),
                {
                    "length": self.dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                },
                True,
            ),
        )
        self._default_pts = self._totuple(value)
        super().__init__(self.sfo, self.sw(), self.offset(), self.default_pts)

    @property
    def fn_mode(self) -> str:
        """Get acquisiton mode in indirect dimensions."""
        return self._fn_mode

    @property
    def dim(self) -> int:
        """Get number of dimensions in the experiment."""
        return self._dim

    def unpack(self, *args) -> Tuple[Any]:
        """Unpack attributes.

        `args` should be strings with names that match attribute names.
        """
        to_underscore = [
            "sw", "offset", "sfo", "nuclei", "dim", "deafult_pts", "fn_mode",
        ]
        ud_args = [f"_{a}" if a in to_underscore else a for a in args]
        if len(args) == 1:
            return getattr(self, ud_args[0], None)
        else:
            return tuple([getattr(self, arg, None) for arg in ud_args])

    def get_timepoints(
        self,
        pts: Optional[Iterable[int]] = None,
        start_time: Optional[Iterable[Union[float, str]]] = None,
        meshgrid: bool = True,
    ) -> Iterable[np.ndarray]:
        """Construct time-points which reflect the experiment parameters.

        Parameters
        ----------
        pts
            The number of points to construct the time-points with in each dimesnion.
            If ``None``, and ``self.default_pts`` is a tuple of ints, it will be
            used.

        start_time
            The start time in each dimension. If set to `None`, the initial
            point in each dimension with be ``0.0``. To set non-zero start times,
            a list of floats or strings can be used. If floats are used, they
            specify the first value in each dimension in seconds. Alternatively,
            strings of the form ``f'{N}dt'``, where ``N`` is an integer, may be
            used, which indicates a cetain multiple of the difference in time
            between two adjacent points.

        meshgrid
            If time-points are being derived for a N-dimensional signal (N > 1),
            setting this argument to ``True`` will return N-dimensional arrays
            corresponding to all combinations of points in each dimension.
        """
        sanity_check(
            (
                "pts", pts, sfuncs.check_int_list, (),
                {
                    "length": self.dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                },
                True,
            ),
            (
                "start_time", start_time, sfuncs.check_start_time, (self.dim,),
                {"len_one_can_be_listless": True}, True,
            )
        )

        if pts is None:
            pts = self.default_pts
        pts = self._totuple(pts)

        if self.dim > 1:
            sanity_check(
                ("meshgrid", meshgrid, sfuncs.check_bool),
            )

        if start_time is None:
            start_time = [0.0] * self.dim
        if not isinstance(start_time, (list, tuple)):
            start_time = [start_time]
        start_time = [
            float(re.match(r"^(-?\d+)dt$", st).group(1)) / sw if isinstance(st, str)
            else st
            for st, sw in zip(start_time, self.sw())
        ]

        tp = tuple(
            [
                np.linspace(0, float(pt - 1) / sw, pt) + st
                for pt, sw, st in zip(pts, self.sw(), start_time)
            ]
        )

        if self.dim > 1 and meshgrid:
            tp = tuple(np.meshgrid(*tp, indexing="ij"))

        return tp

    def get_shifts(
        self,
        pts: Optional[Iterable[int]] = None,
        unit: str = "hz",
        flip: bool = True,
        meshgrid: bool = True,
    ) -> Iterable[np.ndarray]:
        """Construct chemical shifts which reflect the experiment parameters.

        Parameters
        ----------
        pts
            The number of points to construct the time-points with in each dimesnion.
            If ``None``, and ``self.default_pts`` is a tuple of ints, it will be
            used.

        unit
            Must be ``"hz"``, ``"ppm"``.

        flip
            If ``True``, the shifts will be returned in descending order, as is
            conventional in NMR. If `False`, the shifts will be in ascending order.

        meshgrid
            If time-points are being derived for a N-dimensional signal (N > 1),
            setting this argument to ``True`` will return N-dimensional arrays
            corresponding to all combinations of points in each dimension.
            plot/contour plot.
        """
        sanity_check(
            (
                "pts", pts, sfuncs.check_int_list, (),
                {
                    "length": self.dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                },
                True,
            ),
            (
                "unit", unit, sfuncs.check_frequency_unit, (self.sfo is not None,),
                {}, True,
            ),
            ("flip", flip, sfuncs.check_bool),
        )

        if pts is None:
            pts = self.default_pts
        pts = self._totuple(pts)

        if self.dim > 1:
            sanity_check(("meshgrid", meshgrid, sfuncs.check_bool))

        shifts = tuple([
            np.linspace((-sw / 2) + offset, (sw / 2) + offset, pt)
            for pt, sw, offset in zip(pts, self.sw(unit), self.offset(unit))
        ])

        if self.dim > 1 and meshgrid:
            shifts = tuple(np.meshgrid(*shifts, indexing="ij"))

        return tuple([np.flip(s) for s in shifts]) if flip else shifts

    def make_fid(
        self,
        params: np.ndarray,
        pts: Optional[Iterable[int]] = None,
        snr: Union[float, None] = None,
        decibels: bool = True,
        indirect_modulation: Optional[str] = None,
    ) -> np.ndarray:
        r"""Construct a FID, as a summation of damped complex sinusoids.

        Parameters
        ----------
        params
            Parameter array with the following structure:

            * **1-dimensional data:**

              .. code:: python

                 params = numpy.array([
                    [a_1, φ_1, f_1, η_1],
                    [a_2, φ_2, f_2, η_2],
                    ...,
                    [a_m, φ_m, f_m, η_m],
                 ])

            * **2-dimensional data:**

              .. code:: python

                 params = numpy.array([
                    [a_1, φ_1, f1_1, f2_1, η1_1, η2_1],
                    [a_2, φ_2, f1_2, f2_2, η1_2, η2_2],
                    ...,
                    [a_m, φ_m, f1_m, f2_m, η1_m, η2_m],
                 ])

        pts
            The number of points to construct the time-points with in each dimesnion.
            If ``None``, and ``self.default_pts`` is a tuple of ints, it will be
            used.

        snr
            The signal-to-noise ratio. If `None` then no noise will be added
            to the FID.

        decibels
            If `True`, the snr is taken to be in units of decibels. If `False`,
            it is taken to be simply the ratio of the singal power over the
            noise power.

        indirect_modulation
            Acquisition mode in indirect dimension of a 2D experiment. If the
            data is not 1-dimensional, this should be one of:

            * ``None`` - :math:`y \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              e^{\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
            * ``"amp"`` - amplitude modulated pair:
              :math:`y_{\mathrm{cos}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              \cos\left(\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1\right)
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
              :math:`y_{\mathrm{sin}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              \sin\left(\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1\right)
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
            * ``"phase"`` - phase-modulated pair:
              :math:`y_{\mathrm{P}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              e^{\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
              :math:`y_{\mathrm{N}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              e^{\left(-2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`

            ``None`` will lead to an array of shape ``(*pts)``. ``amp`` and ``phase``
            will lead to an array of shape ``(2, *pts)``.
        """
        sanity_check(
            ("params", params, sfuncs.check_parameter_array, (self.dim,)),
            (
                "pts", pts, sfuncs.check_int_list, (),
                {
                    "length": self.dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                },
                True,
            ),
            ("snr", snr, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            ("decibels", decibels, sfuncs.check_bool),
            (
                "indirect_modulation", indirect_modulation,
                sfuncs.check_one_of, ("amp", "phase"), {}, True
            ),
        )

        if pts is None:
            pts = self.default_pts
        else:
            pts = self._totuple(pts)

        offset = self.offset()
        amp = params[:, 0]
        phase = params[:, 1]
        # Center frequencies at 0 based on offset
        freq = [params[:, 2 + i] - offset[i] for i in range(self.dim)]
        damp = [params[:, self.dim + 2 + i] for i in range(self.dim)]

        # Time points in each dimension
        tp = self.get_timepoints(pts, meshgrid=False)

        if self.dim == 1:
            fid = np.einsum(
                "ij,j->i",
                # Vandermonde matrix of signal poles
                np.exp(
                    np.outer(
                        tp[0],
                        2j * np.pi * freq[0] - damp[0],
                    )
                ),
                # Vector of complex amplitudes
                amp * np.exp(1j * phase),
            )

        elif self.dim == 2:
            # Signal pole matrix (dimension 1)
            z1 = np.exp(
                np.outer(
                    tp[0],
                    2j * np.pi * freq[0] - damp[0],
                )
            )
            # Complex amplitude vector
            alpha = amp * np.exp(1j * phase)
            # Transposed signal pole matrix (dimension 2)
            z2 = np.exp(
                np.outer(
                    2j * np.pi * freq[1] - damp[1],
                    tp[1],
                )
            )

            if indirect_modulation in ["amp", "phase"]:
                fid = np.zeros((2, *pts), dtype="complex128")

                if indirect_modulation == "amp":
                    fid[0] = np.einsum("ik,k,kj->ij", z1.real, alpha, z2)
                    fid[1] = np.einsum("ik,k,kj->ij", z1.imag, alpha, z2)

                elif indirect_modulation == "phase":
                    fid[0] = np.einsum("ik,k,kj->ij", z1, alpha, z2)
                    fid[1] = np.einsum(
                        "ik,k,kj->ij",
                        np.exp(
                            np.outer(
                                tp[0],
                                -2j * np.pi * freq[0] - damp[0],
                            )
                        ),
                        alpha,
                        z2,
                    )

            else:
                fid = np.einsum("ik,k,kj->ij", z1, alpha, z2)

        if snr is not None:
            fid += sig._make_noise(fid, snr, decibels)
        return fid

    def generate_random_signal(
        self,
        oscillators: int,
        pts: Optional[Iterable[int]] = None,
        snr: Optional[float] = None,
        decibels: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random synthetic FID.

        Parameters
        ----------
        oscillators
            Number of oscillators.

        pts
            The number of points to construct the time-points with in each dimesnion.
            If ``None``, and ``self.default_pts`` is a tuple of ints, it will be
            used.

        snr
            The signal-to-noise ratio. If `None` then no noise will be added
            to the FID.

        decibels
            If ``True``, the snr is taken to be in units of decibels. If ``False``,
            it is taken to be simply the ratio of the singal power over the
            noise power.

        Returns
        -------
        fid
            The synthetic FID.

        params
            Parameters used to construct the signal
        """
        sanity_check(
            ("oscillators", oscillators, sfuncs.check_int, (), {"min_value": 1}),
            (
                "pts", pts, sfuncs.check_int_list, (),
                {
                    "length": self.dim,
                    "must_be_positive": True,
                },
                self.default_pts is not None,
            ),
            ("snr", snr, sfuncs.check_float, (), {}, True),
            ("decibels", decibels, sfuncs.check_bool),
        )

        if pts is None:
            pts = self.default_pts
        pts = self._totuple(pts)

        # low: 0.0, high: 1.0
        # amplitdues
        params = nrandom.uniform(size=oscillators)
        # phases
        params = np.hstack(
            (params, nrandom.uniform(low=-np.pi, high=np.pi, size=oscillators)),
        )
        # frequencies
        f = [
            nrandom.uniform(low=-s / 2 + o, high=s / 2 + o, size=oscillators)
            for s, o in zip(self.sw(), self.offset())
        ]
        params = np.hstack((params, *f))
        # damping
        eta = [
            nrandom.uniform(low=0.1, high=0.3, size=oscillators)
            for _ in range(self.dim)
        ]
        params = np.hstack((params, *eta))
        params = params.reshape((oscillators, 2 * (self.dim + 1)), order="F")

        return (self.make_fid(params, pts, snr=snr, decibels=decibels), params)

    def oscillator_integrals(
        self,
        params: np.ndarray,
        pts: Optional[Iterable[int]] = None,
        absolute: bool = True,
        scale_relative_to: Optional[int] = None,
    ) -> float:
        """Determine the integral of the FT of oscillators.

        Parameters
        ----------
        params
            Parameter array with the following structure:

            * **1-dimensional data:**

              .. code:: python

                 params = numpy.array([
                    [a_1, φ_1, f_1, η_1],
                    [a_2, φ_2, f_2, η_2],
                    ...,
                    [a_m, φ_m, f_m, η_m],
                 ])

            * **2-dimensional data:**

              .. code:: python

                 params = numpy.array([
                    [a_1, φ_1, f1_1, f2_1, η1_1, η2_1],
                    [a_2, φ_2, f1_2, f2_2, η1_2, η2_2],
                    ...,
                    [a_m, φ_m, f1_m, f2_m, η1_m, η2_m],
                 ])

        pts
            The number of points to construct the signals to be integrated in each
            dimesnion. If ``None``, and ``self.default_pts`` is a tuple of ints, it
            will be used.

        absolute
            Whether or not to take the absolute value of the spectrum before
            integrating.

        scale_relative_to
            If an int, the integral corresponding to ``params[scale_relative_to]``
            is set to ``1``, and other integrals are scaled accordingly.

        Notes
        -----
        The integration is performed using the composite Simpsons rule, provided
        by `scipy.integrate.simps <https://docs.scipy.org/doc/scipy-1.5.4/\
        reference/generated/scipy.integrate.simps.html>`_

        Spacing of points along the frequency axes is set as ``1`` (i.e. ``dx = 1``).
        """
        sanity_check(
            ("params", params, sfuncs.check_parameter_array, (self.dim,)),
            (
                "pts", pts, sfuncs.check_int_list, (),
                {
                    "length": self.dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                },
                self.default_pts is not None,
            ),
            ("absolute", absolute, sfuncs.check_bool),
            (
                "scale_relative_to", scale_relative_to, sfuncs.check_int, (),
                {
                    "min_value": 0,
                    "max_value": params.shape[0] - 1,
                },
                True,
            )
        )

        # Integrals are the spectra initally. They are mutated and converted
        # into the integrals during the for loop.
        integrals = [
            sig.ft(
                self.make_fid(np.expand_dims(p, axis=0), pts),
            )
            for p in params
        ]
        integrals = [np.absolute(x) if absolute else x for x in integrals]

        for axis in reversed(range(integrals[0].ndim)):
            integrals = [integrate.simps(x, axis=axis) for x in integrals]

        if isinstance(scale_relative_to, int):
            integrals = [x / integrals[scale_relative_to] for x in integrals]

        return integrals
