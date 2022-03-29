# expinfo.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 25 Mar 2022 11:39:12 GMT

import re
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np

from nmrespy._freqconverter import FrequencyConverter
from nmrespy._sanity import sanity_check, funcs as sfuncs


def check_fn_mode(obj: Any) -> Optional[str]:
    valids = ["QF", "QSED", "TPPI", "States", "States-TPPI", "Echo-Anitecho"]
    if obj not in valids:
        return "Should be one of " + ", ".join([f"\"{x}\"" for x in valids])


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
        """Create an ExpInfo instance.

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
        sanity_check(("dim", dim, sfuncs.check_positive_int))
        self._dim = dim
        sanity_check(
            ("sw", sw, sfuncs.check_float_list, (self.dim, True, True)),
            ("offset", offset, sfuncs.check_float_list, (self.dim, True), True),
            ("sfo", sfo, sfuncs.check_float_list, (self.dim, True, True, True), True),
            ("nuclei", nuclei, sfuncs.check_nucleus_list, (self.dim, True, True), True),
            ("default_pts", default_pts, sfuncs.check_positive_int_list, (self.dim,), True),  # noqa: E501
        )

        self._fn_mode = None
        if self._dim > 1:
            sanity_check(("fn_mode", fn_mode, check_fn_mode, (), True))
            if fn_mode is None:
                self._fn_mode = "QF"
            else:
                self._fn_mode = fn_mode

        if offset is None:
            offset = tuple(dim * [0.0])

        self.__dict__.update(**kwargs)

        self._sw = self._totuple(sw)
        self._offset = self._totuple(offset)
        self._sfo = self._totuple(sfo)
        self._nuclei = self._totuple(nuclei)
        self._default_pts = self._totuple(default_pts)

        super().__init__(self._sw, self._offset, self._sfo, self._default_pts)

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
        sanity_check(("unit", unit, sfuncs.check_frequency_unit, (self.sfo is not None,)))  # noqa: E501
        return self.convert(self._sw, f"hz->{unit}")

    def offset(self, unit: str = "hz") -> Iterable[float]:
        """Get the transmitter offset frequency.

        Parameters
        ----------
        unit
            Must be ``"hz"`` or ``"ppm"``.
        """
        sanity_check(("unit", unit, sfuncs.check_frequency_unit, (self.sfo is not None,)))  # noqa: E501
        return self.convert(self._offset, f"hz->{unit}")

    @property
    def sfo(self) -> Iterable[Optional[float]]:
        "Get the transmitter frequency (MHz)."
        return self._sfo

    @property
    def nuclei(self) -> Iterable[str]:
        """Get the nuclei associated with each channel."""
        return self._nuclei

    @property
    def default_pts(self) -> Iterable[int]:
        """Get default points associated with each dimension."""
        return self._default_pts

    @property
    def fn_mode(self) -> str:
        """Get acquisiton mode in ndirect dimensions."""
        return self._fn_mode

    @property
    def dim(self) -> int:
        """Get number of dimensions in the experiment."""
        return self._dim

    def unpack(self, *args) -> Tuple[Any]:
        """Unpack attributes.

        `args` should be strings with names that match attribute names.
        """
        to_underscore = ["sw", "offset", "sfo", "nuclei", "dim"]
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
            ("pts", pts, sfuncs.check_points, (self.dim,), self.default_pts is not None),  # noqa: E501
            ("start_time", start_time, sfuncs.check_start_time, (self.dim,), True),
        )
        if pts is None:
            pts = self.default_pts

        if self.dim > 1:
            sanity_check(
                ("meshgrid", meshgrid, sfuncs.check_bool),
            )

        if start_time is None:
            start_time = [0.0] * self.dim

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
            ("pts", pts, sfuncs.check_points, (self.dim,), self.default_pts is not None),  # noqa: E501
            ("unit", unit, sfuncs.check_frequency_unit, ((self.sfo is not None),), True),  # noqa: E501
            ("flip", flip, sfuncs.check_bool),
        )
        if pts is None:
            pts = self.default_pts

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
        parameters: np.ndarray,
        pts: Optional[Iterable[int]] = None,
        snr: Union[float, None] = None,
        decibels: bool = True,
        fn_mode: Optional[str] = None,
    ) -> np.ndarray:
        r"""Construct a FID, as a summation of damped complex sinusoids.

        Parameters
        ----------
        paramaters
            Parameter array with the following structure:

            * **1-dimensional data:**

              .. code:: python

                 parameters = numpy.array([
                    [a_1, φ_1, f_1, η_1],
                    [a_2, φ_2, f_2, η_2],
                    ...,
                    [a_m, φ_m, f_m, η_m],
                 ])

            * **2-dimensional data:**

              .. code:: python

                 parameters = numpy.array([
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

        fn_mode
            Acquisition mode in indirect dimensions of mulit-dimensional experiments.
            If the data is not 1-dimensional, this should be one of ``None``,
            ``QF``, ``QSED``, ``TPPI``, ``States``, ``States-TPPI``, ``Echo-Anitecho``.
            If ``None``, ``self.fn_mode`` will be used.
        """
        sanity_check(
            ("parameters", parameters, sfuncs.check_parameter_array, (self.dim,)),
            ("pts", pts, sfuncs.check_int_list, (self.dim, False, True), self.default_pts is not None),  # noqa: E501
            ("snr", snr, sfuncs.check_positive_float, (), True),
            ("decibels", decibels, sfuncs.check_bool),
            ("fn_mode", fn_mode, check_fn_mode, (), True),
        )

        if pts is None:
            pts = self.default_pts
        else:
            pts = self._totuple(pts)

        if self.dim > 1 and fn_mode is None:
            fn_mode = self.fn_mode
        else:
            fn_mode = fn_mode

        offset = self.offset()
        amp = parameters[:, 0]
        phase = parameters[:, 1]
        # Center frequencies at 0 based on offset
        freq = [parameters[:, 2 + i] - offset[i] for i in range(self.dim)]
        damp = [parameters[:, self.dim + 2 + i] for i in range(self.dim)]

        # Time points in each dimension
        tp = self.get_timepoints(pts, meshgrid=False)

        if self.dim == 1:
            return np.einsum(
                "ij,i->j",
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
            phase0 = np.einsum(
                "ik,k,kj->ij",
                # Signal pole matrix (dimension 1)
                np.exp(
                    np.outer(
                        tp[0],
                        2j * np.pi * freq[0] - damp[0],
                    )
                ),
                # Complex amplitude vector
                amp * np.exp(1j * phase),
                # Transposed signal pole matrix (dimension 2)
                np.exp(
                    np.outer(
                        2j * np.pi * freq[1] - damp[1],
                        tp[1],
                    )
                ),
            )

            if fn_mode == "QF":
                return phase0
            else:
                raise ValueError(f"{fn_mode} yet to be implemented!")
