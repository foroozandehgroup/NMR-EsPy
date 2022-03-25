# expinfo.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 25 Mar 2022 11:39:12 GMT

import re
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np

from nmrespy._colors import END, RED
from nmrespy._sanity import sanity_check, funcs as sfuncs


class ExpInfo:
    """Stores information about NMR experiments."""

    def __init__(
        self,
        dim: int,
        sw: Iterable[float],
        offset: Optional[Iterable[float]] = None,
        sfo: Optional[Iterable[float]] = None,
        nuclei: Optional[Iterable[str]] = None,
        default_pts: Optional[Iterable[int]] = None,
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

        kwargs
            Any extra parameters to be included
        """
        sanity_check(("dim", dim, sfuncs.check_positive_int))
        self._dim = dim
        sanity_check(
            self._sw_check(sw),
            self._offset_check(offset),
            self._sfo_check(sfo),
            self._nuclei_check(nuclei),
            self._default_pts_check(default_pts),
        )

        if offset is None:
            offset = tuple(dim * [0.0])

        self.__dict__.update(kwargs)
        self._sw = self._totuple(sw)
        self._offset = self._totuple(offset)
        self._sfo = self._totuple(sfo)
        self._nuclei = self._totuple(nuclei)
        self._default_pts = self._totuple(default_pts)

    def _sw_check(self, obj: Any) -> Tuple:
        return (
            "sw", obj, sfuncs.check_float_list, (self.dim, True, True),
        )

    def _offset_check(self, obj: Any) -> Tuple:
        return (
            "offset", obj, sfuncs.check_float_list, (self.dim, True), True,
        )

    def _sfo_check(self, obj: Any) -> Tuple:
        return (
            "sfo", obj, sfuncs.check_float_list, (self.dim, True, True, True), True,
        )

    def _nuclei_check(self, obj: Any) -> Tuple:
        return (
            "nuclei", obj, sfuncs.check_nucleus_list, (self.dim, True, True), True,
        )

    def _default_pts_check(self, obj: Any) -> Tuple:
        return (
            "default_pts", obj, sfuncs.check_positive_int_list, (self.dim,), True,
        )

    def _totuple(self, obj: Any) -> Tuple:
        if self._dim == 1 and (not sfuncs.isiter(obj)) and (obj is not None):
            obj = (obj,)
        elif obj is not None:
            obj = tuple(obj)
        return obj

    @property
    def sw(self) -> Iterable[float]:
        """Get sweep width (Hz)."""
        return self._sw

    @sw.setter
    def sw(self, sw: Any) -> None:
        sanity_check(self._sw_check(sw))
        self._sw = self._totuple(sw)
        print(self._sw)

    @property
    def offset(self) -> Iterable[float]:
        """Get transmitter offset frequency (Hz)."""
        return self._offset

    @offset.setter
    def offset(self, offset: Any) -> None:
        sanity_check(self._offset_check(offset))
        self._offset = self._totuple(offset)

    @property
    def sfo(self) -> Iterable[float]:
        """Get transmitter frequency (MHz)."""
        return self._sfo

    @sfo.setter
    def sfo(self, sfo: Any) -> None:
        sanity_check(self._sfo_check(sfo))
        self._sfo = self._totuple(sfo)

    @property
    def nuclei(self) -> Iterable[str]:
        """Get nuclei associated with each channel."""
        return self._nuclei

    @nuclei.setter
    def nuclei(self, nuclei: Any) -> None:
        sanity_check(self._nuclei_check(nuclei))
        self._nuclei = self._totuple(nuclei)

    @property
    def default_pts(self) -> Iterable[int]:
        """Get default points associated with each dimension."""
        return self._default_pts

    @default_pts.setter
    def default_pts(self, default_pts: Any) -> None:
        sanity_check(self._default_pts_check(default_pts))
        self._default_pts = self._totuple(default_pts)

    @property
    def dim(self) -> int:
        """Get number of dimensions in the expeirment."""
        return self._dim

    @dim.setter
    def dim(self, new_value):
        raise ValueError(f"{RED}`dim` cannot be mutated.{END}")

    def unpack(self, *args) -> Tuple[Any]:
        """Unpack attributes.

        `args` should be strings with names that match attribute names.
        """
        to_underscore = ["sw", "offset", "sfo", "nuclei", "dim"]
        ud_args = [f"_{a}" if a in to_underscore else a for a in args]
        if len(args) == 1:
            return self.__dict__[ud_args[0]]
        else:
            return tuple([self.__dict__[arg] for arg in ud_args])

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
            ("pts", pts, sfuncs.check_points, (self.dim,), True),
            ("start_time", start_time, sfuncs.check_start_time, (self.dim,), True),
        )
        pts = self._get_pts(pts)

        if self.dim > 1:
            sanity_check(
                ("meshgrid", meshgrid, sfuncs.check_bool),
            )

        if start_time is None:
            start_time = [0.0] * self.dim

        sw = self.sw
        start_time = [
            float(re.match(r"^(-?\d+)dt$", st).group(1)) / sw_ if isinstance(st, str)
            else st
            for st, sw_ in zip(start_time, sw)
        ]

        tp = tuple(
            [
                np.linspace(0, float(pts_ - 1) / sw_, pts_) + st
                for pts_, sw_, st in zip(pts, sw, start_time)
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
            The unit of the chemical shifts. One of ``"hz"``, ``"ppm"``.

        flip
            If `True`, the shifts will be returned in descending order, as is
            conventional in NMR. If `False`, the shifts will be in ascending order.

        meshgrid
            If time-points are being derived for a N-dimensional signal (N > 1),
            setting this argument to ``True`` will return N-dimensional arrays
            corresponding to all combinations of points in each dimension.
            plot/contour plot.
        """
        sw, offset, sfo, dim = self.unpack("sw", "offset", "sfo", "dim")
        sanity_check(
            ("pts", pts, sfuncs.check_points, (dim,), True),
            ("unit", unit, sfuncs.check_frequency_unit, ((sfo is not None),), True),
            ("flip", flip, sfuncs.check_bool)
        )
        pts = self._get_pts(pts)

        if dim > 1:
            sanity_check(("meshgrid", meshgrid, sfuncs.check_bool))

        shifts = [
            np.linspace((-sw_ / 2) + offset_, (sw_ / 2) + offset_, pts_)
            for pts_, sw_, offset_ in zip(pts, sw, offset)
        ]
        if unit == "ppm":
            for i, (axis, sfo_) in enumerate(zip(shifts, sfo)):
                if sfo_ is not None:
                    shifts[i] = axis / sfo_

        if dim > 1 and meshgrid:
            shifts = np.meshgrid(*shifts, indexing="ij")

        return tuple([np.flip(s) for s in shifts]) if flip else tuple(shifts)

    def _get_pts(self, pts):
        if pts is None:
            if self.default_pts is not None:
                return self.default_pts
            else:
                raise ValueError(
                    f"{RED}You must provide `pts` as a list of ints.{END}"
                )
        else:
            return pts
