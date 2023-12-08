# expinfo.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 08 Dec 2023 05:25:31 PM EST

import datetime
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.random as nrandom
import scipy.integrate as integrate

from nmrespy._files import check_saveable_dir
from nmrespy._freqconverter import FrequencyConverter
from nmrespy._paths_and_links import NMRESPYPATH
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy import sig


class ExpInfo(FrequencyConverter):
    """Stores information about NMR experiments.

    ``ExpInfo`` provides useful methods which require this information, such as
    generating FIDs, creating samples (time-points, chemical shifts), etc. It is
    a parent class to all estimator objects.
    """
    def __init__(
        self,
        dim: int,
        sw: Iterable[float],
        offset: Optional[Iterable[float]] = None,
        sfo: Optional[Optional[Iterable[float]]] = None,
        nuclei: Optional[Optional[Iterable[str]]] = None,
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
        sanity_check(
            (
                "sw", sw, sfuncs.check_float_list, (),
                {
                    "length": dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                },
            ),
            (
                "offset", offset, sfuncs.check_float_list, (),
                {
                    "length": dim,
                    "len_one_can_be_listless": True,
                },
                True,
            ),
            (
                "sfo", sfo, sfuncs.check_float_list, (),
                {
                    "length": dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                    "allow_none": True,
                },
                True,
            ),
            (
                "nuclei", nuclei, sfuncs.check_nucleus_list, (),
                {
                    "length": dim,
                    "len_one_can_be_listless": True,
                    "none_allowed": True
                },
                True,
            ),
            (
                "default_pts", default_pts, sfuncs.check_int_list, (),
                {
                    "length": dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                },
                True,
            ),
        )

        self.__dict__.update(**kwargs)

        if offset is None:
            offset = tuple(dim * [0.0])

        self._dim = dim
        for var, name in zip(
            (sw, offset, sfo, nuclei, default_pts),
            ("sw", "offset", "sfo", "nuclei", "default_pts"),
        ):
            self.__dict__[f"_{name}"] = self._totuple(var)

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

    @property
    def default_pts(self) -> Iterable[int]:
        """Get default points associated with each dimension."""
        return self._default_pts

    # TODO: Want to get rid of this
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
    def fn_mode(self) -> Optional[str]:
        """Get acquisiton mode in indirect dimensions. If ``self.dim == 1``,
        returns ``None``.
        """
        return self._fn_mode

    @property
    def dim(self) -> int:
        """Get number of dimensions in the experiment."""
        return self._dim

    def sw(self, unit: str = "hz") -> Iterable[float]:
        """Get the sweep width.

        Parameters
        ----------
        unit
            Must be ``"hz"`` or ``"ppm"``.
        """
        sanity_check(self._funit_check(unit))
        return self.convert(self._sw, f"hz->{unit}")

    def offset(self, unit: str = "hz") -> Iterable[float]:
        """Get the transmitter offset frequency.

        Parameters
        ----------
        unit
            Must be ``"hz"`` or ``"ppm"``.
        """
        sanity_check(self._funit_check(unit))
        return self.convert(self._offset, f"hz->{unit}")

    @property
    def bf(self) -> Optional[Iterable[Optional[float]]]:
        """Get the basic frequency (MHz).

        For each dimension where :py:meth:`sfo` is not ``None``, this is
        equivalent to ``self.sfo[i] - self.offset()[i]``
        """
        if self.sfo is None:
            return None
        return tuple(
            [
                sfo - (offset * 1e-6) if sfo is not None else None
                for sfo, offset in zip(self.sfo, self.offset())
            ]
        )

    @property
    def sfo(self) -> Optional[Iterable[Optional[float]]]:
        """Get the transmitter frequency (MHz)."""
        return self._sfo

    @property
    def nuclei(self) -> Optional[Iterable[Optional[str]]]:
        """Get the nuclei associated with each channel."""
        return self._nuclei

    @property
    def unicode_nuclei(self) -> Optional[Iterable[Optional[str]]]:
        """Get the nuclei associated with each channel with superscript numbers.

        Examples: ``"1H"`` → ``"¹H"``, ``"15N"`` → ``"¹⁵N"``.
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
        r"""Get the nuclei associated with each channel with for use in LaTeX.

        Examples:

        * ``"1H"`` → ``"\\textsuperscript{1}H"``
        * ``"195Pt"`` → ``"\\textsuperscript{195}Pt"``
        """
        if self._nuclei is None:
            return None

        components = [
            re.search(r"^(\d+)([A-Za-z]+)", nucleus).groups()
            if nucleus is not None else None
            for nucleus in self._nuclei
        ]

        return tuple([
            f"\\textsuperscript{{{c[0]}}}{c[1]}" if c is not None
            else None
            for c in components
        ])

    def _axis_freq_labels(self, unit: str) -> Iterable[str]:
        labels = []
        for sfo, nuc in zip(self.sfo, self.unicode_nuclei):
            if sfo is None:
                u = "Hz"
            else:
                u = unit.replace("h", "H")
            if nuc is None:
                labels.append(u)
            else:
                labels.append(f"{nuc} ({u})")

        return labels

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
            The start time in each dimension. If set to ``None``, the initial
            point in each dimension will be ``0.0``. To set non-zero start times,
            a list of floats or strings can be used.

            * If floats are used, they specify the first value in each
              dimension in seconds.
            * Strings of the form ``f'{N}dt'``, where ``N`` is an integer, may be
              used, which indicates a cetain multiple of the dwell time.

        meshgrid
            If time-points are being derived for a N-dimensional signal (N > 1),
            setting this argument to ``True`` will return N-dimensional arrays
            corresponding to all combinations of points in each dimension. If
            ``False``, an iterable of 1D arrays will be returned.
        """
        sanity_check(
            self._pts_check(pts),
            (
                "start_time", start_time, sfuncs.check_start_time, (self.dim,),
                {"len_one_can_be_listless": True}, True,
            )
        )

        pts = self._process_pts(pts)

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
            The number of points to construct the shifts with in each dimesnion.
            If ``None``, and ``self.default_pts`` is a tuple of ints, it will be
            used.

        unit
            Must be one of ``"hz"`` or ``"ppm"``.

        flip
            If ``True``, the shifts will be returned in descending order, as is
            conventional in NMR. If ``False``, the shifts will be in ascending order.

        meshgrid
            If time-points are being derived for a N-dimensional signal (N > 1),
            setting this argument to ``True`` will return N-dimensional arrays
            corresponding to all combinations of points in each dimension. If
            ``False``, an iterable of 1D arrays will be returned.
        """
        sanity_check(
            self._pts_check(pts),
            self._funit_check(unit),
            ("flip", flip, sfuncs.check_bool),
        )

        pts = self._process_pts(pts)

        if self.dim > 1:
            sanity_check(("meshgrid", meshgrid, sfuncs.check_bool))

        shifts = [
            None if sw is None
            else np.linspace((-sw / 2) + offset, (sw / 2) + offset, pt + 1)
            for pt, sw, offset in zip(pts, self.sw(unit), self.offset(unit))
        ]
        shifts = tuple([
            None if shift_ax is None else shift_ax[:-1]
            for shift_ax in shifts
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
        r"""Construct an FID from an array of oscillator parameters.

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
            If ``True``, the snr is taken to be in units of decibels. If `False`,
            it is taken to be simply the ratio of the singal power over the
            noise power.

        indirect_modulation
            Acquisition mode in the indirect dimension if the data is 2D.
            If the data is 1D, this argument is ignored.

            * ``None`` - hypercomplex dataset:

              .. math::

                  y \left(t_1, t_2\right) = \sum_{m} a_m e^{\mathrm{i} \phi_m}
                  e^{\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
                  e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}

            * ``"amp"`` - amplitude modulated pair:

              .. math::

                  y_{\mathrm{cos}} \left(t_1, t_2\right) = \sum_{m} a_m
                  e^{\mathrm{i} \phi_m} \cos\left(\left(2 \pi \mathrm{i} f_{1,
                  m} - \eta_{1, m}\right) t_1\right) e^{\left(2 \pi \mathrm{i}
                  f_{2, m} - \eta_{2, m}\right) t_2}

              .. math::

                  y_{\mathrm{sin}} \left(t_1, t_2\right) = \sum_{m} a_m
                  e^{\mathrm{i} \phi_m} \sin\left(\left(2 \pi \mathrm{i} f_{1,
                  m} - \eta_{1, m}\right) t_1\right) e^{\left(2 \pi \mathrm{i}
                  f_{2, m} - \eta_{2, m}\right) t_2}

            * ``"phase"`` - phase-modulated pair:

              .. math::

                  y_{\mathrm{P}} \left(t_1, t_2\right) = \sum_{m} a_m
                  e^{\mathrm{i} \phi_m} e^{\left(2 \pi \mathrm{i} f_{1, m} -
                  \eta_{1, m}\right) t_1} e^{\left(2 \pi \mathrm{i} f_{2, m} -
                  \eta_{2, m}\right) t_2}

              .. math::

                  y_{\mathrm{N}} \left(t_1, t_2\right) = \sum_{m} a_m
                  e^{\mathrm{i} \phi_m}
                  e^{\left(-2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
                  e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}

            ``None`` will lead to an array of shape ``(n1, n2)``. ``amp`` and ``phase``
            will lead to an array of shape ``(2, n1, n2)``, with ``fid[0]`` and
            ``fid[1]`` being the two components of the pair.

        See Also
        --------
        * For converting amplitude-modulated data to spectral data, see
          :py:func:`nmrespy.sig.proc_amp_modulated`
        * For converting phase-modulated data to spectral data, see
          :py:func:`nmrespy.sig.proc_phase_modulated`
        """
        sanity_check(
            self._params_check(params),
            self._pts_check(pts),
            ("snr", snr, sfuncs.check_float, (), {"min_value": 0.}, True),
            ("decibels", decibels, sfuncs.check_bool),
        )

        if self.dim > 1:
            sanity_check(
                (
                    "indirect_modulation", indirect_modulation,
                    sfuncs.check_one_of, ("amp", "phase"), {}, True
                ),
            )

        pts = self._process_pts(pts)
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
            fid += sig.make_noise(fid, snr, decibels)

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
            The number of points to construct the signal with in each dimesnion.
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
            self._pts_check(pts),
            ("snr", snr, sfuncs.check_float, (), {}, True),
            ("decibels", decibels, sfuncs.check_bool),
        )
        pts = self._process_pts(pts)

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
            If an int, the integral corresponding to the assigned oscillator is
            set to ``1``, and other integrals are scaled accordingly.

        Notes
        -----
        The integration is performed using the composite Simpsons rule, provided
        by `scipy.integrate.simps <https://docs.scipy.org/doc/scipy-1.5.4/\
        reference/generated/scipy.integrate.simps.html>`_

        Spacing of points along the frequency axes is set as ``1`` (i.e. ``dx = 1``).
        """
        sanity_check(
            self._params_check(params),
            self._pts_check(pts),
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

    @staticmethod
    def _get_dir_number(root_dir: Path) -> int:
        number = 1
        while True:
            if not (root_dir / str(number)).is_dir():
                break
            else:
                number += 1
        return number

    def _convert_to_bruker_format(self, data) -> Tuple[int, np.ndarray]:
        if data.ndim - self.dim == 1:
            data = np.hstack([d for d in data])
        if data.ndim > 1:
            data = data.flatten()
        if data.dtype == "complex128":
            # interleave real and imaginary parts
            data = np.vstack(
                (data.real, data.imag)
            ).reshape((-1,), order="F").astype("<f8")
        data_max = np.amax(data)
        nc = int(np.floor(np.log2(2 ** 29 / data_max)))
        return (
            nc,
            (data * (2 ** nc)).astype("<i4")
        )

    @staticmethod
    def _write_bruker_param_files(
        directory: Path,
        stem: str,
        texts: Iterable[str],
    ) -> None:
        for stem, text in zip([f"{stem}{x}" for x in ("", "2")[:len(texts)]], texts):
            for fname in (stem, f"{stem}s"):
                if fname[-1] == "s":
                    text = text.replace(
                        f"$$ {directory}/{stem}",
                        f"$$ {directory}/{stem}s",
                    )
                with open(directory / fname, "w") as fh:
                    fh.write(text)

    def write_to_bruker(
        self,
        fid: np.ndarray,
        path: Union[str, Path],
        expno: Optional[int] = None,
        procno: Optional[int] = None,
        force_overwrite: bool = False,
        fnmode: Optional[str] = None,
    ) -> None:
        sanity_check(
            ("path", path, check_saveable_dir, (True,)),
            ("expno", expno, sfuncs.check_int, (), {"min_value": 1}, True),
            ("procno", procno, sfuncs.check_int, (), {"min_value": 1}, True),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
        )
        # TODO
        # Ensure sfo and nuclei are not None
        if self.dim == 1:
            fnmode = ""
        if self.dim == 2:
            sanity_check(
                ("fnmode", fnmode, sfuncs.check_one_of, ("amp", "phase"), {}, True),
            )
            if fnmode is None:
                sanity_check(
                    ("fid", fid, sfuncs.check_ndarray, (), {"dim": 2}),
                )
                fnmode = "1"
            elif fnmode in ("amp", "phase"):
                sanity_check(
                    (
                        "fid", fid, sfuncs.check_ndarray, (),
                        {"dim": 3, "shape": [(0, 2)]},
                    ),
                )

        # Path checking and creating
        if not (path := Path(path).resolve()).is_dir():
            path.mkdir()

        if expno is None:
            expno = self._get_dir_number(path)
        acqu_dir = path / str(expno)
        sanity_check(
            ("path & expno", acqu_dir, check_saveable_dir, (force_overwrite,)),
        )
        acqu_dir.mkdir(exist_ok=True)

        if not (pdata_dir := acqu_dir / "pdata").is_dir():
            pdata_dir.mkdir()
        if procno is None:
            procno = self._get_dir_number(pdata_dir)
        proc_dir = pdata_dir / str(procno)
        sanity_check(
            (
                "path & expno & procno", proc_dir, check_saveable_dir,
                (force_overwrite,),
            ),
        )
        proc_dir.mkdir(exist_ok=True)

        # Processing time-domain data
        nc_fid, fid_uncomplex = self._convert_to_bruker_format(fid)
        with open(acqu_dir / ("fid" if self.dim == 1 else "ser"), "wb") as fh:
            fh.write(fid_uncomplex.tobytes())

        # Generating and processing spectral data
        if self.dim == 1:
            fid[0] *= 0.5
            spectrum = sig.ft(fid).real
            nc_spec, spectrum = self._convert_to_bruker_format(spectrum)
            with open(proc_dir / "1r", "wb") as fh:
                fh.write(spectrum.tobytes())

        elif self.dim == 2 and fnmode == "phase":
            spectra = sig.proc_phase_modulated(fid)
            proc_spectra = []
            nc_spec = 0
            for spectrum in spectra:
                nc, spectrum = self._convert_to_bruker_format(spectrum)
                proc_spectra.append(spectrum)
                if nc > nc_spec:
                    nc_spec = nc
            for fname, spectrum in zip(("2rr", "2ri", "2ir", "2ii"), spectra):
                with open(proc_dir / fname, "wb") as fh:
                    fh.write(spectrum.tobytes())
            fnmode = "6"

        elif self.dim == 2 and fnmode == "amp":
            # TODO
            fnmode = "2"

        # Creating acqus and procs files
        acqu_texts = []
        proc_texts = []
        for s in ("", "2")[:self.dim]:
            with open(NMRESPYPATH / f"ts_templates/acqu{s}s", "r") as fh:
                acqu_texts.append(fh.read())
            with open(NMRESPYPATH / f"ts_templates/proc{s}s", "r") as fh:
                proc_texts.append(fh.read())

        tz = datetime.timezone(datetime.timedelta(seconds=-time.timezone))
        now = datetime.datetime.now(tz=tz)
        unix_time = int(
            (now - datetime.datetime(1970, 1, 1, tzinfo=tz)).total_seconds()
        )
        timestamp = (
            f"$$ {now.strftime('%Y-%m-%d %H:%M:%S %z')} "
            f"{os.getlogin()}@{os.uname()[1]}"
        )

        # TODO: Create dummy values in case sfo unknown
        shape = fid.shape
        for i, (acqu_text, proc_text) in enumerate(zip(acqu_texts, proc_texts)):
            s = str(i + 1) if i != 0 else ""
            idx = -(i + 1)
            acqu_subs = {
                "<BF1>": str(self.bf[idx]),
                "<DATE>": str(unix_time),
                "<FnMODE>": fnmode,
                "<NC>": str(-nc_fid),
                "<NUC1>": f"<{self.nuclei[idx]}>",
                "<O1>": str(self.offset()[idx]),
                "<OWNER>": os.getlogin(),
                "<PARMODE>": str(self.dim - 1),
                "<PATH>": f"$$ {acqu_dir}/acqu{s}",
                "<SFO1>": str(self.sfo[idx]),
                "<SW>": str(self.sw(unit="ppm")[idx]),
                "<SW_h>": str(self.sw()[idx]),
                "<TIMESTAMP>": timestamp,
                "<TD>": str(2 * shape[idx] if idx == -1 else shape[idx]),
            }

            proc_subs = {
                "<AXNUC>": f"<{self.nuclei[idx]}>",
                "<FTSIZE>": str(shape[idx]),
                "<NC_proc>": str(-nc_spec),
                "<OFFSET>": str(self.get_shifts(unit="ppm", meshgrid=False)[idx][0]),
                "<OWNER>": os.getlogin(),
                "<PATH>": f"$$ {proc_dir}/proc{s}",
                "<SF>": str(self.bf[idx]),
                "<SI>": str(shape[idx]),
                "<SR>": str(self.offset()[idx]),
                "<SW_p>": str(self.sw()[idx]),
                "<TIMESTAMP>": timestamp,
            }

            for old, new in acqu_subs.items():
                acqu_text = acqu_text.replace(old, new)
            for old, new in proc_subs.items():
                proc_text = proc_text.replace(old, new)
            acqu_texts[i] = acqu_text
            proc_texts[i] = proc_text

        self._write_bruker_param_files(acqu_dir, "acqu", acqu_texts)
        self._write_bruker_param_files(proc_dir, "proc", proc_texts)

    def _process_pts(self, pts: Optional[Union[Iterable[int], int]]) -> Iterable[int]:
        if pts is None:
            return self.default_pts
        return self._totuple(pts)

    # Common sanity checks
    def _funit_check(self, unit: Any, name: str = "unit"):
        return (name, unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,))

    def _params_check(self, params: Any):
        return ("params", params, sfuncs.check_parameter_array, (self.dim,))

    def _pts_check(self, pts: Any):
        return (
            "pts", pts, sfuncs.check_int_list, (),
            {
                "length": self.dim,
                "len_one_can_be_listless": True,
                "min_value": 1,
            },
            self.default_pts is not None,
        )
