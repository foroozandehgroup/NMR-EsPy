# bbqchili.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 21 Jun 2023 11:22:23 BST

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union, Iterable

import nmrespy as ne
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._files import check_existent_dir
from nmrespy._sanity import sanity_check, funcs as sfuncs
import nmrglue as ng
import numpy as np
import scipy.integrate as integrate
import copy
from scipy.optimize import minimize, Bounds

from . import logger

if USE_COLORAMA:
    import colorama
    colorama.init()


class BBQChili(ne.Estimator1D):
    """Estimator class for implementing the BBQChili technique.

    .. note::

        To create an instance of ``BBQChili`` from Bruker data, use
        :py:meth:`new_from_bruker`.
    """

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        pulse_length: float,
        pulse_bandwidth: float,
        prescan_delay: Optional[float] = None,
        datapath: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        data
            Time-domain data to consider.

        expinfo
            Experiment information.

        pulse_length
            The length of the chirp pulse.

        pulse_bandwidth
            The bandwidth of the chirp pulse.

        prescan_delay
            The delay between the chirp pulse and acquisition, if known.

        datapath
            If applicable, the path that the data was derived from.
        """
        sanity_check(
            ("data", data, sfuncs.check_ndarray, (), {"dim": 1}),
            ("expinfo", expinfo, sfuncs.check_expinfo, (), {"dim": 1}),
            (
                "pulse_length", pulse_length, sfuncs.check_float, (),
                {"greater_than_zero": True},
            ),
            (
                "pulse_bandwidth", pulse_bandwidth, sfuncs.check_float, (),
                {"greater_than_zero": True},
            ),
            (
                "prescan_delay", prescan_delay, sfuncs.check_float, (),
                {"greater_than_zero": True}, True,
            ),
            ("datapath", datapath, check_existent_dir, (), {}, True),
        )
        self.pulse_length = pulse_length
        self.pulse_bandwidth = pulse_bandwidth
        self.prescan_delay = prescan_delay
        super().__init__(data, expinfo, datapath)

    @classmethod
    def new_bruker(
        cls,
        directory: Union[str, Path],
        pulse_length: float,
        pulse_bandwidth: float,
        prescan_delay: Optional[float] = None,
        convdta: bool = True,
    ) -> BBQChili:
        """Set up a ``BBQChili`` instance from Bruker data.

        Parameters
        ----------
        pulse_length
            The length of the chirp pulse.

        pulse_bandwidth
            The bandwidth of the chirp pulse.

        prescan_delay
            The delay between the chirp pulse and acquisition, if known.

        directory
            The directory under which the data of interest is stored.
        """
        sanity_check(
            (
                "pulse_length", pulse_length, sfuncs.check_float, (),
                {"greater_than_zero": True}
            ),
            (
                "pulse_bandwidth", pulse_bandwidth, sfuncs.check_float, (),
                {"greater_than_zero": True}
            ),
            (
                "prescan_delay", prescan_delay, sfuncs.check_float, (),
                {"greater_than_zero": True}, True,
            ),
            ("directory", directory, check_existent_dir),
        )
        if isinstance(directory, str):
            directoryPath = Path(directory)

        else:
            directoryPath = directory

        data, expinfo = ne.load.load_bruker(directoryPath)

        if directoryPath.parent.name == "pdata":
            slice_ = slice(0, data.shape[0] // 2)
            data = (2 * ne.sig.ift(data))[slice_]

        elif convdta:
            grpdly = expinfo.parameters["acqus"]["GRPDLY"]
            data = ne.sig.convdta(data, grpdly)

        return cls(
            data,
            expinfo,
            pulse_length,
            pulse_bandwidth,
            prescan_delay,
            datapath=directoryPath,
        )

    @classmethod
    def new_from_parameters(
        cls,
        params: np.ndarray,
        pulse_length: float,
        prescan_delay: float,
        pulse_bandwidth: float,
        pts: int,
        sw: float,
        offset: float,
        sfo: float = 500.,
        nucleus: str = "1H",
        snr: Optional[float] = 20.,
    ) -> BBQChili:
        expinfo = ne.ExpInfo(
            dim=1,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nucleus,
            default_pts=pts,
        )

        data = np.zeros(expinfo.default_pts, dtype="complex")
        obj = cls(
            data,
            expinfo,
            pulse_length,
            pulse_bandwidth,
            prescan_delay,
        )

        base_timepoints = obj.get_timepoints()[0]

        for (a, p, f, d) in params:
            t0 = obj._time_offset(f)
            obj._data += (
                a * np.exp(1j * p) *
                np.exp(
                    ((2j * np.pi * (f - obj.offset()[0])) - d) *
                    (base_timepoints + t0)
                )
            )

        obj._data += ne.sig.make_noise(obj._data, snr=snr)
        return obj

    @logger
    def estimate(self, **kwargs) -> None:
        """See :py:meth:`nmrespy.Estimator1D.estimate`.

        N.B. ``phase_variance`` will be ignored if you give it explicitly. This
        will always be set to ``False``
        """
        kwargs["phase_variance"] = False
        kwargs["_log"] = False
        super().estimate(**kwargs)

    @logger
    def subband_estimate(self, noise_region: Tuple[float, float], **kwargs) -> None:
        """See :py:meth:`nmrespy.Estimator1D.subband_estimate`.

        N.B.  ``phase_variance`` will be ignored if you give it explicitly.
        This will always be set to ``False``
        """
        kwargs["phase_variance"] = False
        kwargs["_log"] = False
        super().subband_estimate(noise_region, **kwargs)

    @logger
    def linear_phase(self) -> np.ndarray:
        """
        First order phase correction of a spectrum generated by a single chirp.
        """
        shifts = self.get_shifts()[0] - self.offset()[0]

        if self.prescan_delay is not None:
            phi1 = shifts * (0.5 * self.pulse_length + self.prescan_delay)
        else:
            phi1 = 0.

        phased = ne.sig.ft(self.data) * np.exp(-2j * np.pi * phi1)
        return phased
    
    @logger
    def quadratic_phase(self) -> np.ndarray:
        """Second order phase correction of a spectrum generated by a single chirp."""
        shifts = self.get_shifts()[0] - self.offset()[0]

        ph1_spectrum = self.linear_phase()

        phi2 = 0.5 * (shifts ** 2) * self.pulse_length / self.pulse_bandwidth
        phased = ph1_spectrum * np.exp(2j * np.pi * phi2)
        return phased

    @logger
    def back_extrapolate(self, 
            params : Optional[np.ndarray] = None,
            deconstructed : bool = False
        ) -> np.ndarray:
        """
        Calculate the offset-dependent time-zero for each oscillator in the
        estimated FID, and reconstruct the time-shifted FID.

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

        deconstructed
            If True, returns an array of the deconstructed FID.
        """
        if not self._results:
            raise NotImplementedError(
                f"{RED}No estimation result is associated with this estimator. "
                "Before running back-extrapolation, you must use either `estimate` "
                f"or `subband_estimate`.{END}"
            )
        
        if params is None:
            oscillator_params = self.get_params()
        else:
            sanity_check(
                self._params_check(params)
            )
            oscillator_params = params

        if deconstructed:
            fid = np.zeros(
                (oscillator_params.shape[0], self.default_pts[0]),
                dtype="complex128"
            )
        else:
            fid = np.zeros(self.default_pts, dtype="complex128")

        base_timepoints, = self.get_timepoints()

        for idx, oscillator in enumerate(oscillator_params):
            a, p, f, d = oscillator
            t0 = self._time_offset(f)
            timepoints = base_timepoints - t0
            oscillatorfid = (
                a * np.exp(1j * p) *
                np.exp((2j * np.pi * (f - self.offset()[0]) - d) * timepoints)
            )
            if deconstructed:
                fid[idx] = oscillatorfid 
            else:
                fid += oscillatorfid

        return fid
    
    @logger
    def peak_integration(
        self,
        peaks : Union[np.ndarray, list],
        spectrum : Optional[np.ndarray] = None,
        pts: Optional[Iterable[int]] = None,
        area : int = 2000,
        scale_relative : Optional[str] = None
    ) -> Tuple[np.ndarray, list]:
        """
        Estimate the areas around specified peaks of the FT spectrum using the composite
        Simpson's rule ``scipy.integrate.simpson``.

        Parameters
        ----------
        peaks
            The list of peaks around which to estimate the integrals.

        spectrum
            The FT spectrum on which to carry out the integration. If ``None``, the 
            default ``self.spectrum`` will be used.

        pts
            The number of points to construct the shifts with in each dimesnion.
            If ``None``, and ``self.default_pts`` is a tuple of ints, it will be
            used.

        area
            The range of frequencies (Hz) around the peak to integrate.

        scale_relative
            ``"median"`` will set the median peak' integral to ``1``, and ``"min"`` will 
            set the smallest peak integral to ``1``, and scale all other integrals 
            accordingly. ``None`` returns the absolute integral values.

        Notes
        -----
        For oscillator integration, see :py:meth:`nmrespy.expinfo.oscillator_integrals`
        """

        pts = self._process_pts(pts)

        assert len(spectrum) == pts[0]

        if spectrum is None:
            spec = self.spectrum
        else:
            spec = spectrum

        # Create index slices for slicing around peaks
        dshift = self.sw()[0] / pts[0]
        idx_range = round(area / dshift) # No. of indices that correspond to the area

        slicer = lambda i : np.s_[(i - idx_range): (i + idx_range +1)]

        # Index of the peaks 
        shifts, = self.get_shifts(pts=pts)
        idx = lambda x : np.argmin(np.abs(shifts - x))

        integrals = [
            integrate.simpson(
                y = spec[slicer(idx(p))],
                dx = dshift
            ).real 
            for p in peaks
        ]
        integrals = np.array(integrals, dtype=np.float64)

        slices = [slicer(idx(p)) for p in peaks]

        if scale_relative == "median":
            integrals = integrals / np.percentile(integrals, 50, method='nearest')
        elif scale_relative == "min":
            integrals = integrals / np.min(integrals)
        elif scale_relative is None:
            pass
        else:
            raise ValueError("Incorrect argument for scale_relative")
        
        return integrals, slices
    
    @logger
    def back_extrapolated_oscillator_integrals(
        self,
        params: np.ndarray,
        pts: Optional[Iterable[int]] = None,
        absolute: bool = True,
        scale_relative_to: Optional[int] = None,
    ) -> list:
        """
        Determine the integral of the FT of the back-extrapolated oscillators.

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

        See also
        --------
        :py:meth: `nmrespy.ExpInfo.oscillator_integrals`
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
        oscillatorfids = self.back_extrapolate(params=params, deconstructed=True)
        integrals = [
            ne.sig.ft(oscillatorfids[i])
            for i in range(oscillatorfids.shape[0])
        ]

        integrals = [np.absolute(x) if absolute else x for x in integrals]

        for axis in reversed(range(integrals[0].ndim)):
            integrals = [integrate.simpson(x, axis=axis) for x in integrals]

        if isinstance(scale_relative_to, int):
            integrals = [x / integrals[scale_relative_to] for x in integrals]

        return integrals
    

    def phase_data(
        self,
        data : Optional[np.ndarray] = None,
        p0 : float = 0.,
        p1 : float = 0.,
        pivot : int = 0
    ) -> Optional[np.ndarray]:
        """
        See :py:meth:`nmrespy.Estimator1D.phase_data`

        Parameters
        ----------
        data
            The FID which to phase. If ``None``, the default ``self.data`` will be used.

        See also
        --------
        :py:meth:`manual_phase_data`
        """
        if data is None:
            super().phase_data(p0, p1, pivot)
        
        else:
            sanity_check(
                ("p0", p0, sfuncs.check_float),
                ("p1", p1, sfuncs.check_float),
                (
                    "pivot", pivot, sfuncs.check_int, (),
                    {"min_value": 0, "max_value": self.data.shape[self.proc_dims[0]] - 1},
                ),
            )

            p0s = [0. for _ in range(self.dim)]
            p1s = [0. for _ in range(self.dim)]
            pivots = [0. for _ in range(self.dim)]

            p0s[self.proc_dims[0]] = p0
            p1s[self.proc_dims[0]] = p1
            pivots[self.proc_dims[0]] = pivot

            _spectrum = copy.deepcopy(data)
            _spectrum[self._first_point_slice] *= 0.5
            _spectrum = ne.sig.ft(_spectrum, axes=self.proc_dims)

            phased_data = ne.sig.ift(
                ne.sig.phase(_spectrum, p0=p0s, p1=p1s, pivot=pivots),
                axes=self.proc_dims,
            )

            return phased_data
        

    def manual_phase_data(
        self,
        spectrum : Optional[np.ndarray] = None,
        max_p1 : float = 10 * np.pi
    ) -> Tuple[float, float]:
        """
        See :py:meth:`nmrespy.Estimator1D.manual_phase_data`

        Unlike py:meth:`nmrespy.Estimator1D.manual_phase_data`, if `spectrum` is specified
        :py:meth:`phase_data` will not be called and has to be called manually.

        Parameters
        ----------
        spectrum
            The FT spectrum on which to carry out the integration. If ``None``, the 
            default ``self.spectrum`` will be used.

        Returns
        -------
        p0
            Zero order phase (rad)

        p1
            First order phase (rad)

        See also
        --------
        :py:meth:`phase_data`
        """
        if spectrum is None:
            super().manual_phase_data(max_p1)

        else:
            p0, p1 = ne.sig.manual_phase_data(spectrum, max_p1=[max_p1])
            p0, p1 = p0[0], p1[0]
            return p0, p1


    def _time_offset(self, freq: float) -> float:
        """Calculate the time-zero for an oscillator of a particular frequency.

        Parameters
        ----------
        freq
            Oscillator frequency
        """
        if self.prescan_delay is not None:
            first_order = self.prescan_delay + 0.5 * self.pulse_length
        else:
            first_order = 0.

        return (
            first_order -
            0.5 * (freq - self.offset()[0]) * self.pulse_length / self.pulse_bandwidth
        )


    def _zero_order_phase_correction(self, data: np.ndarray) -> np.ndarray:
        """Zero-order phase correction on time-domain data.

        Determines the phase which maximises the real component of the initial point.

        Parameters
        ----------
        data
            Data to phase.
        """
        def cost(phi0, data):
            return -(data[0] * np.exp(-1j * phi0)).real

        result = minimize(
            fun=cost,
            x0=np.array([0], dtype="float64"),
            args=(data,),
            method="L-BFGS-B",
            jac=None,
            bounds=Bounds(-np.pi, np.pi),
        )["x"]
        return data * np.exp(-1j * result)
