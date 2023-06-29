# bbqchili.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 21 Jun 2023 11:22:23 BST

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union

import nmrespy as ne
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._files import check_existent_dir
from nmrespy._sanity import sanity_check, funcs as sfuncs
import nmrglue as ng
import numpy as np
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
        data, expinfo = ne.load.load_bruker(directory)

        if directory.parent.name == "pdata":
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
            datapath=directory,
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
    def quadratic_phase(self) -> np.ndarray:
        """Second order phase correction of a spectrum generated by a single chirp."""
        shifts = self.get_shifts()[0] - self.offset()[0]

        if self.prescan_delay is not None:
            phi1 = shifts * (0.5 * self.pulse_length + self.prescan_delay)
        else:
            phi1 = 0.

        phi2 = 0.5 * (shifts ** 2) * self.pulse_length / self.pulse_bandwidth
        phased = ne.sig.ft(self.data) * np.exp(-2j * np.pi * (phi1 - phi2))
        return phased

    @logger
    def back_extrapolate(self) -> np.ndarray:
        """Calculate the offset-dependent time-zero for each oscillator in the
        estimated FID, and reconstruct the time-shifted FID."""
        if not self._results:
            raise NotImplementedError(
                f"{RED}No estimation result is associated with this estimator. "
                "Before running back-extrapolation, you must use either `estimate` "
                f"or `subband_estimate`.{END}"
            )

        fid = np.zeros(self.default_pts, dtype="complex128")
        base_timepoints, = self.get_timepoints()
        for oscillator in self.get_params():
            a, p, f, d = oscillator
            t0 = self._time_offset(f)
            timepoints = base_timepoints - t0
            fid += (
                a * np.exp(1j * p) *
                np.exp((2j * np.pi * (f - self.offset()[0]) - d) * timepoints)
            )

        return fid

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
