# seq_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 28 Jul 2022 16:32:27 BST

import copy
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from nmrespy import ExpInfo, sig
from nmrespy.freqfilter import Filter
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import NonlinearProgramming
from nmrespy._files import check_existent_dir
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)
from . import logger, Estimator, Result
mpl.use("tkAgg")


class EstimatorSeq1D(Estimator):

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ExpInfo,
        increments: Iterable[float],
        increment_qty: Optional[str] = None,
        datapath: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        data
            Time-domain data to estimate. Should be a 2-dimensional NumPy array
            with shape ``(incr, pts)`` where ``incr`` is the number of increments
            in the experiment, and ``pts`` is the number of points in each FID.

        expinfo
            Experiment information.

        datapath
            If applicable, the path that the data was derived from.
        """
        sanity_check(
            ("data", data, sfuncs.check_ndarray, (), {"dim": 2}),
            ("expinfo", expinfo, sfuncs.check_expinfo, (), {"dim": 1}),
            ("datapath", datapath, check_existent_dir, (), {}, True),
            ("increment_qty", increment_qty, sfuncs.check_str, (), {}, True),
        )
        sanity_check(
            (
                "increments", increments, sfuncs.check_float_list, (),
                {"length": data.shape[0]},
            )
        )
        expinfo._default_pts = data.shape[-1]
        super().__init__(data, expinfo, datapath)
        self.increments = np.array(increments)
        self.increment_qty = increment_qty

    @property
    def spectrum(self) -> np.ndarray:
        data = copy.deepcopy(self.data)
        data[:, 0] *= 0.5
        return sig.ft(data, axes=1)

    @property
    def n_increments(self) -> int:
        return self.data.shape[0]

    def view_data(self, projection: str = "2d") -> None:
        # TODO: Docs, time-domain, freq-unit
        sanity_check(
            ("projection", projection, sfuncs.check_one_of, ("2d", "3d")),
        )
        shifts, = self.get_shifts(unit="ppm")
        spectrum = self.spectrum
        fig = plt.figure()
        if projection == "2d":
            ax = fig.add_subplot()
            colors = mpl.cm.get_cmap("rainbow")(np.linspace(0, 1, self.n_increments))
            for spec, incr, color in zip(spectrum, self.increments, colors):
                ax.plot(shifts, spec, color=color, lw=0.8, label=f"{incr:.2f}")
            ax.legend()

        elif projection == "3d":
            ax = fig.add_subplot(projection='3d')
            x, y = np.meshgrid(shifts, self.increments, indexing="ij")
            ax.plot_wireframe(x, y, spectrum.T, lw=0.5)
            ax.set_ylabel(self.increment_qty)

        ax.set_xlim(reversed(ax.get_xlim()))
        unit = "ppm" if self.hz_ppm_valid else "Hz"
        ax.set_xlabel(
            f"{self.latex_nuclei[0]} ({unit})" if self.nuclei[0] is not None
            else unit
        )
        plt.show()

    @logger
    def phase_data(
        self,
        p0: float = 0.0,
        p1: float = 0.0,
        pivot: int = 0,
    ) -> None:
        """Apply first-order phae correction to the estimator's data.

        Parameters
        ----------
        p0
            Zero-order phase correction, in radians.

        p1
            First-order phase correction, in radians.

        pivot
            Index of the pivot.
        """
        sanity_check(
            ("p0", p0, sfuncs.check_float),
            ("p1", p1, sfuncs.check_float),
            ("pivot", pivot, sfuncs.check_index, (self._data.size,)),
        )
        self._data = sig.phase(self._data, [0., p0], [0., p1], [0, pivot])

    @logger
    def estimate(
        self,
        region: Optional[Tuple[float, float]] = None,
        noise_region: Optional[Tuple[float, float]] = None,
        region_unit: str = "hz",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        method: str = "gauss-newton",
        mode: str = "apfd",
        phase_variance: bool = True,
        max_iterations: Optional[int] = None,
        cut_ratio: Optional[float] = 1.1,
        mpm_trim: Optional[int] = 4096,
        nlp_trim: Optional[int] = None,
        fprint: bool = True,
        _log: bool = True,
    ) -> None:
        sanity_check(
            (
                "region_unit", region_unit, sfuncs.check_frequency_unit,
                (self.hz_ppm_valid,),
            ),
            (
                "initial_guess", initial_guess, sfuncs.check_initial_guess,
                (self.dim,), {}, True
            ),
            ("method", method, sfuncs.check_one_of, ("lbfgs", "gauss-newton", "exact")),
            ("phase_variance", phase_variance, sfuncs.check_bool),
            ("mode", mode, sfuncs.check_optimiser_mode),
            (
                "max_iterations", max_iterations, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            ("fprint", fprint, sfuncs.check_bool),
            ("mpm_trim", mpm_trim, sfuncs.check_int, (), {"min_value": 1}, True),
            ("nlp_trim", nlp_trim, sfuncs.check_int, (), {"min_value": 1}, True),
            (
                "cut_ratio", cut_ratio, sfuncs.check_float, (),
                {"greater_than_one": True}, True,
            ),
        )
        sanity_check(
            (
                "region", region, sfuncs.check_region,
                (self.sw(region_unit), self.offset(region_unit)), {}, True,
            ),
            (
                "noise_region", noise_region, sfuncs.check_region,
                (self.sw(region_unit), self.offset(region_unit)), {}, True,
            ),
        )

        if region is None:
            region = self.convert(((0, self._data.size - 1),), "idx->hz")
            noise_region = None
            mpm_signal = nlp_signal = self.data
            mpm_expinfo = nlp_expinfo = self.expinfo

        else:
            for i, fid in enumerate(self.data):
                filt = Filter(
                    fid,
                    self.expinfo,
                    region,
                    noise_region,
                    region_unit=region_unit,
                )

                if i == 0:
                    mpm_fid, mpm_expinfo = filt.get_filtered_fid(cut_ratio=cut_ratio)
                    nlp_fid, nlp_expinfo = filt.get_filtered_fid(cut_ratio=None)
                    mpm_signal = np.zeros(
                        (self.n_increments, mpm_fid.size),
                        dtype="complex128",
                    )
                    nlp_signal = np.zeros(
                        (self.n_increments, nlp_fid.size),
                        dtype="complex128",
                    )

                else:
                    mpm_fid, _ = filt.get_filtered_fid(cut_ratio=cut_ratio)
                    nlp_fid, _ = filt.get_filtered_fid(cut_ratio=None)

                mpm_signal[i] = mpm_fid
                nlp_signal[i] = nlp_fid

            region = filt.get_region()
            noise_region = filt.get_noise_region()

        if (mpm_trim is None) or (mpm_trim > mpm_signal.shape[1]):
            mpm_trim = mpm_signal.shape[1]
        if (nlp_trim is None) or (nlp_trim > nlp_signal.shape[1]):
            nlp_trim = nlp_signal.shape[1]

        mpm_signal = mpm_signal[:, :mpm_trim]
        nlp_signal = nlp_signal[:, :nlp_trim]

        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            oscillators = initial_guess if isinstance(initial_guess, int) else 0
            x0 = MatrixPencil(
                mpm_expinfo,
                mpm_signal[0],
                oscillators=oscillators,
                fprint=fprint,
            ).get_params()

            if x0 is None:
                return self._results.append(
                    Result(
                        self.n_increments * [np.array([[]])],
                        self.n_increments * [np.array([[]])],
                        region,
                        noise_region,
                        self.sfo,
                    )
                )

        params = []
        errors = []
        for i, signal in enumerate(nlp_signal):
            result = NonlinearProgramming(
                nlp_expinfo,
                signal,
                x0,
                method=method,
                mode=mode,
                max_iterations=max_iterations,
                phase_variance=True,
                negative_amps="leave_alone",
                fprint=fprint,
            )
            x0 = result.get_params()
            params.append(x0)
            errors.append(result.get_errors())

        self._results.append(
            Result(
                params,
                errors,
                region,
                noise_region,
                self.sfo,
            )
        )

    def new_bruker(self):
        pass

    def new_synthetic_from_simulation(self):
        pass

    def plot_result(self):
        pass
