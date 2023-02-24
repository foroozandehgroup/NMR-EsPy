# seq_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 24 Feb 2023 11:42:20 GMT

from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import nmrespy as ne
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.estimators import Result
from nmrespy.estimators.onedim import Estimator1D
from nmrespy.freqfilter import Filter
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import nonlinear_programming
from nmrespy.nlp.optimisers import trust_ncg
from nmrespy.plot import make_color_cycle
from nmrespy.nlp._funcs import FunctionFactory


class EstimatorSeq1D(Estimator1D):

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        datapath: Optional[Path] = None,
        increments: Optional[np.ndarray] = None,
        increment_type: Optional[str] = None,
    ) -> None:
        super().__init__(data[0], expinfo, datapath)
        self._data = data
        sanity_check(
            (
                "increments", increments, sfuncs.check_ndarray, (),
                {"dim": 1, "shape": [(0, data.shape[0])]}, True,
            ),
            ("increment_type", increment_type, sfuncs.check_str, (), {}, True),
        )
        self.increments = increments
        self.increment_type = increment_type

    def estimate(
        self,
        region: Optional[Tuple[float, float]] = None,
        noise_region: Optional[Tuple[float, float]] = None,
        region_unit: str = "hz",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        mode: str = "apfd",
        amp_thold: Optional[float] = None,
        phase_variance: bool = True,
        cut_ratio: Optional[float] = 1.1,
        mpm_trim: Optional[int] = None,
        nlp_trim: Optional[int] = None,
        hessian: str = "gauss-newton",
        max_iterations: Optional[int] = None,
        negative_amps: str = "remove",
        output_mode: Optional[int] = 10,
        save_trajectory: bool = False,
        epsilon: float = 1.0e-8,
        eta: float = 0.15,
        initial_trust_radius: float = 1.0,
        max_trust_radius: float = 4.0,
        _log: bool = True,
    ) -> None:
        sanity_check(
            (
                "region_unit", region_unit, sfuncs.check_frequency_unit,
                (self.hz_ppm_valid,),
            ),
            (
                "initial_guess", initial_guess, sfuncs.check_initial_guess,
                (self.dim,), {}, True,
            ),
            ("hessian", hessian, sfuncs.check_one_of, ("gauss-newton", "exact")),
            ("phase_variance", phase_variance, sfuncs.check_bool),
            ("mode", mode, sfuncs.check_optimiser_mode),
            (
                "amp_thold", amp_thold, sfuncs.check_float, (),
                {"greater_than_zero": True}, True,
            ),
            (
                "mpm_trim", mpm_trim, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            (
                "nlp_trim", nlp_trim, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            (
                "cut_ratio", cut_ratio, sfuncs.check_float, (),
                {"min_value": 1.}, True,
            ),
            (
                "max_iterations", max_iterations, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            (
                "negative_amps", negative_amps, sfuncs.check_one_of,
                ("remove", "flip_phase", "ignore"),
            ),
            ("output_mode", output_mode, sfuncs.check_int, (), {"min_value": 0}, True),
            ("save_trajectory", save_trajectory, sfuncs.check_bool),
            (
                "epsilon", epsilon, sfuncs.check_float, (),
                {"min_value": np.finfo(float).eps},
            ),
            ("eta", eta, sfuncs.check_float, (), {"min_value": 0.0, "max_value": 1.0}),
            (
                "initial_trust_radius", initial_trust_radius, sfuncs.check_float, (),
                {"greater_than_zero": True},
            ),
        )
        sanity_check(
            self._region_check(region, region_unit, "region"),
            self._region_check(noise_region, region_unit, "noise_region"),
            (
                "max_trust_radius", max_trust_radius, sfuncs.check_float, (),
                {"min_value": initial_trust_radius},
            ),
        )

        if region is None:
            region_unit = "hz"
            region = self._full_region
            noise_region = None
            mpm_fid = self.data[0]
            initial_fid = self.data[0]
            other_fids = [fid for fid in self.data[1:]]
            mpm_expinfo = nlp_expinfo = self.expinfo

        else:
            region = self._process_region(region)
            noise_region = self._process_region(noise_region)
            initial_filter = Filter(
                self.data[0],
                self.expinfo,
                region,
                noise_region,
                region_unit=region_unit,
            )

            mpm_fid, mpm_expinfo = initial_filter.get_filtered_fid(cut_ratio=cut_ratio)
            initial_fid, nlp_expinfo = initial_filter.get_filtered_fid(cut_ratio=None)

            other_fids = []
            for fid in self.data[1:]:
                filter_ = Filter(
                    fid,
                    self.expinfo,
                    region,
                    noise_region,
                    region_unit=region_unit,
                )

                other_fids.append(filter_.get_filtered_fid(cut_ratio=None)[0])

            region = initial_filter.get_region()
            noise_region = initial_filter.get_noise_region()

        mpm_trim = self._get_trim("mpm", mpm_trim, mpm_fid.shape[-1])
        nlp_trim = self._get_trim("nlp", nlp_trim, initial_fid.shape[-1])

        mpm_fid = mpm_fid[:mpm_trim]
        initial_fid = initial_fid[:nlp_trim]
        other_fids = [fid[:nlp_trim] for fid in other_fids]

        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            oscillators = initial_guess if isinstance(initial_guess, int) else 0
            x0 = MatrixPencil(
                mpm_expinfo,
                mpm_fid,
                oscillators=oscillators,
                fprint=isinstance(output_mode, int),
            ).get_params()
            # TODO deal with case of no oscillators

        if max_iterations is None:
            if hessian == "exact":
                max_iterations = self.default_max_iterations_exact_hessian
            elif hessian == "gauss-newton":
                max_iterations = self.default_max_iterations_gn_hessian

        initial_result = nonlinear_programming(
            nlp_expinfo,
            initial_fid,
            x0,
            phase_variance=phase_variance,
            hessian=hessian,
            mode=mode,
            amp_thold=amp_thold,
            max_iterations=max_iterations,
            negative_amps="flip_phase",
            output_mode=output_mode,
            save_trajectory=save_trajectory,
            tolerance=epsilon,
            eta=eta,
            initial_trust_radius=initial_trust_radius,
            max_trust_radius=max_trust_radius,
        )
        results = [
            Result(
                x0,
                # initial_result.x,
                initial_result.errors,
                region,
                noise_region,
                self.sfo,
            )
        ]

        x0 = initial_result.x

        for fid in other_fids:
            result = nonlinear_programming(
                nlp_expinfo,
                fid,
                x0,
                phase_variance=phase_variance,
                hessian=hessian,
                mode="a",
                amp_thold=amp_thold,
                max_iterations=max_iterations,
                negative_amps="ignore",
                output_mode=output_mode,
                save_trajectory=save_trajectory,
                tolerance=epsilon,
                eta=eta,
                initial_trust_radius=initial_trust_radius,
                max_trust_radius=max_trust_radius,
            )

            results.append(
                Result(
                    result.x,
                    result.errors,
                    region,
                    noise_region,
                    self.sfo,
                )
            )

            x0 = result.x

        self._results.append(results)

    def fit(self, osc: int, func: str, index: int = -1,) -> Tuple[float, float]:
        sanity_check(
            self._index_check(index),
            ("func", func, sfuncs.check_one_of, ("T1",)),
        )
        res = self.get_results(indices=[index])[0]
        n_oscs = res[0].get_params().shape[0]
        sanity_check(
            (
                "osc", osc, sfuncs.check_int, (),
                {"min_value": 0, "max_value": n_oscs - 1},
            ),
        )

        integrals = self.integrals(osc, index=index)
        x0 = np.array([integrals[-1], 1.])

        if func == "T1":
            function_factory = FunctionFactoryInvRec

        result = trust_ncg(
            x0=x0,
            function_factory=function_factory,
            args=(integrals, self.increments),
        ).x

        return result

    def integrals(self, osc: int, index: int = -1) -> np.ndarray:
        sanity_check(
            self._index_check(index),
        )
        res = self.get_results(indices=[index])[0]
        n_oscs = res[0].get_params().shape[0]
        sanity_check(
            (
                "osc", osc, sfuncs.check_int, (),
                {"min_value": 0, "max_value": n_oscs - 1},
            ),
        )

        res, = self.get_results(indices=[index])
        return np.array(
            [
                self.oscillator_integrals(
                    np.expand_dims(r.get_params()[osc], axis=0),
                    absolute=False,
                )
                for r in res
            ]
        )[:, 0].real

    def plot_result(
        self,
        index: int = -1,
        xaxis_unit: str = "hz",
        oscillator_colors: Any = None,
        elev: float = 45.,
        azim: float = 45.,
        **kwargs,
    ):
        sanity_check(
            self._index_check(index),
            self._funit_check(xaxis_unit, "xaxis_unit"),
            (
                "oscillator_colors", oscillator_colors, sfuncs.check_oscillator_colors,
                (), {}, True,
            ),
            ("elev", elev, sfuncs.check_float),
            ("azim", azim, sfuncs.check_float),
        )

        result, = self.get_results([index])
        region, = result[0].get_region(unit=xaxis_unit)
        slice_ = slice(
            *self.convert([region], f"{xaxis_unit}->idx")[0]
        )
        shifts, = self.get_shifts(unit=xaxis_unit)
        shifts = shifts[slice_]

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(projection="3d")

        params_set = [res.get_params() for res in result]
        spectra = []
        oscillators = []
        for (fid, params) in zip(self.data, params_set):
            fid[0] *= 0.5
            spectra.append(ne.sig.ft(fid).real[slice_])
            incr_oscillators = []
            for p in params:
                p = np.expand_dims(p, axis=0)
                osc = self.make_fid(p)
                osc[0] *= 0.5
                incr_oscillators.append(ne.sig.ft(osc).real[slice_])
            oscillators.append(incr_oscillators)

        span = self._get_data_span(
            spectra +
            [osc for incr_oscillators in oscillators for osc in incr_oscillators]
        )

        noscs = len(oscillators[0])

        for spec, incr, oscs in zip(
            reversed(spectra), reversed(self.increments), reversed(oscillators)
        ):
            colors = make_color_cycle(oscillator_colors, noscs)
            y = np.full(shifts.shape, incr)
            ax.plot(shifts, y, spec, color="#000000")
            for osc in oscs:
                ax.plot(shifts, y, osc, color=next(colors), lw=0.6)

        # azim at 270 provies a face-on view of the spectra.
        ax.view_init(elev=elev, azim=270. + azim)

        # Configure x-axis
        ax.set_xlim(shifts[0], shifts[-1])
        nuc = self.unicode_nuclei
        unit = xaxis_unit.replace("h", "H")
        if nuc is None:
            xlabel = unit
        else:
            xlabel = f"{nuc[-1]} ({unit})"
        ax.set_xlabel(xlabel)

        # Configure y-axis
        ax.set_ylim(self.increments[0], self.increments[-1])
        if self.increment_type is not None:
            ax.set_ylabel(self.increment_type)

        # Configure z-axis
        h = span[1] - span[0]
        bottom = span[0] - 0.03 * h
        top = span[1] + 0.03 * h
        ax.set_zlim(bottom, top)
        ax.set_zticks([])

        return fig, ax

    @staticmethod
    def _get_data_span(data: Iterable[np.ndarray]) -> Tuple[float, float]:
        return (
            min([np.amin(datum) for datum in data]),
            max([np.amax(datum) for datum in data]),
        )


class EstimatorInvRec(EstimatorSeq1D):

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        increments: np.ndarray,
        datapath: Optional[Path] = None,
    ) -> None:
        super().__init__(
            data, expinfo, datapath, increments, increment_type="$\\tau$ (s)")

    def fit(self, osc: int, index: int = -1) -> Tuple[float, float]:
        r"""Fit estimation result for a given oscillator across increments in
        order to predict the longitudinal relaxtation time, :math:`T_1`. The
        function that is fit is:

        .. math::

            x\left(I_0, T_1\right) =
            I_0 \left[ 1 - 2 \exp\left( \frac{\tau}{T_1} \right) \right].

        where :math:`I_0` is the predicted integral of the oscillator peak in
        the limit of :math:`\tau \rightarrow \infty`.

        Parameters
        ----------
        index
            The result index.

        osc
            The index of the oscillator to considier.

        Returns
        -------
        I0

        T1
            The predicted longitudinal relaxation time (s).
        """
        I0, T1 = super().fit(osc, func="T1", index=index)
        return I0, T1

    def model(
        self,
        I0: float,
        T1: float,
        taus: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        sanity_check(
            ("I0", I0, sfuncs.check_float),
            ("T1", T1, sfuncs.check_float),
            ("taus", taus, sfuncs.check_ndarray, (), {"dim": 1}, True),
        )

        if taus is None:
            taus = self.increments
        return I0 * (1 - 2 * np.exp(-taus / T1))

    @staticmethod
    def _obj_grad_hess(
        theta: np.ndarray,
        *args: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        r"""Objective, gradient and Hessian for fitting inversion recovery data.
        The model to be fit is given by

        .. math::

            I = I_0 \left[ 1 - 2 \exp\left( \frac{\tau}{T_1} \right) \right].

        Parameters
        ----------
        theta
            Parameters of the model: :math:`I_0` and :math:`T_1`.

        args
            Comprises two items:

            * integrals across each increment.
            * delays (:math:`\tau`).
        """
        I0, T1 = theta
        integrals, taus = args

        t_over_T1 = taus / T1
        t_over_T1_sq = taus / (T1 ** 2)
        t_over_T1_cb = taus / (T1 ** 3)
        exp_t_over_T1 = np.exp(-t_over_T1)
        y_minus_x = integrals - I0 * (1 - 2 * exp_t_over_T1)
        n = taus.size

        # Objective
        obj = np.sum(y_minus_x.T ** 2)

        # Grad
        d1 = np.zeros((n, 2))
        d1[:, 0] = 1 - 2 * exp_t_over_T1
        d1[:, 1] = -2 * I0 * t_over_T1_sq * exp_t_over_T1
        grad = -2 * y_minus_x.T @ d1

        # Hessian
        d2 = np.zeros((n, 2, 2))
        off_diag = -2 * t_over_T1_sq * exp_t_over_T1
        d2[:, 0, 1] = off_diag
        d2[:, 1, 0] = off_diag
        d2[:, 1, 1] = 2 * I0 * t_over_T1_cb * exp_t_over_T1 * (2 - t_over_T1)

        hess = -2 * (np.einsum("i,ijk->jk", y_minus_x, d2) - d1.T @ d1)

        return obj, grad, hess


class FunctionFactoryInvRec(FunctionFactory):
    def __init__(self, theta: np.ndarray, *args) -> None:
        super().__init__(theta, EstimatorInvRec._obj_grad_hess, *args)
