# seq_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 23 Feb 2023 00:27:06 GMT

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import nmrespy as ne
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.freqfilter import Filter
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import nonlinear_programming
from nmrespy.nlp.optimisers import trust_ncg
from nmrespy.nlp._funcs import FunctionFactory
from . import _Estimator1DProc, Result


class EstimatorSeq1D(_Estimator1DProc):

    default_mpm_trim = 4096
    default_nlp_trim = None
    default_max_iterations_exact_hessian = 100
    default_max_iterations_gn_hessian = 200

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        datapath: Optional[Path] = None,
        increments: Optional[np.ndarray] = None,
        unit: Optional[str] = None,
    ) -> None:
        super().__init__(data[0], expinfo, datapath)
        self._data = data
        sanity_check(
            (
                "increments", increments, sfuncs.check_ndarray, (),
                {"dim": 1, "shape": [(0, data.shape[0])]}, True,
            ),
            ("unit", unit, sfuncs.check_str, (), {}, True),
        )
        self.increments = increments
        self.unit = unit

    def get_shifts(
        self,
        pts: Optional[int] = None,
        unit: str = "hz",
        flip: bool = True,
        meshgrid: bool = True,
    ) -> Iterable[np.ndarray]:
        sanity_check(
            self._pts_check(pts),
            self._funit_check(unit),
            ("flip", flip, sfuncs.check_bool),
            ("meshgrid", meshgrid, sfuncs.check_bool),
        )
        shifts = super().get_shifts(pts, unit, flip)[0]

        if meshgrid:
            return tuple(np.meshgrid(shifts, self.increments, indexing="ij"))
        else:
            return (shifts, self.increments)

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

    def plot_result(self):
        result = self.get_results([0])[0]
        region = result[0].get_region()
        shifts = self.get_shifts(meshgrid=False)[0]
        params_set = [res.get_params() for res in result]
        for i, (fid, params) in enumerate(zip(self.data, params_set)):
            fig, ax = plt.subplots()
            fid[0] *= 0.5
            spec = ne.sig.ft(fid)
            ax.plot(shifts, spec)
            for p in params:
                p = np.expand_dims(p, axis=0)
                osc = self.make_fid(p)
                osc[0] *= 0.5
                osc_spec = ne.sig.ft(osc)
                ax.plot(shifts, osc_spec)
            ax.set_xlim(region[0][0], region[0][1])
            fig.savefig(Path(f"~/fig{i}.pdf").expanduser())


class EstimatorInvRec(EstimatorSeq1D):

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        increments: np.ndarray,
        datapath: Optional[Path] = None,
    ) -> None:
        super().__init__(data, expinfo, datapath, increments, unit="s")

    def fit(self, index: int, osc: int) -> Tuple[float, float]:
        res = self.get_results(indices=[index])[0]
        integrals = np.array(
            [
                np.real(
                    self.oscillator_integrals(
                        np.expand_dims(r.get_params()[osc], axis=0),
                        absolute=False,
                    )
                )
                for r in res
            ]
        )[:, 0]
        x0 = np.array([integrals[-1], 1.])

        I0, T1 = trust_ncg(
            x0=x0,
            function_factory=FunctionFactoryInvRec,
            args=(integrals, self.increments),
        ).x

        return I0, T1

    @staticmethod
    def obj_grad_hess(theta: np.ndarray, *args: Tuple[np.ndarray, np.ndarray]) -> float:
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
        super().__init__(theta, EstimatorInvRec.obj_grad_hess, *args)
