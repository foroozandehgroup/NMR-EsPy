# seq_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sun 26 Feb 2023 14:48:55 GMT

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import nmrespy as ne
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.estimators import Result, logger
from nmrespy.estimators.onedim import Estimator1D
from nmrespy.freqfilter import Filter
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import nonlinear_programming
from nmrespy.nlp._funcs import FunctionFactory
from nmrespy.nlp.optimisers import trust_ncg
from nmrespy.plot import make_color_cycle


class EstimatorSeq1D(Estimator1D):

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        datapath: Optional[Path] = None,
        increments: Optional[np.ndarray] = None,
        increment_label: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        data
            The data associated with the binary file in `path`.

        datapath
            The path to the directory containing the NMR data.

        expinfo
            Experiment information.

        increments
            The values of increments used to acquire the 1D signals. Examples would
            include:

            * Delay times in an inversion recovery experiment.
            * Gradient strengths in a diffusion experiment.

        increment_label
            A label to describe what the increment is. This will appear in relavent
            plots. For example, a suitable value could be ``"$G_z$
            (Gcm\\textsuperscript{-1})"`` for a diffusion experiment.
        """
        super().__init__(data[0], expinfo, datapath)
        self._data = data
        sanity_check(
            (
                "increments", increments, sfuncs.check_ndarray, (),
                {"dim": 1, "shape": [(0, data.shape[0])]}, True,
            ),
            ("increment_label", increment_label, sfuncs.check_str, (), {}, True),
        )
        self.increments = increments
        self.increment_label = increment_label

    def view_data(
        self,
        domain: str = "freq",
        components: str = "real",
        freq_unit: str = "hz",
    ) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        if domain == "freq":
            x, = self.get_shifts(unit=freq_unit)
            zs = copy.deepcopy(self.data)
            zs[:, 0] *= 0.5
            zs = ne.sig.ft(zs, axes=-1)

        elif domain == "time":
            x, = self.get_timepoints()
            z = self.data

        ys = self.increments

        if components in ("real", "both"):
            for (y, z) in zip(ys, zs):
                y_arr = np.full(z.shape, y)
                ax.plot(x, y_arr, z.real, color="k")
        if components in ("imag", "both"):
            for (y, z) in zip(ys, zs):
                y_arr = np.full(z.shape, y)
                ax.plot(x, y_arr, z.imag, color="#808080")

        if domain == "freq":
            xlabel, = self._axis_freq_labels(freq_unit)
        elif domain == "time":
            xlabel = "$t$ (s)"

        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.increment_label)
        ax.set_zticks([])

        plt.show()

    def view_data_increment(
        self,
        increment: int = 0,
        domain: str = "freq",
        components: str = "real",
        freq_unit: str = "hz",
    ) -> None:
        sanity_check(
            (
                "increment", increment, sfuncs.check_int, (),
                {"min_value": 0, "max_value": self.increments - 1},
            ),
        )
        data_cp = copy.deepcopy(self.data)
        self._data = self.data[increment]

        super().view_data(domain=domain, components=components, freq_unit=freq_unit)

        self._data = data_cp

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
        r"""Estimate a specified region of the signal.

        The basic steps that this method carries out are:

        * (Optional, but highly advised) Generate a frequency-filtered "sub-FID"
          corresponding to a specified region of interest for each increment.
        * (Optional) For the first increment in the data, generate an initial
          guess using the Minimum Description Length (MDL) [#]_ and Matrix
          Pencil Method (MPM) [#]_ [#]_ [#]_ [#]_
        * Apply numerical optimisation to determine a final estimate of the signal
          parameters for the first increment. The optimisation routine employed
          is the Trust Newton Conjugate Gradient (NCG) algorithm ([#]_ ,
          Algorithm 7.2).
        * For each successive increment, use the previous increment's estiamtion
          result as the initial guess, and optimise the amplitudes only, again using
          the NCG algorithm.

        Parameters
        ----------
        region
            The frequency range of interest. Should be of the form ``[left, right]``
            where ``left`` and ``right`` are the left and right bounds of the region
            of interest in Hz or ppm (see ``region_unit``). If ``None``, the
            full signal will be considered, though for sufficently large and
            complex signals it is probable that poor and slow performance will
            be realised.

        noise_region
            If ``region`` is not ``None``, this must be of the form ``[left, right]``
            too. This should specify a frequency range where no noticeable signals
            reside, i.e. only noise exists.

        region_unit
            One of ``"hz"`` or ``"ppm"`` Specifies the units that ``region``
            and ``noise_region`` have been given as.

        initial_guess
            * If ``None``, an initial guess will be generated using the MPM
              with the MDL being used to estimate the number of oscillators
              present.
            * If an int, the MPM will be used to compute the initial guess with
              the value given being the number of oscillators.
            * If a NumPy array, this array will be used as the initial guess.

        hessian
            Specifies how to construct the Hessian matrix.

            * If ``"exact"``, the exact Hessian will be used.
            * If ``"gauss-newton"``, the Hessian will be approximated as is
              done with the Gauss-Newton method. See the *"Derivation from
              Newton's method"* section of `this article
              <https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm>`_.

        mode
            A string containing a subset of the characters ``"a"`` (amplitudes),
            ``"p"`` (phases), ``"f"`` (frequencies), and ``"d"`` (damping factors).
            Specifies which types of parameters should be considered for optimisation.
            In most scenarios, you are likely to want the default value, ``"apfd"``.

        amp_thold
            A value that imposes a threshold for deleting oscillators of
            negligible ampltiude.

            * If ``None``, does nothing.
            * If a float, oscillators with amplitudes satisfying :math:`a_m <
              a_{\mathrm{thold}} \lVert \boldsymbol{a} \rVert_2` will be
              removed from the parameter array, where :math:`\lVert
              \boldsymbol{a} \rVert_2` is the Euclidian norm of the vector of
              all the oscillator amplitudes. It is advised to set ``amp_thold``
              at least a couple of orders of magnitude below 1.

        phase_variance
            Whether or not to include the variance of oscillator phases in the cost
            function. This should be set to ``True`` in cases where the signal being
            considered is derived from well-phased data.

        mpm_trim
            Specifies the maximal size allowed for the filtered signal when
            undergoing the Matrix Pencil. If ``None``, no trimming is applied
            to the signal. If an int, and the filtered signal has a size
            greater than ``mpm_trim``, this signal will be set as
            ``signal[:mpm_trim]``.

        nlp_trim
            Specifies the maximal size allowed for the filtered signal when undergoing
            nonlinear programming. By default (``None``), no trimming is applied to
            the signal. If an int, and the filtered signal has a size greater than
            ``nlp_trim``, this signal will be set as ``signal[:nlp_trim]``.

        max_iterations
            A value specifiying the number of iterations the routine may run
            through before it is terminated. If ``None``, a default number
            of maximum iterations is set, based on the the data dimension and
            the value of ``hessian``.

        negative_amps
            Indicates how to treat oscillators which have gained negative
            amplitudes during the optimisation.

            * ``"remove"`` will result in such oscillators being purged from
              the parameter estimate. The optimisation routine will the be
              re-run recursively until no oscillators have a negative
              amplitude.
            * ``"flip_phase"`` will retain oscillators with negative
              amplitudes, but the the amplitudes will be multiplied by -1,
              and a π radians phase shift will be applied.
            * ``"ignore"`` will do nothing (negative amplitude oscillators will remain).

        output_mode
            Dictates what information is sent to stdout.

            * If ``None``, nothing will be sent.
            * If ``0``, only a message on the outcome of the optimisation will
              be sent.
            * If a positive int ``k``, information on the cost function,
              gradient norm, and trust region radius is sent every kth
              iteration.

        save_trajectory
            If ``True``, a list of parameters at each iteration will be saved, and
            accessible via the ``trajectory`` attribute.

            .. warning:: Not implemented yet!

        epsilon
            Sets the convergence criterion. Convergence will occur when
            :math:`\lVert \boldsymbol{g}_k \rVert_2 < \epsilon`.

        eta
            Criterion for accepting an update. An update will be accepted if
            the ratio of the actual reduction and the predicted reduction is
            greater than ``eta``:

            .. math ::

                \rho_k = \frac{f(x_k) - f(x_k - p_k)}{m_k(0) - m_k(p_k)} > \eta

        initial_trust_radius
            The initial value of the radius of the trust region.

        max_trust_radius
            The largest permitted radius for the trust region.

        _log
            Ignore this!

        References
        ----------
        .. [#] Yingbo Hua and Tapan K Sarkar. “Matrix pencil method for estimating
           parameters of exponentially damped/undamped sinusoids in noise”. In:
           IEEE Trans. Acoust., Speech, Signal Process. 38.5 (1990), pp. 814–824.

        .. [#] Yung-Ya Lin et al. “A novel detection–estimation scheme for noisy
           NMR signals: applications to delayed acquisition data”. In: J. Magn.
           Reson. 128.1 (1997), pp. 30–41.

        .. [#] Yingbo Hua. “Estimating two-dimensional frequencies by matrix
           enhancement and matrix pencil”. In: [Proceedings] ICASSP 91: 1991
           International Conference on Acoustics, Speech, and Signal Processing.
           IEEE. 1991, pp. 3073–3076.

        .. [#] Fang-Jiong Chen et al. “Estimation of two-dimensional frequencies
           using modified matrix pencil method”. In: IEEE Trans. Signal Process.
           55.2 (2007), pp. 718–724.

        .. [#] M. Wax, T. Kailath, Detection of signals by information theoretic
           criteria, IEEE Transactions on Acoustics, Speech, and Signal Processing
           33 (2) (1985) 387–392.

        .. [#] Jorge Nocedal and Stephen J. Wright. Numerical optimization. 2nd
               ed. Springer series in operations research. New York: Springer,
               2006.
        """
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
                initial_result.x,
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

    def _fit(
        self,
        func: str,
        oscs: Optional[Iterable[int]] = None,
        index: int = -1,
    ) -> Iterable[np.ndarray]:
        sanity_check(
            self._index_check(index),
            ("func", func, sfuncs.check_one_of, ("T1",)),
        )
        res = self.get_results(indices=[index])[0]
        n_oscs = res[0].get_params().shape[0]
        sanity_check(
            self._oscs_check(oscs, n_oscs),
        )
        oscs = self._proc_oscs(oscs, n_oscs)

        integrals = self.integrals(oscs, index=index)

        if func == "T1":
            function_factory = FunctionFactoryInvRec

        results = []
        errors = []
        for integs in integrals:
            if func == "T1":
                x0 = np.array([integs[-1], 1.])

            nlp_result = trust_ncg(
                x0=x0,
                function_factory=function_factory,
                args=(integs, self.increments),
                output_mode=None,
            )
            results.append(nlp_result.x)
            errors.append(nlp_result.errors / np.sqrt(len(self.increments)))

        return results, errors

    def integrals(
        self,
        oscs: Optional[Iterable[int]] = None,
        index: int = -1,
    ) -> Iterable[np.ndarray]:
        sanity_check(
            self._index_check(index),
        )
        res = self.get_results(indices=[index])[0]
        n_oscs = res[0].get_params().shape[0]
        sanity_check(
            self._oscs_check(oscs, n_oscs),
        )
        oscs = self._proc_oscs(oscs, n_oscs)
        res, = self.get_results(indices=[index])
        return [
            np.array(
                [
                    self.oscillator_integrals(
                        np.expand_dims(r.get_params()[osc], axis=0),
                        absolute=False,
                    )
                    for r in res
                ]
            )[:, 0].real
            for osc in oscs
        ]

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
        if self.increment_label is not None:
            ax.set_ylabel(self.increment_label)

        # Configure z-axis
        h = span[1] - span[0]
        bottom = span[0] - 0.03 * h
        top = span[1] + 0.03 * h
        ax.set_zlim(bottom, top)
        ax.set_zticks([])

        return fig, ax

    @logger
    def plot_result_increment(
        self,
        increment: int = 0,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, np.ndarray[mpl.axes.Axes]]:
        sanity_check(
            (
                "increment", increment, sfuncs.check_int, (),
                {"min_value": 0, "max_value": len(self.increments) - 1},
            ),
        )

        # Temporarily fudge the `_results`, and `_data` attributes so
        # `Estimator1D.plot_result` will work
        results_cp = copy.deepcopy(self._results)
        data_cp = copy.deepcopy(self.data)

        self._results = [r[increment] for r in results_cp]
        self._data = self.data[increment]

        if "plot_model" not in kwargs:
            kwargs["plot_model"] = False

        fig, axs = super().plot_result(**kwargs)

        self._results = results_cp
        self._data = data_cp

        return fig, axs

    def _oscs_check(self, x: Any, n_oscs: int):
        return (
            "oscs", x, sfuncs.check_int_list, (),
            {
                "min_value": 0,
                "max_value": n_oscs - 1,
                "len_one_can_be_listless": True,
            }, True,
        )

    def _proc_oscs(
        self,
        oscs: Optional[Union[int, Iterable[int]]],
        n_oscs: int,
    ) -> Iterable[int]:
        if oscs is None:
            oscs = list(range(n_oscs))
        elif isinstance(oscs, int):
            oscs = [oscs]
        return oscs


class EstimatorInvRec(EstimatorSeq1D):
    """Estimation class for the consideration of datasets acquired by an inversion
    recovery experiment, for the purpose of determining longitudinal relaxation
    times (:math:`T_1`)."""

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        delays: np.ndarray,
        datapath: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        data
            The data associated with the binary file in `path`.

        expinfo
            Experiment information.

        delays
            Delays used in the inversion recovery experiment.

        datapath
            The path to the directory containing the NMR data.
        """
        super().__init__(
            data, expinfo, datapath, increments=delays, increment_label="$\\tau$ (s)")

    @classmethod
    def new_spinach(
        cls,
        shifts: Iterable[float],
        couplings: Optional[Iterable[Tuple(int, int, float)]],
        n_delays: int,
        max_delay: float,
        t1s: Union[Iterable[float], float],
        pts: int,
        sw: float,
        t2s: Union[Iterable[float], float] = 5.,
        offset: float = 0.,
        sfo: float = 500.,
        nucleus: str = "1H",
        snr: Optional[float] = 20.,
    ) -> EstimatorInvRec:
        sanity_check(
            ("shifts", shifts, sfuncs.check_float_list),
            ("n_delays", n_delays, sfuncs.check_int, (), {"min_value": 1}),
            ("max_delay", max_delay, sfuncs.check_float, (), {"greater_than_zero": True}),  # noqa: E501
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}),
            ("sw", sw, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("offset", offset, sfuncs.check_float),
            ("sfo", sfo, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("nucleus", nucleus, sfuncs.check_nucleus),
            ("snr", snr, sfuncs.check_float),
        )

        nspins = len(shifts)
        if isinstance(t1s, float):
            t1s = nspins * [t1s]
        if isinstance(t2s, float):
            t2s = nspins * [t2s]

        sanity_check(
            (
                "couplings", couplings, sfuncs.check_spinach_couplings, (nspins,),
                {}, True,
            ),
            (
                "t1s", t1s, sfuncs.check_float_list, (),
                {"length": nspins, "must_be_positive": True},
            ),
            (
                "t2s", t2s, sfuncs.check_float_list, (),
                {"length": nspins, "must_be_positive": True},
            ),
        )

        if couplings is None:
            couplings = []

        r1s = [1 / t1 for t1 in t1s]
        r2s = [1 / t2 for t2 in t2s]

        fid = cls._run_spinach(
            "invrec_sim", shifts, couplings, float(n_delays), float(max_delay), r1s,
            r2s, pts, sw, offset, sfo, nucleus, to_double=[4, 5],
        ).reshape((pts, n_delays)).T

        if snr is not None:
            fid = ne.sig.add_noise(fid, snr)

        expinfo = ne.ExpInfo(
            dim=1,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nucleus,
            default_pts=pts,
        )

        return cls(fid, expinfo, np.linspace(0, max_delay, n_delays))

    def fit(
        self,
        oscs: Optional[Iterable[int]] = None,
        index: int = -1
    ) -> Iterable[np.ndarray]:
        r"""Fit estimation result for the given oscillators across increments in
        order to predict the longitudinal relaxtation time, :math:`T_1`.

        For the oscillators specified, the integrals of the oscilators' peaks are
        determined at each increment, and the following function is fit:

        .. math::

            I \left(I_{\infty}, T_1, \tau\right) =
            I_{\infty} \left[ 1 - 2 \exp\left( \frac{\tau}{T_1} \right) \right].

        where :math:`I` is the peak integral when the delay is :math:`\tau`, and
        :math:`I_{\infty} = \lim_{\tau \rightarrow \infty} I`.

        Parameters
        ----------
        oscs
            The indices of the oscillators to considier. If ``None``, all oscillators
            are consdiered.

        index
            The result index. By default, the last result acquired is considered.

        Returns
        -------
        Iterable[np.ndarray]
            Iterable (list) of numpy arrays of shape ``(2,)``. For each array,
            the first element corresponds to :math:`I_{\infty}`, and the second
            element corresponds to :math:`T_1`.
        """
        result, errors = self._fit("T1", oscs, index=index)
        return result, errors

    def model(
        self,
        Iinfty: float,
        T1: float,
        delays: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Return the function

        .. math::

            \boldsymbol{I}\left(I_{\infty}, T_1, \boldsymbol{\tau} \right) =
                I_{\infty} \left[
                    1 - 2 \exp\left(- \frac{\boldsymbol{\tau}}{T_1} \right)
                \right]

        Parameters
        ----------
        Iinfty
            :math:`I_{\infty}`.

        T1
            :math:`T_1`.

        delays
            The delays to consider (:math:`\boldsymbol{\tau}`). If ``None``,
            ``self.increments`` will be used.
        """
        sanity_check(
            ("Iinfty", Iinfty, sfuncs.check_float),
            ("T1", T1, sfuncs.check_float),
            ("delays", delays, sfuncs.check_ndarray, (), {"dim": 1}, True),
        )

        if delays is None:
            delays = self.increments
        return Iinfty * (1 - 2 * np.exp(-delays / T1))

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

    def plot_fit_single_oscillator(
        self,
        osc: int,
        index: int = -1,
        fit_increments: int = 100,
        color: Any = "#808080",
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        sanity_check(
            self._index_check(index),
            (
                "fit_increments", fit_increments, sfuncs.check_int, (),
                {"min_value": len(self.increments)},
            ),
            ("color", color, sfuncs.check_mpl_color),
        )
        n_oscs = self.get_results(indices=[index])[0][0].get_params().shape[0]
        sanity_check(
            ("osc", osc, sfuncs.check_int, (), {"min_value": 0, "max_value": n_oscs - 1}),  # noqa: E501
        )
        params, errors = self.fit([osc], index)
        Iinfty, T1 = params[0]
        Iinfty_error, T1_error = errors[0]

        fig, ax = plt.subplots(**kwargs)
        x = np.linspace(self.increments[0], self.increments[-1], fit_increments)
        ax.plot(x, self.model(Iinfty, T1, x), color=color)
        ax.scatter(self.increments, self.integrals(oscs=[osc], index=index), color=color)  # noqa: E501
        ax.set_xlabel(self.increment_label)
        ax.set_ylabel("$I$")

        text = (
            f"$I_{{\\infty}} = \\num{{{Iinfty:.4g}}} \\pm "
            f"\\num{{{Iinfty_error:.4g}}}$\n"
            f"$T_1 = \\num{{{T1:.4g}}} \\pm "
            f"\\num{{{T1_error:.4g}}}$ s"
        )
        ax.text(0.02, 0.98, text, ha="left", va="top", transform=ax.transAxes)

        return fig, ax

    def plot_fit_multi_oscillators(
        self,
        oscs: Optional[Iterable[int]] = None,
        index: int = -1,
        fit_increments: int = 100,
        xs: str = "osc_idx",
        T1_labels: bool = True,
        colors: Any = None,
        azim: float = 45.,
        elev: float = 45.,
        label_zshift: float = 0.,
        label_fontsize: int = 7,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        sanity_check(
            self._index_check(index),
            (
                "fit_increments", fit_increments, sfuncs.check_int, (),
                {"min_value": len(self.increments)},
            ),
            ("xs", xs, sfuncs.check_one_of, ("osc_idx", "hz", "ppm")),
            ("colors", colors, sfuncs.check_oscillator_colors, (), {}, True),
            ("elev", elev, sfuncs.check_float),
            ("azim", azim, sfuncs.check_float),
            ("label_zshift", label_zshift, sfuncs.check_float),
            ("label_fontsize", label_fontsize, sfuncs.check_int, (), {"min_value": 1}),
        )
        res_params = self.get_results(indices=[index])[0][0].get_params()
        n_oscs = res_params.shape[0]
        sanity_check(self._oscs_check(oscs, n_oscs))
        oscs = self._proc_oscs(oscs, n_oscs)

        params, errors = self.fit(oscs, index)
        integrals = self.integrals(oscs, index=index)

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(projection="3d")

        colors = make_color_cycle(colors, len(oscs))
        zorder = n_oscs
        span_data = []

        if xs == "osc_idx":
            xs = oscs
            xlabel = "osc"
        else:
            nuc = self.unicode_nuclei
            unit = xs.replace("h", "H")
            if nuc is None:
                xlabel = unit
            else:
                xlabel = f"{nuc[-1]} ({unit})"
            if xs == "hz":
                xs = res_params[oscs, 2]
            elif xs == "ppm":
                xs = res_params[oscs, 2] / self.sfo[-1]

        for x, (Iinfty, T1), integs in zip(xs, params, integrals):
            color = next(colors)
            x_scatter = np.full(self.increments.shape, x)
            y_scatter = self.increments
            z_scatter = integs
            span_data.append(z_scatter)
            ax.scatter(
                x_scatter,
                y_scatter,
                z_scatter,
                color=color,
                zorder=zorder,
                lw=0.,
            )

            x_fit = np.full(fit_increments, x)
            y_fit = np.linspace(self.increments[0], self.increments[-1], fit_increments)
            z_fit = self.model(Iinfty, T1, y_fit)
            span_data.append(z_fit)
            ax.plot(x_fit, y_fit, z_fit, color=color, zorder=zorder)

            ax.text(
                x_fit[-1],
                y_fit[-1],
                z_fit[-1] + label_zshift,
                f"{T1:.3g}",
                color=color,
                zdir="z",
                ha="center",
                fontsize=label_fontsize,
            )

            zorder -= 1

        # azim at 270 provies a face-on view of the spectra.
        ax.view_init(elev=elev, azim=270. + azim)

        ax.set_xlabel(xlabel)
        ax.set_xlim(reversed(ax.get_xlim()))

        # Configure y-axis
        ax.set_ylim(self.increments[0], self.increments[-1])
        if self.increment_label is not None:
            ax.set_ylabel(self.increment_label)

        span = self._get_data_span(span_data)

        # Configure z-axis
        h = span[1] - span[0]
        bottom = span[0] - 0.03 * h
        top = span[1] + 0.03 * h
        ax.set_zlim(bottom, top)
        ax.set_zlabel("$I$")

        return fig, ax


class FunctionFactoryInvRec(FunctionFactory):
    def __init__(self, theta: np.ndarray, *args) -> None:
        super().__init__(theta, EstimatorInvRec._obj_grad_hess, *args)
