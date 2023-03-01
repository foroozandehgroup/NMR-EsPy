# seq_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 01 Mar 2023 19:51:16 GMT

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import nmrespy as ne
from nmrespy._misc import proc_kwargs_dict
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.estimators import Result, logger
from nmrespy.estimators.onedim import Estimator1D
from nmrespy.freqfilter import Filter
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import nonlinear_programming
from nmrespy.nlp._funcs import FunctionFactory
from nmrespy.nlp.optimisers import trust_ncg
from nmrespy.plot import make_color_cycle


# Patch Axes3D to prevent annoying padding
# https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new


class EstimatorSeq1D(Estimator1D):

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        datapath: Optional[Path] = None,
        increments: Optional[np.ndarray] = None,
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
        """
        super().__init__(data[0], expinfo, datapath)
        self._data = data
        sanity_check(
            (
                "increments", increments, sfuncs.check_ndarray, (),
                {"dim": 1, "shape": [(0, data.shape[0])]}, True,
            ),
        )
        self.increments = increments

    @property
    def increment_label(self) -> str:
        return getattr(self, "_increment_label", "")

    @property
    def fit_labels(self) -> Optional[str]:
        return getattr(self, "_fit_labels", None)

    @property
    def fit_units(self) -> Optional[str]:
        return getattr(self, "_fit_units", None)

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
        """View the data (FID or spectrum) of a paricular increment in the dataset
        with an interacitve matplotlib plot.

        Parameters
        ----------
        increment
            The increment to view. The data displayed is ``self.data[increment]``.

        domain
            Must be ``"freq"`` or ``"time"``.

        components
            Must be ``"real"``, ``"imag"``, or ``"both"``.

        freq_unit
            Must be ``"hz"`` or ``"ppm"``. If ``domain`` is ``"freq"``, this
            determines which unit to set chemical shifts to.
        """
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
        indices: Optional[Iterable[int]] = None,
        oscs: Optional[Iterable[int]] = None,
    ) -> Iterable[np.ndarray]:
        sanity_check(
            ("func", func, sfuncs.check_one_of, ("T1",)),
            self._indices_check(indices),
        )
        params = self.get_params(indices=indices)
        n_oscs = params[0].shape[0]
        sanity_check(
            self._oscs_check(oscs, n_oscs),
        )
        oscs = self._proc_oscs(oscs, n_oscs)

        integrals = self.integrals(indices, oscs)
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
                initial_trust_radius=0.1,
                max_trust_radius=0.1,
                output_mode=5,
            )
            results.append(nlp_result.x)
            errors.append(nlp_result.errors / np.sqrt(len(self.increments)))

        return results, errors

    def integrals(
        self,
        indices: Optional[Iterable[int]] = None,
        oscs: Optional[Iterable[int]] = None,
    ) -> Iterable[np.ndarray]:
        """Get the integrals associated with specified oscillators for a given result.

        Parameters
        ----------
        oscs
            Oscillators to get integrals for. By default (``None``) all oscillators
            are considered.

        index
            The index of the result to edit. Index ``0`` corresponds to the
            first result obtained using the estimator, ``1`` corresponds to the
            next, etc. You can also use ``-1`` for the most recent result,
            ``-2`` for the second most recent, etc. By default, the most
            recently obtained result will be considered.
        """
        sanity_check(
            self._indices_check(indices),
        )
        params = self.get_params(indices)
        n_oscs = params[0].shape[0]
        sanity_check(
            self._oscs_check(oscs, n_oscs),
        )
        oscs = self._proc_oscs(oscs, n_oscs)
        params = [p[oscs] for p in params]

        params = np.array(params).transpose(1, 0, 2)
        return [
            np.array(self.oscillator_integrals(osc, absolute=False)).real
            for osc in np.array(params)
        ]

    def plot_fit_single_oscillator(
        self,
        osc: int,
        index: int = -1,
        fit_increments: int = 100,
        fit_line_kwargs: Dict = None,
        scatter_kwargs: Dict = None,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Plot the fit across increments for a particular oscillator.

        osc
            Index for the oscillator of interest.

        index
            Index for the result of interest. By default (``-1``), the last acquired
            result is used.

        fit_increments
            The number of points in the fit line.

        fit_line_kwargs
            Keyword arguments for the fit line. Use this to specify features such
            as color, linewidth etc. All keys should be valid arguments for
            `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.

        scatter_kwargs
            Keyword arguments for the scatter plot. All keys should be valid arguments
            for
            `matplotlib.axes.Axes.scatter
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html>`_.

        kwargs
            Keyword arguments for
            `matplotlib.pyplot.figure
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_

        Returns
        -------
        fig
            The figure generated

        ax
            The axes assigned to the figure.
        """
        sanity_check(
            self._index_check(index),
            (
                "fit_increments", fit_increments, sfuncs.check_int, (),
                {"min_value": len(self.increments)},
            ),
        )
        n_oscs = self.get_params(indices=[index])[0].shape[0]
        sanity_check(
            ("osc", osc, sfuncs.check_int, (), {"min_value": 0, "max_value": n_oscs - 1}),  # noqa: E501
        )

        fit_line_kwargs = proc_kwargs_dict(
            fit_line_kwargs,
            default={"color": "#808080"},
        )
        scatter_kwargs = proc_kwargs_dict(
            scatter_kwargs,
            default={"color": "k", "marker": "x"},
        )

        params, errors = self.fit([index], [osc])
        params = params[0]
        errors = errors[0]

        fig, ax = plt.subplots(ncols=1, nrows=1, **kwargs)
        x = np.linspace(self.increments[0], self.increments[-1], fit_increments)
        ax.plot(x, self.model(*params, x), **fit_line_kwargs)
        integrals, = self.integrals([index], [osc])
        ax.scatter(self.increments, integrals, **scatter_kwargs)
        ax.set_xlabel(self.increment_label)
        ax.set_ylabel("$\\int$", rotation="horizontal")

        text = "\n".join(
            [
                f"$p_{{{i}}} = {para:.4g} \\pm {err:.4g}$U{i}"
                for i, (para, err) in enumerate(zip(params, errors), start=1)
            ]
        )
        if self.fit_labels is not None:
            for i, (flab, ulab) in enumerate(
                zip(self.fit_labels, self.fit_units),
                start=1,
            ):
                text = text.replace(f"p_{{{i}}}", flab)
                text = text.replace(f"U{i}", f" {ulab}" if ulab != "" else "")
        else:
            i = 0
            while True:
                if (to_rm := f"U{i}") in text:
                    text = text.replace(to_rm, "")
                else:
                    break

        ax.text(0.02, 0.98, text, ha="left", va="top", transform=ax.transAxes)

        return fig, ax

    def plot_fit_multi_oscillators(
        self,
        indices: Optional[Iterable[int]] = None,
        oscs: Optional[Iterable[int]] = None,
        fit_increments: int = 100,
        xaxis_unit: str = "ppm",
        xaxis_ticks: Optional[Union[Iterable[int], Iterable[Iterable[int, Iterable[float]]]]] = None,  # noqa: E501
        region_separation: float = 0.02,
        labels: bool = True,
        colors: Any = None,
        azim: float = 45.,
        elev: float = 45.,
        label_zshift: float = 0.,
        fit_line_kwargs: Optional[Dict] = None,
        scatter_kwargs: Optional[Dict] = None,
        label_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Create a 3D figure showing fits for multiple oscillators.

        The x-, y-, and z-axes comprise chemical shifts/oscillator indices,
        increment values, and integral values, respectively.

        Parameters
        ----------
        indices
            The indices of results to return. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            By default (``None``), all results are considered.

        oscs
            Oscillators to consider. By default (``None``) all oscillators are
            considered.

        fit_increments
            The number of points in the fit lines.

        xaxis_unit
            The unit of the x-axis. Should be one of:

            * ``"hz"`` or ``"ppm"``: Fits are plotted at the oscillators'
              frequecnies, either in Hz or ppm.
            * ``"osc_idx"``: Fits are evenly spaced about the x-axis, featuring
              at the oscillators' respective indices.

        xaxis_ticks
            Specifies custom x-axis ticks for each region, overwriting the default
            ticks. Keeping this ``None`` (default) will use default ticks. Otherwise:

            * If ``xaxis_unit`` if one of ``"hz"`` or ``"ppm"``, this should be
              of the form: ``[(i, (a, b, ...)), (j, (c, d, ...)), ...]``
              where ``i`` and ``j`` are ints indicating the region under consideration,
              and ``a``-``d`` are floats indicating the tick values.
            * If ``xaxis_unit`` is ``osc_idx``, this should be a list of ints,
              all of which correspond to a valid oscillator index.

        region_separation
            The extent by which adjacent regions are separated in the figure,
            in axes coordinates.

        labels
            If ``True``, the values extracted from the fits are written by each
            fit.

        colors
            Colors to give to the fits. Behaves in the same way as
            ``oscillator_colors`` in :py:meth:`~nmrespy.Estimator1D.plot_result`.

        elev
            Elevation angle for the plot view.

        azim
            Azimuthal angle for the plot view.

        label_zshift
            Vertical shift for the labels of T1 values (if the exist, see
            ``labels``). Given in axes coordinates.

        fit_line_kwargs
            Keyword arguments for the fit line. Use this to specify features such
            as color, linewidth etc. All keys should be valid arguments for
            `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.
            If ``"color"`` is included, it is ignored (colors are processed
            based on the ``colors`` argument.

        scatter_kwargs
            Keyword arguments for the scatter plot. All keys should be valid arguments
            for
            `matplotlib.axes.Axes.scatter
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html>`_.
            If ``"color"`` is included, it is ignored (colors are procecessed
            based on the ``colors`` argument.

        label_kwargs
            Keyword arguments for fit labels. All keys should be valid arguments for
            `matplotlib.text.Text
            <https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text>`_
            If ``"color"`` is included, it is ignored (colors are procecessed
            based on the ``colors`` argument. ``"horizontalalignment"``, ``"ha"``,
            ``"verticalalignment"``, and ``"va"`` are also ignored.

        kwargs
            Keyword arguments for
            `matplotlib.pyplot.figure
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_
        """
        sanity_check(
            self._indices_check(indices),
            (
                "fit_increments", fit_increments, sfuncs.check_int, (),
                {"min_value": len(self.increments)},
            ),
            ("xaxis_unit", xaxis_unit, sfuncs.check_one_of, ("osc_idx", "hz", "ppm")),
            ("labels", labels, sfuncs.check_bool),
            ("colors", colors, sfuncs.check_oscillator_colors, (), {}, True),
            ("elev", elev, sfuncs.check_float),
            ("azim", azim, sfuncs.check_float),
            ("label_zshift", label_zshift, sfuncs.check_float),
        )
        if xaxis_unit != "osc_idx":
            sanity_check(self._funit_check(xaxis_unit, "xaxis_unit"))

        scatter_kwargs = proc_kwargs_dict(
            scatter_kwargs,
            default={"lw": 0., "depthshade": False},
            to_pop=("color", "c", "zorder"),
        )
        fit_line_kwargs = proc_kwargs_dict(
            fit_line_kwargs,
            to_pop=("color", "zorder"),
        )
        label_kwargs = proc_kwargs_dict(
            label_kwargs,
            to_pop=("ha", "horizontalalignment", "va", "verticalalignment", "color", "zdir"),  # noqa: E501
        )

        indices = self._process_indices(indices)
        params = self.get_params(
            indices=indices,
            funit="ppm" if xaxis_unit == "ppm" else "hz",
        )
        n_oscs = params[0].shape[0]
        sanity_check(self._oscs_check(oscs, n_oscs))
        oscs = self._proc_oscs(oscs, n_oscs)
        params = [p[oscs] for p in params]

        integrals = self.integrals(indices, oscs)
        fit, errors = self.fit(indices, oscs)

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(projection="3d")

        colors = make_color_cycle(colors, len(oscs))
        zorder = n_oscs

        if xaxis_unit == "osc_idx":
            sanity_check(
                (
                    "xaxis_ticks", xaxis_ticks, sfuncs.check_list_with_elements_in,
                    (oscs,), {}, True,
                )
            )
            xs = oscs
            xlabel = "oscillator index"

        else:
            merge_indices, merge_regions = self._plot_regions(indices, xaxis_unit)
            sanity_check(
                (
                    "xaxis_ticks", xaxis_ticks, sfuncs.check_xaxis_ticks,
                    (merge_regions,), {}, True,
                ),
                (
                    "region_separation", region_separation, sfuncs.check_float,
                    (), {"min_value": 0., "max_value": 1 / (len(merge_regions) - 1)},
                ),
            )

            merge_region_spans = self._get_3d_xaxis_spans(
                merge_regions,
                region_separation,
            )

            xaxis_ticks, xaxis_ticklabels = self._get_3d_xticks_and_labels(
                xaxis_ticks,
                indices,
                xaxis_unit,
                merge_regions,
                merge_region_spans,
            )
            xs = self._transform_freq_to_xaxis(
                params[0][:, 2],
                merge_regions,
                merge_region_spans,
            )

            xlabel, = self._axis_freq_labels(xaxis_unit)

        for x, params, integs in zip(xs, fit, integrals):
            color = next(colors)
            x_scatter = np.full(self.increments.shape, x)
            y_scatter = self.increments
            z_scatter = integs
            ax.scatter(
                x_scatter,
                y_scatter,
                z_scatter,
                color=color,
                zorder=zorder,
                **scatter_kwargs,
            )

            x_fit = np.full(fit_increments, x)
            y_fit = np.linspace(self.increments[0], self.increments[-1], fit_increments)
            z_fit = self.model(*params, y_fit)
            ax.plot(
                x_fit,
                y_fit,
                z_fit,
                color=color,
                zorder=zorder,
                **fit_line_kwargs,
            )

            ax.text(
                x_fit[-1],
                y_fit[-1],
                z_fit[-1] + label_zshift,
                f"{params[-1]:.3g}",
                color=color,
                zdir="z",
                ha="center",
                **label_kwargs,
            )

            zorder -= 1

        # azim at 270 provies a face-on view of the spectra.
        ax.view_init(elev=elev, azim=270. + azim)

        ax.set_xlabel(xlabel)

        if xaxis_ticks is not None:
            ax.set_xticks(xaxis_ticks)
            ax.set_xticklabels(xaxis_ticklabels)

        # Configure y-axis
        ax.set_ylim(self.increments[0], self.increments[-1])
        ax.set_ylabel(self.increment_label)

        # Configure z-axis
        ax.set_zlabel("$\\int$")

        if xaxis_unit != "osc_idx":
            self._set_3d_axes_xspine_and_lims(ax, merge_region_spans)
        else:
            ax.set_xlim(n_oscs - 1, 0)

        return fig, ax

    def plot_result(
        self,
        indices: Optional[Iterable[int]] = None,
        xaxis_unit: str = "hz",
        xaxis_ticks: Optional[Iterable[float]] = None,
        region_separation: float = 0.02,
        oscillator_colors: Any = None,
        elev: float = 45.,
        azim: float = 45.,
        spec_line_kwargs: Optional[Dict] = None,
        osc_line_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """Generate a figure of the estimation result.

        A 3D plot is generated, showing the estimation result for each increment in
        the data.

        Paramters
        ---------
        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        xaxis_unit
            The unit to express chemical shifts in. Should be ``"hz"`` or ``"ppm"``.

        xaxis_ticks
            Specifies custom x-axis ticks for each region, overwriting the default
            ticks. Should be of the form: ``[(i, (a, b, ...)), (j, (c, d, ...)), ...]``
            where ``i`` and ``j`` are ints indicating the region under consideration,
            and ``a``-``d`` are floats indicating the tick values.

        region_separation
            The extent by which adjacent regions are separated in the figure,
            in axes coordinates.

        oscillator_colors
            Describes how to color individual oscillators. The following
            is a complete list of options:

            * If a `valid matplotlib colour
              <https://matplotlib.org/stable/tutorials/colors/colors.html>`_ is
              given, all oscillators will be given this color.
            * If a string corresponding to a `matplotlib colormap
              <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_
              is given, the oscillators will be consecutively shaded by linear
              increments of this colormap.
            * If an iterable object containing valid matplotlib colors is
              given, these colors will be cycled.
              For example, if ``oscillator_colors = ['r', 'g', 'b']``:

              + Oscillators 0, 3, 6, ... would be :red:`red (#FF0000)`
              + Oscillators 1, 4, 7, ... would be :green:`green (#008000)`
              + Oscillators 2, 5, 8, ... would be :blue:`blue (#0000FF)`

            * If ``None``, the default colouring method will be applied, which
              involves cycling through the following colors:

              + :oscblue:`#1063E0`
              + :oscorange:`#EB9310`
              + :oscgreen:`#2BB539`
              + :oscred:`#D4200C`

        elev
            Elevation angle for the plot view.

        azim
            Azimuthal angle for the plot view.

        spec_line_kwargs
            Keyword arguments for the spectrum lines. All keys should be valid
            arguments for
            `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.

        osc_line_kwargs
            Keyword arguments for the oscillator lines. All keys should be valid
            arguments for
            `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.
            If ``"color"`` is included, it is ignored (colors are processed based on
            the ``oscillator_colors`` argumnet.

        kwargs
            Keyword arguments for
            `matplotlib.pyplot.figure
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_
        """
        sanity_check(
            self._indices_check(indices),
            self._funit_check(xaxis_unit, "xaxis_unit"),
            (
                "oscillator_colors", oscillator_colors, sfuncs.check_oscillator_colors,
                (), {}, True,
            ),
            ("elev", elev, sfuncs.check_float),
            ("azim", azim, sfuncs.check_float),
        )

        spec_line_kwargs = proc_kwargs_dict(
            spec_line_kwargs,
            default={"linewidth": 0.8, "color": "k"},
        )
        osc_line_kwargs = proc_kwargs_dict(
            osc_line_kwargs,
            default={"linewidth": 0.5},
            to_pop=("color"),
        )

        indices = self._process_indices(indices)

        merge_indices, merge_regions = self._plot_regions(indices, xaxis_unit)

        sanity_check(
            (
                "xaxis_ticks", xaxis_ticks, sfuncs.check_xaxis_ticks,
                (merge_regions,), {}, True,
            ),
            (
                "region_separation", region_separation, sfuncs.check_float,
                (), {"min_value": 0., "max_value": 1 / (len(merge_regions) - 1)},
            ),
        )

        merge_region_spans = self._get_3d_xaxis_spans(
            merge_regions,
            region_separation,
        )

        xaxis_ticks, xaxis_ticklabels = self._get_3d_xticks_and_labels(
            xaxis_ticks,
            indices,
            xaxis_unit,
            merge_regions,
            merge_region_spans,
        )

        spectra = []
        for fid in self.data:
            fid[0] *= 0.5
            spectra.append(ne.sig.ft(fid).real)

        params = self.get_params(indices)

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(projection="3d")

        colorcycle = make_color_cycle(oscillator_colors, params[0].shape[0])
        for i, (spectrum, p, incr) in enumerate(
            zip(
                reversed(spectra), reversed(params), reversed(self.increments)
            )
        ):
            cc = copy.deepcopy(colorcycle)

            for (idx, region, span) in zip(
                merge_indices, merge_regions, merge_region_spans
            ):
                slice_ = slice(
                    *self.convert([region], f"{xaxis_unit}->idx")[0]
                )
                x = np.linspace(span[0], span[1], slice_.stop - slice_.start)
                y = np.full(x.shape, incr)
                spec = spectrum[slice_]
                ax.plot(x, y, spec, **spec_line_kwargs)
                for osc_params in p:
                    osc_params = np.expand_dims(osc_params, axis=0)
                    osc = self.make_fid(osc_params)
                    osc[0] *= 0.5
                    osc_spec = ne.sig.ft(osc).real[slice_]
                    ax.plot(x, y, osc_spec, color=next(cc), **osc_line_kwargs)

        ax.set_xticks(xaxis_ticks)
        ax.set_xticklabels(xaxis_ticklabels)
        ax.set_xlim(reversed(ax.get_xlim()))
        ax.set_xlabel(self._axis_freq_labels(xaxis_unit)[-1])
        ax.set_ylabel(self.increment_label)
        ax.set_zticks([])
        # azim at 270 provies a face-on view of the spectra.
        ax.view_init(elev=elev, azim=270. + azim)

        # x-axis spine with breaks.
        self._set_3d_axes_xspine_and_lims(ax, merge_region_spans)

        return fig, ax

    def _plot_regions(
        self,
        indices: Iterable[int],
        xaxis_unit: str,
    ) -> Iterable[Tuple[float, float]]:
        if isinstance(self._results[0], list):
            # Temporarily change `self._results` so `Estimator1D._plot_regions` works
            results_cp = copy.deepcopy(self._results)
            self._results = [res[0] for res in self._results]
            merge_indices, merge_regions = super()._plot_regions(indices, xaxis_unit)
            self._results = results_cp
        else:
            merge_indices, merge_regions = super()._plot_regions(indices, xaxis_unit)

        return merge_indices, merge_regions

    def _get_3d_xticks_and_labels(
        self,
        xaxis_ticks: Optional[Iterable[int, Iterable[float]]],
        indices: Iterable[int],
        xaxis_unit: str,
        merge_regions: Iterable[Tuple[float, float]],
        merge_region_spans: Iterable[Tuple[float, float]],
    ) -> Tuple[Iterable[float], Iterable[str]]:
        # Get the default xticks from `Estimator1D.plot_result`.
        # Filter any away that are outside the region
        default_xaxis_ticks = []
        _, _axs = self.plot_result_increment(indices=indices, region_unit=xaxis_unit)
        for i, (_ax, region) in enumerate(zip(_axs[0], merge_regions)):
            mn, mx = min(region), max(region)
            default_xaxis_ticks.append(
                (i, [x for x in _ax.get_xticks() if mn <= x <= mx])
            )

        flatten = lambda lst: [x for sublist in lst for x in sublist]  # noqa: E731
        if xaxis_ticks is None:
            xaxis_ticks = default_xaxis_ticks
        else:
            for i, _ in enumerate(merge_regions):
                found = any([x[0] == i for x in xaxis_ticks])
                if not found:
                    xaxis_ticks.append(default_xaxis_ticks[i])

        xaxis_ticks = flatten([x[1] for x in xaxis_ticks])

        # Scale xaxis ticks so the lie in the range [0-1]
        xaxis_ticks_scaled = self._transform_freq_to_xaxis(
            [ticks for ticks in xaxis_ticks],
            merge_regions,
            merge_region_spans,
        )

        # TODO Better formatting of tick labels
        xaxis_ticklabels = [f"{xtick:.3g}" for xtick in xaxis_ticks]

        return xaxis_ticks_scaled, xaxis_ticklabels

    @staticmethod
    def _transform_freq_to_xaxis(
        values: Iterable[float],
        merge_regions: Iterable[Tuple[float, float]],
        merge_region_spans: Iterable[Tuple[float, float]],
    ) -> Iterable[float]:
        transformed_values = []
        coefficients = []
        for region, span in zip(merge_regions, merge_region_spans):
            min_region, max_region = min(region), max(region)
            m1 = 1 / (max_region - min_region)
            c1 = -min_region * m1
            m2 = max(span) - min(span)
            c2 = min(span)
            coefficients.append([m1, c1, m2, c2])

        scale = lambda x, coeffs: (x * coeffs[0] + coeffs[1]) * coeffs[2] + coeffs[3]  # noqa: E501, E731

        for value in values:
            # Determine which region the value lies in
            for region, coeffs in zip(merge_regions, coefficients):
                if min(region) <= value <= max(region):
                    transformed_values.append(scale(value, coeffs))
                    break

        return transformed_values

    @staticmethod
    def _get_3d_xaxis_spans(
        merge_regions: Iterable[float, float],
        region_separation: float,
    ) -> Iterable[Tuple[float, float]]:
        n_regions = len(merge_regions)
        # x-axis will span [0, 1].
        # Figure out sections of this range to assign each region
        plot_width = 1 - (n_regions - 1) * region_separation
        merge_region_widths = [max(mr) - min(mr) for mr in merge_regions]
        merge_region_sum = sum(merge_region_widths)
        merge_region_widths = [
            mrw / merge_region_sum * plot_width
            for mrw in merge_region_widths
        ]
        merge_region_lefts = [
            1 - (i * region_separation) - sum(merge_region_widths[:i])
            for i in range(n_regions)
        ]
        merge_region_spans = [
            (mrl, mrl - mrw)
            for mrl, mrw in zip(merge_region_lefts, merge_region_widths)
        ]
        return merge_region_spans

    def _set_3d_axes_xspine_and_lims(
        self,
        ax: mpl.axes.Axes,
        merge_region_spans: Iterable[Tuple[float, float]],
    ) -> None:
        xspine = ax.w_xaxis.line
        xspine_kwargs = {
            "lw": xspine.get_linewidth(),
            "color": xspine.get_color(),
        }
        ax.w_xaxis.line.set_visible(False)
        curr_zlim = ax.get_zlim()
        spine_y = 2 * [self.increments[0]]
        spine_z = 2 * [curr_zlim[0]]
        for span in merge_region_spans:
            ax.plot(span, spine_y, spine_z, **xspine_kwargs)

        ax.set_xlim(1, 0)
        ax.set_ylim(self.increments[0], self.increments[-1])
        ax.set_zlim(curr_zlim)

    @logger
    def plot_result_increment(
        self,
        increment: int = 0,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, np.ndarray[mpl.axes.Axes]]:
        """Generate a figure of the estimation result for a particular increment.

        The figure created is of the same form as those generated by
        :py:meth:`~nmrespy.Estimator1D.plot_result`.

        Parameters
        ----------
        increment
            The increment to view. By default, the first increment in used (``0``).

        kwargs
            All kwargs are accepted by :py:meth:`nmrespy.Estimator1D.plot_result`.

        See Also
        --------
        :py:meth:`nmrespy.Estimator1D.plot_result`.
        """
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

        kwargs = proc_kwargs_dict(kwargs, default={"plot_model": False})
        fig, axs = super().plot_result(**kwargs)

        self._results = results_cp
        self._data = data_cp

        return fig, axs

    def get_params(
        self,
        indices: Optional[Iterable[int]] = None,
        merge: bool = True,
        funit: str = "hz",
        sort_by: str = "f-1",
    ) -> Optional[Union[Iterable[np.ndarray], np.ndarray]]:
        if isinstance(self._results[0], list):
            results_cp = copy.deepcopy(self._results)
            params = []
            for i, _ in enumerate(self.increments):
                self._results = [res[i] for res in results_cp]
                params.append(super().get_params(indices, merge, funit, sort_by))
                self._results = results_cp
        else:
            params = super().get_params(indices, merge, funit, sort_by)

        return params

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

    _increment_label = "$\\tau$ (s)"
    # Rendered in math mode
    _fit_labels = ["I_{\\infty}", "T_1"]
    # Rendered outside of math mode
    _fit_units = ["", "s"]

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
        super().__init__(data, expinfo, datapath, increments=delays)

    @classmethod
    def new_spinach(
        cls,
        shifts: Iterable[float],
        couplings: Optional[Iterable[Tuple(int, int, float)]],
        n_delays: int,
        max_delay: float,
        t1s: Union[Iterable[float], float],
        t2s: Union[Iterable[float], float],
        pts: int,
        sw: float,
        offset: float = 0.,
        sfo: float = 500.,
        nucleus: str = "1H",
        snr: Optional[float] = 20.,
    ) -> EstimatorInvRec:
        r"""Create a new instance from an inversion-recovery Spinach simulation.

        A data is acquired with linear increments of delay:

        .. math::

            \boldsymbol{\tau} =
                \left[
                    0,
                    \frac{\tau_{\text{max}}}{N_{\text{delays}} - 1},
                    \frac{2 \tau_{\text{max}}}{N_{\text{delays}} - 1},
                    \cdots,
                    \tau_{\text{max}}
                \right]

        with :math:`\tau_{\text{max}}` being ``max_delay`` and
        :math:`N_{\text{delays}}` being ``n_delays``.


        See :ref:`SPINACH_INSTALL` for requirments to use this method.

        Parameters
        ----------
        shifts
            A list of tuple of chemical shift values for each spin.

        couplings
            The scalar couplings present in the spin system. Given ``shifts`` is of
            length ``n``, couplings should be an iterable with entries of the form
            ``(i1, i2, coupling)``, where ``1 <= i1, i2 <= n`` are the indices of
            the two spins involved in the coupling, and ``coupling`` is the value
            of the scalar coupling in Hz. ``None`` will set all spins to be
            uncoupled.

        t1s
            The :math:`T_1` times for each spin. Should be either a list of floats
            with the same length as ``shifts``, or a float. If a float, all spins will
            be assigned the same :math:`T_1`. Note that :math:`T_1 = 1 / R_1`.

        t2s
            The :math:`T_2` times for each spin. See ``t1s`` for the required form.
            Note that :math:`T_2 = 1 / R_2`.

        n_delays
            The number of delays.

        max_delay
            The largest delay, in seconds.

        pts
            The number of points the signal comprises.

        sw
            The sweep width of the signal (Hz).

        offset
            The transmitter offset (Hz).

        sfo
            The transmitter frequency (MHz).

        nucleus
            The identity of the nucleus. Should be of the form ``"<mass><sym>"``
            where ``<mass>`` is the atomic mass and ``<sym>`` is the element symbol.
            Examples:

            * ``"1H"``
            * ``"13C"``
            * ``"195Pt"``

        snr
            The signal-to-noise ratio of the resulting signal, in decibels. ``None``
            produces a noiseless signal.
        """
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
        indices: Optional[Iterable[int]] = None,
        oscs: Optional[Iterable[int]] = None,
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
        result, errors = self._fit("T1", indices, oscs)
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


class FunctionFactoryInvRec(FunctionFactory):
    def __init__(self, theta: np.ndarray, *args) -> None:
        super().__init__(theta, EstimatorInvRec._obj_grad_hess, *args)
