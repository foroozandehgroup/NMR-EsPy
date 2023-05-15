# twodim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sat 13 May 2023 19:40:01 BST

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import nmrespy as ne
from nmrespy.load import load_bruker
from nmrespy._colors import RED, GRE, END, USE_COLORAMA
from nmrespy._files import check_existent_dir
from nmrespy._misc import boxed_text, proc_kwargs_dict
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)

from nmrespy.estimators import logger
from nmrespy.estimators import Estimator, Result
from nmrespy.estimators.jres import ContourApp

if USE_COLORAMA:
    import colorama
    colorama.init()


class Estimator2D(Estimator):
    """Estimator class for 2D data comprising a pair of States signals."""

    default_mpm_trim = (64, 64)
    default_nlp_trim = (128, 128)
    default_max_iterations_exact_hessian = 40
    default_max_iterations_gn_hessian = 100

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        modulation: str,
        datapath: Optional[Path] = None,
    ) -> None:
        super().__init__(data, expinfo, datapath)
        sanity_check(("modulation", modulation, sfuncs.check_one_of, ("amp", "phase")))
        self.modulation = modulation

    @classmethod
    def new_bruker(
        cls,
        directory: Union[str, Path],
        convdta: bool = True,
    ) -> Estimator2D:
        sanity_check(
            ("directory", directory, check_existent_dir),
            ("convdta", convdta, sfuncs.check_bool),
        )
        directory = Path(directory).expanduser()
        data, expinfo = load_bruker(directory)
        pts_2 = data.shape[1] // 2
        data_shape = (2, data.shape[0], pts_2)

        if data.ndim != 2:
            raise ValueError(f"{RED}Data dimension should be 2.{END}")
        data_proc = np.zeros(data_shape, dtype="complex128")

        print(data)
        data_proc[0] = data[:, :pts_2]
        data_proc[1] = data[:, pts_2:]

        if convdta:
            grpdly = expinfo.parameters["acqus"]["GRPDLY"]
            data = ne.sig.convdta(data, grpdly)

        return cls(data_proc, expinfo, "phase", datapath=directory)

    @classmethod
    def new_spinach_hsqc(
        cls,
        shifts: Iterable[float],
        couplings: Optional[Iterable[Tuple(int, int, float)]],
        isotopes: Iterable[str],
        pts: Tuple[int, int],
        sw: Tuple[float, float],
        offset: Tuple[float, float],
        field: float = 11.74,
        nuclei: Tuple[str, str] = ("13C", "1H"),
        snr: Optional[float] = 30.,
        lb: float = 6.91,
    ) -> Estimator2D:
        r"""Create a new instance from a Spinach HSQC simulation.

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

        isotopes
            A list of the identities of each nucleus. This should be a list with the
            same length as ``shifts``, with all elements being one of the
            elements in ``nuclei``.

        pts
            The number of points the signal comprises.

        sw
            The sweep width of the signal (Hz).

        offset
            The transmitter offset (Hz).

        field
            The magnetic field strength (T).

        nucleus
            The identity of the nuclei targeted in the pulse sequence.

        snr
            The signal-to-noise ratio of the resulting signal, in decibels. ``None``
            produces a noiseless signal.

        lb
            Line broadening (exponential damping) to apply to the signal.
            The first point will be unaffected by damping, and the final point will
            be multiplied by ``np.exp(-lb)``. The default results in the final
            point being decreased in value by a factor of roughly 1000.
        """

        sanity_check(
            ("shifts", shifts, sfuncs.check_float_list),
            ("pts", pts, sfuncs.check_int_list, (), {"length": 2, "min_value": 1}),
            ("sw", sw, sfuncs.check_float_list, (), {"length": 2, "must_be_positive": True}),  # noqa: E501
            ("offset", offset, sfuncs.check_float_list, (), {"length": 2}),
            ("field", field, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("nuclei", nuclei, sfuncs.check_nucleus_list, (), {"length": 2}),
            ("snr", snr, sfuncs.check_float, (), {}, True),
            ("lb", lb, sfuncs.check_float, (), {"greater_than_zero": True})
        )
        nspins = len(shifts)
        sanity_check(
            ("couplings", couplings, sfuncs.check_spinach_couplings, (nspins,), {}, True),  # noqa: E501
            ("isotopes", isotopes, sfuncs.check_nucleus_list, (), {"length": nspins}),
        )

        if couplings is None:
            couplings = []

        fid, sfo = cls._run_spinach(
            "hsqc_sim", shifts, couplings, isotopes, field, pts, sw, offset, nuclei,
            to_int=[4], to_double=[5, 6],
        )
        fid_pos = np.array(fid["pos"]).T
        fid_neg = np.array(fid["neg"]).T
        fid = np.zeros((2, *fid_pos.shape), dtype="complex128")
        fid[0] = fid_pos
        fid[1] = fid_neg
        sfo = tuple([x[0] for x in sfo])

        if snr is not None:
            fid = ne.sig.add_noise(fid, snr)

        fid = ne.sig.exp_apodisation(fid, lb, axes=(1, 2))

        expinfo = ne.ExpInfo(
            dim=2,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nuclei,
            default_pts=fid.shape[1:],
        )

        return cls(fid, expinfo, "phase")

    @property
    def spectrum(self) -> np.ndarray:
        if self.modulation == "amp":
            return ne.sig.proc_amp_modulated(self.data).real
        elif self.modulation == "phase":
            return ne.sig.proc_phase_modulated(self.data)[0]

    @logger
    def estimate(
        self,
        region: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        noise_region: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        region_unit: str = "hz",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        mode: str = "apfd",
        amp_thold: Optional[float] = None,
        phase_variance: bool = True,
        cut_ratio: Optional[float] = 1.1,
        mpm_trim: Optional[Tuple[int, int]] = None,
        nlp_trim: Optional[Tuple[int, int]] = None,
        hessian: str = "gauss-newton",
        max_iterations: Optional[int] = None,
        negative_amps: str = "remove",
        output_mode: Optional[int] = 10,
        save_trajectory: bool = False,
        epsilon: float = 1.0e-8,
        eta: float = 0.15,
        initial_trust_radius: float = 1.0,
        max_trust_radius: float = 4.0,
        check_neg_amps_every: int = 10,
        _log: bool = True,
    ) -> None:
        r"""Estimate a specified region of the signal.

        The basic steps that this method carries out are:

        * (Optional, but highly advised) Generate a frequency-filtered "sub-FID"
          corresponding to a specified region of interest.
        * (Optional) Generate an initial guess using the Minimum Description
          Length (MDL) [#]_ and Matrix Pencil Method (MPM) [#]_ [#]_ [#]_ [#]_
        * Apply numerical optimisation to determine a final estimate of the signal
          parameters. The optimisation routine employed is the Trust Newton Conjugate
          Gradient (NCG) algorithm ([#]_ , Algorithm 7.2).

        Parameters
        ----------
        region
            The frequency range of interest in each dimension.
            Should be of the form ``[[left_f1, right_f1], [left_f2, right_f2]]``
            where ``left_f<n>`` and ``right_f<n>`` are the left and right
            bounds of the region of interest for dimension ``<n>``, in Hz or ppm
            (see ``region_unit``). If ``None``, the full signal will be
            considered, though for sufficently large and complex signals it is
            probable that poor and slow performance will be realised.

        noise_region
            If ``region`` is not ``None``, this must be of the form
            ``[[left_f1, right_f1], [left_f2, right_f2]]``
            too. This should specify a region where no noticeable signals
            reside, i.e. only noise exists.

        region_unit
            One of ``"hz"`` or ``"ppm"`` Specifies the units that ``region``
            and ``noise_region`` have been given as.

        initial_guess
            * If ``None``, an initial guess will be generated using the MMEMPM
              with the MDL applied on the first direct-dimension FID being used
              to estimate the number of oscillators present.
            * If an int, the MMEMPM will be used to compute the initial guess with
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
            ``signal[:mpm_trim[0], :mpm_trim[1]]``.

        nlp_trim
            Specifies the maximal size allowed for the filtered signal when undergoing
            nonlinear programming. By default (``None``), no trimming is applied to
            the signal. If an int, and the filtered signal has a size greater than
            ``nlp_trim``, this signal will be set as
            ``signal[:nlp_trim[0], :nlp_trim[1]]``.

        max_iterations
            A value specifiying the number of iterations the routine may run
            through before it is terminated. If ``None``, a default number
            of maximum iterations is set, based on the value of ``hessian``.

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

        check_neg_amps_every
            For every iteration that is a multiple of this, negative amplitudes
            will be checked for and dealt with if found.

        _log
            Ignore this!

        References
        ----------
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
                "mpm_trim", mpm_trim, sfuncs.check_int_list, (),
                {"length": 2, "min_value": 1}, True,
            ),
            (
                "nlp_trim", nlp_trim, sfuncs.check_int_list, (),
                {"length": 2, "min_value": 1}, True,
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
            (
                "region", region, sfuncs.check_region,
                (self.sw(unit=region_unit), self.offset(unit=region_unit)), {}, True,
            ),
            (
                "noise_region", noise_region, sfuncs.check_region,
                (self.sw(unit=region_unit), self.offset(unit=region_unit)), {}, True,
            ),
            (
                "max_trust_radius", max_trust_radius, sfuncs.check_float, (),
                {"min_value": initial_trust_radius},
            ),
            (
                "check_neg_amps_every", check_neg_amps_every, sfuncs.check_int, (),
                {"min_value": 1, "max_value": max_iterations},
            ),
        )

        if output_mode != 0:
            if region is None:
                txt = "ESTIMATING ENTIRE SIGNAL"
            else:
                unit = region_unit.replace("h", "H")
                txt = (
                    f"ESTIMATING REGION: "
                    f"{region[0][0]} - {region[0][1]} {unit} (F1) "
                    f"{region[1][0]} - {region[1][1]} {unit} (F2)"
                )
            print(boxed_text(txt, GRE))

        (
            region,
            noise_region,
            mpm_expinfo,
            nlp_expinfo,
            mpm_signal,
            nlp_signal,
        ) = self._filter_signal(
            region, noise_region, region_unit, mpm_trim, nlp_trim, cut_ratio,
        )

        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            oscillators = initial_guess if isinstance(initial_guess, int) else 0
            x0 = ne.mpm.MatrixPencil(
                mpm_expinfo,
                mpm_signal,
                oscillators=oscillators,
                output_mode=isinstance(output_mode, int),
            ).get_params()

            if x0 is None:
                self._results.append(
                    Result(
                        np.array([[]]),
                        np.array([[]]),
                        region,
                        noise_region,
                        self.sfo,
                    )
                )
                return

        if max_iterations is None:
            if hessian == "exact":
                max_iterations = self.default_max_iterations_exact_hessian
            elif hessian == "gauss-newton":
                max_iterations = self.default_max_iterations_gn_hessian

        optimiser_kwargs = {
            "phase_variance": phase_variance,
            "hessian": hessian,
            "mode": mode,
            "amp_thold": amp_thold,
            "max_iterations": max_iterations,
            "negative_amps": negative_amps,
            "output_mode": output_mode,
            "save_trajectory": save_trajectory,
            "epsilon": epsilon,
            "eta": eta,
            "initial_trust_radius": initial_trust_radius,
            "max_trust_radius": max_trust_radius,
            "check_neg_amps_every": check_neg_amps_every,
        }

        result = ne.nlp.nonlinear_programming(
            nlp_expinfo,
            nlp_signal,
            x0,
            **optimiser_kwargs,
        )

        self._results.append(
            Result(
                result.x,
                result.errors,
                region,
                noise_region,
                self.sfo,
                result.trajectory,
            )
        )

    def get_data_1d(self, dim: int = 1) -> np.ndarray:
        if self.modulation == "amp":
            data_2d = self.data[0] + 1j * self.self.data[1]
        else:
            data_2d = self.data[0]

        if dim == 0:
            return data_2d[:, 0]
        else:
            return data_2d[0]

    def get_spectrum_1d(self, dim: int = 1) -> np.ndarray:
        return ne.sig.ft(self.get_data_1d(dim))

    def get_1d_expinfo(self, dim: int = 1) -> ne.ExpInfo:
        sanity_check(("dim", dim, sfuncs.check_one_of, (0, 1)))
        return ne.ExpInfo(
            dim=1,
            sw=self.sw()[dim],
            offset=self.offset()[dim],
            sfo=self.sfo[dim],
            nuclei=self.nuclei[dim],
            default_pts=self.data.shape[dim + 1],
        )

    def view_data_1d(
        self,
        dim: int = 1,
        **kwargs,
    ) -> None:
        sanity_check(
            ("dim", dim, sfuncs.check_one_of, (0, 1)),
        )
        estimator_1d = self.get_estimator_1d(dim)
        estimator_1d.view_data(**kwargs)

    def view_data(
        self,
        domain: str = "freq",
    ) -> None:
        spectrum = self.spectrum
        app = ContourApp(spectrum, self.expinfo)
        app.mainloop()

    def get_estimator_1d(self, dim: int = 1) -> ne.Estimator1D:
        data_1d = self.get_data_1d(dim)
        expinfo_1d = self.get_1d_expinfo(dim)
        estimator_1d = ne.Estimator1D(data_1d, expinfo_1d)

        if not self._results:
            return estimator_1d

        results_1d = []
        freq_idx = 2 + dim
        damp_idx = 4 + dim
        for result_2d in self.get_results():
            params_2d, errors_2d = result_2d.get_params(), result_2d.get_errors()
            params_1d, errors_1d = [
                np.zeros((params_2d.shape[0], 4), dtype="float64") for _ in range(2)
            ]
            params_1d[:, :2], errors_1d[:, :2] = params_2d[:, :2], errors_2d[:, :2]
            params_1d[:, 2], errors_1d[:, 2] = params_2d[:, freq_idx], errors_2d[:, freq_idx]  # noqa: E501
            params_1d[:, 3], errors_1d[:, 3] = params_2d[:, damp_idx], errors_2d[:, damp_idx]  # noqa: E501
            region_1d = (result_2d.get_region()[dim],)
            noise_region_1d = (result_2d.get_noise_region()[dim],)
            result_1d = Result(
                params_1d,
                errors_1d,
                region_1d,
                noise_region_1d,
                sfo=(self.sfo[dim],),
            )
            results_1d.append(result_1d)
        estimator_1d._results = results_1d

        return estimator_1d

    def plot_result(
        self,
        index: int = -1,
        region_unit: str = "hz",
        contour_base: Optional[float] = None,
        contour_nlevels: Optional[int] = None,
        contour_factor: Optional[float] = None,
        oscillator_colors: Any = None,
        contour_kwargs: Optional[Dict] = None,
        scatter_kwargs: Optional[Dict] = None,
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        sanity_check(
            self._index_check(index),
            self._funit_check(region_unit, "region_unit"),
            (
                "contour_base", contour_base, sfuncs.check_float, (),
                {"min_value": 0.}, True,
            ),
            (
                "contour_nlevels", contour_nlevels, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            (
                "contour_factor", contour_factor, sfuncs.check_float, (),
                {"min_value": 1.}, True,
            ),
            (
                "oscillator_colors", oscillator_colors, sfuncs.check_oscillator_colors,
                (), {}, True,
            ),
        )

        contour_kwargs = proc_kwargs_dict(
            contour_kwargs,
            default={"linewidths": 0.4, "colors": "k"},
            to_pop=["levels"],
        )
        scatter_kwargs = proc_kwargs_dict(
            scatter_kwargs,
            default={"marker": "o", "s": 8},
            to_pop=["color"],
        )

        index, = self._process_indices([index])
        result = self.get_results(indices=[index])[0]
        region = result.get_region(unit=region_unit)
        params = result.get_params(funit=region_unit)
        n_oscs = params.shape[0]
        full_shifts_2d_y, full_shifts_2d_x = self.get_shifts(unit=region_unit)
        full_spectrum = self.spectrum

        conv = f"{region_unit}->idx"
        slice_ = [slice(r[0], r[1], None) for r in self.expinfo.convert(region, conv)]
        spectrum = full_spectrum[slice_[0], slice_[1]]
        shifts_2d_x = full_shifts_2d_x[slice_[0], slice_[1]]
        shifts_2d_y = full_shifts_2d_y[slice_[0], slice_[1]]
        freqs = []
        for osc in params:
            freqs.append((osc[2], osc[3]))

        if all(
            [isinstance(x, (float, int))
             for x in (contour_base, contour_nlevels, contour_factor)]
        ):
            contour_levels = [
                contour_base * contour_factor ** i
                for i in range(contour_nlevels)
            ]
        else:
            contour_levels = None

        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.contour(
            shifts_2d_x,
            shifts_2d_y,
            spectrum,
            levels=contour_levels,
            **contour_kwargs,
        )

        colors = ne.plot.make_color_cycle(oscillator_colors, n_oscs)
        for (f1, f2) in freqs:
            ax.scatter(
                x=f2,
                y=f1,
                color=next(colors),
                zorder=100,
                **scatter_kwargs,
            )

        ylabel, xlabel = self._axis_freq_labels(unit=region_unit)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xlim(shifts_2d_x[0, 0], shifts_2d_x[0, -1])
        ax.set_ylim(shifts_2d_y[0, 0], shifts_2d_y[-1, 0])

        return fig, ax


    def plot_result_1d(
        self,
        dim: int = 1,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, np.ndarray[mpl.axes.Axes]]:
        """Generate a figure of the estimation result, looking at one dimension.

        Parameters
        ----------
        dim
            The dimension to consider. ``0`` corresponds to the indirect dimension,
            ``1`` corresponds to the direct dimension.

        kwargs
            For a description of all other arguments, see
            :py:meth:`Estimator1D.plot_result`.
        """
        sanity_check(
            ("dim", dim, sfuncs.check_one_of, (0, 1)),
        )
        estimator_1d = self.get_estimator_1d(dim)
        fig, axs, _ = estimator_1d.plot_result(**kwargs)
        return fig, axs

    def _filter_signal(
        self,
        region: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
        noise_region: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
        region_unit: str,
        mpm_trim: Optional[Tuple[int, int]],
        nlp_trim: Optional[Tuple[int, int]],
        cut_ratio: Optional[float],
    ) -> None:
        # This method is uused by `Estimator1D` and `Estimator2DJ`.
        # It is overwritten by `EstimatorSeq1D`.
        if region is None:
            region_unit = "hz"
            region = self._full_region
            noise_region = None
            mpm_signal = nlp_signal = self.data
            mpm_expinfo = nlp_expinfo = self.expinfo

        else:
            filter_ = ne.freqfilter.Filter(
                self.data,
                self.expinfo,
                region,
                noise_region,
                region_unit=region_unit,
                twodim_dtype=self.modulation,
            )

            region = filter_.get_region()
            noise_region = filter_.get_noise_region()
            mpm_signal, mpm_expinfo = filter_.get_filtered_fid(cut_ratio=cut_ratio)
            nlp_signal, nlp_expinfo = filter_.get_filtered_fid(cut_ratio=None)

        mpm_trim = self._get_trim("mpm", mpm_trim, mpm_signal.shape)
        nlp_trim = self._get_trim("nlp", nlp_trim, nlp_signal.shape)

        return (
            region,
            noise_region,
            mpm_expinfo,
            nlp_expinfo,
            mpm_signal[:mpm_trim[0], :mpm_trim[1]],
            nlp_signal[:nlp_trim[0], :nlp_trim[1]],
        )

    @property
    def _full_region(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return tuple(
            self.convert(
                [(0, x - 1) for x in self.data.shape[1:]],
                "idx->hz",
            )
        )

    def _get_trim(
        self,
        type_: str,
        trim: Optional[Tuple[int, int]],
        shape: Tuple[float, float],
    ) -> Tuple[int, int]:
        default = (
            self.default_mpm_trim if type_ == "mpm" else
            self.default_nlp_trim
        )
        if trim is None:
            return [min(def_, s) for def_, s in zip(default, shape)]
        else:
            return [min(t, s) for t, s in zip(trim, shape)]
