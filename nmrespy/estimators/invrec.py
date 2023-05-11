# invrec.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 10 May 2023 00:19:07 BST

from __future__ import annotations
import copy
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
import numpy as np

import nmrespy as ne
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.estimators.seq_onedim import EstimatorSeq1D
from nmrespy.nlp._funcs import FunctionFactory


class FunctionFactoryInvRec(FunctionFactory):
    def __init__(self, theta: np.ndarray, *args) -> None:
        super().__init__(theta, EstimatorInvRec._obj_grad_hess, *args)


class EstimatorInvRec(EstimatorSeq1D):
    """Estimation class for the consideration of datasets acquired by an inversion
    recovery experiment, for the purpose of determining longitudinal relaxation
    times (:math:`T_1`)."""

    _increment_label = "$\\tau$ (s)"
    _fit_labels = ["$a_{\\infty}$", "$T_1$"]
    _fit_units = ["", "s"]
    function_factory = FunctionFactoryInvRec

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
        super().__init__(data, expinfo, delays, datapath)

    @classmethod
    def new_from_parameters(
        cls,
        params: np.ndarray,
        t1s: Union[Iterable[float], float],
        delays: np.ndarray,
        pts: int,
        sw: float,
        offset: float,
        sfo: float = 500.,
        nucleus: str = "1H",
        snr: Optional[float] = 20.,
    ) -> EstimatorInvRec:
        """Generate an estimator instance with sythetic data created from an
        array of oscillator parameters.

        Parameters
        ----------
        params
            Parameter array with the following structure:

              .. code:: python

                 params = numpy.array([
                    [a_1, φ_1, f_1, η_1],
                    [a_2, φ_2, f_2, η_2],
                    ...,
                    [a_m, φ_m, f_m, η_m],
                 ])

        t1s
            The longitudinal relaxation times associated with each oscillator in the
            parameter array. Should be a list of floats with ``len(t1s) ==
            params.shape[0]``.

        increments
           List of inversion recovery delays to generate the data from.

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
            Examples: ``"1H"``, ``"13C"``, ``"195Pt"``

        snr
            The signal-to-noise ratio (dB). If ``None`` then no noise will be added
            to the FID.
        """
        sanity_check(
            ("params", params, sfuncs.check_parameter_array, (1,)),
            ("delays", delays, sfuncs.check_ndarray, (), {"dim": 1}, True),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}),
            ("sw", sw, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("offset", offset, sfuncs.check_float, (), {}, True),
            ("nucleus", nucleus, sfuncs.check_nucleus),
            ("sfo", sfo, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            ("snr", snr, sfuncs.check_float, (), {"greater_than_zero": True}, True),
        )
        noscs = params.shape[0]
        sanity_check(
            (
                "t1s", t1s, sfuncs.check_float_list, (),
                {"length": noscs, "must_be_positive": True},
            ),
        )

        expinfo = ne.ExpInfo(
            dim=1,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nucleus,
            default_pts=pts,
        )

        a0s = params[:, 0]
        t1s = np.array(t1s)
        data = np.zeros((delays.size, pts), dtype="complex")
        for i, tau in enumerate(delays):
            print(i)
            amps = a0s * (1 - 2 * np.exp(-tau / t1s))
            p = copy.deepcopy(params)
            p[:, 0] = amps
            data[i] = expinfo.make_fid(p, snr=snr)
            print(data[i])

        return cls(data, expinfo, delays)

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

    @classmethod
    def new_bruker(
        cls,
        directory: Union[str, Path],
        delay_file: Optional[str] = None,
        convdta: bool = True,
    ) -> EstimatorInvRec:
        data, expinfo, delays, datapath = cls._new_bruker_pre(
            directory, delay_file, convdta,
        )
        return cls(data, expinfo, delays, datapath)

    @staticmethod
    def _proc_mpm_signal(mpm_signal: np.ndarray) -> np.ndarray:
        return ne.sig.ift(-1 * ne.sig.ft(mpm_signal))

    @staticmethod
    def _proc_nlp_signal(nlp_signal: np.ndarray) -> np.ndarray:
        nlp_signal[0] = ne.sig.ift(-1 * ne.sig.ft(nlp_signal[0]))
        return nlp_signal

    @staticmethod
    def _proc_first_result(result: np.ndarray) -> np.ndarray:
        result[:, 0] *= -1
        return result

    def get_x0(
        self,
        amplitudes: np.ndarray,
        increments: np.ndarray,
    ) -> np.ndarray:
        # TODO: Could probably improve...
        return np.array([-np.amin(amplitudes), 1.])

    def fit(
        self,
        indices: Optional[Iterable[int]] = None,
        oscs: Optional[Iterable[int]] = None,
        neglect_increments: Optional[Iterable[int]] = None,
    ) -> Iterable[np.ndarray]:
        r"""Fit estimation result for the given oscillators across increments in
        order to predict the longitudinal relaxtation time, :math:`T_1`.

        For the oscillators specified, the following function is fit:

        .. math::

            a \left(a_{\infty}, T_1, \tau\right) =
            a_{\infty} \left[ 1 - 2 \exp\left( \frac{\tau}{T_1} \right) \right].

        where :math:`a` is the oscillator amplitude when the delay is :math:`\tau`, and
        :math:`a_{\infty} = \lim_{\tau \rightarrow \infty} a`.

        Parameters
        ----------
        oscs
            The indices of the oscillators to considier. If ``None``, all oscillators
            are consdiered.

        index
            The result index. By default, the last result acquired is considered.

        neglect_increments
            Increments of the dataset to neglect. Default, all increments are included
            in the fit.

        Returns
        -------
        Iterable[np.ndarray]
            Iterable (list) of numpy arrays of shape ``(2,)``. For each array,
            the first element corresponds to :math:`I_{\infty}`, and the second
            element corresponds to :math:`T_1`.
        """
        result, errors = self._fit(indices, oscs, neglect_increments)
        return result, errors

    def model(
        self,
        a_infty: float,
        T1: float,
        delays: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Return the function

        .. math::

            \boldsymbol{a}\left(a_{\infty}, T_1, \boldsymbol{\tau} \right) =
                a_{\infty} \left[
                    1 - 2 \exp\left(- \frac{\boldsymbol{\tau}}{T_1} \right)
                \right]

        Parameters
        ----------
        a_infty
            :math:`I_{\infty}`.

        T1
            :math:`T_1`.

        delays
            The delays to consider (:math:`\boldsymbol{\tau}`). If ``None``,
            ``self.increments`` will be used.
        """
        sanity_check(
            ("a_infty", a_infty, sfuncs.check_float),
            ("T1", T1, sfuncs.check_float),
            ("delays", delays, sfuncs.check_ndarray, (), {"dim": 1}, True),
        )

        if delays is None:
            delays = self.increments
        return a_infty * (1 - 2 * np.exp(-delays / T1))

    @staticmethod
    def _obj_grad_hess(
        theta: np.ndarray,
        *args: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        r"""Objective, gradient and Hessian for fitting inversion recovery data.
        The model to be fit is given by

        .. math::

            a = a_0 \left[ 1 - 2 \exp\left( \frac{\tau}{T_1} \right) \right].

        Parameters
        ----------
        theta
            Parameters of the model: :math:`a_0` and :math:`T_1`.

        args
            Comprises two items:

            * integrals across each increment.
            * delays (:math:`\tau`).
        """
        a0, T1 = theta
        amplitudes, taus = args

        t_over_T1 = taus / T1
        t_over_T1_sq = taus / (T1 ** 2)
        t_over_T1_cb = taus / (T1 ** 3)
        exp_t_over_T1 = np.exp(-t_over_T1)
        y_minus_x = amplitudes - a0 * (1 - 2 * exp_t_over_T1)
        n = taus.size

        # Objective
        obj = np.sum(y_minus_x.T ** 2)

        # Grad
        d1 = np.zeros((n, 2))
        d1[:, 0] = 1 - 2 * exp_t_over_T1
        d1[:, 1] = -2 * a0 * t_over_T1_sq * exp_t_over_T1
        grad = -2 * y_minus_x.T @ d1

        # Hessian
        d2 = np.zeros((n, 2, 2))
        off_diag = -2 * t_over_T1_sq * exp_t_over_T1
        d2[:, 0, 1] = off_diag
        d2[:, 1, 0] = off_diag
        d2[:, 1, 1] = 2 * a0 * t_over_T1_cb * exp_t_over_T1 * (2 - t_over_T1)

        hess = -2 * (np.einsum("i,ijk->jk", y_minus_x, d2) - d1.T @ d1)

        return obj, grad, hess
