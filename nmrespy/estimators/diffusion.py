# diffusion.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 09 May 2023 18:30:15 BST

from __future__ import annotations
import copy
from pathlib import Path
import re
from typing import Iterable, Optional, Tuple, Union

import numpy as np

import nmrespy as ne
from nmrespy._files import check_existent_path
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.estimators.seq_onedim import EstimatorSeq1D
from nmrespy.nlp._funcs import FunctionFactory


# gyromagnetic ratios for common nuclei, in MHz T-1
GAMMAS = {
    k: v / (2 * np.pi)
    for k, v in zip(
        ("1H", "13C", "15N", "19F"),
        (2.6752218744e8, 6.728284e7, -2.71261804e7, 2.518148e8),
    )
}


class FunctionFactoryDiffusion(FunctionFactory):
    def __init__(self, theta: np.ndarray, *args) -> None:
        super().__init__(theta, _EstimatorDiffusion._obj_grad_hess, *args)


# TODO: ABC
class _EstimatorDiffusion(EstimatorSeq1D):
    """Estimation class for the consideration of datasets acquired by diffusion NMR."""

    _increment_label = "$g$ (Gcm$^{-1}$)"
    _fit_labels = ["$a_{0}$", "$D$"]
    _fit_units = ["", "m$^2$s$^{-1}$"]
    function_factory = FunctionFactoryDiffusion

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        gradients: np.ndarray,
        small_delta: float,
        big_delta: float,
        shape_function: Optional[np.ndarray] = None,
        sigma: float = 1.,
        lambda_: float = 0.5,
        kappa: float = 0.33333333,
        gamma: Optional[float] = None,
        datapath: Optional[Path] = None,
    ) -> None:
        r"""
        Parameters
        ----------
        data
            The data associated with the estimator.

        expinfo
            Experiment information.

        gradients
            Gradients used in the experiment, in G cm⁻¹.

        small_delta
            The length of diffusion-encoding gradients (:math:`\delta`), in
            seconds

        big_delta
            The length of the diffusion delay (:math:`\Delta`), in seconds.

        shape_function
            An array of points denoting the profile of the diffusion-encoding gradient.
            If ``None``, the values supplied to ``sigma``, ``lambda_`` and ``kappa``
            will be used in the Stejskal-Tanner equation.

        sigma
            The shape factor of the diffusion-encoding gradient.

            .. math::

                \sigma = \int_0^1 s(\epsilon) \mathrm{d} \epsilon

            where :math:`s` is the shape function of the diffusion-encoding gradient,
            and :math:`\epsilon` is the level of gradient progress.

        lambda_

            .. math::

                \lambda = \frac{1}{\sigma}
                    \int_0^1 \left(
                    \int_0^{\epsilon^{\prime}} s(\epsilon^{\prime}) \mathrm{d} \epsilon
                    \right) \mathrm{d} \epsilon

        kappa

            .. math::

                \kappa = \frac{1}{\sigma^2}
                    \int_0^1 \left(
                    \int_0^{\epsilon^{\prime}} s(\epsilon^{\prime}) \mathrm{d} \epsilon
                    \right)^2 \mathrm{d} \epsilon

        gamma
            The gyromagnetic ratio of the nucleus, in 10⁶ s⁻¹ T⁻¹. If ``None``,
            an attempt will be made to extract this, based on ``expinfo.nuclei[0]``.
            **If** ``expinfo.nuclei[0]`` is ``None``, or a a string other than
            ``"1H"``, ``"13C"``, ``"15N"``, ``"19F"``, you will have to provide
            ``gamma`` manually.

        datapath
            The path to the directory containing the NMR data.
        """
        super().__init__(data, expinfo, gradients, datapath=datapath)

        sanity_check(
            (
                "small_delta", small_delta, sfuncs.check_float, (),
                {"greater_than_zero": True},
            ),
            (
                "big_delta", big_delta, sfuncs.check_float, (),
                {"greater_than_zero": True},
            ),
            (
                "shape_function", shape_function, sfuncs.check_ndarray, (),
                {"dim": 1}, True,
            ),
        )

        if shape_function is None:
            sanity_check(
                (
                    "sigma", sigma, sfuncs.check_float, (),
                    {"greater_than_zero": True, "max_value": 1.},
                ),
                (
                    "lambda_", lambda_, sfuncs.check_float, (),
                    {"greater_than_zero": True, "max_value": 1.},
                ),
                (
                    "kappa", kappa, sfuncs.check_float, (),
                    {"greater_than_zero": True, "max_value": 1.},
                ),
            )
        else:
            sigma, lambda_, kappa = self._process_shape_function(shape_function)

        if gamma is None:
            nuc, = self.nuclei
            sanity_check(
                (
                    "Trying to extract gamma based on self.nuclei[0]",
                    nuc, sfuncs.check_one_of, [x for x in GAMMAS.keys()],
                ),
            )
            self.gamma = GAMMAS[nuc]

        else:
            sanity_check(("gamma", gamma, sfuncs.check_float))
            self.gamma = gamma

        self.small_delta = small_delta
        self.big_delta = big_delta
        self.sigma = sigma
        self.lambda_ = lambda_
        self.kappa = kappa
        self.shape_function = shape_function

    @staticmethod
    def _new_bruker_pre(
        directory: Union[str, Path],
        increment_file: Optional[str] = None,
        gradient_file: str = "gpnam1",
        convdta: bool = True,
    ) -> Tuple[
        np.ndarray,
        ne.ExpInfo,
        np.ndarray,
        float,
        float,
        np.ndarray,
        Path,
    ]:
        data, expinfo, gradients, datapath = super()._new_bruker_pre(
            directory, increment_file, convdta,
        )
        acqus = expinfo.parameters["acqus"]
        small_delta = float(acqus["P"][30]) * 1.e-6  # μs -> s
        big_delta = float(acqus["D"][20])

        sanity_check(
            (
                "gradient_file", gradfile := datapath / gradient_file,
                check_existent_path,
            ),
        )

        # TODO: Error handling
        with open(gradfile, "r") as fh:
            gradfile_txt = fh.read()
        data_regex = re.compile(r"##XYDATA= \(X\+\+\(Y\..Y\)\)(.*)##END", re.DOTALL)
        data_str = re.search(data_regex, gradfile_txt).group(1).lstrip("\n").rstrip("\n")  # noqa: E501
        shape_function = np.array([float(x) for x in data_str.split("\n")])

        return (
            data,
            expinfo,
            gradients,
            small_delta,
            big_delta,
            shape_function,
            datapath,
        )

    @classmethod
    def new_from_parameters(
        cls,
        params: np.ndarray,
        diffusion_constants: Union[float, np.ndarray],
        gradients: np.ndarray,
        pts: int,
        sw: float,
        offset: float,
        big_delta: float = 1.,
        small_delta: float = 0.001,
        shape_function: Optional[np.ndarray] = None,
        sigma: float = 1.,
        lambda_: float = 0.5,
        kappa: float = 0.3333333,
        sfo: float = 500.,
        nucleus: str = "1H",
        snr: Optional[float] = 20.,
    ) -> _EstimatorDiffusion:
        """Generate an estimator instance with synthetic data created from a
        specification of oscillator parameters, gradient stengthsm and diffusion
        constants associated with each oscillator.

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

        diffusion_constants
            Specifies the diffusion constant associated with each oscillator.
            Should be the same length as ``params.shape[0]``.

        gradients
            Array of gradient strengths used for diffusion-encoding, in G cm⁻¹.

        pts
            The number of points the signal comprises.

        sw
            The sweep width of the signal (Hz).

        offset
            The transmitter offset (Hz).

        sfo
            The transmitter frequency (MHz).

        nucleus
            The identity of the nucleus. Should be one of ``"1H"``, ``"13C"``,
            ``"15N"``, ``"19F"``.

        snr
            The signal-to-noise ratio (dB). If ``None`` then no noise will be added
            to the FID.

        other
            For other arguments, see :py:class:`EstimatorDiffusionMonopolar`.

        Notes
        -----
        The default arguments for ``sigma``, ``lambda_`` and ``kappa`` correspond to
        the assumption that rectangular gradient pulses are used. Any gradient
        shape can be accommodated by specifying ``shape_funtion``.
        """
        sanity_check(
            ("params", params, sfuncs.check_parameter_array, (1,)),
            ("gradients", gradients, sfuncs.check_ndarray, (), {"dim": 1}),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}),
            ("sw", sw, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("offset", offset, sfuncs.check_float, (), {}, True),
            ("nucleus", nucleus, sfuncs.check_nucleus),
            ("sfo", sfo, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            ("nucleus", nucleus, sfuncs.check_one_of, ("1H", "13C", "15N", "19F")),
            ("snr", snr, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            (
                "small_delta", small_delta, sfuncs.check_float, (),
                {"greater_than_zero": True},
            ),
            (
                "big_delta", big_delta, sfuncs.check_float, (),
                {"greater_than_zero": True},
            ),
            (
                "shape_function", shape_function, sfuncs.check_ndarray, (),
                {"dim": 1}, True,
            ),
        )
        sanity_check(
            (
                "diffusion_constants", diffusion_constants, sfuncs.check_ndarray,
                (), {"dim": 1, "shape": [(0, params.shape[0])]},
            ),
        )

        if shape_function is None:
            sanity_check(
                (
                    "sigma", sigma, sfuncs.check_float, (),
                    {"greater_than_zero": True, "max_value": 1.},
                ),
                (
                    "lambda_", lambda_, sfuncs.check_float, (),
                    {"greater_than_zero": True, "max_value": 1.},
                ),
                (
                    "kappa", kappa, sfuncs.check_float, (),
                    {"greater_than_zero": True, "max_value": 1.},
                ),
            )
        else:
            sigma, lambda_, kappa = cls._process_shape_function(shape_function)

        expinfo = ne.ExpInfo(
            dim=1,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nucleus,
            default_pts=pts,
        )
        data = np.zeros((gradients.size, pts), dtype="complex128")

        self = cls(
            data=data,
            expinfo=expinfo,
            gradients=gradients,
            small_delta=small_delta,
            big_delta=big_delta,
            sigma=sigma,
            lambda_=lambda_,
            kappa=kappa,
        )

        factors = np.exp(-self.c * np.outer(gradients ** 2, diffusion_constants))
        for i, factor in enumerate(factors):
            p = copy.deepcopy(params)
            p[:, 0] *= factor
            data[i] += self.make_fid(p, snr=snr)

        return self

    @staticmethod
    def _process_shape_function(
        shape_function: np.ndarray,
    ) -> Tuple[float, float, float]:
        n = shape_function.size
        cum_dist = np.cumsum(shape_function) / n
        sigma = cum_dist[-1]
        half_first_point = 0.5 * cum_dist[0]
        lambda_ = (1 / (n * sigma)) * np.sum(cum_dist) - half_first_point
        kappa = (1 / (n * sigma ** 2)) * np.sum(cum_dist ** 2) - half_first_point
        return sigma, lambda_, kappa

    @classmethod
    def new_bruker(
        cls,
        directory: Union[str, Path],
        increment_file: Optional[str] = None,
        gradient_file: str = "gpnam1",
        convdta: bool = True,
    ) -> _EstimatorDiffusion:
        (
            data,
            expinfo,
            gradients,
            small_delta,
            big_delta,
            shape_function,
            datapath,
        ) = cls._new_bruker_pre(
            directory, increment_file, gradient_file, convdta,
        )

        return cls(
            data,
            expinfo,
            gradients,
            small_delta,
            big_delta,
            shape_function=shape_function,
            datapath=datapath,
        )

    @property
    def c(self) -> float:
        r"""Proportionality constant in Stejskal-Tanner, in units 10⁻⁴ m⁻² s.

        .. math::

            c = \gamma^2 \delta^2 \sigma^2 \Delta^{\prime}
        """
        return (
            1.e-4 *
            (self.gamma * self.small_delta * self.sigma) ** 2 *
            self.big_delta_prime
        )

    def get_x0(
        self,
        amplitudes: np.ndarray,
        increments: np.ndarray,
    ) -> np.ndarray:
        idx2, idx1 = np.argpartition(amplitudes, -2)[-2:]
        a1, a2 = amplitudes[idx1], amplitudes[idx2]
        g1, g2 = increments[idx1], increments[idx2]
        a0_init = a1 - (g1 * ((a2 - a1) / (g2 - g1)))
        cD_init = -(1 / (g1 ** 2)) * np.log(a1 / a0_init)
        return np.array([a0_init, cD_init])

    def fit(
        self,
        indices: Optional[Iterable[int]] = None,
        oscs: Optional[Iterable[int]] = None,
        neglect_increments: Optional[Iterable[int]] = None,
    ) -> Iterable[np.ndarray]:
        r"""Fit estimation result for the given oscillators across increments in
        order to predict the translational diffusion coefficient, :math:`D`.

        For the oscillators specified, the following function is fit:

        .. math::

            a \left(a_{0}, D \vert g\right) =
            a_{0} \exp\left(-c D g^2\right).

        where :math:`a` is the oscillator amplitude when the gradient is
        :math:`g`, and :math:`a_{0} = \lim_{g \rightarrow 0} a`.

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
            the first element corresponds to :math:`I_{0}`, and the second
            element corresponds to :math:`D`.
        """
        fits, errors = self._fit(indices, oscs, neglect_increments)
        c = self.c
        for fit, err in zip(fits, errors):
            fit[1] /= c
            err[1] /= c

        return fits, errors

    def model(
        self,
        a0: float,
        D: float,
        gradients: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Return the function

        .. math::

            \boldsymbol{a}\left(a_{0}, D, \boldsymbol{g} \right) =
                a_0 \exp\left(
                    - c D \boldsymbol{a}^2
                \right)

        Parameters
        ----------
        a0
            :math:`a_0`.

        D
            :math:`D`.

        gradients
            The gradients to consider (:math:`\boldsymbol{g}`). If ``None``,
            ``self.increments`` will be used.
        """
        sanity_check(
            ("a0", a0, sfuncs.check_float),
            ("D", D, sfuncs.check_float),
            ("gradients", gradients, sfuncs.check_ndarray, (), {"dim": 1}, True),
        )

        if gradients is None:
            gradients = self.increments

        return a0 * np.exp(-self.c * D * (gradients ** 2))

    @staticmethod
    def _obj_grad_hess(
        theta: np.ndarray,
        *args: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        r"""Objective, gradient and Hessian for fitting inversion recovery data.
        The model to be fit is given by

        .. math::

            a = a_0 \exp\left(-c D g^2\right).

        Parameters
        ----------
        theta
            Parameters of the model: :math:`a_0` and :math:`D`.

        args
            Comprises two items:

            * Amplitudes across each increment.
            * Gradients (:math:`g`).
            * c (:math:`c`).
        """
        a0, cD = theta
        amplitudes, gradients = args

        G_sq = gradients ** 2
        cDG_sq = cD * G_sq
        exp_minus_cDG_sq = np.exp(-cDG_sq)
        y_minus_x = amplitudes - a0 * exp_minus_cDG_sq
        n = gradients.size

        # Objective
        obj = np.sum(y_minus_x.T ** 2)

        # Grad
        d1 = np.zeros((n, 2))
        d1[:, 0] = exp_minus_cDG_sq
        d1[:, 1] = -I0 * G_sq * exp_minus_cDG_sq
        grad = -2 * y_minus_x.T @ d1

        # Hessian
        d2 = np.zeros((n, 2, 2))
        off_diag = -G_sq * exp_minus_cDG_sq
        d2[:, 0, 1] = off_diag
        d2[:, 1, 0] = off_diag
        d2[:, 1, 1] = (G_sq ** 2) * exp_minus_cDG_sq

        hess = -2 * (np.einsum("i,ijk->jk", y_minus_x, d2) - d1.T @ d1)

        return obj, grad, hess


class EstimatorDiffusionMonopolar(_EstimatorDiffusion):
    """Estimator for the consideration of diffusion NMR data acquired by an
    experiment with monopolar gradients.
    """

    @property
    def big_delta_prime(self) -> float:
        return self.big_delta + 2 * (self.kappa - self.lambda_) * self.small_delta


class EstimatorDiffusionBipolar(_EstimatorDiffusion):
    """Estimator for the consideration of diffusion NMR data acquired by an
    experiment with bipolar gradients.

    .. note::

        This class has equivalent behaviour to
        :py:class:`EstimatorDiffusionMonopolar`, except for :py:meth:`big_delta_prime`.
    """

    @property
    def big_delta_prime(self) -> float:
        r"""
        .. math::

            \Delta^{\prime} =
                \Delta + \frac{\delta \left(2 \kappa - 2 \lambda - 1\right)}{4} -
                \frac{1}{2}
        """
        return (
            self.big_delta +
            (self.small_delta * (2 * self.kappa - 2 * self.lambda_ - 1) / 4) -
            0.5
        )


class EstimatorDiffusionOneshot(_EstimatorDiffusion):
    """Estimator for the consideration of diffusion NMR data acquired by the "one-shot"
    experiment (`<10.1002/mrc.1107>`_).

    .. note::

        This class has equivalent behaviour to
        :py:class:`EstimatorDiffusionMonopolar`, except for it's init method,
        :py:meth:`new_from_parameters`, and :py:meth:`big_delta_prime`.
    """

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        gradients: np.ndarray,
        small_delta: float,
        big_delta: float,
        alpha: float,
        tau: float,
        shape_function: Optional[np.ndarray] = None,
        sigma: float = 1.,
        lambda_: float = 0.5,
        kappa: float = 0.33333333,
        gamma: Optional[float] = None,
        datapath: Optional[Path] = None,
    ) -> None:
        r"""
        Parameters
        ----------
        data
            The data associated with the estimator.

        expinfo
            Experiment information.

        gradients
            Gradients used in the experiment, in G cm⁻¹.

        small_delta
            The length of diffusion-encoding gradients (:math:`\delta`), in
            seconds

        big_delta
            The length of the diffusion delay (:math:`\Delta`), in seconds.

        alpha
            Unbalancing factor of bipolar gradient pairs

        tau
            Delay between the midpoints of the gradient pulses within a given
            diffusion-encoding period.

        shape_function
            An array of points denoting the profile of the diffusion-encoding gradient.
            If ``None``, the values supplied to ``sigma``, ``lambda_`` and ``kappa``
            will be used in the Stejskal-Tanner equation.

        sigma
            The shape factor of the diffusion-encoding gradient.

            .. math::

                \sigma = \int_0^1 s(\epsilon) \mathrm{d} \epsilon

            where :math:`s` is the shape function of the diffusion-encoding gradient,
            and :math:`\epsilon` is the level of gradient progress.

        lambda_

            .. math::

                \lambda = \frac{1}{\sigma}
                    \int_0^1 \left(
                    \int_0^{\epsilon^{\prime}} s(\epsilon^{\prime}) \mathrm{d} \epsilon
                    \right) \mathrm{d} \epsilon

        kappa

            .. math::

                \kappa = \frac{1}{\sigma^2}
                    \int_0^1 \left(
                    \int_0^{\epsilon^{\prime}} s(\epsilon^{\prime}) \mathrm{d} \epsilon
                    \right)^2 \mathrm{d} \epsilon

        gamma
            The gyromagnetic ratio of the nucleus, in 10⁶ s⁻¹ T⁻¹. If ``None``,
            an attempt will be made to extract this, based on ``expinfo.nuclei[0]``.
            **If** ``expinfo.nuclei[0]`` is ``None``, or a a string other than
            ``"1H"``, ``"13C"``, ``"15N"``, ``"19F"``, you will have to provide
            ``gamma`` manually.

        datapath
            The path to the directory containing the NMR data.
        """

        super().__init__(
            data,
            expinfo,
            gradients,
            small_delta,
            big_delta,
            shape_function,
            sigma,
            lambda_,
            kappa,
            gamma,
            datapath=datapath,
        )
        sanity_check(
            ("alpha", alpha, sfuncs.check_float),
            ("tau", tau, sfuncs.check_float),
        )
        self.alpha = alpha
        self.tau = tau

    @classmethod
    def new_bruker(
        cls,
        directory: Union[str, Path],
        increment_file: Optional[str] = None,
        gradient_file: str = "gpnam1",
        convdta: bool = True,
    ) -> EstimatorDiffusionOneshot:
        (
            data,
            expinfo,
            gradients,
            small_delta,
            big_delta,
            shape_function,
            datapath,
        ) = cls._new_bruker_pre(
            directory, increment_file, gradient_file, convdta,
        )
        small_delta *= 2.
        acqus = expinfo.parameters["acqus"]
        alpha = float(acqus["CNST"][14])
        tau = float(acqus["CNST"][17])

        return cls(
            data,
            expinfo,
            gradients,
            small_delta,
            big_delta,
            shape_function=shape_function,
            alpha=alpha,
            tau=tau,
            datapath=datapath,
        )

    @classmethod
    def new_from_parameters(
        cls,
        params: np.ndarray,
        diffusion_constants: Union[float, np.ndarray],
        gradients: np.ndarray,
        pts: int,
        sw: float,
        offset: float,
        sfo: float = 500.,
        nucleus: str = "1H",
        snr: Optional[float] = 20.,
        big_delta: float = 0.1,
        small_delta: float = 0.001,
        tau: float = 0.001,
        alpha: float = 0.2,
        shape_function: Optional[np.ndarray] = None,
        sigma: float = 1.,
        lambda_: float = 0.5,
        kappa: float = 0.3333333,
    ) -> EstimatorDiffusionOneshot:
        """Generate an estimator instance with synthetic data created from a
        specification of oscillator parameters, gradient stengthsm and diffusion
        constants associated with each oscillator.

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

        diffusion_constants
            Specifies the diffusion constant associated with each oscillator.
            Should be the same length as ``params.shape[0]``.

        gradients
            Array of gradient strengths used for diffusion-encoding, in G cm⁻¹.

        pts
            The number of points the signal comprises.

        sw
            The sweep width of the signal (Hz).

        offset
            The transmitter offset (Hz).

        sfo
            The transmitter frequency (MHz).

        nucleus
            The identity of the nucleus. Should be one of ``"1H"``, ``"13C"``,
            ``"15N"``, ``"19F"``.

        snr
            The signal-to-noise ratio (dB). If ``None`` then no noise will be added
            to the FID.

        other
            For other arguments, see :py:class:`EstimatorDiffusionOneshot`.

        Notes
        -----
        The default arguments for ``sigma``, ``lambda_`` and ``kappa`` correspond to
        the assumption that rectangular gradient pulses are used. Any gradient
        shape can be accommodated by specifying ``shape_funtion``.
        """
        monopolar_class = EstimatorDiffusionMonopolar.new_from_parameters(
            params=params,
            diffusion_constants=diffusion_constants,
            gradients=gradients,
            pts=pts,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nucleus=nucleus,
            snr=snr,
            big_delta=big_delta,
            small_delta=small_delta,
            shape_function=shape_function,
            sigma=sigma,
            lambda_=lambda_,
            kappa=kappa,
        )

        return cls(
            data=monopolar_class.data,
            expinfo=monopolar_class.expinfo,
            gradients=monopolar_class.increments,
            small_delta=monopolar_class.small_delta,
            big_delta=monopolar_class.big_delta,
            alpha=alpha,
            tau=tau,
            sigma=monopolar_class.sigma,
            lambda_=monopolar_class.lambda_,
            kappa=monopolar_class.kappa,
            gamma=monopolar_class.gamma,
        )

    @property
    def big_delta_prime(self) -> float:
        r"""
        .. math::

            \Delta^{\prime} =
                \Delta +
                \frac{\delta \left(\kappa - \lambda\right)\left(\alpha^2 + 1\right)}
                {2} +
                \frac{\left(\delta + 2 \tau\right) \left(\alpha^2 - 1\right)}{4}
        """
        return (
            self.big_delta +
            ((self.small_delta * (self.kappa - self.lambda_) * (self.alpha ** 2 + 1)) / 2) +  # noqa: E501
            (((self.small_delta + 2 * self.tau) * (self.alpha ** 2 - 1)) / 4)
        )
