# diffusion.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 10 Mar 2023 18:30:01 GMT

from __future__ import annotations
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


class _EstimatorDiffusion(EstimatorSeq1D):
    """Estimation class for the consideration of datasets acquired by monopolar
    diffusion NMR."""

    _increment_label = "$g$ (Gcm$^{-1}$)"
    _fit_labels = ["$I_{0}$", "$D$"]
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

        lambda_
            .. todo::
                Describe

        kappa
            .. todo::
                Describe

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

    @classmethod
    def _new_bruker_pre(
        cls,
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
        integrals: np.ndarray,
        increments: np.ndarray,
    ) -> np.ndarray:
        idx2, idx1 = np.argpartition(integrals, -2)[-2:]
        I1, I2 = integrals[idx1], integrals[idx2]
        g1, g2 = increments[idx1], increments[idx2]
        I0_init = I1 - (g1 * ((I2 - I1) / (g2 - g1)))
        cD_init = -(1 / (g1 ** 2)) * np.log(I1 / I0_init)
        return np.array([I0_init, cD_init])

    def fit(
        self,
        indices: Optional[Iterable[int]] = None,
        oscs: Optional[Iterable[int]] = None,
        neglect_increments: Optional[Iterable[int]] = None,
    ) -> Iterable[np.ndarray]:
        r"""Fit estimation result for the given oscillators across increments in
        order to predict the translational diffusion coefficient, :math:`D`.

        For the oscillators specified, the integrals of the oscilators' peaks are
        determined at each increment, and the following function is fit:

        .. math::

            I \left(I_{0}, D, g\right) =
            I_{0} \exp\left(-c D g^2\right).

        where :math:`I` is the peak integral when the gradient is :math:`G`, and
        :math:`I_{0} = \lim_{G \rightarrow 0} I`.

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
        I0: float,
        D: float,
        gradients: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Return the function

        .. math::

            \boldsymbol{I}\left(I_{0}, D, \boldsymbol{G} \right) =
                I_0 \exp\left(
                    - c D \boldsymbol{G}^2
                \right)

        Parameters
        ----------
        I0
            :math:`I_0`.

        D
            :math:`D`.

        gradients
            The gradients to consider (:math:`\boldsymbol{G}`). If ``None``,
            ``self.increments`` will be used.
        """
        sanity_check(
            ("I0", I0, sfuncs.check_float),
            ("D", D, sfuncs.check_float),
            ("gradients", gradients, sfuncs.check_ndarray, (), {"dim": 1}, True),
        )

        if gradients is None:
            gradients = self.increments

        return I0 * np.exp(-self.c * D * (gradients ** 2))

    @staticmethod
    def _obj_grad_hess(
        theta: np.ndarray,
        *args: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        r"""Objective, gradient and Hessian for fitting inversion recovery data.
        The model to be fit is given by

        .. math::

            I = I_0 \exp\left(-c D G^2\right).

        Parameters
        ----------
        theta
            Parameters of the model: :math:`I_0` and :math:`D`.

        args
            Comprises two items:

            * integrals across each increment.
            * gradients (:math:`G`).
            * c (:math:`c`).
        """
        I0, cD = theta
        integrals, gradients = args

        G_sq = gradients ** 2
        cDG_sq = cD * G_sq
        exp_minus_cDG_sq = np.exp(-cDG_sq)
        y_minus_x = integrals - I0 * exp_minus_cDG_sq
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

    @property
    def big_delta_prime(self) -> float:
        return self.big_delta + 2 * (self.kappa - self.lambda_) * self.small_delta


class EstimatorDiffusionBipolar(_EstimatorDiffusion):

    @property
    def big_delta_prime(self) -> float:
        return (
            self.big_delta +
            (self.small_delta * (2 * self.kappa - 2 * self.lambda_ - 1) / 4) -
            0.5
        )


class EstimatorDiffusionOneshot(_EstimatorDiffusion):

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

    @property
    def big_delta_prime(self) -> float:
        return (
            self.big_delta +
            ((self.small_delta * (self.kappa - self.lambda_) * (self.alpha ** 2 + 1)) / 2) +  # noqa: E501
            (((self.small_delta + 2 * self.tau) * (self.alpha ** 2 - 1)) / 4)
        )
