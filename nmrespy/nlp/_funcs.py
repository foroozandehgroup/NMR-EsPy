# _funcs.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 14 Mar 2022 14:39:24 GMT

"""Definitions of fidelities, gradients, and Hessians."""

from typing import Dict, Tuple
import numpy as np


args_type = Tuple[np.ndarray, np.ndarray, int, np.ndarray, list, bool]


class ObjGrad:
    """Object which computes and memoises the fidelity and gradient.

    Parameters
    ----------
    fun
        Callable which computes the objective and gradient.
    """

    def __init__(self, fun: callable):
        self.fun = fun
        self.theta = None
        self.obj = None
        self.grad = None

    def _compute_if_needed(self, theta, *args):
        """Determine if quantities need to be computed.

        Determines whether parameters array has changed, or the objective has not
        been computed yet. If either of these are so, obj and grad are
        computed.
        """
        if not np.all(theta == self.theta) or self.obj is None:
            self.obj, self.grad = self.fun(theta, *args)

    def objective(self, theta, *args) -> float:
        """Return the objective. Compute if necessary.

        Parameters
        ----------
        theta
            Parameter array.

        args
            Extra arguments to feed to the function which computes the objective
            and gradient.

        Returns
        -------
        objective: float
        """
        self._compute_if_needed(theta, *args)
        return self.obj

    def gradient(self, theta, *args) -> np.ndarray:
        """Return the gradient. Compute if necessary.

        Parameters
        ----------
        theta
            Parameter array.

        args
            Extra arguments to feed to the function which computes the objective
            and gradient.

        Returns
        -------
        gradient: numpy.ndarray
        """
        self._compute_if_needed(theta, *args)
        return self.grad


class ObjGradHess(ObjGrad):
    """Object which computes and memoises the fidelity, gradient and Hessian.

    Parameters
    ----------
    fun
        Callable which computes the objective and gradient.
    """

    def __init__(self, fun: callable):
        super().__init__(fun)
        self.hess = None

    def _compute_if_needed(self, theta, *args):
        """Determine if quantities need to be computed.

        Determines whether parameters array has changed, or the objective has not
        been computed yet. If either of these are so, obj, grad and hess
        computed.
        """
        if not np.all(theta == self.theta) or self.obj is None:
            self.obj, self.grad, self.hess = self.fun(theta, *args)

    def hessian(self, theta, *args) -> np.ndarray:
        """Return the hessian. Compute if necessary.

        Parameters
        ----------
        theta
            Parameter array.

        args
            Extra arguments to feed to the function which computes the objective,
            gradient and hessian.

        Returns
        -------
        hessian: numpy.ndarray
        """
        self._compute_if_needed(theta, *args)
        return self.hess


def first_derivatives_1d(
    model_per_osc: np.ndarray, idx: list, deriv_functions: Dict[str, callable],
) -> np.ndarray:
    """Compute all partial derivatives of a 1D model signal.

    Parameters
    ----------
    model_per_osc
        ``(N, M)`` shape array containing ``N``-length vectors corresponding to each
        oscillator's contribution to the model.

    idx
        List denoting the parameter types which are to be optimised.

        * ``0`` - amplitude
        * ``1`` - phase
        * ``2`` - frequency
        * ``3`` - damping factor

    deriv_functions
        Dictionary containing callables to compute partial derivatives.

    Returns
    -------
    d1: np.ndarray
        ``(N, p * M)`` shape array containing all first partial derivatives,
         where ``p = len(idx)``.

    See Also
    --------
    :py:func:`obj_grad_1d`
    :py:func:`obj_grad_true_hess_1d`
    :py:func:`obj_grad_gauss_newton_hess_1d`
    """
    p = len(idx)
    m = model_per_osc.shape[1]
    d1 = np.zeros((model_per_osc.shape[0], m * p), dtype="complex")

    ad = deriv_functions["a"]
    pd = deriv_functions["p"]
    fd = deriv_functions["f"]
    dd = deriv_functions["d"]

    # At points denoted ---x y z---, the parameters to be
    # optimised have been established, and the parameters are stated:
    # a: amplitudes
    # φ: phases
    # f: frequencies
    # η: damping factors
    if 0 in idx:
        d1[:, :m] = ad(model_per_osc)  # a
        if 1 in idx:
            d1[:, m : 2 * m] = pd(model_per_osc)  # φ
            if 2 in idx:
                d1[:, 2 * m : 3 * m] = fd(model_per_osc)  # f
                if 3 in idx:
                    # ---a φ f η--- (all parameters)
                    d1[:, 3 * m :] = dd(model_per_osc)  # η
                # ---a φ f---
            elif 3 in idx:
                # ---a φ η---
                d1[:, 2 * m :] = dd(model_per_osc)  # η
            # ---a φ---
        elif 2 in idx:
            d1[:, m : 2 * m] = fd(model_per_osc)  # f
            if 3 in idx:
                # ---a f η---
                d1[:, 2 * m :] = dd(model_per_osc)  # η
            # ---a f---
        elif 3 in idx:
            # ---a η---
            d1[:, m:] = dd(model_per_osc)  # η
        # ---a only---
    elif 1 in idx:
        d1[:, :m] = pd(model_per_osc)  # φ
        if 2 in idx:
            d1[:, m : 2 * m] = fd(model_per_osc)  # f
            if 3 in idx:
                # ---φ f η---
                d1[:, 2 * m :] = dd(model_per_osc)  # η
            # ---φ f---
        elif 3 in idx:
            # ---φ η---
            d1[:, m:] = dd(model_per_osc)  # η
        # ---φ only---
    elif 2 in idx:
        d1[:, :m] = fd(model_per_osc)  # f
        if 3 in idx:
            # ---f η---
            d1[:, m:] = dd(model_per_osc)  # η
        # ---f only---
    else:
        # ---η only---
        d1[:, :] = dd(model_per_osc)  # η

    return d1


def second_derivatives_1d(
    d1: np.ndarray, idx: list, deriv_functions: Dict[str, callable],
) -> np.ndarray:
    """Compute all partial derivatives of a 2D model signal.

    Parameters
    ----------
    d1
        ``(N, p * M)`` shape array containing ``N``-length vectors corresponding to
        all first derivatives of the model, with ``p = len(idx)``.

    idx
        List denoting the parameter types which are to be optimised.

        * ``0`` - amplitude
        * ``1`` - phase
        * ``2`` - frequency
        * ``3`` - damping factor

    deriv_functions
        Dictionary containing callables to compute partial derivatives.

    Returns
    -------
    d2: np.ndarray
        ``(N, (p * (p + 1) * M) // 2)`` shape array containing all non-trivially zero
        second partial derivatives.

    See Also
    --------
    :py:func:`obj_grad_true_hess_1d`
    :py:func:`obj_grad_gauss_newton_hess_1d`
    """
    p = len(idx)
    n, m = d1.shape[0], d1.shape[1] // p
    # int((p * (p + 1)) / 2) --> p-th triangle number
    # gives array of:
    # --> nx10m if all oscs are to be optimised
    # --> nx6m if one type is passive
    # --> nx3m if two types are passive
    # --> nxm if three types are passive
    d2 = np.zeros((n, m * p * (p + 1) // 2), dtype="complex")

    ad = deriv_functions["a"]
    pd = deriv_functions["p"]
    fd = deriv_functions["f"]
    dd = deriv_functions["d"]

    # At points denoted ---x y z---, the parameters to be
    # optimised have been established, and the parameters are stated:
    # a: amplitudes
    # φ: phases
    # f: frequencies
    # η: damping factors
    if 0 in idx:
        # (a-a is trivially zero)
        if 1 in idx:
            d2[:, m : 2 * m] = pd(d1[:, :m])  # a-φ
            if 2 in idx:
                d2[:, 2 * m : 3 * m] = ad(d1[:, 2 * m : 3 * m])  # a-f
                if 3 in idx:
                    # ---a φ f η---
                    d2[:, 3 * m : 4 * m] = ad(d1[:, 3 * m : 4 * m])  # a-η
                    d2[:, 4 * m : 5 * m] = pd(d1[:, m : 2 * m])  # φ-φ
                    d2[:, 5 * m : 6 * m] = pd(d1[:, 2 * m : 3 * m])  # φ-f
                    d2[:, 6 * m : 7 * m] = pd(d1[:, 3 * m : 4 * m])  # φ-η
                    d2[:, 7 * m : 8 * m] = fd(d1[:, 2 * m : 3 * m])  # f-f
                    d2[:, 8 * m : 9 * m] = dd(d1[:, 2 * m : 3 * m])  # f-η
                    d2[:, 9 * m :] = dd(d1[:, 3 * m :])  # η-η
                else:
                    # ---a φ f---
                    d2[:, 3 * m : 4 * m] = pd(d1[:, m : 2 * m])  # φ-φ
                    d2[:, 4 * m : 5 * m] = pd(d1[:, 2 * m :])  # φ-f
                    d2[:, 5 * m :] = fd(d1[:, 2 * m :])  # f-f
            elif 3 in idx:
                # ---a φ η---
                d2[:, 2 * m : 3 * m] = ad(d1[:, 2 * m :])  # a-η
                d2[:, 3 * m : 4 * m] = pd(d1[:, m : 2 * m])  # φ-φ
                d2[:, 4 * m : 5 * m] = pd(d1[:, 2 * m :])  # φ-η
                d2[:, 5 * m :] = dd(d1[:, 2 * m :])  # η-η
            else:
                # ---a φ---
                d2[:, 2 * m :] = pd(d1[:, m:])  # φ-φ
        elif 2 in idx:
            d2[:, m : 2 * m] = ad(d1[:, m : 2 * m])  # a-f
            if 3 in idx:
                # ---a f η---
                d2[:, 2 * m : 3 * m] = ad(d1[:, 2 * m :])  # a-η
                d2[:, 3 * m : 4 * m] = fd(d1[:, m : 2 * m])  # f-f
                d2[:, 4 * m : 5 * m] = dd(d1[:, m : 2 * m])  # f-η
                d2[:, 5 * m :] = dd(d1[:, 2 * m :])  # η-η
            else:
                # ---a f---
                d2[:, 2 * m :] = fd(d1[:, m:])  # f-f
        elif 3 in idx:
            # ---a η---
            d2[:, m :] = dd(d1[:, :m])  # a-η
            d2[:, 2 * m :] = dd(d1[:, m:])  # η-η
        # ---a only---
    elif 1 in idx:
        d2[:, :m] = pd(d1[:, :m])  # φ-φ
        if 2 in idx:
            d2[:, m : 2 * m] = pd(d1[:, m : 2 * m])  # φ-f
            if 3 in idx:
                # ---φ f η---
                d2[:, 2 * m : 3 * m] = pd(d1[:, 2 * m :])  # φ-η
                d2[:, 3 * m : 4 * m] = fd(d1[:, m : 2 * m])  # f-f
                d2[:, 4 * m : 5 * m] = dd(d1[:, m : 2 * m])  # f-η
                d2[:, 5 * m :] = dd(d1[:, 2 * m :])  # η-η
            else:
                # ---φ f---
                d2[:, 2 * m :] = fd(d1[:, m:])  # f-f
        elif 3 in idx:
            # ---φ η---
            d2[:, m : 2 * m] = pd(d1[:, m:])  # φ-η
            d2[:, 2 * m :] = dd(d1[:, m:])  # η-η
        # ---φ only---
    elif 2 in idx:
        d2[:, :m] = fd(d1[:, :m])  # f-f
        if 3 in idx:
            # ---f η---
            d2[:, m : 2 * m] = dd(d1[:, :m])  # f-η
            d2[:, 2 * m :] = dd(d1[:, m:])  # η-η
        # ---f only---
    else:
        # ---η only---
        d2 = dd(d1)  # η-η

    return d2


def obj_1d(active: np.ndarray, *args: args_type) -> float:
    """Compute the objective for 1D data.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` - frequencies.
              + ``3`` - damping factors.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    obj: float
        Value of the objective.
    """
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    Z = np.exp(np.outer(tp[0], 2j * np.pi * theta[2 * m : 3 * m] - theta[3 * m :]))
    alpha = (theta[:m] * np.exp(1j * theta[m : 2 * m]))
    model = Z @ alpha
    diff = data - model

    obj = np.real(diff.conj().T @ diff)

    if phasevar:
        # If 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in idx else 0
        phases = theta[i * m : (i + 1) * m]
        mu = np.einsum("i->", phases) / m
        obj += np.einsum("i->", (phases - mu) ** 2) / (np.pi * m)

    return obj


def obj_grad_1d(active: np.ndarray, *args: args_type) -> Tuple[float, np.ndarray]:
    """Compute the objective and gradient for 1D data.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` - frequencies.
              + ``3`` - damping factors.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    obj: float
        Value of the objective.

    grad: numpy.ndarray
        Gradient of the objective.
    """
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    # N x M array comprising M N-length vectors.
    # Each vector is the model produced by a single oscillator.
    # Broadcasting of signal pole matrix and complex amplitude vector
    model_per_osc = np.exp(
        np.outer(tp[0], (1j * 2 * np.pi * theta[2 * m : 3 * m] - theta[3 * m :]))
    ) * (theta[:m] * np.exp(1j * theta[m : 2 * m]))

    deriv_functions = {
        "a": lambda x: x / theta[:m],
        "p": lambda x: 1j * x,
        "f": lambda x: np.einsum("ij,i->ij", x, (1j * 2 * np.pi * tp[0])),
        "d": lambda x: np.einsum("ij,i->ij", x, -tp[0])
    }

    # ∂x/∂θᵢ
    d1 = first_derivatives_1d(model_per_osc, idx, deriv_functions)

    model = np.einsum("ij->i", model_per_osc)
    diff = data - model

    # --- ℱ(θ) ---
    obj = np.real(diff.conj().T @ diff)
    # --- ∇ℱ(θ) ---
    grad = -2 * np.real(d1.conj().T @ diff)

    if phasevar:
        # If 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in idx else 0
        phases = theta[i * m : (i + 1) * m]
        mu = np.einsum("i->", phases) / m

        # Var(φ)
        obj += np.einsum("i->", (phases - mu) ** 2) / (np.pi * m)
        # ∂Var(φ)/∂φᵢ
        grad[i * m : (i + 1) * m] += 0.8 * ((2 / m) * (phases - mu)) / np.pi
    return obj, grad


def hess_1d(active: np.ndarray, *args: args_type) -> np.ndarray:
    """Hessian of cost function for 1D data.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` - frequencies.
              + ``3`` - damping factors.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    hess
        Hessian of cost function.
    """
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    # N x M array comprising M N-length vectors.
    # Each vector is the model produced by a single oscillator.
    # Broadcasting of signal pole matrix and complex amplitude vector
    model_per_osc = np.exp(
        np.outer(tp[0], (1j * 2 * np.pi * theta[2 * m : 3 * m] - theta[3 * m :]))
    ) * (theta[:m] * np.exp(1j * theta[m : 2 * m]))

    deriv_functions = {
        "a": lambda x : x / theta[:m],
        "p": lambda x: 1j * x,
        "f": lambda x: np.einsum("ij,i->ij", x, (1j * 2 * np.pi * tp[0])),
        "d": lambda x: np.einsum("ij,i->ij", x, -tp[0])
    }
    # ∂x/∂θᵢ
    d1 = first_derivatives_1d(model_per_osc, idx, deriv_functions)
    # ∂²x/∂θᵢ∂θⱼ values that are not trivially zero.
    # If θᵢ and θⱼ correspond to different oscillators, these second derivatives
    # will always be zero.
    d2 = second_derivatives_1d(d1, idx, deriv_functions)

    model = np.einsum("ij->i", model_per_osc)
    diff = data - model

    # Non-trivially zero values of the second Hessian term:
    # (y - x)† ∂²x/∂θᵢ∂θⱼ
    diagonals = -2 * np.real(np.einsum("ji,j->i", d2.conj(), diff))
    # Determine indices of elements withn non-trivially zero second Hessian term
    # (specfically, those in upper triangle).
    p = len(idx)
    hess_shape = (p * m, p * m)
    hess = np.zeros(hess_shape)
    diag_indices = _generate_diagonal_indices(p, m)
    hess[diag_indices] = diagonals

    main_diagonals = _diagonal_indices(hess_shape[0], k=0)
    # Division by 2 to ensure elements on main diagonal aren't doubled
    # after transposition
    hess[main_diagonals] = hess[main_diagonals] / 2

    # Transpose (hessian is symmetric)
    hess += hess.T

    # Add component containing first derivatives
    hess += 2 * np.real(np.einsum("ki,kj->ij", d1.conj(), d1))

    if phasevar:
        # If 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in idx else 0
        hess[i * m : (i + 1) * m, i * m : (i + 1) * m] -= 2 / (m ** 2 * np.pi)
        hess[
            main_diagonals[0][i * m : (i + 1) * m],
            main_diagonals[1][i * m : (i + 1) * m],
        ] += 2 / (np.pi * m)

    return hess


def obj_finite_diff_grad_hess_1d(
    active: np.ndarray, h: float, *args: args_type
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the objective, gradient and hessian for 1D data.

    The gradient and Hessian are computed using fintie difference.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    h
        Finite difference parameter

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` - frequencies.
              + ``3`` - damping factors.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    obj: float
        Value of the objective.

    grad: numpy.ndarray
        Finite difference gradient of the objective.

    hess: numpy.ndarray
        Finite difference Hessian of the objective.
    """
    return _finite_diff(active, 1, h, *args)


def obj_grad_gauss_newton_hess_1d(
    active: np.ndarray, *args: args_type
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the objective, gradient and hessian for 1D data.

    The Hessian is computed using the Gauss-Newton technique.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` - frequencies.
              + ``3`` - damping factors.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    obj: float
        Value of the objective.

    grad: numpy.ndarray
        Gradient of the objective.

    hess: numpy.ndarray
        Gauss-Newton Hessian of the objective.
    """
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    # N x M array comprising M N-length vectors.
    # Each vector is the model produced by a single oscillator.
    # Broadcasting of signal pole matrix and complex amplitude vector
    model_per_osc = np.exp(
        np.outer(tp[0], (1j * 2 * np.pi * theta[2 * m : 3 * m] - theta[3 * m :]))
    ) * (theta[:m] * np.exp(1j * theta[m : 2 * m]))

    deriv_functions = {
        "a": lambda x: x / theta[:m],
        "p": lambda x: 1j * x,
        "f": lambda x: np.einsum("ij,i->ij", x, (1j * 2 * np.pi * tp[0])),
        "d": lambda x: np.einsum("ij,i->ij", x, -tp[0])
    }

    # Jacobian: -∂x/∂θᵢ
    jac = first_derivatives_1d(model_per_osc, idx, deriv_functions)

    model = np.einsum("ij->i", model_per_osc)
    diff = data - model

    # --- ℱ(θ) ---
    obj = np.real(diff.conj().T @ diff)
    # --- ∇ℱ(θ) ---
    grad = -2 * np.real(diff.conj().T @ jac)
    # --- ∇²ℱ(θ) ---
    hess = 2 * np.real(jac.conj().T @ jac)

    if phasevar:
        # If 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in idx else 0
        phases = theta[i * m : (i + 1) * m]
        mu = np.einsum("i->", phases) / m
        # Var(φ)
        obj += np.einsum("i->", (phases - mu) ** 2) / (np.pi * m)
        # ∂Var(φ)/∂φᵢ
        grad[i * m : (i + 1) * m] += 0.8 * ((2 / m) * (phases - mu)) / np.pi
        # ∂²Var(φ)/∂φᵢ∂φⱼ
        hess[i * m : (i + 1) * m, i * m : (i + 1) * m] -= 2 / (m ** 2 * np.pi)
        main_diagonals = _diagonal_indices(hess.shape[0], k=0)
        hess[
            main_diagonals[0][i * m : (i + 1) * m],
            main_diagonals[1][i * m : (i + 1) * m],
        ] += 2 / (np.pi * m)

    return obj, grad, hess


def obj_grad_true_hess_1d(active: np.ndarray, *args):
    """Compute the objective, gradient and hessian for 1D data.

    The Hessian is computed exactly.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` - frequencies.
              + ``3`` - damping factors.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    obj: float
        Value of the objective.

    grad: numpy.ndarray
        Gradient of the objective.

    hess: numpy.ndarray
        Exact Hessian of the objective.
    """
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    # N x M array comprising M N-length vectors.
    # Each vector is the model produced by a single oscillator.
    # Broadcasting of signal pole matrix and complex amplitude vector
    model_per_osc = np.exp(
        np.outer(tp[0], (1j * 2 * np.pi * theta[2 * m : 3 * m] - theta[3 * m :]))
    ) * (theta[:m] * np.exp(1j * theta[m : 2 * m]))

    deriv_functions = {
        "a": lambda x: x / theta[:m],
        "p": lambda x: 1j * x,
        "f": lambda x: np.einsum("ij,i->ij", x, (1j * 2 * np.pi * tp[0])),
        "d": lambda x: np.einsum("ij,i->ij", x, -tp[0])
    }
    # ∂x/∂θᵢ
    d1 = first_derivatives_1d(model_per_osc, idx, deriv_functions)
    # ∂²x/∂θᵢ∂θⱼ values that are not trivially zero.
    # If θᵢ and θⱼ correspond to different oscillators, these second derivatives
    # will always be zero.
    d2 = second_derivatives_1d(d1, idx, deriv_functions)

    model = np.einsum("ij->i", model_per_osc)
    diff = data - model

    # --- ℱ(θ) ---
    obj = np.real(diff.conj().T @ diff)
    # --- ∇ℱ(θ) ---
    grad = -2 * np.real(diff.conj().T @ d1)
    # --- ∇²ℱ(θ) ---
    # Non-trivially zero values of the second Hessian term:
    # (y - x)† ∂²x/∂θᵢ∂θⱼ
    diagonals = -2 * np.real(np.einsum("ji,j->i", d2.conj(), diff))
    # Determine indices of elements withn non-trivially zero second Hessian term
    # (specfically, those in upper triangle).
    p = len(idx)
    hess_shape = (p * m, p * m)
    hess = np.zeros(hess_shape)
    diag_indices = _generate_diagonal_indices(p, m)
    hess[diag_indices] = diagonals

    main_diagonals = _diagonal_indices(hess_shape[0], k=0)
    # Division by 2 to ensure elements on main diagonal aren't doubled
    # after transposition
    hess[main_diagonals] = hess[main_diagonals] / 2

    # Transpose (hessian is symmetric)
    hess += hess.T

    # Add component containing first derivatives
    # ∂x†/∂θᵢ ∂x/∂θⱼ
    hess += 2 * np.real(np.einsum("ki,kj->ij", d1.conj(), d1))
    if phasevar:
        # If 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in idx else 0
        phases = theta[i * m : (i + 1) * m]
        mu = np.einsum("i->", phases) / m
        # Var(φ)
        obj += np.einsum("i->", (phases - mu) ** 2) / (np.pi * m)
        # ∂Var(φ)/∂φᵢ
        grad[i * m : (i + 1) * m] += 0.8 * ((2 / m) * (phases - mu)) / np.pi
        # ∂²Var(φ)/∂φᵢ∂φⱼ
        hess[i * m : (i + 1) * m, i * m : (i + 1) * m] -= 2 / (m ** 2 * np.pi)
        hess[
            main_diagonals[0][i * m : (i + 1) * m],
            main_diagonals[1][i * m : (i + 1) * m],
        ] += 2 / (np.pi * m)

    return obj, grad, hess


def first_derivatives_2d(
    model_per_osc: np.ndarray, idx: list, deriv_functions: Dict[str, callable],
) -> np.ndarray:
    """Compute all partial derivatives of a 2D model signal.

    Parameters
    ----------
    model_per_osc
        ``(N1, N2, M)`` shape array containing ``(N1, N2)``-shape matrices
        corresponding to each oscillator's contribution to the model.

    idx
        List denoting the parameter types which are to be optimised.

        * ``0`` - amplitude
        * ``1`` - phase
        * ``2`` and ``3`` - frequencies 1 and 2
        * ``4`` and ``5`` - damping factors 1 and 2

    deriv_functions
        Dictionary containing callables to compute partial derivatives.

    Returns
    -------
    d1: np.ndarray
        ``(N1, N2, p * M)`` shape array containing all first partial derivatives,
         where ``p = len(idx)``.

    See Also
    --------
    :py:func:`obj_grad_2d`
    :py:func:`obj_grad_true_hess_2d`
    :py:func:`obj_grad_gauss_newton_hess_2d`
    """
    p = len(idx)
    m = model_per_osc.shape[-1]
    d1 = np.zeros((*model_per_osc.shape[:2], m * p), dtype="complex")

    ad = deriv_functions["a"]
    pd = deriv_functions["p"]
    f1d = deriv_functions["f1"]
    f2d = deriv_functions["f2"]
    d1d = deriv_functions["d1"]
    d2d = deriv_functions["d2"]

    # At points denoted ---x y z---, the parameters to be
    # optimised have been established, and the parameters are stated:
    # a: amplitudes
    # φ: phases
    # f1 and f2: frequencies
    # η1 and η2: damping factors
    if 0 in idx:
        d1[..., :m] = ad(model_per_osc)  # a
        if 1 in idx:
            d1[..., m : 2 * m] = pd(model_per_osc)  # φ
            if 2 in idx:
                d1[..., 2 * m : 3 * m] = f1d(model_per_osc)  # f1
                d1[..., 3 * m : 4 * m] = f2d(model_per_osc)  # f2
                if 4 in idx:
                    # ---a φ f1 f2 η1 η2--- (all parameters)
                    d1[..., 4 * m : 5 * m] = d1d(model_per_osc)  # η1
                    d1[..., 5 * m :] = d2d(model_per_osc)  # η2
                # ---a φ f1 f2---
            elif 4 in idx:
                # ---a φ η1 η2---
                d1[..., 2 * m : 3 * m] = d1d(model_per_osc)  # η1
                d1[..., 3 * m :] = d2d(model_per_osc)  # η2
            # ---a φ---
        elif 2 in idx:
            d1[..., m : 2 * m] = f1d(model_per_osc)  # f1
            d1[..., 2 * m : 3 * m] = f2d(model_per_osc)  # f2
            if 4 in idx:
                # ---a f1 f2 η1 η2---
                d1[..., 3 * m : 4 * m] = d1d(model_per_osc)  # η1
                d1[..., 4 * m : 5 * m] = d2d(model_per_osc)  # η2
        elif 4 in idx:
            # ---a η1 η2---
            d1[..., m : 2 * m] = d1d(model_per_osc)  # η1
            d1[..., 2 * m :] = d2d(model_per_osc)  # η2
        # ---a only---
    elif 1 in idx:
        d1[..., :m] = pd(model_per_osc)  # φ
        if 2 in idx:
            d1[..., m : 2 * m] = f1d(model_per_osc)  # f1
            d1[..., 2 * m : 3 * m] = f2d(model_per_osc)  # f2
            if 4 in idx:
                # ---φ f1 f2 η1 η2---
                d1[..., 3 * m : 4 * m] = d1d(model_per_osc)  # η1
                d1[..., 4 * m :] = d2d(model_per_osc)  # η2
            # ---φ f1 f2---
        elif 4 in idx:
            # ---φ η1 η2---
            d1[..., m : 2 * m] = d1d(model_per_osc)  # η1
            d1[..., 2 * m :] = d2d(model_per_osc)  # η2
        # ---φ only---
    elif 2 in idx:
        d1[..., :m] = f1d(model_per_osc)  # f1
        d1[..., m : 2 * m] = f2d(model_per_osc)  # f2
        if 4 in idx:
            # ---f1 f2 η1 η2---
            d1[..., 2 * m : 3 * m] = d1d(model_per_osc)  # η1
            d1[..., 3 * m :] = d2d(model_per_osc)  # η2
        # ---f1 f2---
    else:
        # ---η1 η2---
        d1[..., :m] = d1d(model_per_osc)  # η1
        d1[..., m:] = d2d(model_per_osc)  # η2

    return d1


def second_derivatives_2d(
    d1: np.ndarray, idx: list, deriv_functions: Dict[str, callable],
) -> np.ndarray:
    """Compute all second partial derivatives of a 2D model signal.

    Parameters
    ----------
    d1
        ``(N1, N2, p * M)`` shape array containing all first partial derivatives of
        the model, with ``p = len(idx)``.

    idx
        List denoting the parameter types which are to be optimised.

        * ``0`` - amplitude.
        * ``1`` - phase.
        * ``2`` and ``3`` - frequencies 1 and 2.
        * ``3`` and ``4`` - damping factors 1 and 2.

    deriv_functions
        Dictionary containing callables to compute partial derivatives.

    Returns
    -------
    d2: np.ndarray
        ``(N1, N2, (p * (p + 1) * M) // 2)`` shape array containing all
        non-trivially zero second partial derivatives.

    See Also
    --------
    :py:func:`obj_grad_true_hess_2d`
    :py:func:`obj_grad_gauss_newton_hess_2d`
    """
    p = len(idx)
    n1, n2 = d1.shape[:2]
    m = d1.shape[-1] // p
    # int((p * (p + 1)) / 2) --> p-th triangle number
    # gives array of:
    # --> nx21m if all oscs are to be optimised
    # --> nx15m if one type is passive
    # --> nx10m if two types are passive
    # --> nx6m if three types are passive
    # --> nx3m if four types are passive
    # --> nxm if five types are passive
    d2 = np.zeros((n1, n2, m * (p * (p + 1) // 2)), dtype="complex")

    ad = deriv_functions["a"]
    pd = deriv_functions["p"]
    f1d = deriv_functions["f1"]
    f2d = deriv_functions["f2"]
    d1d = deriv_functions["d1"]
    d2d = deriv_functions["d2"]

    if 0 in idx:
        # (a-a is trivially zero)
        if 1 in idx:
            d2[..., m : 2 * m] = pd(d1[..., :m])  # a-φ
            if 2 in idx:
                d2[..., 2 * m : 3 * m] = ad(d1[..., 2 * m : 3 * m])  # a-f1
                d2[..., 3 * m : 4 * m] = ad(d1[..., 3 * m : 4 * m])  # a-f2
                if 4 in idx:
                    # ---a φ f1 f2 η1 η2--- (all parameters)
                    d2[..., 4 * m : 5 * m] = ad(d1[..., 4 * m : 5 * m])  # a-η1
                    d2[..., 5 * m : 6 * m] = ad(d1[..., 5 * m :])  # a-η2
                    d2[..., 6 * m : 7 * m] = pd(d1[..., m : 2 * m])  # φ-φ
                    d2[..., 7 * m : 8 * m] = pd(d1[..., 2 * m : 3 * m])  # φ-f1
                    d2[..., 8 * m : 9 * m] = pd(d1[..., 3 * m : 4 * m])  # φ-f2
                    d2[..., 9 * m : 10 * m] = pd(d1[..., 4 * m : 5 * m])  # φ-η1
                    d2[..., 10 * m : 11 * m] = pd(d1[..., 5 * m :])  # φ-η2
                    d2[..., 11 * m : 12 * m] = f1d(d1[..., 2 * m : 3 * m])  # f1-f1
                    d2[..., 12 * m : 13 * m] = f1d(d1[..., 3 * m : 4 * m])  # f1-f2
                    d2[..., 13 * m : 14 * m] = f1d(d1[..., 4 * m : 5 * m])  # f1-η1
                    d2[..., 14 * m : 15 * m] = f1d(d1[..., 5 * m :])  # f1-η2
                    d2[..., 15 * m : 16 * m] = f2d(d1[..., 3 * m : 4 * m])  # f2-f2
                    d2[..., 16 * m : 17 * m] = f2d(d1[..., 4 * m : 5 * m])  # f2-η1
                    d2[..., 17 * m : 18 * m] = f2d(d1[..., 5 * m :])  # f2-η2
                    d2[..., 18 * m : 19 * m] = d1d(d1[..., 4 * m : 5 * m])  # η1-η1
                    d2[..., 19 * m : 20 * m] = d1d(d1[..., 5 * m :])  # η1-η2
                    d2[..., 20 * m : 21 * m] = d2d(d1[..., 5 * m :])  # η2-η2
                else:
                    # ---a φ f1 f2---
                    d2[..., 4 * m : 5 * m] = pd(d1[..., m : 2 * m])  # φ-φ
                    d2[..., 5 * m : 6 * m] = pd(d1[..., 2 * m : 3 * m])  # φ-f1
                    d2[..., 6 * m : 7 * m] = pd(d1[..., 3 * m :])  # φ-f2
                    d2[..., 7 * m : 8 * m] = f1d(d1[..., 2 * m : 3 * m])  # f1-f1
                    d2[..., 8 * m : 9 * m] = f1d(d1[..., 3 * m :])  # f1-f2
                    d2[..., 9 * m :] = f2d(d1[..., 3 * m :])  # f1-f2
            elif 4 in idx:
                # ---a φ η1 η2---
                d2[..., 2 * m : 3 * m] = ad(d1[..., 2 * m : 3 * m])  # a-η1
                d2[..., 3 * m : 4 * m] = ad(d1[..., 3 * m :])  # a-η2
                d2[..., 4 * m : 5 * m] = pd(d1[..., m : 2 * m])  # φ-φ
                d2[..., 5 * m : 6 * m] = pd(d1[..., 2 * m : 3 * m])  # φ-η1
                d2[..., 6 * m : 7 * m] = pd(d1[..., 3 * m :])  # φ-η2
                d2[..., 7 * m : 8 * m] = d1d(d1[..., 2 * m : 3 * m])  # η1-η1
                d2[..., 8 * m : 9 * m] = d1d(d1[..., 3 * m :])  # η1-η2
                d2[..., 9 * m :] = d2d(d1[..., 3 * m :])  # η2-η2
            else:
                # ---a φ---
                d2[..., 2 * m :] = pd(d1[..., m:])  # φ-φ
        elif 2 in idx:
            d2[..., m : 2 * m] = ad(d1[..., m : 2 * m])  # a-f1
            d2[..., 2 * m : 3 * m] = ad(d1[..., 2 * m : 3 * m])  # a-f2
            if 4 in idx:
                # ---a f1 f2 η1 η2---
                d2[..., 3 * m : 4 * m] = ad(d1[..., 3 * m : 4 * m])  # a-η1
                d2[..., 4 * m : 5 * m] = ad(d1[..., 4 * m : 5 * m])  # a-η2
                d2[..., 5 * m : 6 * m] = f1d(d1[..., m : 2 * m])  # f1-f1
                d2[..., 6 * m : 7 * m] = f1d(d1[..., 2 * m : 3 * m])  # f1-f2
                d2[..., 7 * m : 8 * m] = f1d(d1[..., 3 * m : 4 * m])  # f1-η1
                d2[..., 8 * m : 9 * m] = f1d(d1[..., 4 * m :])  # f1-η2
                d2[..., 9 * m : 10 * m] = f2d(d1[..., 2 * m : 3 * m])  # f2-f2
                d2[..., 10 * m : 11 * m] = f2d(d1[..., 3 * m : 4 * m])  # f2-η1
                d2[..., 11 * m : 12 * m] = f2d(d1[..., 4 * m :])  # f2-η2
                d2[..., 12 * m : 13 * m] = d1d(d1[..., 3 * m : 4 * m])  # η1-η1
                d2[..., 13 * m : 14 * m] = d1d(d1[..., 4 * m :])  # η1-η2
                d2[..., 14 * m : 15 * m] = d2d(d1[..., 4 * m :])  # η2-η2
            else:
                # ---a f1 f2---
                d2[..., 3 * m : 4 * m] = f1d(d1[..., m : 2 * m])  # f1-f1
                d2[..., 4 * m : 5 * m] = f1d(d1[..., 2 * m :])  # f1-f2
                d2[..., 5 * m :] = f2d(d1[..., 2 * m :])  # f2-f2
        elif 4 in idx:
            # ---a η1 η2---
            d2[..., m : 2 * m] = ad(d1[..., m : 2 * m])  # a-η1
            d2[..., 2 * m : 3 * m] = ad(d1[..., 2 * m :])  # a-η2
            d2[..., 3 * m : 4 * m] = d1d(d1[..., m : 2 * m])  # η1-η1
            d2[..., 4 * m : 5 * m] = d1d(d1[..., 2 * m :])  # η1-η2
            d2[..., 5 * m :] = d2d(d1[..., 2 * m :])  # η2-η2
        # ---a only---
    elif 1 in idx:
        d2[..., :m] = pd(d1[..., :m])  # φ-φ
        if 2 in idx:
            d2[..., m : 2 * m] = pd(d1[..., m : 2 * m])  # φ-f1
            d2[..., 2 * m : 3 * m] = pd(d1[..., 2 * m : 3 * m])  # φ-f2
            if 4 in idx:
                # ---φ f1 f2 η1 η2---
                d2[..., 3 * m : 4 * m] = pd(d1[..., 3 * m : 4 * m])  # φ-η1
                d2[..., 4 * m : 5 * m] = pd(d1[..., 4 * m :])  # φ-η2
                d2[..., 5 * m : 6 * m] = f1d(d1[..., m : 2 * m])  # f1-f1
                d2[..., 6 * m : 7 * m] = f1d(d1[..., 2 * m : 3 * m])  # f1-f2
                d2[..., 7 * m : 8 * m] = f1d(d1[..., 3 * m : 4 * m])  # f1-η1
                d2[..., 8 * m : 9 * m] = f1d(d1[..., 4 * m :])  # f1-η2
                d2[..., 9 * m : 10 * m] = f2d(d1[..., 2 * m : 3 * m])  # f2-f2
                d2[..., 10 * m : 11 * m] = f2d(d1[..., 3 * m : 4 * m])  # f2-η1
                d2[..., 11 * m : 12 * m] = f2d(d1[..., 4 * m :])  # f2-η2
                d2[..., 12 * m : 13 * m] = d1d(d1[..., 3 * m : 4 * m])  # η1-η1
                d2[..., 13 * m : 14 * m] = d1d(d1[..., 4 * m :])  # η1-η2
                d2[..., 14 * m : 15 * m] = d2d(d1[..., 4 * m :])  # η2-η2
            else:
                # ---φ f1 f2---
                d2[..., 3 * m : 4 * m] = f1d(d1[..., m : 2 * m])  # f1-f1
                d2[..., 4 * m : 5 * m] = f1d(d1[..., 2 * m :])  # f1-f2
                d2[..., 5 * m :] = f2d(d1[..., 2 * m :])  # f2-f2
        elif 4 in idx:
            # ---φ η1 η2---
            d2[..., m : 2 * m] = pd(d1[..., m : 2 * m])  # φ-η1
            d2[..., 2 * m : 3 * m] = pd(d1[..., 2 * m :])  # φ-η2
            d2[..., 3 * m : 4 * m] = d1d(d1[..., m : 2 * m])  # η1-η1
            d2[..., 4 * m : 5 * m] = d1d(d1[..., 2 * m :])  # η1-η2
            d2[..., 5 * m :] = d2d(d1[..., 2 * m :])  # η2-η2
        # ---φ only---
    elif 2 in idx:
        d2[..., :m] = f1d(d1[..., :m])  # f1-f1
        d2[..., m : 2 * m] = f1d(d1[..., m : 2 * m])  # f1-f2
        if 4 in idx:
            # ---f1 f2 η1 η2---
            d2[..., 2 * m : 3 * m] = f1d(d1[..., 2 * m : 3 * m])  # f1-η1
            d2[..., 3 * m : 4 * m] = f1d(d1[..., 3 * m :])  # f1-η2
            d2[..., 4 * m : 5 * m] = f2d(d1[..., m : 2 * m])  # f2-f2
            d2[..., 5 * m : 6 * m] = f2d(d1[..., 2 * m : 3 * m])  # f2-η1
            d2[..., 6 * m : 7 * m] = f2d(d1[..., 3 * m :])  # f2-η2
            d2[..., 7 * m : 8 * m] = d1d(d1[..., 2 * m : 3 * m])  # η1-η1
            d2[..., 8 * m : 9 * m] = d1d(d1[..., 3 * m :])  # η1-η2
            d2[..., 9 * m :] = d2d(d1[..., 3 * m :])  # η2-η2
        else:
            # ---f1 f2---
            d2[..., 2 * m :] = f2d(d1[..., m:])  # f2-f2
    else:
        # ---η1 η2---
        d2[..., :m] = d1d(d1[..., :m])  # η1-η1
        d2[..., m : 2 * m] = d1d(d1[..., m:])  # η1-η2
        d2[..., 2 * m :] = d2d(d1[..., m:])  # η2-η2

    return d2


def obj_2d(active: np.ndarray, *args: args_type) -> float:
    """Compute the objective for 2D data.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` and ``3`` - frequencies 1 and 2.
              + ``3`` and ``4`` - damping factors 1 and 2.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    obj: float
        Value of the objective.
    """
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    model = (
        # Z1
        np.exp(
            np.outer(
                tp[0],
                2j * np.pi * theta[2 * m : 3 * m] - theta[4 * m : 5 * m],
            )
        ) @
        # α
        np.diag(
            theta[:m] * np.exp(1j * theta[m : 2 * m])
        ) @
        # Z2T
        np.exp(
            np.outer(
                2j * np.pi * theta[3 * m : 4 * m] - theta[5 * m :],
                tp[1],
            )
        )
    )
    diff = data - model

    # Tr(AB) = Σᵢⱼ aᵢⱼbⱼᵢ
    obj = np.real(np.einsum("ij,ij->", diff.conj(), diff))

    if phasevar:
        phases = theta[m : 2 * m]
        mu = np.einsum("i->", phases) / m
        obj += np.einsum("i->", (phases - mu) ** 2) / (np.pi * m)

    return obj


def obj_grad_2d(active: np.ndarray, *args: args_type) -> Tuple[float, np.ndarray]:
    """Compute the objective and gradient for 2D data.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` and ``3`` - frequencies 1 and 2.
              + ``3`` and ``4`` - damping factors 1 and 2.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    obj: float
        Value of the objective.

    grad: numpy.ndarray
        Gradient of the objective.
    """
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    model_per_osc = np.einsum(
        "ik,kj->ijk",
        # Y
        np.exp(
            np.outer(
                tp[0],
                2j * np.pi * theta[2 * m : 3 * m] - theta[4 * m : 5 * m]
            ),
        ),
        np.einsum(
            "ij,i->ij",
            # Z
            np.exp(
                np.outer(
                    2j * np.pi * theta[3 * m : 4 * m] - theta[5 * m : 6 * m],
                    tp[1],
                )
            ),
            # α
            theta[:m] * np.exp(1j * theta[m : 2 * m])
        ),
    )

    deriv_functions = {
        "a": lambda x: x / theta[:m],
        "p": lambda x: 1j * x,
        "f1": lambda x: np.einsum("ijk,i->ijk", x, 2j * np.pi * tp[0]),
        "f2": lambda x: np.einsum("ijk,j->ijk", x, 2j * np.pi * tp[1]),
        "d1": lambda x: np.einsum("ijk,i->ijk", x, -tp[0]),
        "d2": lambda x: np.einsum("ijk,j->ijk", x, -tp[1]),
    }

    d1 = first_derivatives_2d(model_per_osc, idx, deriv_functions)

    model = np.einsum("ijk->ij", model_per_osc)
    diff = data - model

    # --- ℱ(θ) ---
    # Tr(AB) = Σᵢⱼ aᵢⱼbⱼᵢ
    obj = np.real(np.einsum("ij,ij->", diff.conj(), diff))
    # --- ∇ℱ(θ) ---
    grad = -2 * np.real(np.einsum("ij,ijk->k", diff.conj(), d1))

    if phasevar:
        # If 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in idx else 0
        phases = theta[i * m : (i + 1) * m]
        mu = np.einsum("i->", phases) / m
        # Var(φ)
        obj += np.einsum("i->", (phases - mu) ** 2) / (np.pi * m)
        # ∂Var(φ)/∂φᵢ
        grad[i * m : (i + 1) * m] += 0.8 * ((2 / m) * (phases - mu)) / np.pi

    return obj, grad


def hess_2d(active: np.ndarray, *args: args_type) -> np.ndarray:
    """Hessian of cost function for 2D data.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` and ``3`` - frequencies 1 and 2.
              + ``4`` and ``5`` - damping factors 1 and 2.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    hess
        Hessian of cost function.
    """
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    model_per_osc = np.einsum(
        "ik,kj->ijk",
        # Y
        np.exp(
            np.outer(
                tp[0],
                2j * np.pi * theta[2 * m : 3 * m] - theta[4 * m : 5 * m]
            ),
        ),
        np.einsum(
            "ij,i->ij",
            # Z
            np.exp(
                np.outer(
                    2j * np.pi * theta[3 * m : 4 * m] - theta[5 * m : 6 * m],
                    tp[1],
                )
            ),
            # α
            theta[:m] * np.exp(1j * theta[m : 2 * m])
        ),
    )

    deriv_functions = {
        "a": lambda x: x / theta[:m],
        "p": lambda x: 1j * x,
        "f1": lambda x: np.einsum("ijk,i->ijk", x, 2j * np.pi * tp[0]),
        "f2": lambda x: np.einsum("ijk,j->ijk", x, 2j * np.pi * tp[1]),
        "d1": lambda x: np.einsum("ijk,i->ijk", x, -tp[0]),
        "d2": lambda x: np.einsum("ijk,j->ijk", x, -tp[1]),
    }

    d1 = first_derivatives_2d(model_per_osc, idx, deriv_functions)
    d2 = second_derivatives_2d(d1, idx, deriv_functions)

    diagonals = -2 * np.real(
        np.einsum(
            "ijk,ij->k",
            d2.conj(),
            data - np.einsum("ijk->ij", model_per_osc),
        )
    )

    p = len(idx)
    hess_shape = (p * m, p * m)
    hess = np.zeros(hess_shape)
    diag_indices = _generate_diagonal_indices(p, m)
    hess[diag_indices] = diagonals
    main_diag_indices = _diagonal_indices(hess_shape[0], k=0)
    # Division by 2 to ensure elements on main diagonal aren't doubled
    # after transposition.
    hess[main_diag_indices] /= 2
    hess += hess.T
    hess += 2 * np.real(np.einsum("ijk,ijl->kl", d1.conj(), d1))

    if phasevar:
        # If 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in idx else 0
        # ∂²Var(φ)/∂φᵢ∂φⱼ
        hess[i * m : (i + 1) * m, i * m : (i + 1) * m] -= 2 / (m ** 2 * np.pi)
        hess[
            main_diag_indices[0][i * m : (i + 1) * m],
            main_diag_indices[1][i * m : (i + 1) * m],
        ] += 2 / (np.pi * m)

    return hess


def obj_finite_diff_grad_hess_2d(
    active: np.ndarray, h: int, *args: args_type
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the objective, gradient and hessian for 2D data.

    The gradient and Hessian are computed using fintie difference.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    h
        Finite difference parameter

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` and ``3`` - frequencies 1 and 2.
              + ``4`` and ``5`` - damping factors 1 and 2.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    obj: float
        Value of the objective.

    grad: numpy.ndarray
        Finite difference gradient of the objective.

    hess: numpy.ndarray
        Finite difference Hessian of the objective.
    """
    return _finite_diff(active, 2, h, *args)


def obj_grad_gauss_newton_hess_2d(
    active: np.ndarray, *args: args_type
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the objective, gradient and hessian for 2D data.

    The Hessian is computed using the Gauss-Newton technique.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` and ``3`` - frequencies 1 and 2.
              + ``3`` and ``4`` - damping factors 1 and 2.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    obj: float
        Value of the objective.

    grad: numpy.ndarray
        Gradient of the objective.

    hess: numpy.ndarray
        Gauss-Newton Hessian of the objective.
    """
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    model_per_osc = np.einsum(
        "ik,kj->ijk",
        # Y
        np.exp(
            np.outer(
                tp[0],
                2j * np.pi * theta[2 * m : 3 * m] - theta[4 * m : 5 * m]
            ),
        ),
        np.einsum(
            "ij,i->ij",
            # Z
            np.exp(
                np.outer(
                    2j * np.pi * theta[3 * m : 4 * m] - theta[5 * m : 6 * m],
                    tp[1],
                )
            ),
            # α
            theta[:m] * np.exp(1j * theta[m : 2 * m])
        ),
    )

    deriv_functions = {
        "a": lambda x: x / theta[:m],
        "p": lambda x: 1j * x,
        "f1": lambda x: np.einsum("ijk,i->ijk", x, 2j * np.pi * tp[0]),
        "f2": lambda x: np.einsum("ijk,j->ijk", x, 2j * np.pi * tp[1]),
        "d1": lambda x: np.einsum("ijk,i->ijk", x, -tp[0]),
        "d2": lambda x: np.einsum("ijk,j->ijk", x, -tp[1]),
    }

    d1 = first_derivatives_2d(model_per_osc, idx, deriv_functions)

    model = np.einsum("ijk->ij", model_per_osc)
    diff = data - model

    # --- ℱ(θ) ---
    # Tr(AB) = Σᵢⱼ aᵢⱼbⱼᵢ
    obj = np.real(np.einsum("ij,ij->", diff.conj(), diff))
    # --- ∇ℱ(θ) ---
    grad = -2 * np.real(np.einsum("ij,ijk->k", diff.conj(), d1))
    # --- ∇²ℱ(θ) ---
    hess = 2 * np.real(np.einsum("ijk,ijl->kl", d1.conj(), d1))

    return obj, grad, hess


def obj_grad_true_hess_2d(active: np.ndarray, *args):
    """Compute the objective, gradient and hessian for 2D data.

    The Hessian is computed exactly.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters that are
          active:

              + ``0`` - amplitudes.
              + ``1`` - phases.
              + ``2`` and ``3`` - frequencies 1 and 2.
              + ``3`` and ``4`` - damping factors 1 and 2.

        * **phase_variance:** If ``True``, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    obj: float
        Value of the objective.

    grad: numpy.ndarray
        Gradient of the objective.

    hess: numpy.ndarray
        Exact Hessian of the objective.
    """
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    model_per_osc = np.einsum(
        "ik,kj->ijk",
        # Y
        np.exp(
            np.outer(
                tp[0],
                2j * np.pi * theta[2 * m : 3 * m] - theta[4 * m : 5 * m]
            ),
        ),
        np.einsum(
            "ij,i->ij",
            # Z
            np.exp(
                np.outer(
                    2j * np.pi * theta[3 * m : 4 * m] - theta[5 * m : 6 * m],
                    tp[1],
                )
            ),
            # α
            theta[:m] * np.exp(1j * theta[m : 2 * m])
        ),
    )

    deriv_functions = {
        "a": lambda x: x / theta[:m],
        "p": lambda x: 1j * x,
        "f1": lambda x: np.einsum("ijk,i->ijk", x, 2j * np.pi * tp[0]),
        "f2": lambda x: np.einsum("ijk,j->ijk", x, 2j * np.pi * tp[1]),
        "d1": lambda x: np.einsum("ijk,i->ijk", x, -tp[0]),
        "d2": lambda x: np.einsum("ijk,j->ijk", x, -tp[1]),
    }

    d1 = first_derivatives_2d(model_per_osc, idx, deriv_functions)
    d2 = second_derivatives_2d(d1, idx, deriv_functions)

    model = np.einsum("ijk->ij", model_per_osc)
    diff = data - model

    # --- ℱ(θ) ---
    # Tr(AB) = Σᵢⱼ aᵢⱼbⱼᵢ
    obj = np.real(np.einsum("ij,ij->", diff.conj(), diff))
    # --- ∇ℱ(θ) ---
    grad = -2 * np.real(np.einsum("ij,ijk->k", diff.conj(), d1))
    # --- ∇²ℱ(θ) ---
    diagonals = -2 * np.real(np.einsum("ijk,ij->k", d2.conj(), diff))
    p = len(idx)
    hess_shape = (p * m, p * m)
    hess = np.zeros(hess_shape)
    diag_indices = _generate_diagonal_indices(p, m)
    hess[diag_indices] = diagonals
    main_diag_indices = _diagonal_indices(hess_shape[0], k=0)
    # Division by 2 to ensure elements on main diagonal aren't doubled
    # after transposition.
    hess[main_diag_indices] /= 2
    hess += hess.T
    hess += 2 * np.real(np.einsum("ijk,ijl->kl", d1.conj(), d1))

    if phasevar:
        # If 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in idx else 0
        phases = theta[i * m : (i + 1) * m]
        mu = np.einsum("i->", phases) / m
        # Var(φ)
        obj += np.einsum("i->", (phases - mu) ** 2) / (np.pi * m)
        # ∂Var(φ)/∂φᵢ
        grad[i * m : (i + 1) * m] += 0.8 * ((2 / m) * (phases - mu)) / np.pi
        # ∂²Var(φ)/∂φᵢ∂φⱼ
        hess[i * m : (i + 1) * m, i * m : (i + 1) * m] -= 2 / (m ** 2 * np.pi)
        hess[
            main_diag_indices[0][i * m : (i + 1) * m],
            main_diag_indices[1][i * m : (i + 1) * m],
        ] += 2 / (np.pi * m)

    return obj, grad, hess


def _finite_diff(
    active: np.ndarray, dim: int, h: float, *args: args_type
) -> Tuple[float, np.ndarray, np.ndarray]:
    data, tp, m, passive, idx, phasevar = args

    if dim == 1:
        obj_func = obj_1d
    elif dim == 2:
        obj_func = obj_2d

    # Number of parameters in `active`
    p = len(idx) * m

    obj = obj_func(active, *args)
    grad = np.zeros(p)
    hess = np.zeros((p, p))

    # Deviation using centered finite difference
    dev = 0.5 * h
    for i in range(p):
        uvi = _unitvec(p, i)
        grad[i] = (
            obj_func(active + dev * uvi, *args) -
            obj_func(active - dev * uvi, *args)
        ) / h

        # Only compute Hessian elements in upper right triangle.
        # Hessian is symmetric, so lower left triangle can be generated via
        # transpose.
        for j in range(i, p):
            uvj = _unitvec(p, j)
            hess[i, j] = (
                obj_func(active + dev * uvi + dev * uvj, *args) -
                obj_func(active + dev * uvi - dev * uvj, *args) -
                obj_func(active - dev * uvi + dev * uvj, *args) +
                obj_func(active - dev * uvi - dev * uvj, *args)
            ) / (h ** 2)
            # If element on diagonal, half so that when transpose in applied
            # doubling doesn't occur.
            if i == j:
                hess[i, j] /= 2

    hess = hess + hess.T

    if phasevar:
        phases = active[m : 2 * m]
        mu = np.einsum("i->", phases) / m
        obj += np.einsum("i->", (phases - mu) ** 2) / (np.pi * m)
        grad[m : 2 * m] += 0.8 * ((2 / m) * (phases - mu)) / np.pi
        hess[m : 2 * m, m : 2 * m] -= 2 / (m ** 2 * np.pi)
        main_diagonals = _diagonal_indices(p, k=0)
        hess[
            main_diagonals[0][m : 2 * m],
            main_diagonals[1][m : 2 * m],
        ] += 2 / (np.pi * m)

    return obj, grad, hess


def _construct_parameters(
    active: np.ndarray, passive: np.ndarray, m: int, idx: list
) -> np.ndarray:
    """Construct the full parameter vector from active and passive vectors.

    Parameters
    ----------
    active
        Active parameter vector.

    passive
        Passive parameter vector.

    m
        Number of oscillators.

    idx
        Indicates the columns (axis 1) for active parameters.

    Returns
    -------
    params
        Full parameter vector with correct ordering
    """
    # number of columns in active parameter array
    p = int((active.shape[0] + passive.shape[0]) / m)

    params = np.zeros(p * m)

    for i in range(p):
        if i in idx:
            params[i * m : (i + 1) * m] = active[:m]
            active = active[m:]
        else:
            params[i * m : (i + 1) * m] = passive[:m]
            passive = passive[m:]

    return params


def _unitvec(n: int, idx: int) -> np.ndarray:
    vec = np.zeros(n)
    vec[idx] = 1
    return vec


def _generate_diagonal_indices(p: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Determine Hessian positions with non-zero second-derivatives.

    Only indices in the top-right of the Hessian are generated.

    Parameters
    ----------
    p
        The number of parameter 'groups'. For an N-dimensional signal this
        will be 2N + 2

    m
        Number of oscillators in parameter estimate

    Returns
    -------
    idx_0
        0-axis coordinates of indices.

    idx_1
        1-axis coordinates of indices.
    """
    idx_0 = []  # axis 0 indices (rows)
    idx_1 = []  # axis 1 indices (columns)
    for i in range(p):
        for j in range(p - i):
            idx_0.append(_diagonal_indices(p * m, k=j * m)[0][i * m : (i + 1) * m])
            idx_1.append(_diagonal_indices(p * m, k=j * m)[1][i * m : (i + 1) * m])

    return np.hstack(idx_0), np.hstack(idx_1)


def _diagonal_indices(size: int, k: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return the indices of an array's kth diagonal.

    A generalisation of `numpy.diag_indices <https://numpy.org/doc\
    /stable/reference/generated/numpy.diag_indices.html>`_,
    which can only be used to obtain the indices along the main diagonal of
    an array.

    Parameters
    ----------
    size
        The size of the array (indices are obtained for a ``(size, size)`` shape
        array).

    k
        Displacement from the main diagonal

    Returns
    -------
    rows
        0-axis coordinates of indices.

    cols
        1-axis coordinates of indices.
    """
    rows, cols = np.diag_indices(size)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols
