# nlp.funcs.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Definitions of fidelities, gradients, and Hessians."""

from typing import Tuple

import numpy as np


args_type = Tuple[np.ndarray, np.ndarray, int, np.ndarray, list, bool]


def f_1d(active: np.ndarray, *args: args_type) -> float:
    """Cost function for 1D data.

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
          active: ``0`` - amplitudes, ``1`` - phases, ``2`` - frequencies,
          ``3`` - damping factors.
        * **phase_variance:** If `True`, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    func
        Value of the cost function
    """
    # unpack arguments
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    # signal pole matrix
    Z = np.exp(
        np.outer(
            tp[0], (1j * 2 * np.pi * theta[2 * m:3 * m] - theta[3 * m:])
        )
    )

    model = Z @ (theta[:m] * np.exp(1j * theta[m:2 * m]))
    diff = data - model
    func = np.real(np.vdot(diff, diff))

    if phasevar:
        # if 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        phases = theta[m:2 * m] if 0 in idx else theta[:m]
        mu = np.einsum('i->', phases) / m
        func += ((np.einsum('i->', phases ** 2) / m) - mu ** 2) / np.pi

    return func


def g_1d(active: np.ndarray, *args: args_type) -> np.ndarray:
    """Gradient of cost function for 1D data.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters are present
          in the active oscillators (para_act): ``0`` - amplitudes, ``1`` -
          phases, ``2`` - frequencies, ``3`` - damping factors.
        * **phase_variance:** If True, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    grad
        Gradient of cost function, with ``grad.shape = (4 * m,)``.
    """
    # unpack arguments
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    # signal pole matrix
    Z = np.exp(
        np.outer(
            tp[0], (1j * 2 * np.pi * theta[2 * m:3 * m] - theta[3 * m:])
        )
    )

    # N x M array comprising M N-length vectors
    # each vector is the model produced by a single oscillator
    model_per_osc = Z * (theta[:m] * np.exp(1j * theta[m:2 * m]))

    # first derivative components
    d1 = np.zeros((*data.shape, m * len(idx)), dtype='complex')

    # ================================================================
    # This section computes all the first derivatives.
    # It is able to handle all parameter types, or any subset of
    # parameter types.
    # At points in the code denoted ---x y z---, the parameters to be
    # optimised has been established, and the parameters are stated:
    # a: amplitudes
    # φ: phases
    # f: frequencies
    # η: damping factors
    # ================================================================

    def ad(arr):
        """Differentiate wrt amplitude."""
        return arr / theta[:m]

    def pd(arr):
        """Differentiate wrt phase."""
        return 1j * arr

    def fd(arr):
        """Differentiate wrt frequency."""
        return np.einsum('ij,i->ij', arr, (1j * 2 * np.pi * tp[0]))

    def dd(arr):
        """Differentiate wrt damping factor."""
        return np.einsum('ij,i->ij', arr, -tp[0])

    if 0 in idx:
        d1[:, :m] = ad(model_per_osc)  # a

        if 1 in idx:
            d1[:, m:2 * m] = pd(model_per_osc)  # φ

            if 2 in idx:
                d1[:, 2 * m:3 * m] = fd(model_per_osc)  # f

                if 3 in idx:
                    # ---a φ f η--- (all parameters)
                    d1[:, 3 * m:] = dd(model_per_osc)  # η

                # ---a φ f---

            elif 3 in idx:
                # ---a φ η---
                d1[:, 2 * m:] = dd(model_per_osc)  # η

            # ---a φ---

        elif 2 in idx:
            d1[:, m:2 * m] = fd(model_per_osc)  # f

            if 3 in idx:
                # ---a f η---
                d1[:, 2 * m:] = dd(model_per_osc)  # η

            # ---a f---

        elif 3 in idx:
            # ---a η---
            d1[:, m:] = dd(model_per_osc)  # η

        # ---a only---

    elif 1 in idx:
        d1[:, :m] = pd(model_per_osc)  # φ

        if 2 in idx:
            d1[:, m:2 * m] = fd(model_per_osc)  # f

            if 3 in idx:
                # ---φ f η---
                d1[:, 2 * m:] = dd(model_per_osc)  # η

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

    # sum model_per_osc to generate data-length vector,
    # and take residual
    diff = data - np.einsum('ij->i', model_per_osc)
    grad = -2 * np.real(d1.conj().T @ diff)

    if phasevar:
        # if 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in idx else 0
        phases = theta[i * m:(i + 1) * m]
        mu = np.einsum('i->', phases) / m
        grad[i * m:(i + 1) * m] += ((2 / m) * (phases - mu)) / np.pi

    return grad


def h_1d(active: np.ndarray, *args: args_type) -> np.ndarray:
    """Hessian of cost function for 1D data.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters are present
          in the active oscillators (para_act): ``0`` - amplitudes, ``1`` -
          phases, ``2`` - frequencies, ``3`` - damping factors.
        * **phase_variance:** If True, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    hess
        Hessian of cost function, with ``hess.shape = (4 * m,4 * m)``.
    """
    # unpack arguments
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    # signal pole matrix
    Z = np.exp(
        np.outer(
            tp[0], (1j * 2 * np.pi * theta[2 * m:3 * m] - theta[3 * m:])
        )
    )

    # N x M array comprising M N-length vectors
    # each vector is the model produced by a single oscillator
    model_per_osc = Z * (theta[:m] * np.exp(1j * theta[m:2 * m]))

    p = len(idx)

    # first derivative components
    d1 = np.zeros((*data.shape, m * p), dtype='complex')

    # int((p * (p + 1)) / 2) --> p-th triangle number
    # gives array of:
    # --> nx10m if all oscs are to be optimised
    # --> nx6m if one type is passive
    # --> nx3m if two types are passive
    # --> nxm if three types are passive
    d2 = np.zeros((*data.shape, m * int((p * (p + 1)) / 2)), dtype='complex')

    # ================================================================
    # This section computes all the first and second derivatives.
    # It is able to handle all parameter types, or any subset of
    # parameter types.
    # At points in the code denoted ---x y z---, the parameters to be
    # optimised has been established, and the parameters are stated:
    # a: amplitudes
    # φ: phases
    # f: frequencies
    # η: damping factors
    # ================================================================

    def ad(arr):
        """Differentiate wrt amplitude."""
        return arr / theta[:m]

    def pd(arr):
        """Differentiate wrt phase."""
        return 1j * arr

    def fd(arr):
        """Differentiate wrt frequency."""
        return np.einsum('ij,i->ij', arr, (1j * 2 * np.pi * tp[0]))

    def dd(arr):
        """Differentiate wrt damping factor."""
        return np.einsum('ij,i->ij', arr, -tp[0])

    if 0 in idx:
        d1[:, :m] = ad(model_per_osc)  # a
        # (a-a is trivially zero)

        if 1 in idx:
            d1[:, m:2 * m] = pd(model_per_osc)  # φ
            d2[:, m:2 * m] = pd(d1[:, :m])      # a-φ

            if 2 in idx:
                d1[:, 2 * m:3 * m] = fd(model_per_osc)       # f
                d2[:, 2 * m:3 * m] = ad(d1[:, 2 * m:3 * m])  # a-f

                if 3 in idx:
                    # ---a φ f η---
                    d1[:, 3 * m:] = dd(model_per_osc)            # η
                    d2[:, 3 * m:4 * m] = ad(d1[:, 3 * m:4 * m])  # a-η
                    d2[:, 4 * m:5 * m] = pd(d1[:, :m])           # φ-φ
                    d2[:, 5 * m:6 * m] = pd(d1[:, 2 * m:3 * m])  # φ-f
                    d2[:, 6 * m:7 * m] = pd(d1[:, 3 * m:4 * m])  # φ-η
                    d2[:, 7 * m:8 * m] = fd(d1[:, 2 * m:3 * m])  # f-f
                    d2[:, 8 * m:9 * m] = dd(d1[:, 2 * m:3 * m])  # f-η
                    d2[:, 9 * m:] = dd(d1[:, 3 * m:])            # η-η

                else:
                    # ---a φ f---
                    d2[:, 3 * m:4 * m] = pd(d1[:, :m])      # φ-φ
                    d2[:, 4 * m:5 * m] = pd(d1[:, 2 * m:])  # φ-f
                    d2[:, 5 * m:] = fd(d1[:, 2 * m:])       # f-f

            elif 3 in idx:
                # ---a φ η---
                d1[:, 2 * m:] = dd(model_per_osc)       # η
                d2[:, 2 * m:3 * m] = ad(d1[:, 2 * m:])  # a-η
                d2[:, 3 * m:4 * m] = pd(d1[:, :m])      # φ-φ
                d2[:, 4 * m:5 * m] = pd(d1[:, 2 * m:])  # φ-η
                d2[:, 5 * m:] = dd(d1[:, 2 * m:])       # η-η

            else:
                # ---a φ---
                d2[:, 2 * m:] = pd(d1[:, m:])  # φ-φ

        elif 2 in idx:
            d1[:, m:2 * m] = fd(model_per_osc)   # f
            d2[:, m:2 * m] = ad(d1[:, m:2 * m])  # a-f

            if 3 in idx:
                # ---a f η---
                d1[:, 2 * m:] = dd(model_per_osc)        # η
                d2[:, 2 * m:3 * m] = ad(d1[:, 2 * m:])   # a-η
                d2[:, 3 * m:4 * m] = fd(d1[:, m:2 * m])  # f-f
                d2[:, 4 * m:5 * m] = dd(d1[:, m:2 * m])  # f-η
                d2[:, 5 * m:] = dd(d1[:, 2 * m:])        # η-η

            else:
                # ---a f---
                d2[:, 2 * m:] = fd(d1[:, m:])  # f-f

        elif 3 in idx:
            # ---a η---
            d1[:, m:] = dd(model_per_osc)  # η
            d2[:, 2 * m:] = dd(d1[:, m:])  # η-η

        # ---a only---

    elif 1 in idx:
        d1[:, :m] = pd(model_per_osc)  # φ
        d2[:, :m] = pd(d1[:, :m])      # φ-φ

        if 2 in idx:
            d1[:, m:2 * m] = fd(model_per_osc)   # f
            d2[:, m:2 * m] = pd(d1[:, m:2 * m])  # φ-f

            if 3 in idx:
                # ---φ f η---
                d1[:, 2 * m:] = dd(model_per_osc)        # η
                d2[:, 2 * m:3 * m] = pd(d1[:, 2 * m:])   # φ-η
                d2[:, 3 * m:4 * m] = fd(d1[:, m:2 * m])  # f-f
                d2[:, 4 * m:5 * m] = dd(d1[:, m:2 * m])  # f-η
                d2[:, 5 * m:] = dd(d1[:, 2 * m:])        # η-η

            else:
                # ---φ f---
                d2[:, 2 * m:] = fd(d1[:, m:])  # f-f

        elif 3 in idx:
            # ---φ η---
            d1[:, m:] = dd(model_per_osc)   # η
            d2[:, m:2 * m] = pd(d1[:, m:])  # φ-η
            d2[:, 2 * m:] = dd(d1[:, m:])   # η-η

        # ---φ only---

    elif 2 in idx:
        d1[:, :m] = fd(model_per_osc)  # f
        d2[:, :m] = fd(d1[:, :m])      # f-f

        if 3 in idx:
            # ---f η---
            d1[:, m:] = dd(model_per_osc)   # η
            d2[:, m:2 * m] = dd(d1[:, :m])  # f-η
            d2[:, 2 * m:] = dd(d1[:, m:])   # η-η

        # ---f only---

    else:
        # ---η only---
        d1 = dd(model_per_osc)  # η
        d2 = dd(d1)             # η-η

    diff = data - np.einsum('ij->i', model_per_osc)
    diagonals = -2 * np.real(np.einsum('ji,j->i', d2.conj(), diff))

    # determine indices of elements which have non-zero second derivs
    # (specfically, those in upper triangle)
    diag_indices = _generate_diagonal_indices(p, m)
    hess = np.zeros((p * m, p * m))
    hess[diag_indices] = diagonals

    main_diagonals = _diagonal_indices(hess, k=0)
    # division by 2 to ensure elements on main diagonal aren't doubled
    # after transposition
    hess[main_diagonals] = hess[main_diagonals] / 2

    # transpose (hessian is symmetric)
    hess += hess.T

    # add component containing first derivatives
    hess += 2 * np.real(np.einsum('ki,kj->ij', d1.conj(), d1))

    if phasevar:
        # if 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in idx else 0
        hess[i * m:(i + 1) * m, i * m:(i + 1) * m] -= (2 / (m ** 2 * np.pi))
        hess[main_diagonals[0][i * m:(i + 1) * m],
             main_diagonals[1][i * m:(i + 1) * m]] \
            += 2 / (np.pi * m)

    return hess


def f_2d(active: np.ndarray, *args: args_type) -> float:
    """Cost function for 2D data.

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
          active: ``0`` - amplitudes, ``1`` - phases, ``2`` - frequencies,
          ``3`` - damping factors.
        * **phase_variance:** If `True`, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    func
        Value of the cost function
    """
    # unpack arguments
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # optimisable and non-optimisable parameters.
    theta = _construct_parameters(active, passive, m, idx)

    Y = np.exp(np.outer(
        tp[0], 1j * 2 * np.pi * theta[2 * m:3 * m] - theta[4 * m:5 * m]
    ))
    Z = np.exp(np.outer(
        1j * 2 * np.pi * theta[3 * m:4 * m] - theta[5 * m:6 * m], tp[1]
    ))
    A = np.diag(theta[:m] * np.exp(1j * theta[m:2 * m]))

    model = Y @ A @ Z
    func = np.real(
        np.einsum('ii', (data - model).conj().T @ (data - model))
    )

    if phasevar:
        # If 0 is in idx, phases will be between m and 2 * m, as amps
        # also present. If not, phases will be between 0 and m.
        phases = theta[m:2 * m] if 0 in idx else theta[:m]
        mu = np.einsum('i->', phases) / m
        func += ((np.einsum('i->', phases ** 2) / m) - mu ** 2) / np.pi

    return func


def g_2d(active: np.ndarray, *args: args_type) -> np.ndarray:
    """Gradient of cost function for 2D data.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args
        Contains elements in the following order:

        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters are present
          in the active oscillators (para_act): ``0`` - amplitudes, ``1`` -
          phases, ``2`` - frequencies, ``3`` - damping factors.
        * **phase_variance:** If True, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    grad
        Gradient of cost function, with ``grad.shape = (6*M,)``.
    """
    # unpack arguments
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    Y = np.exp(np.outer(
        tp[0], 1j * 2 * np.pi * theta[2 * m:3 * m] - theta[4 * m:5 * m]
    ))
    Z = np.exp(np.outer(
        1j * 2 * np.pi * theta[3 * m:4 * m] - theta[5 * m:6 * m], tp[1]
    ))
    alpha = theta[:m] * np.exp(1j * theta[m:2 * m])
    model_per_osc = np.einsum(
        'ik,kj->ijk', Y, np.einsum('ij,i->ij', Z, alpha)
    )

    # first derivative components
    d1 = np.zeros((*data.shape, m * len(idx)), dtype='complex')

    # ================================================================
    # This section computes all the first derivatives.
    # It is able to handle all parameter types, or any subset of
    # parameter types.
    # At points in the code denoted ---x y z---, the parameters to be
    # optimised has been established, and the parameters are stated:
    # a: amplitudes
    # φ: phases
    # f1 f2: frequencies in dims 1 and 2
    # η1 η2: damping factors in dims 1 and 2
    # ================================================================

    def ad(arr):
        """Differentiate wrt amplitude."""
        return arr / theta[:m]

    def pd(arr):
        """Differentiate wrt phase."""
        return 1j * arr

    def f1d(arr):
        """Differentiate wrt frequency 1."""
        return np.einsum(
            'ijk,i->ijk', arr, 1j * 2 * np.pi * tp[0]
        )

    def f2d(arr):
        """Differentiate wrt frequency 2."""
        return np.einsum(
            'ijk,j->ijk', arr, 1j * 2 * np.pi * tp[1]
        )

    def d1d(arr):
        """Differentiate wrt damping factor 1."""
        return np.einsum('ijk,i->ijk', arr, -tp[0])

    def d2d(arr):
        """Differentiate wrt damping factor 2."""
        return np.einsum('ijk,j->ijk', arr, -tp[1])

    if 0 in idx:
        d1[..., :m] = ad(model_per_osc)  # a

        if 1 in idx:
            d1[..., m:2 * m] = pd(model_per_osc)  # φ

            if 2 in idx:
                d1[..., 2 * m:3 * m] = f1d(model_per_osc)  # f1
                d1[..., 3 * m:4 * m] = f2d(model_per_osc)  # f2

                if 4 in idx:
                    # ---a φ f1 f2 η1 η2--- (all parameters)
                    d1[..., 4 * m:5 * m] = d1d(model_per_osc)  # η1
                    d1[..., 5 * m:] = d2d(model_per_osc)       # η2

                # ---a φ f1 f2---

            elif 4 in idx:
                # ---a φ η1 η2---
                d1[..., 2 * m:3 * m] = d1d(model_per_osc)  # η1
                d1[..., 3 * m:] = d2d(model_per_osc)       # η2

            # ---a φ---

        elif 2 in idx:
            d1[..., m:2 * m] = f1d(model_per_osc)      # f1
            d1[..., 2 * m:3 * m] = f2d(model_per_osc)  # f2

            if 4 in idx:
                # ---a f1 f2 η1 η2---
                d1[..., 3 * m:4 * m] = d1d(model_per_osc)  # η1
                d1[..., 4 * m:] = d2d(model_per_osc)       # η2

            # ---a f1 f2---

        elif 4 in idx:
            # ---a η1 η2---
            d1[..., m:2 * m] = d1d(model_per_osc)  # η1
            d1[..., 2 * m:] = d2d(model_per_osc)   # η2

        # ---a only---

    elif 1 in idx:
        d1[..., :m] = pd(model_per_osc)  # φ

        if 2 in idx:
            d1[..., m:2 * m] = f1d(model_per_osc)      # f1
            d1[..., 2 * m:3 * m] = f2d(model_per_osc)  # f2

            if 4 in idx:
                # ---φ f1 f2 η1 η2---
                d1[..., 3 * m:4 * m] = d1d(model_per_osc)  # η1
                d1[..., 4 * m:] = d2d(model_per_osc)       # η2

            # ---φ f1 f2---

        elif 4 in idx:
            # ---φ η1 η2---
            d1[..., m:2 * m] = d1d(model_per_osc)  # η1
            d1[..., 2 * m:] = d2d(model_per_osc)   # η2

        # ---φ only---

    elif 2 in idx:
        d1[..., :m] = f1d(model_per_osc)       # f1
        d1[..., m:2 * m] = f2d(model_per_osc)  # f2

        if 4 in idx:
            # ---f1 f2 η1 η2---
            d1[..., 2 * m:3 * m] = d1d(model_per_osc)  # η1
            d1[..., 3 * m:] = d2d(model_per_osc)       # η2

        # ---f1 f2 only---

    else:
        # ---η1 η2 only---
        d1[..., :m] = d1d(model_per_osc)  # η1
        d1[..., m:] = d2d(model_per_osc)  # η2

    diff = data - np.einsum('ijk->ij', model_per_osc)
    grad = -2 * np.real(
        np.einsum(
            'iik->k',
            np.einsum(
                'li,ljk->ijk',
                diff.conj(),
                d1
            )
        )
    )

    if phasevar:
        i = 1 if 0 in idx else 0
        phases = theta[i * m:(i + 1) * m]
        mu = np.einsum('i->', phases) / m
        grad[i * m:(i + 1) * m] += 2 * (phases - mu) / (m * np.pi)

    return grad


def h_2d(active: np.ndarray, *args: args_type) -> np.ndarray:
    """Hessian of cost function for 2D data.

    Parameters
    ----------
    active
        Array of active parameters (parameters to be optimised).

    args
        Contains elements in the following order:
        * **data:** Array of the original FID data.
        * **tp:** The time-points the signal was sampled at
        * **m:** Number of oscillators
        * **passive:** Passive parameters (not to be optimised).
        * **idx:** Indicates the types of parameters are present
          in the active oscillators (para_act): ``0`` - amplitudes, ``1`` -
          phases, ``2`` - frequencies, ``3`` - damping factors.
        * **phase_variance:** If True, include the oscillator phase
          variance to the cost function.
​
    Returns
    -------
    hess
        Hessian of cost function, with ``hess.shape = (6*M, 6*M)``.
    """
    # unpack arguments
    data, tp, m, passive, idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, idx)

    Y = np.exp(np.outer(
        tp[0], 1j * 2 * np.pi * theta[2 * m:3 * m] - theta[4 * m:5 * m]
    ))
    Z = np.exp(np.outer(
        1j * 2 * np.pi * theta[3 * m:4 * m] - theta[5 * m:6 * m], tp[1]
    ))
    alpha = theta[:m] * np.exp(1j * theta[m:2 * m])
    model_per_osc = np.einsum(
        'ik,kj->ijk', Y, np.einsum('ij,i->ij', Z, alpha)
    )

    # Note ordering of 2nd derivative blocks:
    # aa   aφ   af1  af2  aη1  aη2  φφ   φf1  φf2  φη1  φη2  f1f1 f1f2 ...
    # 0    1    2    3    4    5    6    7    8    9    10   11   12   ...

    # ...  f1η1 f1η2 f2f2 f2η1 f2η2 η1η1 η1η2 η2η2
    # ...  13   14   15   16   17   18   19   20

    p = len(idx)

    # first derivative components
    d1 = np.zeros((*data.shape, m * p), dtype='complex')

    # int((p * (p + 1)) / 2) --> p-th triangle number
    # gives array of:
    # --> nx21m if all oscs are to be optimised
    # --> nx15m if one type is passive
    # --> nx10m if two types are passive
    # --> nx6m if three types are passive
    # --> nx3m if four types are passive
    # --> nxm if five types are passive
    d2 = np.zeros((*data.shape, m * int((p * (p + 1)) / 2)), dtype='complex')

    # ================================================================
    # This section computes all the first and second derivatives.
    # It is able to handle all parameter types, or any subset of
    # parameter types.
    # At points in the code denoted ---x y z---, the parameters to be
    # optimised has been established, and the parameters are stated:
    # a: amplitudes
    # φ: phases
    # f1 f2: frequencies in dims 1 and 2
    # η1 η2: damping factors in dims 1 and 2
    # ================================================================

    def ad(arr):
        """Differentiate wrt amplitude."""
        return arr / theta[:m]

    def pd(arr):
        """Differentiate wrt phase."""
        return 1j * arr

    def f1d(arr):
        """Differentiate wrt frequency 1."""
        return np.einsum(
            'ijk,i->ijk', arr, 1j * 2 * np.pi * tp[0]
        )

    def f2d(arr):
        """Differentiate wrt frequency 2."""
        return np.einsum(
            'ijk,j->ijk', arr, 1j * 2 * np.pi * tp[1]
        )

    def d1d(arr):
        """Differentiate wrt damping factor 1."""
        return np.einsum('ijk,i->ijk', arr, -tp[0])

    def d2d(arr):
        """Differentiate wrt damping factor 2."""
        return np.einsum('ijk,j->ijk', arr, -tp[1])

    if 0 in idx:
        d1[..., :m] = ad(model_per_osc)  # a
        # (a-a is trivially zero)

        if 1 in idx:
            d1[..., m:2 * m] = pd(model_per_osc)  # φ
            d2[..., m:2 * m] = pd(d1[..., :m])    # a-φ

            if 2 in idx:
                d1[..., 2 * m:3 * m] = f1d(model_per_osc)        # f1
                d1[..., 3 * m:4 * m] = f2d(model_per_osc)        # f2
                d2[..., 2 * m:3 * m] = ad(d1[..., 2 * m:3 * m])  # a-f1
                d2[..., 3 * m:4 * m] = ad(d1[..., 3 * m:4 * m])  # a-f2

                if 4 in idx:
                    # ---a φ f1 f2 η1 η2--- (all parameters)
                    d1[..., 4 * m:5 * m] = d1d(model_per_osc)           # η1
                    d1[..., 5 * m:] = d2d(model_per_osc)                # η2
                    d2[..., 4 * m:5 * m] = ad(d1[..., 4 * m:5 * m])     # a-η1
                    d2[..., 5 * m:6 * m] = ad(d1[..., 5 * m:])          # a-η2
                    d2[..., 6 * m:7 * m] = pd(d1[..., m:2 * m])         # φ-φ
                    d2[..., 7 * m:8 * m] = pd(d1[..., 2 * m:3 * m])     # φ-f1
                    d2[..., 8 * m:9 * m] = pd(d1[..., 3 * m:4 * m])     # φ-f2
                    d2[..., 9 * m:10 * m] = pd(d1[..., 4 * m:5 * m])    # φ-η1
                    d2[..., 10 * m:11 * m] = pd(d1[..., 5 * m:])        # φ-η2
                    d2[..., 11 * m:12 * m] = f1d(d1[..., 2 * m:3 * m])  # f1-f1
                    d2[..., 12 * m:13 * m] = f1d(d1[..., 3 * m:4 * m])  # f1-f2
                    d2[..., 13 * m:14 * m] = f1d(d1[..., 4 * m:5 * m])  # f1-η1
                    d2[..., 14 * m:15 * m] = f1d(d1[..., 5 * m:])       # f1-η2
                    d2[..., 15 * m:16 * m] = f2d(d1[..., 3 * m:4 * m])  # f2-f2
                    d2[..., 16 * m:17 * m] = f2d(d1[..., 4 * m:5 * m])  # f2-η1
                    d2[..., 17 * m:18 * m] = f2d(d1[..., 5 * m:])       # f2-η2
                    d2[..., 18 * m:19 * m] = d1d(d1[..., 4 * m:5 * m])  # η1-η1
                    d2[..., 19 * m:20 * m] = d1d(d1[..., 5 * m:])       # η1-η2
                    d2[..., 20 * m:21 * m] = d2d(d1[..., 5 * m:])       # η2-η2

                else:
                    # ---a φ f1 f2---
                    d2[..., 4 * m:5 * m] = pd(d1[..., m:2 * m])       # φ-φ
                    d2[..., 5 * m:6 * m] = pd(d1[..., 2 * m:3 * m])   # φ-f1
                    d2[..., 6 * m:7 * m] = pd(d1[..., 3 * m:])        # φ-f2
                    d2[..., 7 * m:8 * m] = f1d(d1[..., 2 * m:3 * m])  # f1-f1
                    d2[..., 8 * m:9 * m] = f1d(d1[..., 3 * m:])       # f1-f2
                    d2[..., 9 * m:] = f2d(d1[..., 3 * m:])            # f1-f2

            elif 4 in idx:
                # ---a φ η1 η2---
                d1[..., 2 * m:3 * m] = d1d(model_per_osc)         # η1
                d1[..., 3 * m:] = d2d(model_per_osc)              # η2
                d2[..., 2 * m:3 * m] = ad(d1[..., 2 * m:3 * m])   # a-η1
                d2[..., 3 * m:4 * m] = ad(d1[..., 3 * m:])        # a-η2
                d2[..., 4 * m:5 * m] = pd(d1[..., m:2 * m])       # φ-φ
                d2[..., 5 * m:6 * m] = pd(d1[..., 2 * m:3 * m])   # φ-η1
                d2[..., 6 * m:7 * m] = pd(d1[..., 3 * m:])        # φ-η2
                d2[..., 7 * m:8 * m] = d1d(d1[..., 2 * m:3 * m])  # η1-η1
                d2[..., 8 * m:9 * m] = d1d(d1[..., 3 * m:])       # η1-η2
                d2[..., 9 * m:] = d2d(d1[..., 3 * m:])            # η2-η2

            else:
                # ---a φ---
                d2[..., 2 * m:] = pd(d1[..., m:])    # φ-φ

        elif 2 in idx:
            d1[..., m:2 * m] = f1d(model_per_osc)            # f1
            d1[..., 2 * m:3 * m] = f2d(model_per_osc)        # f2
            d2[..., m:2 * m] = ad(d1[..., m:2 * m])          # a-f1
            d2[..., 2 * m:3 * m] = ad(d1[..., 2 * m:3 * m])  # a-f2

            if 4 in idx:
                # ---a f1 f2 η1 η2---
                d1[..., 3 * m:4 * m] = d1d(model_per_osc)           # η1
                d1[..., 4 * m:5 * m] = d2d(model_per_osc)           # η2
                d2[..., 3 * m:4 * m] = ad(d1[..., 3 * m:4 * m])     # a-η1
                d2[..., 4 * m:5 * m] = ad(d1[..., 4 * m:5 * m])     # a-η2
                d2[..., 5 * m:6 * m] = f1d(d1[..., m:2 * m])        # f1-f1
                d2[..., 6 * m:7 * m] = f1d(d1[..., 2 * m:3 * m])    # f1-f2
                d2[..., 7 * m:8 * m] = f1d(d1[..., 3 * m:4 * m])    # f1-η1
                d2[..., 8 * m:9 * m] = f1d(d1[..., 4 * m:])         # f1-η2
                d2[..., 9 * m:10 * m] = f2d(d1[..., 2 * m:3 * m])   # f2-f2
                d2[..., 10 * m:11 * m] = f2d(d1[..., 3 * m:4 * m])  # f2-η1
                d2[..., 11 * m:12 * m] = f2d(d1[..., 4 * m:])       # f2-η2
                d2[..., 12 * m:13 * m] = d1d(d1[..., 3 * m:4 * m])  # η1-η1
                d2[..., 13 * m:14 * m] = d1d(d1[..., 4 * m:])       # η1-η2
                d2[..., 14 * m:15 * m] = d2d(d1[..., 4 * m:])       # η2-η2

            else:
                # ---a f1 f2---
                d2[..., 3 * m:4 * m] = f1d(d1[..., m:2 * m])  # f1-f1
                d2[..., 4 * m:5 * m] = f1d(d1[..., 2 * m:])   # f1-f2
                d2[..., 5 * m:] = f2d(d1[..., 2 * m:])        # f2-f2

        elif 4 in idx:
            # ---a η1 η2---
            d1[..., m:2 * m] = d1d(model_per_osc)         # η1
            d1[..., 2 * m:] = d2d(model_per_osc)          # η2
            d2[..., m:2 * m] = ad(d1[..., m:2 * m])       # a-η1
            d2[..., 2 * m:3 * m] = ad(d1[..., 2 * m:])    # a-η2
            d2[..., 3 * m:4 * m] = d1d(d1[..., m:2 * m])  # η1-η1
            d2[..., 4 * m:5 * m] = d1d(d1[..., 2 * m:])   # η1-η2
            d2[..., 5 * m:] = d2d(d1[..., 2 * m:])        # η2-η2

        # ---a only---

    elif 1 in idx:
        d1[..., :m] = pd(model_per_osc)  # φ
        d2[..., :m] = pd(d1[..., :m])    # φ-φ

        if 2 in idx:
            d1[..., m:2 * m] = f1d(model_per_osc)            # f1
            d1[..., 2 * m:3 * m] = f2d(model_per_osc)        # f2
            d2[..., m:2 * m] = pd(d1[..., m:2 * m])          # φ-f1
            d2[..., 2 * m:3 * m] = pd(d1[..., 2 * m:3 * m])  # φ-f2

            if 4 in idx:
                # ---φ f1 f2 η1 η2---
                d1[..., 3 * m:4 * m] = d1d(model_per_osc)           # η1
                d1[..., 4 * m:] = d2d(model_per_osc)                # η2
                d2[..., 3 * m:4 * m] = pd(d1[..., 3 * m:4 * m])    # φ-η1
                d2[..., 4 * m:5 * m] = pd(d1[..., 4 * m:])          # φ-η2
                d2[..., 5 * m:6 * m] = f1d(d1[..., m:2 * m])        # f1-f1
                d2[..., 6 * m:7 * m] = f1d(d1[..., 2 * m:3 * m])    # f1-f2
                d2[..., 7 * m:8 * m] = f1d(d1[..., 3 * m:4 * m])    # f1-η1
                d2[..., 8 * m:9 * m] = f1d(d1[..., 4 * m:])         # f1-η2
                d2[..., 9 * m:10 * m] = f2d(d1[..., 2 * m:3 * m])   # f2-f2
                d2[..., 10 * m:11 * m] = f2d(d1[..., 3 * m:4 * m])  # f2-η1
                d2[..., 11 * m:12 * m] = f2d(d1[..., 4 * m:])       # f2-η2
                d2[..., 12 * m:13 * m] = d1d(d1[..., 3 * m:4 * m])  # η1-η1
                d2[..., 13 * m:14 * m] = d1d(d1[..., 4 * m:])       # η1-η2
                d2[..., 14 * m:15 * m] = d2d(d1[..., 4 * m:])       # η2-η2

            else:
                # ---φ f1 f2---
                d2[..., 3 * m:4 * m] = f1d(d1[..., m:2 * m])  # f1-f1
                d2[..., 4 * m:5 * m] = f1d(d1[..., 2 * m:])   # f1-f2
                d2[..., 5 * m:] = f2d(d1[..., 2 * m:])        # f2-f2

        elif 4 in idx:
            # ---φ η1 η2---
            d1[..., m:2 * m] = d1d(model_per_osc)         # η1
            d1[..., 2 * m:] = d2d(model_per_osc)          # η2
            d2[..., m:2 * m] = pd(d1[..., m:2 * m])       # φ-η1
            d2[..., 2 * m:3 * m] = pd(d1[..., 2 * m:])    # φ-η2
            d2[..., 3 * m:4 * m] = d1d(d1[..., m:2 * m])  # η1-η1
            d2[..., 4 * m:5 * m] = d1d(d1[..., 2 * m:])   # η1-η2
            d2[..., 5 * m:] = d2d(d1[..., 2 * m:])        # η2-η2

        # ---φ only---

    elif 2 in idx:
        d1[..., :m] = f1d(model_per_osc)              # f1
        d1[..., m:2 * m] = f2d(model_per_osc)         # f2
        d2[..., :m] = f1d(d1[..., :m])                # f1-f1
        d2[..., m:2 * m] = f1d(d1[..., m:2 * m])      # f1-f2

        if 4 in idx:
            # ---f1 f2 η1 η2---
            d1[..., 2 * m:3 * m] = d1d(model_per_osc)         # η1
            d1[..., 3 * m:] = d2d(model_per_osc)              # η2
            d2[..., 2 * m:3 * m] = f1d(d1[..., 2 * m:3 * m])  # f1-η1
            d2[..., 3 * m:4 * m] = f1d(d1[..., 3 * m:])       # f1-η2
            d2[..., 4 * m:5 * m] = f2d(d1[..., m:2 * m])      # f2-f2
            d2[..., 5 * m:6 * m] = f2d(d1[..., 2 * m:3 * m])  # f2-η1
            d2[..., 6 * m:7 * m] = f2d(d1[..., 3 * m:])       # f2-η2
            d2[..., 7 * m:8 * m] = d1d(d1[..., 2 * m:3 * m])  # η1-η1
            d2[..., 8 * m:9 * m] = d1d(d1[..., 3 * m:])       # η1-η2
            d2[..., 9 * m:] = d2d(d1[..., 3 * m:])            # η2-η2

        else:
            # ---f1 f2---
            d2[..., 2 * m:] = f2d(d1[..., m:])  # f2-f2

    else:
        # ---η1 η2---
        d1[..., :m] = d1d(model_per_osc)     # η1
        d1[..., m:] = d2d(model_per_osc)     # η2
        d2[..., :m] = d1d(d1[..., :m])       # η1-η1
        d2[..., m:2 * m] = d1d(d1[..., m:])  # η1-η2
        d2[..., 2 * m:] = d2d(d1[..., m:])   # η2-η2

    diff = data - np.einsum('ijk->ij', model_per_osc)
    diagonals = -2 * np.real(np.einsum('jki,jk->i', d2.conj(), diff))

    # determine indices of elements which have non-zero second derivs
    # (specfically, those in upper triangle)
    diag_indices = _generate_diagonal_indices(p, m)
    hess = np.zeros((p * m, p * m))
    hess[diag_indices] = diagonals

    main_diagonals = _diagonal_indices(hess, k=0)
    # division by 2 to ensure elements on main diagonal aren't doubled
    # after transposition
    hess[main_diagonals] = hess[main_diagonals] / 2

    # transpose (hessian is symmetric)
    hess += hess.T

    # add component containing first derivatives
    hess += 2 * np.real(np.einsum('kli,klj->ij', d1.conj(), d1))

    if phasevar:
        # if 0 in idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in idx else 0
        hess[i * m:(i + 1) * m, i * m:(i + 1) * m] -= (2 / (m ** 2 * np.pi))
        hess[main_diagonals[0][i * m:(i + 1) * m],
             main_diagonals[1][i * m:(i + 1) * m]] \
            += 2 / (np.pi * m)

    return hess


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
            params[i * m:(i + 1) * m] = active[:m]
            active = active[m:]
        else:
            params[i * m:(i + 1) * m] = passive[:m]
            passive = passive[m:]

    return params


def _generate_diagonal_indices(
    p: int, m: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    arr = np.zeros((p * m, p * m))  # dummy array with same shape as Hessian
    idx_0 = []  # axis 0 indices (rows)
    idx_1 = []  # axis 1 indices (columns)
    for i in range(p):
        for j in range(p - i):
            idx_0.append(_diagonal_indices(arr, k=j * m)[0][i * m:(i + 1) * m])
            idx_1.append(_diagonal_indices(arr, k=j * m)[1][i * m:(i + 1) * m])

    return np.hstack(idx_0), np.hstack(idx_1)


def _diagonal_indices(
    arr: np.ndarray, k: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the indices of an array's kth diagonal.

    A generalisation of `numpy.diag_indices_from <https://numpy.org/doc\
    /stable/reference/generated/numpy.diag_indices_from.html>`_,
    which can only be used to obtain the indices along the main diagonal of
    an array.

    Parameters
    ----------
    arr
        Square array (Hessian matrix)

    k
        Displacement from the main diagonal

    Returns
    -------
    rows
        0-axis coordinates of indices.

    cols
        1-axis coordinates of indices.
    """
    rows, cols = np.diag_indices_from(arr)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols
