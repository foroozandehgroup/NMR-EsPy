# nlp.funcs.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Definitions of fidelities, gradients, and Hessians."""

import numpy as np


def f_1d(active, *args):
    """
    Cost function for 1D data

    Parameters
    ----------
    active : numpy.ndarray
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** `numpy.ndarray.` Array of the original FID data.
        * **tp:** `numpy.ndarray.` The time-points the signal was sampled at
        * **m:** `int.` Number of oscillators
        * **passive:** `numpy.ndarray.` Passive parameters (not to be
          optimised).
        * **active_idx:** `list.` Indicates the types of parameters are present
          in the active oscillators (para_act): ``0`` - amplitudes, ``1`` -
          phases, ``2`` - frequencies, ``3`` - damping factors.
        * **phase_variance:** `Bool.` If True, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    func : float
        Value of the cost function
    """

    # unpack arguments
    data, tp, m, passive, active_idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, active_idx)

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
        # if 0 in active_idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        phases = theta[m:2 * m] if 0 in active_idx else theta[:m]
        mu = np.einsum('i->', phases) / m
        func += ((np.einsum('i->', phases ** 2) / m) - mu ** 2) / np.pi

    return func


def g_1d(active, *args):
    """
    Gradient of cost function for 1D data

    Parameters
    ----------
    active : numpy.ndarray
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** `numpy.ndarray.` Array of the original FID data.
        * **tp:** `numpy.ndarray.` The time-points the signal was sampled at
        * **m:** `int.` Number of oscillators
        * **passive:** `numpy.ndarray.` Passive parameters (not to be
          optimised).
        * **active_idx:** `list.` Indicates the types of parameters are present
          in the active oscillators (para_act): ``0`` - amplitudes, ``1`` -
          phases, ``2`` - frequencies, ``3`` - damping factors.
        * **phase_variance:** `Bool.` If True, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    grad : numpy.ndarray
        Gradient of cost function, with ``grad.shape = (4 * m,)``.
    """

    # unpack arguments
    data, tp, m, passive, active_idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, active_idx)

    # signal pole matrix
    Z = np.exp(
        np.outer(
            tp[0], (1j * 2 * np.pi * theta[2 * m:3 * m] - theta[3 * m:])
        )
    )

    # N x M array comprising M N-length vectors
    # each vector is the model produced by a single oscillator
    model_per_oscillator = Z * (theta[:m] * np.exp(1j * theta[m:2 * m]))

    # first derivative components
    fd = np.zeros((*data.shape, m * len(active_idx)), dtype='complex')

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

    if 2 in active_idx:
        def freq_derivative(arr):
            return np.einsum('ij,i->ij', arr, (1j * 2 * np.pi * tp[0]))

    if 3 in active_idx:
        def damp_derivative(arr):
            return np.einsum('ij,i->ij', arr, -tp[0])

    if 0 in active_idx:
        fd[:, :m] = model_per_oscillator / theta[:m]  # a

        if 1 in active_idx:
            fd[:, m:2 * m] = 1j * model_per_oscillator  # φ

            if 2 in active_idx:
                fd[:, 2 * m:3 * m] = freq_derivative(model_per_oscillator)  # f

                if 3 in active_idx:
                    # ---a φ f η--- (all parameters)
                    fd[:, 3 * m:] = damp_derivative(model_per_oscillator)  # η

                # ---a φ f---

            elif 3 in active_idx:
                # ---a φ η---
                fd[:, 2 * m:] = damp_derivative(model_per_oscillator)  # η

            # ---a φ---

        elif 2 in active_idx:
            fd[:, m:2 * m] = freq_derivative(model_per_oscillator)  # f

            if 3 in active_idx:
                # ---a f η---
                fd[:, 2 * m:] = damp_derivative(model_per_oscillator)  # η

            # ---a f---

        elif 3 in active_idx:
            # ---a η---
            fd[:, m:] = damp_derivative(model_per_oscillator)  # η

        # ---a only---

    elif 1 in active_idx:
        fd[:, :m] = 1j * model_per_oscillator  # φ

        if 2 in active_idx:
            fd[:, m:2 * m] = freq_derivative(model_per_oscillator)  # f

            if 3 in active_idx:
                # ---φ f η---
                fd[:, 2 * m:] = damp_derivative(model_per_oscillator)  # η

            # ---φ f---

        elif 3 in active_idx:
            # ---φ η---
            fd[:, m:] = damp_derivative(model_per_oscillator)  # η

        # ---φ only---

    elif 2 in active_idx:
        fd[:, :m] = freq_derivative(model_per_oscillator)  # f

        if 3 in active_idx:
            # ---f η---
            fd[:, m:] = damp_derivative(model_per_oscillator)  # η

        # ---f only---

    else:
        # ---η only---
        fd[:, :] = damp_derivative(model_per_oscillator)  # η

    # sum model_per_oscillator to generate data-length vector,
    # and take residual
    diff = data - np.einsum('ij->i', model_per_oscillator)
    grad = -2 * np.real(fd.conj().T @ diff)

    if phasevar:
        # if 0 in active_idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in active_idx else 0
        phases = theta[i * m:(i + 1) * m]
        mu = np.einsum('i->', phases) / m
        grad[i * m:(i + 1) * m] += ((2 / m) * (phases - mu)) / np.pi

    return grad


def h_1d(active, *args):
    """
    Hessian of cost function for 1D data

    Parameters
    ----------
    active : numpy.ndarray
        Array of active parameters (parameters to be optimised).

    args : list_iterator
        Contains elements in the following order:

        * **data:** `numpy.ndarray.` Array of the original FID data.
        * **tp:** `numpy.ndarray.` The time-points the signal was sampled at
        * **m:** `int.` Number of oscillators
        * **passive:** `numpy.ndarray.` Passive parameters (not to be
          optimised).
        * **active_idx:** `list.` Indicates the types of parameters are present
          in the active oscillators (para_act): ``0`` - amplitudes, ``1`` -
          phases, ``2`` - frequencies, ``3`` - damping factors.
        * **phase_variance:** `Bool.` If True, include the oscillator phase
          variance to the cost function.

    Returns
    -------
    hess : numpy.ndarray
        Hessian of cost function, with ``hess.shape = (4 * m,4 * m)``.
    """

    # unpack arguments
    data, tp, m, passive, active_idx, phasevar = args

    # reconstruct correctly ordered parameter vector from
    # active and passive parameters.
    theta = _construct_parameters(active, passive, m, active_idx)

    # signal pole matrix
    Z = np.exp(
        np.outer(
            tp[0], (1j * 2 * np.pi * theta[2 * m:3 * m] - theta[3 * m:])
        )
    )

    # N x M array comprising M N-length vectors
    # each vector is the model produced by a single oscillator
    model_per_oscillator = Z * (theta[:m] * np.exp(1j * theta[m:2 * m]))

    p = len(active_idx)

    # first derivative components
    fd = np.zeros((*data.shape, m * p), dtype='complex')

    # int((p * (p + 1)) / 2) --> p-th triangle number
    # gives array of:
    # --> nx10m if all oscs are to be optimised
    # --> nx6m if one type is passive
    # --> nx3m if two types are passive
    # --> nxm if three types are passive
    sd = np.zeros((*data.shape, m * int((p * (p + 1)) / 2)), dtype='complex')

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

    # functions for taking derivatives wrt amp, phase, freq, damping
    if 0 in active_idx:
        def amp_derivative(arr):
            return arr / theta[:m]

    if 1 in active_idx:
        def phase_derivative(arr):
            return 1j * arr

    if 2 in active_idx:
        def freq_derivative(arr):
            return np.einsum('ij,i->ij', arr, (1j * 2 * np.pi * tp[0]))

    if 3 in active_idx:
        def damp_derivative(arr):
            return np.einsum('ij,i->ij', arr, -tp[0])

    if 0 in active_idx:
        fd[:, :m] = amp_derivative(model_per_oscillator)  # a
        # (a-a is trivially zero)

        if 1 in active_idx:
            fd[:, m:2 * m] = phase_derivative(model_per_oscillator)  # φ
            sd[:, m:2 * m] = phase_derivative(fd[:, :m])  # a-φ

            if 2 in active_idx:
                fd[:, 2 * m:3 * m] = freq_derivative(model_per_oscillator)  # f
                sd[:, 2 * m:3 * m] = amp_derivative(fd[:, 2 * m:3 * m])  # a-f

                if 3 in active_idx:
                    # ---a φ f η---
                    fd[:, 3 * m:] = \
                        damp_derivative(model_per_oscillator)  # η
                    sd[:, 3 * m:4 * m] = \
                        amp_derivative(fd[:, 3 * m:4 * m])  # a-η
                    sd[:, 4 * m:5 * m] = \
                        phase_derivative(fd[:, :m])  # φ-φ
                    sd[:, 5 * m:6 * m] = \
                        phase_derivative(fd[:, 2 * m:3 * m])  # φ-f
                    sd[:, 6 * m:7 * m] = \
                        phase_derivative(fd[:, 3 * m:4 * m])  # φ-η
                    sd[:, 7 * m:8 * m] = \
                        freq_derivative(fd[:, 2 * m:3 * m])  # f-f
                    sd[:, 8 * m:9 * m] = \
                        damp_derivative(fd[:, 2 * m:3 * m])  # f-η
                    sd[:, 9 * m:] = \
                        damp_derivative(fd[:, 3 * m:])  # η-η

                else:
                    # ---a φ f---
                    sd[:, 3 * m:4 * m] = phase_derivative(fd[:, :m])  # φ-φ
                    sd[:, 4 * m:5 * m] = phase_derivative(fd[:, 2 * m:])  # φ-f
                    sd[:, 5 * m:] = freq_derivative(fd[:, 2 * m:])  # f-f

            elif 3 in active_idx:
                # ---a φ η---
                fd[:, 2 * m:] = damp_derivative(model_per_oscillator)  # η
                sd[:, 2 * m:3 * m] = amp_derivative(fd[:, 2 * m:])  # a-η
                sd[:, 3 * m:4 * m] = phase_derivative(fd[:, :m])  # φ-φ
                sd[:, 4 * m:5 * m] = phase_derivative(fd[:, 2 * m:])  # φ-η
                sd[:, 5 * m:] = damp_derivative(fd[:, 2 * m:])  # η-η

            else:
                # ---a φ---
                sd[:, 2 * m:] = phase_derivative(fd[:, m:])  # φ-φ

        elif 2 in active_idx:
            fd[:, m:2 * m] = freq_derivative(model_per_oscillator)  # f
            sd[:, m:2 * m] = amp_derivative(fd[:, m:2 * m])  # a-f

            if 3 in active_idx:
                # ---a f η---
                fd[:, 2 * m:] = damp_derivative(model_per_oscillator)  # η
                sd[:, 2 * m:3 * m] = amp_derivative(fd[:, 2 * m:])  # a-η
                sd[:, 3 * m:4 * m] = freq_derivative(fd[:, m:2 * m])  # f-f
                sd[:, 4 * m:5 * m] = damp_derivative(fd[:, m:2 * m])  # f-η
                sd[:, 5 * m:] = damp_derivative(fd[:, 2 * m:])  # η-η

            else:
                # ---a f---
                sd[:, 2 * m:] = freq_derivative(fd[:, m:])  # f-f

        elif 3 in active_idx:
            # ---a η---
            fd[:, m:] = damp_derivative(model_per_oscillator)  # η
            sd[:, 2 * m:] = damp_derivative(fd[:, m:])  # η-η

        # ---a only---

    elif 1 in active_idx:
        fd[:, :m] = phase_derivative(model_per_oscillator)  # φ
        sd[:, :m] = phase_derivative(fd[:, :m])  # φ-φ

        if 2 in active_idx:
            fd[:, m:2 * m] = freq_derivative(model_per_oscillator)  # f
            sd[:, m:2 * m] = phase_derivative(fd[:, m:2 * m])  # φ-f

            if 3 in active_idx:
                # ---φ f η---
                fd[:, 2 * m:] = damp_derivative(model_per_oscillator)  # η
                sd[:, 2 * m:3 * m] = phase_derivative(fd[:, 2 * m:])  # φ-η
                sd[:, 3 * m:4 * m] = freq_derivative(fd[:, m:2 * m])  # f-f
                sd[:, 4 * m:5 * m] = damp_derivative(fd[:, m:2 * m])  # f-η
                sd[:, 5 * m:] = damp_derivative(fd[:, 2 * m:])  # η-η

            else:
                # ---φ f---
                sd[:, 2 * m:] = freq_derivative(fd[:, m:])  # f-f

        elif 3 in active_idx:
            # ---φ η---
            fd[:, m:] = damp_derivative(model_per_oscillator)  # η
            sd[:, m:2 * m] = phase_derivative(fd[:, m:])  # φ-η
            sd[:, 2 * m:] = damp_derivative(fd[:, m:])  # η-η

        # ---φ only---

    elif 2 in active_idx:
        fd[:, :m] = freq_derivative(model_per_oscillator)  # f
        sd[:, :m] = freq_derivative(fd[:, :m])  # f-f

        if 3 in active_idx:
            # ---f η---
            fd[:, m:] = damp_derivative(model_per_oscillator)  # η
            sd[:, m:2 * m] = damp_derivative(fd[:, :m])  # f-η
            sd[:, 2 * m:] = damp_derivative(fd[:, m:])  # η-η

        # ---f only---

    else:
        # ---η only---
        fd = damp_derivative(model_per_oscillator)  # η
        sd = damp_derivative(fd)  # η-η

    diff = data - np.einsum('ij->i', model_per_oscillator)

    diagonals = -2 * np.real(sd.conj().T @ diff)

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
    hess += 2 * np.real(np.einsum('ki,kj->ij', fd.conj(), fd))

    if phasevar:
        # if 0 in active_idx, phases will be between m and 2m, as amps
        # also present if not, phases will be between 0 and m
        i = 1 if 0 in active_idx else 0
        hess[i * m:(i + 1) * m, i * m:(i + 1) * m] -= (2 / (m ** 2 * np.pi))
        hess[main_diagonals[0][i * m:(i + 1) * m],
             main_diagonals[1][i * m:(i + 1) * m]] \
            += 2 / (np.pi * m)

    return hess


def _construct_parameters(active, passive, m, idx):
    """
    Constructs the full parameter vector from active and passive sub-vectors.

    Parameters
    ----------
    active : numpy.ndarray
        Active parameter vector.

    passive : numpy.ndarray
        Passive parameter vector.

    m : int
        Number of oscillators.

    idx : list
        Indicates the columns (axis 1) for active parameters.

    Returns
    -------
    parameters - numpy.ndarray
        Full parameter vector with correct ordering
    """

    # number of columns in active parameter array
    p = int((active.shape[0] + passive.shape[0]) / m)

    parameters = np.zeros(p * m)

    for i in range(p):
        if i in idx:
            parameters[i * m:(i + 1) * m] = active[:m]
            active = active[m:]
        else:
            parameters[i * m:(i + 1) * m] = passive[:m]
            passive = passive[m:]

    return parameters


def _generate_diagonal_indices(p, m):
    """
    Determines all array indicies that correspond to positions in which
    non-zero second derivatives reside, in the top right half of the Hessian.

    Parameters
    ----------
    p : int
        The number of parameter 'groups'. For an N-dimensional signal this
        will be 2N + 2

    m : int
        Number of oscillators in parameter estimate

    Returns
    -------

    idx_0 : numpy.ndarray
        0-axis coordinates of indices.

    idx_1 : numpy.ndarray
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


def _diagonal_indices(arr, k=0):
    """
    Returns the indices of an array's kth diagonal. A generalisation of
    numpy.diag_indices_from(), which can only be used to obtain the indices
    along the main diagonal of an array.

    Parameters
    ----------
    arr : numpy.ndarray
        Square array (Hessian matrix)

    k : int
        Displacement from the main diagonal

    Returns
    -------
    rows : numpy.ndarray
        0-axis coordinates of indices.

    cols : numpy.ndarray
        1-axis coordinates of indices.
    """

    rows, cols = np.diag_indices_from(arr)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols
