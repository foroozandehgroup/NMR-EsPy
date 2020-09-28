#!/usr/bin/python3
# nlp.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# module to support the NMRespyInfo.nonlinear_programming() method

from copy import deepcopy
import time

import numpy as np
from numpy.linalg import norm, inv
from scipy.optimize import minimize

from ._cols import *
if USE_COLORAMA:
    import colorama
from ._timing import _print_time


def nlp(data, dim, theta0, sw, off, phase_variance, method, mode, bound, maxit,
        amp_thold, freq_thold, fprint, first_call, start):
    """
    nlp.nlp(data, dim, theta0, sw, offset, trim, phase_variance, method, mode, bound,
            maxit, amp_thold, freq_thold, fprint, first_call)

    Description
    ———————————
    A nonlinear programming routine to generate signal parameter estimates.

    Parameters
    ——————————
    data - numpy.ndarray
        The data to analyse (not normalised).
    dim - int
        The signal dimension. Will be 1 or 2.
    theta0 - numpy.ndarray
        Initial parameter guess, with shape (M, 4) for 1D data or (M, 6)
        for 2D data.
    sw - tuple
        The sweep width (Hz) in each dimension.
    off - tuple
        The transmitter offset frequency (Hz) in each dimension.
    trim - tuple
        Specifies the number of initial data points to consider in
        each dimension.
    phase_variance - Bool
        Specifies whether or not to include the variance of oscillator
        phases into the NLP routine
    method - str
        Optimisation routine. Should be 'trust_region' or 'lbfgs'
    mode - str
        String composed of any combination of characters 'a', 'p', 'f', 'd'.
        Used to determine which parameter types to optimise, and which to
        fix.
    bound - Bool
        Specifies whether or not to bound the parameters during optimisation.
    maxit - int
        A value specifiying the number of iterations the routine may run
        through before it is terminated. The estimate a termination will be
        returned.
    amp_thold - float or None
        A threshold such that any oscillators have amplitudes lower than
        this, they will be deleted. See _rm_negligible_amps() for more
        information.
    freq_thold - float or None
        A threshold such that any pair of oscillators that have frequencies
        with a difference smaller than it, they will be merged.
    first_call - Bool
        Specifies whether nlp() has been called for the first time in a
        recursive loop. Used just for printing terminal output.
    start - float or None
        Time at which 1st call to nlp() started.

    Returns
    ———————
    x - numpy.ndarray
        The result of the NLP routine, with shape (M_new, 4) for 1D data or
        (M_new, 6) for 2D data, where M_new ≤ M.
    """

    if first_call:
        if fprint:
            start = time.time()
            print(f'=============================\n'
                  f'{G}Nonlinear Programming Started{END}\n'
                  f'=============================')

    # normalise data
    nm = norm(data)
    data_n = data / nm
    theta0_cor = _correct_freqs(theta0, off) # correct freqs (center at 0)
    theta0_cor[..., 0] = theta0_cor[..., 0] / nm # scale amplitudes
    M = theta0_cor.shape[0] # nuber of oscillators (M)

    # time-points
    n = data_n.shape # number of points in data
    tp = []
    for s, n_ in zip(sw, n):
        tp.append(np.linspace(0, float(n_-1)/s, n_))
    tp = tuple(tp)

    # based on mode, determine array of parameters to optimise (active),
    # and array of parameters to fix (passive).
    idx_act, idx_pas = _get_mode_indices(mode, dim)
    theta0_act = theta0_cor[..., idx_act]
    theta0_pas = theta0_cor[..., idx_pas]
    p_act = len(idx_act) # number of active paraemters per oscillator
    p_pas = len(idx_pas) # number of passive paraemters per oscillator

    # reshape passive and active parameter arrays as veectors
    # Fortran (columnwise) style ordering
    theta0_act = np.reshape(theta0_act, (p_act * M), order='F')
    theta0_pas = np.reshape(theta0_pas, (p_pas * M), order='F')

    # generate optimiser boundary conditions
    bounds = None
    if bound:
        bounds = _get_bounds(M, sw, off, idx_act)


    # label for level of optimiser verboseness
    fp = 0 # no terminal output from scipy.minimize()
    if fprint:
        fp = 3 # rather verbose terminal output from scipy.minimize()

    # arguments to pass to cost funct, grad, and hessian
    opt_args = (data_n, tp, M, theta0_pas, idx_act, phase_variance)

    # callables for cost function, grad and hessian
    if dim == 1:
        cf = {'f' : _f_1d, 'g' : _g_1d, 'h' : _h_1d}
    elif dim == 2:
        cf = {'f' : _f_2d, 'g' : _g_2d, 'h' : _h_2d}

    if method == 'trust_region':
        res = minimize(fun=cf['f'], x0=theta0_act,  args=opt_args,
                       method='trust-constr', jac=cf['g'], hess=cf['h'],
                       bounds=bounds,
                       options={'maxiter': maxit, 'verbose': fp})

    elif method == 'lbfgs':
        fp = int(fp / 3) # make fp = 1 for verbose output
        res = minimize(fun=cf['f'], x0=theta0_act,  args=opt_args,
                       method='L-BFGS-B', jac=cf['g'], bounds=bounds,
                       options={'maxiter': maxit, 'iprint': fp})

    theta_act = res['x'] # extract solution from result dict

    # # derive estimation errors
    # func = cf['f'](theta_act, data_n, tp, M, theta0_pas, idx_act, False)
    # inv_hess = inv(cf['h'](theta_act, data_n, tp, M, theta0_pas, idx_act, False))
    # se = np.sqrt(func) * np.sqrt(np.diag(inv_hess) / (np.prod(n)-1))
    # se[:M] = se[:M] * nm
    # theta_act[:M] = theta_act[:M] * nm


    theta_act = np.reshape(theta_act, (M, p_act), order='F')
    theta0_pas = np.reshape(theta0_pas, (M, p_pas), order='F')

    # construct array
    p = p_act + p_pas # total number of parameters per oscillator
    theta = np.zeros((M, p))
    theta[..., idx_act] = theta_act
    theta[..., idx_pas] = theta0_pas
    theta[..., 0] = theta[..., 0] * nm # rescale amps
    theta[..., 1] = (theta[..., 1] + np.pi) % (2 * np.pi) - np.pi # wrap phases

    # removal of negligibale amplitude oscillators
    theta = _rm_negligible_amps(theta, amp_thold, fprint)
    # correct freqs (move offset back to original position)
    theta = _correct_freqs(theta, off)
    # removal of -ve amp oscillators
    theta, term = _rm_negative_amps(theta, fprint)

    if term:
        # error estimate
        if fprint:
            finish = time.time()
            print(f'==============================\n'
                  f'{G}Nonlinear Programming Complete{END}\n'
                  f'==============================')
            _print_time(finish-start)

        return theta[np.argsort(theta[..., 2])]

    else:
        # Re-run nlp recusively until solution has no -ve amps
        return nlp(data, dim, theta, sw, off, phase_variance, method, mode,
                   bound, maxit, amp_thold, freq_thold, fprint, False, start)



def _f_1d(para_act, *args):
    """
    _f_1d(para, *args)

    Decription
    ———————————
    Determines the cost function to be minimised for non-linear
    optimisation (1D signal)
    Given by:
    (y - x(θ))†(y - x(θ)) + Var(φ)
    where y is the (normalised) data, and x is the model, θ is the
    parameter array, and φ is the oscillator phases.

    Parameters
    ——————————
    para_act - numpy.ndarray
        Array of active parameters (paramters to be optimised).
    args - list_iterator
        Additional arguments, with entries:
        data: ndarray - Array of the original FID data.
        tp: ndarray - Array of time-points the FID was sampled at.
        M: int - Number of oscillations.
        para_pas - Passive parameters (not to be optimised)
        idx: list - List indicating the types of parameters are present in the
                    active oscillators (para_act).
                    0 - amplitudes
                    1 - phases
                    2 - frequencies
                    3 - damping factors
        phase_vairance: Bool - If True, include the oscillator phase variance
                               to the cost function.

    Returns
    ———————
    func - float
        Value of the cost function
    """

    # unpack arguments
    data = args[0]
    tp = args[1][0]
    M = args[2]
    para_pas = args[3]
    idx = args[4]
    phase_variance = args[5]

    # reconstruct correctly ordered parameter vector from
    # optimisable and non-optimisable parameters.
    para = _construct_para(para_act, para_pas, M, idx)

    # determine the signal pole matrix
    Z = np.exp(np.outer(tp, (1j*2*np.pi*para[2*M:3*M] - para[3*M:])))
    model = np.matmul(Z, (para[:M] * np.exp(1j * para[M:2*M])))
    diff = model - data
    func = np.real(np.vdot(diff, diff))

    if phase_variance:
        if 0 in idx:
            mu = np.sum(para[M:2*M]) / M
            func += (np.sum(para[M:2*M] ** 2) / M) - (mu ** 2)
        else:
            mu = np.sum(para[:M]) / M
            func += (np.sum(para[:M] ** 2) / M) - (mu ** 2)

    return func


def _g_1d(para_act, *args):
    """
    _g_1d(para, *args)

    Decription
    ———————————
    Determines the gradient of the cost function to be minimised for
    non-linear optimisation (1D signal).
    i-th element given by:
    2Re[(y - x(θ))† dx(θ)/∂ᵢ] + ∂Var(φ)/∂θᵢ
    where y is the (normalised) data, and x is the model, θ is the
    parameter array, and φ is the oscillator phases.

    Parameters
    ——————————
    para_act - numpy.ndarray
        Array of active parameters (paramters to be optimised).
    args - list_iterator
        Additional arguments, with entries:
        data: ndarray - Array of the original FID data.
        tp: ndarray - Array of time-points the FID was sampled at.
        M: int - Number of oscillations.
        para_pas - Passive parameters (not to be optimised)
        idx: list - List indicating the types of parameters are present in the
                    active oscillators (para_act).
                    0 - amplitudes
                    1 - phases
                    2 - frequencies
                    3 - damping factors
        phase_vairance: Bool - If True, include the oscillator phase variance
                               to the cost function.

    Returns
    ———————
    grad - numpy.ndarray
        Gradient of the cost function.
    """

    # unpack arguments
    data = args[0]
    tp = args[1][0]
    M = args[2]
    para_pas = args[3]
    idx = args[4]
    phase_variance = args[5]

    # reconstruct correctly ordered parameter vector from
    # optimisable and non-optimisable parameters.
    para = _construct_para(para_act, para_pas, M, idx)

    Z = np.exp(np.outer(tp, (1j*2*np.pi*para[2*M:3*M] - para[3*M:4*M])))
    model = Z * (para[:M] * np.exp(1j * para[M:2*M]))

    # derivative components
    comps = []
    # derivatives
    if 0 in idx:
        comps.append(model / para[:M])
    if 1 in idx:
        comps.append(1j * model)
    if 2 in idx:
        comps.append(np.einsum('ij,i->ij', model, 1j*2*np.pi*tp))
    if 3 in idx:
        comps.append(np.einsum('ij,i->ij', model, -tp))

    deriv = np.hstack(comps)

    diff = data - np.sum(model, axis=1)
    grad = -2 * np.real(np.matmul(deriv.conj().T, diff))

    if phase_variance:
        # amplitudes are being optimised (phases between M and 2M)
        if 0 in idx:
            mu = np.sum(para[M:2*M]) / M
            grad[M:2*M] = grad[M:2*M] + ((2 / M) * (para[M:2*M] - mu))
        # amplitudes are not being optimised (phases between 0 and M)
        else:
            mu = np.sum(para[:M]) / M
            grad[:M] = grad[:M] + ((2 / M) * (para[:M] - mu))

    return grad


def _h_1d(para_act, *args):
    """
    _h_1d(para, *args)

    Decription
    ———————————
    Determines the Hessian of the cost function to be minimised for
    non-linear optimisation (1D signal)
    i,j-th element given by:
    2Re[(y - x(θ))† ∂²x(θ)/∂θᵢ∂θⱼ - ∂x(θ)†/∂θᵢ ∂x(θ)/∂θⱼ] + ∂²Var(φ)/∂θᵢ∂θⱼ
    where y is the (normalised) data, and x is the model, θ is the
    parameter array, and φ is the oscillator phases.

    Parameters
    ——————————
    para_act - numpy.ndarray
        Array of active parameters (paramters to be optimised).
    args - list_iterator
        Additional arguments, with entries:
        data: ndarray - Array of the original FID data.
        tp: ndarray - Array of time-points the FID was sampled at.
        M: int - Number of oscillations.
        para_pas - Passive parameters (not to be optimised)
        idx: list - List indicating the types of parameters are present in the
                    active oscillators (para_act).
                    0 - amplitudes
                    1 - phases
                    2 - frequencies
                    3 - damping factors
        phase_vairance: Bool - If True, include the oscillator phase variance
                               to the cost function.


    Returns
    ———————
    hess - numpy.ndarray
        Hessian of the cost function.
    """

    # unpack arguments
    data = args[0]
    tp = args[1][0]
    M = args[2]
    para_pas = args[3]
    idx = args[4]
    phase_variance = args[5]

    # reconstruct correctly ordered parameter vector from
    # optimisable and non-optimisable parameters.
    para = _construct_para(para_act, para_pas, M, idx)

    Z = np.exp(np.outer(tp, (1j*2*np.pi*para[2*M:3*M] - para[3*M:4*M])))
    model = Z * (para[:M] * np.exp(1j * para[M:2*M]))

    # first derivatives and diagonal second derivatives
    comps1 = []
    comps2 = []
    if 0 in idx:
        comps1.append(model / para[:M]) # a
        comps2.append(np.zeros(comps1[0].shape, dtype=complex)) # a-a
    if 1 in idx:
        comps1.append(1j * model) # φ
        comps2.append(1j * comps1[-1]) # φ-φ
    if 2 in idx:
        comps1.append(np.einsum('ij,i->ij', model, 1j*2*np.pi*tp)) # f
        comps2.append(np.einsum('ij,i->ij', comps1[-1], 1j*2*np.pi*tp)) # f-f
    if 3 in idx:
        comps1.append(np.einsum('ij,i->ij', model, -tp)) # η
        comps2.append(np.einsum('ij,i->ij', comps1[-1], -tp)) # η-η

    # non-digonal second derivatives
    if 0 in idx:
        if 1 in idx:
            comps2.insert(1, 1j * comps1[0]) # a-φ
            if 2 in idx:
                comps2.insert(2, comps1[2] / para[:M]) # a-f
                comps2.insert(4, 1j * comps1[2]) # φ-f
                if 3 in idx:
                    comps2.insert(3, comps1[3] / para[:M]) # a-η
                    comps2.insert(6, 1j * comps1[3]) # φ-η
                    comps2.insert(8, np.einsum('ij,i->ij', comps1[2], -tp)) # f-η
            elif 3 in idx:
                comps2.insert(2, comps1[2] / para[:M]) # a-η
                comps2.insert(4, 1j * comps1[2]) # φ-η
        elif 2 in idx:
            comps2.insert(1, comps1[1] / para[:M]) # a-f
            if 3 in idx:
                comps2.insert(2, comps1[2] / para[:M]) # a-η
                comps2.insert(4, np.einsum('ij,i->ij', comps1[1], -tp)) # f-η
        elif 3 in idx:
            comps2.insert(1, comps1[1] / para[:M]) # a-η

    elif 1 in idx:
        if 2 in idx:
            comps2.insert(1, 1j * comps1[1]) # φ-f
            if 3 in idx:
                comps2.insert(2, 1j * comps1[2]) # φ-η
                comps2.insert(4, np.einsum('ij,i->ij', comps1[1], -tp)) # f-η
        elif 3 in idx:
            comps2.insert(1, 1j * comps1[1]) # φ-η

    elif 2 in idx:
        if 3 in idx:
            comps2.insert(1, np.einsum('ij,i->ij', comps1[0], -tp)) # f-η

    deriv1 =np.hstack(comps1)
    deriv2 = np.hstack(comps2)

    diff = data - np.sum(model, axis=1)
    diags = -2 * np.real(np.matmul(deriv2.conj().T, diff))

    p = len(idx) # number of parameter types

    # determine indices of elements which have non-zero second derivs
    # (specfically, those in upper triangle)
    diag_idx_0, diag_idx_1 = _generate_diag_indices(p, M)
    hess = np.zeros((p*M, p*M))
    hess[diag_idx_0, diag_idx_1] = diags

    # division by 2 to ensure elements on main diagonal aren't doubled
    # after transposition
    hess[_diag_indices(hess, k=0)] = hess[_diag_indices(hess, k=0)] / 2
    # transose (hessian is symmetric)
    hess += hess.T
    hess += 2 * np.real(np.einsum('ki,kj->ij', deriv1.conj(), deriv1))

    if phase_variance:
        # amplitudes are being optimised (phases between M and 2M)
        if 0 in idx:
            hess[M:2*M, M:2*M] += (2/(M**2))
            hess[_diag_indices(hess, k=0)[0][M:2*M],
                 _diag_indices(hess, k=0)[1][M:2*M]] += 2 / M
        # amplitudes are not being optimised (phases between 0 and M)
        else:
            hess[:M, :M] += (2/(M**2))
            hess[_diag_indices(hess, k=0)[0][:M],
                 _diag_indices(hess, k=0)[1][:M]] += 2 / M

    return hess


def _f_2d(para_act, *args):
    """
    _f_2d(para, *args)

    Decription
    ———————————
    Determines the cost function to be minimised for
    non-linear optimisation (2D signal)
    Given by:
    ‖Y - X(θ)‖² + Var(φ)
    where Y is the (normalised) data, X is the model, θ is the
    parameter array, and φ is the oscillator phases.

    Parameters
    ——————————
    para_act - numpy.ndarray
        Array of active parameters (paramters to be optimised).
    args - list_iterator
        Additional arguments, with entries:
        data: ndarray - Array of the original FID data.
        tp: tuple - Contains 2 numpy.ndarrays with time-points the FID was
                    sampled at, in each dimesnion
        M: int - Number of oscillations.
        para_pas - Passive parameters (not to be optimised)
        idx: list - List indicating the types of parameters are present in the
                    active oscillators (para_act).
                    0 - amplitudes
                    1 - phases
                    2 & 3 - frequencies
                    4 & 5 - damping factors
        phase_vairance: Bool - If True, include the oscillator phase variance
                               to the cost function.

    Returns
    ———————
    func - float
        Value of the cost function
    """

    # unpack arguments
    data = args[0]
    tp = args[1]
    M = args[2]
    para_pas = args[3]
    idx = args[4]
    phase_variance = args[5]

    # reconstruct correctly ordered parameter vector from
    # optimisable and non-optimisable parameters.
    para = _construct_para(para_act, para_pas, M, idx)

    Y = np.exp(np.outer(tp[0], (1j*2*np.pi*para[2*M:3*M] - para[4*M:5*M])))
    Z = np.exp(np.outer((1j*2*np.pi*para[3*M:4*M] - para[5*M:6*M]), tp[1]))
    A = np.diag(para[:M] * np.exp(1j * para[M:2*M]))

    model = np.matmul(Y, np.matmul(A, Z))
    func = np.real(np.trace(np.matmul((data - model).conj().T, (data - model))))

    if phase_variance:
        if 0 in idx:
            mu = np.sum(para[M:2*M]) / M
            func += (np.sum(para[M:2*M] ** 2) / M) - (mu ** 2)
        else:
            mu = np.sum(para[:M]) / M
            func += (np.sum(para[:M] ** 2) / M) - (mu ** 2)

    return func


def _g_2d(para_act, *args):
    """
    _g_2d(para, *args)

    Decription
    ———————————
    Determines the grad function of cost function to be minimised for
    non-linear optimisation (2D signal)
    i-th element Given by:
    2Re[Tr[(Y - X(θ))† dX(θ)/∂θᵢ]] + ∂Var(φ)/∂θᵢ
    where Y is the (normalised) data, X is the model, θ is the
    parameter array, and φ is the oscillator phases.

    Parameters
    ——————————
    para_act - numpy.ndarray
        Array of active parameters (paramters to be optimised).
    args - list_iterator
        Additional arguments, with entries:
        data: ndarray - Array of the original FID data.
        tp: tuple - Contains 2 numpy.ndarrays with time-points the FID was
                    sampled at, in each dimesnion
        M: int - Number of oscillations.
        para_pas - Passive parameters (not to be optimised)
        idx: list - List indicating the types of parameters are present in the
                    active oscillators (para_act).
                    0 - amplitudes
                    1 - phases
                    2 & 3 - frequencies
                    4 & 5 - damping factors
        phase_vairance: Bool - If True, include the oscillator phase variance
                               to the cost function.

    Returns
    ———————
    grad - numpy.ndarray
        Gradient of the cost function
    """

    # unpack arguments
    data = args[0]
    tp = args[1]
    M = args[2]
    para_pas = args[3]
    idx = args[4]
    phase_variance = args[5]

    # reconstruct correctly ordered parameter vector from
    # active and passsive parameters.
    para = _construct_para(para_act, para_pas, M, idx)

    Y = np.exp(np.outer(tp[0], (1j*2*np.pi*para[2*M:3*M] - para[4*M:5*M])))
    Z = np.exp(np.outer((1j*2*np.pi*para[3*M:4*M] - para[5*M:6*M]), tp[1]))
    alpha = para[:M] * np.exp(1j * para[M:2*M])
    ZA = np.einsum('ij,i->ij', Z, alpha)
    model = np.einsum('ik,kj->ijk', Y, ZA)

    # derivatives
    comps = []
    if 0 in idx:
        comps.append(model / para[:M]) # a
    if 1 in idx:
        comps.append(1j * model) # φ
    if 2 in idx:
        comps.append(np.einsum('ijk,i->ijk', model, 1j*2*np.pi*tp[0])) # f1
        comps.append(np.einsum('ijk,j->ijk', model, 1j*2*np.pi*tp[1])) # f2
    if 4 in idx:
        comps.append(np.einsum('ijk,i->ijk', model, -tp[0])) # η1
        comps.append(np.einsum('ijk,j->ijk', model, -tp[1])) # η2

    deriv = np.dstack(comps)

    diff = data - np.sum(model, axis=2)
    grad = -2*np.real(np.einsum('iik->k', (np.einsum('li,ljk->ijk',
                                            diff.conj(), deriv))))

    if phase_variance:
        if 0 in idx:
            grad[M:2*M] = grad[M:2*M] + (2 * (para[M:2*M] -
                                         (np.sum(para[M:2*M]) / M))) / M
        else:
            grad[:M] = grad[:M] + (2 * (para[:M] -
                                   (np.sum(para[:M]) / M))) / M

    return grad


def _h_2d(para_act, *args):
    """
    _h_2d(para, *args)

    Description
    ———————————
    Determines the Hessian function of cost function to be minimised for
    non-linear optimisation (2D signal)
    i,j-th element Given by:
    2Re[Tr[(Y - X(θ))† ∂²X(θ)/∂θᵢ∂θⱼ - ∂X(θ)†/∂θᵢ ∂X(θ)/∂θⱼ]] + ∂²Var(φ)/∂θᵢ∂θⱼ
    where Y is the (normalised) data, X is the model, θ is the
    parameter array, and φ is the oscillator phases.

    Parameters
    ——————————
    para_act - numpy.ndarray
        Array of active parameters (paramters to be optimised).
    args - list_iterator
        Additional arguments, with entries:
        data: ndarray - Array of the original FID data.
        tp: tuple - Contains 2 numpy.ndarrays with time-points the FID was
                    sampled at, in each dimesnion
        M: int - Number of oscillations.
        para_pas - Passive parameters (not to be optimised)
        idx: list - List indicating the types of parameters are present in the
                    active oscillators (para_act).
                    0 - amplitudes
                    1 - phases
                    2 & 3 - frequencies
                    4 & 5 - damping factors
        phase_vairance: Bool - If True, include the oscillator phase variance
                               to the cost function.

    Returns
    ———————
    hess - numpy.ndarray
        Hessian of the cost function.
    """
    # TODO: Consider ways to improve performance

    # unpack arguments
    data = args[0]
    tp = args[1]
    M = args[2]
    para_pas = args[3]
    idx = args[4]
    phase_variance = args[5]

    # reconstruct correctly ordered parameter vector from
    # active and passsive parameters.
    para = _construct_para(para_act, para_pas, M, idx)

    Y = np.exp(np.outer(tp[0], (1j*2*np.pi*para[2*M:3*M] - para[4*M:5*M])))
    Z = np.exp(np.outer((1j*2*np.pi*para[3*M:4*M] - para[5*M:6*M]), tp[1]))
    alpha = para[:M] * np.exp(1j * para[M:2*M])
    ZA = np.einsum('ij,i->ij', Z, alpha)
    model = np.einsum('ik,kj->ijk', Y, ZA)

    # Note ordering of 2nd derivative blocks:
    # aa   aφ   af1  af2  aη1  aη2  φφ   φf1  φf2  φη1  φη2  f1f1 f1f2 ...
    # 0    1    2    3    4    5    6    7    8    9    10   11   12   ...

    # ...  f1η1 f1η2 f2f2 f2η1 f2η2 η1η1 η1η2 η2η2
    # ...  13   14   15   16   17   18   19   20

    # first derivatives and diagonal second derivatives
    comps1 = []
    comps2 = []
    if 0 in idx:
        comps1.append(model / para[:M]) # a
        comps2.append(np.zeros(comps1[0].shape, dtype=complex)) # a-a
    if 1 in idx:
        comps1.append(1j * model) # φ
        comps2.append(1j * comps1[-1]) # φ-φ
    if 2 in idx:
        comps1.append(np.einsum('ijk,i->ijk', model, 1j*2*np.pi*tp[0])) # f1
        comps2.append(np.einsum('ijk,i->ijk', comps1[-1], 1j*2*np.pi*tp[0])) # f1-f1
        comps1.append(np.einsum('ijk,j->ijk', model, 1j*2*np.pi*tp[1])) # f2
        comps2.append(np.einsum('ijk,i->ijk', comps1[-1], 1j*2*np.pi*tp[0])) # f2-f2
    if 4 in idx:
        comps1.append(np.einsum('ijk,i->ijk', model, -tp[0])) # η1
        comps2.append(np.einsum('ijk,i->ijk', comps1[-1], -tp[0])) # η1-η1
        comps1.append(np.einsum('ijk,j->ijk', model, -tp[1])) # η2
        comps2.append(np.einsum('ijk,j->ijk', comps1[-1], -tp[1])) # η2-η2

    # off-diagonal second derivatives
    if 0 in idx:
        if 1 in idx:
            comps2.insert(1, 1j * comps1[0]) # a-φ
            if 2 in idx:
                comps2.insert(2, comps1[2] / para[:M]) # a-f1
                comps2.insert(3, comps1[3] / para[:M]) # a-f2
                comps2.insert(5, 1j * comps1[2]) # φ-f1
                comps2.insert(6, 1j * comps1[3]) # φ-f2
                comps2.insert(8, np.einsum('ijk,i->ijk', comps1[3], 1j*2*np.pi*tp[0])) # f1-f2
                if 4 in idx:
                    comps2.insert(4, comps1[4] / para[:M]) # a-η1
                    comps2.insert(5, comps1[5] / para[:M]) # a-η2
                    comps2.insert(9, 1j * comps1[4]) # φ-η1
                    comps2.insert(10, 1j * comps1[5]) # φ-η2
                    comps2.insert(13, np.einsum('ijk,i->ijk', comps1[2], -tp[0])) # f1-η1
                    comps2.insert(14, np.einsum('ijk,j->ijk', comps1[2], -tp[1])) # f1-η2
                    comps2.insert(16, np.einsum('ijk,i->ijk', comps1[3], -tp[0])) # f2-η1
                    comps2.insert(17, np.einsum('ijk,j->ijk', comps1[3], -tp[1])) # f2-η2
                    comps2.insert(19, np.einsum('ijk,i->ijk', comps1[5], -tp[0])) # η1-η2
            elif 4 in idx:
                comps2.insert(2, comps1[2] / para[:M]) # a-η1
                comps2.insert(2, comps1[3] / para[:M]) # a-η2
                comps2.insert(5, 1j * comps1[2]) # φ-η1
                comps2.insert(6, 1j * comps1[3]) # φ-η1
                comps2.insert(8, np.einsum('ijk,i->ijk', comps1[3], -tp[0])) # η1-η2
        elif 2 in idx:
            comps2.insert(1, comps1[1] / para[:M]) # a-f1
            comps2.insert(2, comps1[2] / para[:M]) # a-f2
            comps2.insert(4, np.einsum('ijk,i->ijk', comps1[2], 1j*2*np.pi*tp[0])) # f1-f2
            if 4 in idx:
                comps2.insert(3, comps1[3] / para[:M]) # a-η1
                comps2.insert(4, comps1[4] / para[:M]) # a-η2
                comps2.insert(7, np.einsum('ijk,i->ijk', comps1[1], -tp[0])) # f1-η1
                comps2.insert(8, np.einsum('ijk,j->ijk', comps1[1], -tp[1])) # f1-η2
                comps2.insert(10, np.einsum('ijk,i->ijk', comps1[2], -tp[0])) # f2-η1
                comps2.insert(11, np.einsum('ijk,j->ijk', comps1[2], -tp[1])) # f2-η1
                comps2.insert(13, np.einsum('ijk,i->ijk', comps1[4], -tp[0])) # η1-η2
        elif 4 in idx:
            comps2.insert(1, comps1[1] / para[:M]) # a-η1
            comps2.insert(2, comps1[2] / para[:M]) # a-η2
            comps2.insert(4, np.einsum('ijk,i->ijk', comps1[2], -tp[0])) # η1-η2

    elif 1 in idx:
        if 2 in idx:
            comps2.insert(1, 1j * comps1[1]) # φ-f1
            comps2.insert(2, 1j * comps1[2]) # φ-f2
            comps2.insert(4, np.einsum('ijk,i->ijk', comps1[2], 1j*2*np.pi*tp[0])) # f1-f2
            if 4 in idx:
                comps2.insert(3, 1j * comps1[3]) # φ-η1
                comps2.insert(4, 1j * comps1[4]) # φ-η2
                comps2.insert(7, np.einsum('ijk,i->ijk', comps1[1], -tp[0])) # f1-η1
                comps2.insert(8, np.einsum('ijk,j->ijk', comps1[1], -tp[1])) # f1-η2
                comps2.insert(10, np.einsum('ijk,i->ijk', comps1[2], -tp[0])) # f2-η1
                comps2.insert(11, np.einsum('ijk,j->ijk', comps1[2], -tp[1])) # f2-η2
                comps2.insert(13, np.einsum('ijk,i->ijk', comps1[4], -tp[0])) # η1-η2
        elif 4 in idx:
            comps2.insert(1, 1j * comps1[1]) # φ-η1
            comps2.insert(2, 1j * comps1[2]) # φ-η2
            comps2.insert(4, np.einsum('ijk,i->ijk', comps1[2], -tp[0])) # η1-η2

    elif 2 in idx:
        comps2.insert(1, np.einsum('ijk,i->ijk', comps1[1], 1j*2*np.pi*tp[0])) # f1-f2
        if 4 in idx:
            comps2.insert(2, np.einsum('ijk,i->ijk', comps1[0], -tp[0])) # f1-η1
            comps2.insert(3, np.einsum('ijk,j->ijk', comps1[0], -tp[1])) # f1-η2
            comps2.insert(5, np.einsum('ijk,i->ijk', comps1[1], -tp[0])) # f2-η1
            comps2.insert(6, np.einsum('ijk,j->ijk', comps1[1], -tp[1])) # f2-η2
            comps2.insert(8, np.einsum('ijk,i->ijk', comps1[3], -tp[0])) # η1-η2

    elif 4 in idx:
        comps2.insert(1, p.einsum('ijk,i->ijk', comps1[1], -tp[0])) # η1-η2

    deriv1 = np.dstack(comps1)
    deriv2 = np.dstack(comps2)
    del comps1, comps2 # clear up memory

    diff = data - np.sum(model, axis=2)
    diags = -2*np.real(np.einsum('jki,jk->i', deriv2.conj(), diff))

    del deriv2 # clear up memory

    p = len(idx) # number of parameter types
    hess = np.zeros((p*M, p*M)) # generate Hessian of correct shape

    # determine indices of elements which have non-zero second derivs
    # (specfically, those in upper triangle)
    diag_idx_0, diag_idx_1 = _generate_diag_indices(p, M)

    hess[diag_idx_0, diag_idx_1] = diags

    hess = np.real(hess)
    hess += np.transpose(hess)
    hess += 2 * np.real(np.einsum('lij,lik -> jk', deriv1.conj(), deriv1))

    if phase_variance:
        # amplitudes are being optimised (phases between M and 2M)
        if 0 in idx:
            hess[M:2*M, M:2*M] += (2/(M**2))
            hess[_diag_indices(hess, k=0)[0][M:2*M],
                 _diag_indices(hess, k=0)[1][M:2*M]] += 2 / M
        # amplitudes are not being optimised (phases between 0 and M)
        else:
            hess[:M, :M] += (2/(M**2))
            hess[_diag_indices(hess, k=0)[0][:M],
                 _diag_indices(hess, k=0)[1][:M]] += 2 / M

    return hess


def _correct_freqs(para, offset):
    """
    _correct_freqs(para, offset, dim)

    Decription
    ———————————
    Alter frequencies from those that correspond to the correct
    spectral values to those that will create the correct corresponding
    FID.

    Parameters
    ——————————
    para - numpy.ndarray
        Parameter array of shape (M, 4) or (M, 6)
    offset - tuple
        Transmitter offset frequency (Hz) in each dimension.
    dim - int
        Signal dimension. Will be 1 or 2.

    Returns
    ———————
    para_cor - numpy.ndarray
        Parameter array with corrected frequencies (same shape as para).
    """
    para_cor = deepcopy(para)
    dim = (para_cor.shape[1]/2) - 1
    for i, off in enumerate(offset):
        para_cor[..., i+2] = -para_cor[..., i+2] + off

    return para_cor


def _get_mode_indices(mode, dim):
    """
    _get_mode_indices(mode, dim)

    Description
    ———————————
    Determine which columns of theta0 to retain when mode is not 'all'

    Parameters
    ——————————
    mode - str
        A string whose characters dictate which parameter types to keep.
        Will be any permutation of any number of 'a', 'p', 'f', and 'd'
    dim - int
        Signal dimesnion

    Returns
    ———————
    idx_act - list
        Indices of columns (i.e axis-1 elements) containing parameters that
        will be optimised (active).
    idx_pas - list
        Indices of columns (i.e axis-1 elements) containing parameters that
        will not be optimised (passive).
    """

    idx_act = []
    for char in mode:
        if char == 'a':
            idx_act.append(0)
        elif char == 'p':
            idx_act.append(1)
        elif char == 'f':
            for i in range(dim):
                idx_act.append(2 + i)
        elif char == 'd':
            for i in range(dim):
                idx_act.append(2 + dim + i)

    idx_pas = []
    for i in range((dim + 1) * 2):
        if i not in idx_act:
            idx_pas.append(i)
    return idx_act, idx_pas


def _get_bounds(M, sw, offset, idx):
    """
    _get_bounds(M, sw, offset, idx)

    Decription
    ———————————
    Constructs a list of bounding constraints to set for each parameter,
    if bounds are desired in the NLP routine.
    Bounds are as follows:
    * amplitudes: 0 < a < ∞
    * phases: -π < φ < π
    * frequencies: offset - sw/2 < f < offset + sw/2
    * damping: 0 < η < ∞

    Parameters
    ——————————
    M - int
        Number of oscillators
    sw - tuple
        Sweep width (Hz) in each dimension.
    offset - tuple
        Transmitter offset frequency (Hz) in each dimension.
    idx - list
        List of indices corresponding to elements along axis-1 of theta0
        to consider.

    Returns
    ———————
    bounds - list
        List of 4*M (dim = 1) or 6*M (dim = 2) elements. Each element
        is a tuple specifying the lower and upper bounds of an
        individual parameter.
    """

    # constuct a list of all bounds, and then trim at the end if
    # mode was not 'apfd' (dictated by idx)
    all_ = [(0, np.inf)] * M # amps
    all_ += [(-np.pi, np.pi)] * M # phases

    dim = 0 # counts number of times the upcoming loop is cycled
    for s, o in zip(sw, offset):
        all_ += [((o - s/2), (o + s/2))] * M # frequencies
        dim += 1
    all_ += [(0, np.inf)] * (dim * M) # damping factors

    bounds = []
    for i in idx:
        bounds += all_[i*M:(i+1)*M]

    return bounds


def _construct_para(para_act, para_pas, M, idx):
    """
    _construct_para(para_act, para_pas, M, idx)

    Decription
    ———————————
    Constructs the full parameter vector from active and passive
    sub-vectors.

    Parameters
    ——————————
    para_act - numpy.ndarray
        Active parameters.
    para_pas - numpy.ndarray
        Passive parameters.
    M - int
        Number of oscillators
    idx - list
        Indicates the relative locations of active parameter blocks.

    Returns
    ———————
    para - numpy.ndarray
        Full parameter vector with correct ordering (i.e. [a, φ, f, η])
    """

    p = int((para_act.shape[0] + para_pas.shape[0]) / M)
    para = np.zeros(p*M)
    for i in range(p):
        if i in idx:
            para[i*M:(i+1)*M] = para_act[:M]
            para_act = para_act[M:]
        else:
            para[i*M:(i+1)*M] = para_pas[:M]
            para_pas = para_pas[M:]
    return para


def _generate_diag_indices(p, M):
    """
    _generate_diag_indices(p, M)

    Decription
    ———————————
    Determines all array indicies that correspond to positions in which
    non-zero second derivatives reside, in the top right half of the Hessian.

    Parameters
    ——————————
    p - int
        The number of parameter types (i.e. a, φ, f, η → p = 4). For an ND
        signal it will be 2N + 2
    M - int
        Number of oscillators in parameter estimate

    Returns
    ———————
    diag_idx_0 - numpy.ndarray
        0-axis coordinates of indices.
    diag_idx_1
        1-axis coordinates of indices.
    """

    arr = np.zeros((p*M, p*M)) # dummy array with same shape as Hessian
    diag_idx_0 = []
    diag_idx_1 = []
    for i in range(p):
        for j in range(p - i):
            diag_idx_0.append(_diag_indices(arr, k=j*M)[0][i*M:(i+1)*M])
            diag_idx_1.append(_diag_indices(arr, k=j*M)[1][i*M:(i+1)*M])

    return np.hstack(diag_idx_0), np.hstack(diag_idx_1)


def _diag_indices(a, k=0):
    """
    _diag_indices(a, k=0)

    Decription
    ———————————
    Returns the indices of an array's kth diagonal. A generalisation of
    numpy.diag_indices_from(), which can only be used to obtain the indices
    along the main diagonal of an array. Used in calculation of the Hessian;
    see _h_1d() and _h_2d().

    Parameters
    ——————————
    a - numpy.ndarray
        Square array (Hessian matrix)
    k - int
        Displacement from the main diagonal

    Returns
    ———————
    rows - numpy.ndarray
        Row elements of diagonal indices
    cols - numpy.ndarray
        Column elements of diagonal indices
    """

    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def _rm_negligible_amps(x, amp_thold, fprint):
    """
    _rm_negligible_amps(x, amp_thold, fprint)

    Decription
    ———————————
    Determines which oscillators (if any) have amplitudes which are
    smaller than a threshold, and removes these from the parameter array

    Parameters
    ——————————
    x - numpy.ndarray
        Parameter array, with shape (M, 4) or (M, 6).
    amp_thold - float or None
        Specifies threshold. Any oscillators satisfying:
        amplitudes ≤ amp_thold ✕ ‖amps‖
        will be removed. If None, will simply return x
    fprint - Bool
        Dictates whether or not to print information to the terminal

    Returns
    ———————
    x_new - numpy.ndarray
        Parameter array, with shape (M_new, 4) or (M_new, 6), with
        M_new ≤ M
    """
    if amp_thold is None:
        return x

    thold = amp_thold * norm(x[:, 0])
    rm_ind = np.nonzero(x[:, 0] < thold) # indices of neg. amp. oscillators
    x_new = np.delete(x, rm_ind, axis=0)
    if np.array_equal(x, x_new):
        pass
    else:
        if fprint:
            print(f'{O}Oscillations with negligible amplitude'
                  f' removed.{END}\nUpdated number of oscillators:'
                  f' {x.shape[0]}')
    return x_new


def _rm_negative_amps(x, fprint):
    """
    _rm_negative_amps(x, fprint)

    Decription
    ———————————
    Determines which oscillators (if any) have negative amplitudes, and
    removes these from the parameter array. Also returns a Boolean used
    by nlp() to decide whether to re-run the optimisation routine.

    Parameters
    ——————————
    x - numpy.ndarray
        Parameter array, with shape (M, 4) or (M, 6).
    fprint - Bool
        Dictates whether or not to print information to the terminal

    Returns
    ———————
    x_new - numpy.ndarray
        Parameter array, with shape (M_new, 4) or (M_new, 6), with
        M_new ≤ M.
    term - Bool
        If True, nlp() will terminate. If False, nlp() will run again.
    """

    rm_ind = np.nonzero(x[:, 0] < 0.0) # indices of -ve amp. oscillators
    x_new = np.delete(x, rm_ind, axis=0)

    if np.array_equal(x, x_new):
        term =  True
    else:
        term = False
        if fprint:
            print(f'{O}Negative amplitudes detected!'
                  f' These have been removed{END}\n'
                  f'Updated no. of oscillators: {x_new.shape[0]}')

    return x_new, term

# “Some men are born mediocre, some men achieve mediocrity,
# and some men have mediocrity thrust upon them.”
# ———————————————————————————————Joseph Heller, Catch-22———
