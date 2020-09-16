#!/usr/bin/python3
# mpm.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# 1d and 2d matrix pencil routines.

from copy import deepcopy
import time

import numpy as np
from numpy.linalg import norm, svd, eig, pinv, solve, inv
from scipy.linalg import hankel
from scipy import sparse
from scipy.sparse.linalg import svds

from ._timing import _print_time
from ._cols import *
if USE_COLORAMA:
    import colorama


def mpm_1d(data, M_in, sw, offset, fprint):
    """1D Matrix Pencil Method, with the option of model order selection
    using the Minimum Description Length (MDL)

    Parameters
    ----------
    data : numpy.ndarray
        Signal to be considered (unnormalised).

    M_in : int
        The number of oscillators. If ``0``, the number of oscilators will
        be estimated using the MDL.

    sw : float
        The experiment sweep width in Hz.

    offset : float
        The experiment transmitter offset frequency in Hz.

    fprint : bool
        Flag specifiying whether to print infomation to the terminal as
        the method runs.

    Returns
    -------
    para : numpy.ndarray
        Array of oscillators parameters.

    Referecnes
    ----------
    [1] Yingbo Hua and Tapan K Sarkar. “Matrix pencil method for estimating
    parameters of exponentially damped/undamped sinusoids in noise”. In:
    IEEE Trans. Acoust., Speech, Signal Process. 38.5 (1990), pp. 814–824.

    [2] Yung-Ya Lin et al. “A novel detection–estimation scheme for noisy NMR
    signals: applications to delayed acquisition data”. In: J. Magn. Reson.
    128.1 (1997), pp. 30–41.
    """

    if fprint:
        start = time.time()
        print(f'=============\n{G}ITMPM started{END}\n=============')

    # normalise data
    nm = norm(data)
    data_norm = data/nm

    # data size and pencil parameter
    N = data_norm.shape[0]
    L = _pencil_parameter(N, fprint)

    # Hankel matrix of data, with dimensions (N-L) * L
    r = data_norm[0:N-L]
    c = data_norm[N-L-1:N]
    Y = _hankel(r, c, fprint)

    # singular value decomposition of Y
    s, Vh = _svd(Y, fprint)

    # number of oscillators
    M = _mdl(M_in, N, L, s, fprint)

    # determine signal poles
    V = np.transpose(Vh)
    z = _signal_poles_1d(V, M, fprint)

    # determine complex amplitudes
    alpha = _complex_amplitudes_1d(z, data_norm, fprint)

    # extract amps, phases, freqs, and damps
    amp = np.abs(alpha) * nm
    phase = np.arctan2(np.imag(alpha), np.real(alpha))
    freq = -(sw / (2 * np.pi)) * np.imag(np.log(z)) + offset
    damp = - sw * np.real(np.log(z[:]))
    x0 = (np.vstack((amp, phase, freq, damp))).T

    if fprint:
        finish = time.time()
        print(f'==============\n{G}ITMPM complete{END}\n==============')
        _print_time(finish-start)

    # removal of terms with negative damping factors
    x0 = _negative_damping(x0, fprint)

    # return parameters, ordered by frequency
    return x0[np.argsort(x0[:, 2])]


def mpm_2d(data, M_in, sw, offset, fprint):
    """
    mpm_2d(data, M_in, sw, offset, fprint)

    ———Description—————————————————————————————
    Modified Matrix Enhancement Matrix Pencil Method for 2D time-domain
    signal analysis.

    ———Parameters——————————————————————————————
    data - ndarray
        2D data array to be considered (unnormalised)

    M_in - int
        The number of oscillators. If 0, the number of oscilators will
        be estimated using the MDL on the first dataslice through the
        direct dimension

    sw - tuple
        The experiment sweep width for both dimensions, in Hz. sw should
        be a tuple of two floats.

    offset - tuple
        The experiment transmitter offset frequencies for both dimensions,
        in Hz. offset should be a tuple of two floats.

    fprint - bool
        Flag specifiying whether to print infomation to the terminal as
        the method runs.

    ———Returns—————————————————————————————————
    para - ndarray
        Array of oscillators parameters. If M_in is 0, the shape of para
        will be (MDL, 6), where MDL is the result of using the minimum
        description length. If M_in > 0, the shape of para will be
        (M_in, 6).
        Each element along axis-0 will have elements ordered as follows:
        [amplitude, phase, frequency 1, frequency 2, damping 1, damping 2]

    ———Referecnes——————————————————————————————
    [1] Yingbo Hua. “Estimating two-dimensional frequencies by matrix
    enhancement and matrix pencil”. In: [Proceedings] ICASSP 91: 1991
    International Conference on Acoustics, Speech, and Signal Processing.
    IEEE. 1991, pp. 3073–3076.

    [2] Fang-Jiong Chen et al. “Estimation of two-dimensional frequencies
    using modified matrix pencil method”. In: IEEE Trans. Signal Process.
    55.2 (2007), pp. 718–724.
    """

    if fprint:
        start = time.time()
        print(f'==============\n{G}MMEMPM started{END}\n==============')

    # normalise data
    data_norm = norm(data)
    data_n = data/data_norm

    N = data_n.shape

    # pencil parameters
    K, L = _pencil_parameter(N, fprint)

    # consider first slice in the direct dim to feed into MDL
    data_mdl = data_n[0,:]
    hankel_mdl = _hankel(data_mdl[:N[1]-L], data_mdl[N[1]-L-1:], False)
    s_mdl, _ = _svd(hankel_mdl, False)

    # number of oscillators
    # TODO: more robust algorithm for 2D data?
    M = _mdl(M_in, N[1], L, s_mdl, fprint)

    # construct enhanced matrix, Xe
    # TODO: make more efficient!
    Xe = _enhanced_matrix(data_n, K, L, fprint)

    # compute the M largest singular vectors of Xe
    U = _svds(Xe, M, fprint)
    Xe = Xe.todense()

    poles = _signal_poles_2d(U, M, K, L, fprint)

    alpha = _complex_amplitudes_2d(poles, data_n, Xe, K, L, fprint)

    # extract amps, phases, freqs, and damps
    amp = np.abs(alpha) * data_norm
    phase = np.arctan2(np.imag(alpha), np.real(alpha))
    freq1 = (sw[0] / (2 * np.pi)) * np.imag(np.log(poles[:, 0])) \
                 + offset[0]
    freq2 = (sw[1] / (2 * np.pi)) * np.imag(np.log(poles[:, 1])) \
                 + offset[1]
    damp1 = -sw[0] * np.real(np.log(poles[:, 0]))
    damp2 = -sw[1] * np.real(np.log(poles[:, 1]))
    x0 = (np.vstack((amp, phase, freq1, freq2, damp1, damp2))).T

    if fprint:
        finish = time.time()
        print(f'===============\n{G}MMEMPM Complete{END}\n===============')
        _print_time(finish-start)

    return x0[np.argsort(x0[..., 2])]


def _pencil_parameter(N, fprint):
    """
    _pencil_parameter(N, fprint)

    ———Description—————————————————————————————
    Determines the pencil parameter(s), for the MPM/MMEMPM

    ———Parameters——————————————————————————————
    N - int or tuple
        The number of data points in each dimension.
    fprint - Bool
        Flag specifiying whether to print infomation to the terminal.

    ———Returns—————————————————————————————————
    pen_par - int or tuple
        Pencil parameter(s)
    """

    if type(N) is int:
        pen_par = int(np.floor(N/3)) # optimal when between N/2 and N/3
        if fprint:
            print(f'Pencil parameter: {pen_par}\n')

    if type(N) is tuple:
        K = int(np.floor((N[0] + 1) / 2))
        L = int(np.floor((N[1] + 1) / 2))

        if fprint:
            print(f'Pencil Parameters: {K} & {L}')

        pen_par = (K, L)

    return pen_par


def _hankel(column, row, fprint):
    """
    _hankel(column, row, fprint)

    ———Description—————————————————————————————
    Wrapper around scipy.linalg.hankel(). Constructs Hankel matrix,
    and prints information if fprint is True.

    ———Parameters——————————————————————————————
    column - numpy.ndarray
        First column of the matrix.
    row - numpy.ndarray
        Last row of the matrix
    fprint - Bool
        Flag specifiying whether to print infomation to the terminal.

    ———Returns—————————————————————————————————
    H - numpy.ndarry
        Hankel matrix with shape (len(column), len(row))
    """

    H = hankel(column, row)

    if fprint:
        print("Hankel data matrix constructed.")
        print(f'\tSize:   {H.shape[0]} x {H.shape[1]}')
        print(f'\tMemory: {round(H.nbytes/(2**30), 4)}GiB\n')

    return H


def _svd(array, fprint):
    """
    _svd(array, fprint)

    ———Description—————————————————————————————
    Wrapper around numpy.linalg.svd(). Computes svd, and gives time.
    Also neglects left singular vectors which are not needed.

    ———Parameters——————————————————————————————
    array - numpy.ndarray
        Matrix to consider
    fprint - Bool
        Flag specifiying whether to print infomation (inc. time taken)
        to the terminal.

    ———Returns—————————————————————————————————
    s - ndarry
        Array of singular values
    Vh - ndarray
        Hermitian conjuage array of right singular vectors
    """

    if fprint:
        start = time.time()
        print('Performing Singular Value Decomposition...')

    _, s, Vh = svd(array)

    if fprint:
        finish = time.time()
        _print_time(finish-start)

    return s, Vh


def _mdl(M_in, N, L, s, fprint):
    """
    _mdl(M_in, N, L, s, fprint)

    ———Description—————————————————————————————
    Computes the MDL, with the option of printing information to the
    terminal.

    ———Parameters——————————————————————————————
    M_in - int
        If 0, the MDL is computed, if >0, the MDL is not computed
    N - int
        Number of data points
    L - int
        Pencil parameter
    s - numpy.ndarray
        Array of singular values
    fprint - Bool
        Flag specifiying whether to print infomation to the terminal.

    ———Returns—————————————————————————————————
    M - int
        The number of oscillations. If M_in is 0, M will be determined
        using the MDL. If M_in is >0, M will be set to M_in.
    """

    if M_in == 0:
        mdl = np.zeros(L)
        if fprint:
            print('Estimating number of oscillations using MDL...')
        for k in range(L):
            mdl[k] = - N * np.sum(np.log(s[k:L])) \
                     + N * (L-k) * np.log((np.sum(s[k:L]) / (L-k))) \
                     + (k * np.log(N) * (2*L-k)) / 2

        M = np.argmin(mdl)

    else:
        M = M_in
        if fprint:
            print('Number of oscillations has been pre-defined by the user')

    if fprint:
        print(f'Number of oscillations: {M}')
        print()

    return M


def _signal_poles_1d(V, M, fprint):
    """
    _signal_poles_1d(V, M, fprint)

    ———Description—————————————————————————————
    Computes the signal poles of a 1D signal, given a matrix of singular
    values and the number of poles, with the option of printing information
    to the terminal.

    ———Parameters——————————————————————————————
    V - numpy.ndarray
        Matrix of right singular vectors, obtained from _svd()
    M - int
        Number of oscillations
    fprint - Bool
        Flag specifiying whether to print infomation (inc. time taken)
        to the terminal.

    ———Returns—————————————————————————————————
    z - numpy.ndarray
        Array of signal poles, of shape (M,)
    """

    if fprint:
        start = time.time()
        print('Determining signal poles...')

    V = V[:, :M] # retain s. vecs corresponding to M largest s. vals
    V1 = V[:-1, :] # remove last column
    V2 = V[1:, :] # remove first column

    # determine first M signal poles (others should be 0)
    z, _ = eig(np.matmul(V2, pinv(V1)))
    z = z[:M]

    if fprint:
        finish = time.time()
        _print_time(finish-start)

    return z


def _permutation_matrix(K, L):
    """
    _permutation_matrix(K, L)

    ———Description—————————————————————————————
    Computation of the permutation matrix P for use in MMEMPM.

    ———Parameters——————————————————————————————
    K - int
        Pencil parameter corresponding to dimension 1
    L - int
        Pencil parameter corresponding to dimension 2

    ———Returns—————————————————————————————————
    P - numpy.ndarray
        Permutation matrix of shape (K*L, K*L)
    """

    # create first row of matrrix: [1, 0, 0, ..., 0]
    first = sparse.lil_matrix((1,K*L))
    first[0,0] = 1

    # seed the permutation matrix
    P = deepcopy(first)

    # first block of K rows of permutation matrix
    for k in range(1,K):
        # create new row to add
        row = sparse.hstack((first[:, -(k*L):], first[:, :-(k*L)]))
        # combine
        P = sparse.vstack((P, row))

    # first K-sized block of matrix
    first_block = deepcopy(P).tolil()

    # stack K-sized blocks to form (K * L) row matrix
    for l in range(1, L):
        # create new row to add
        row = sparse.hstack((first_block[:, -l:], first_block[:, :-l]))
        # combine
        P = sparse.vstack((P, row))

    return P.todense()


def _signal_poles_2d(U, M, K, L, fprint):
    """
    _signal_poles_2d(U, M, K, L, fprint)

    ———Description—————————————————————————————
    Computes the signal poles of a 2D signal, given a matrix of singular
    values, the number of poles, and pencil parameters,with the option of
    printing information (inc. time taken) to the terminal.

    ———Parameters——————————————————————————————
    U - numpy.ndarray
        Matrix of left singular vectors, obtained from _svds()
    M - int
        Number of oscillations
    K - int
        Pencil parameter corresponding to dimension 1
    L - int
        Pencil parameter corresponding to dimension 2
    fprint - Bool
        Flag specifiying whether to print infomation (inc. time taken)
        to the terminal.

    ———Returns—————————————————————————————————
    z - numpy.ndarray
        Array of signal poles, of shape (M,)
    """

    if fprint:
        start = time.time()
        print('Determining signal poles...')

    # retain only M principle left s. vecs
    Us = U[:, :M]
    r = Us.shape[0]
    U1 = Us[:r-L, :] # last L rows deleted
    U2 = Us[L:, :] # first L rows deleted
    # determine dim 1 signal poles and eigenvectors
    eig_y, vec_y = eig(np.matmul(pinv(U1), U2))

    P = _permutation_matrix(K, L)
    Usp = np.matmul(P, Us) # shuffled singular value matrix
    rp = Usp.shape[0]
    U1p = Usp[:rp-K, :] # last K rows deleted
    U2p = Usp[K:, :] # first K rows deleted

    # determine dim 2 signal poles
    eig_z = np.diag(np.matmul(solve(vec_y, pinv(U1p)),np.matmul(U2p, vec_y)))
    poles = np.hstack((eig_y, eig_z)).reshape((M,2), order='F')

    if fprint:
        finish = time.time()
        _print_time(finish-start)

    return poles


def _complex_amplitudes_1d(z, data, fprint):
    """
    _complex_amplitudes_1d(z, data, N, fprint)

    ———Description—————————————————————————————
    Computes the complex amplitudes of a 1D signal, given the signal poles
    and data, with the option of printing information (inc. time taken) to
    the terminal.

    ———Parameters——————————————————————————————
    z - numpy.ndarray
        Array of signal poles, of shape (M,)
    data - numpy.ndarray
        Signal of interest
    fprint - Bool
        Flag specifiying whether to print infomation (inc. time taken)
        to the terminal.

    ———Returns—————————————————————————————————
    alpha - numpy.ndarray
        Array of complex amplitudes, of shape (M,)
    """

    if fprint:
        start = time.time()
        print('Determining complex amplitudes...')

    N = data.shape[0]
    n = np.arange(N)
    Z = (np.power.outer(z, n)).T # Vandermonde matrix of poles
    alpha = np.matmul(pinv(Z), data)

    if fprint:
        finish = time.time()
        _print_time(finish-start)

    return alpha

def _complex_amplitudes_2d(poles, data, Xe, K, L, fprint):
    """
    _complex_amplitudes_2d(poles, Xe, K, L, fprint)

    ———Description—————————————————————————————
    Computes the complex amplitudes of a 2D signal, given the signal poles,
    enhanced matrix, and pencil parameters, with the option of printing
    information (inc. time taken) to the terminal.

    ———Parameters——————————————————————————————
    poles - numpy.ndarray
        Array of signal poles, of shape (M, 2), where M is the number
        of oscillations.
    Xe - numpy.ndarray
        Enchanced block hankel matrix
    K - int
        Pencil parameter corresponding to dimension 1
    L - int
        Pencil parameter corresponding to dimension 2
    fprint - Bool
        Flag specifiying whether to print infomation (inc. time taken)
        to the terminal.

    ———Returns—————————————————————————————————
    alpha - numpy.ndarray
        Array of signal poles, of shape (M,), where M is poles.shape[0]
    """

    if fprint:
        start = time.time()
        print('Determining complex amplitudes...')

    N = data.shape
    ZL = np.power.outer(poles[:, 1], np.arange(L)).T
    ZR = np.power.outer(poles[:, 1], np.arange(N[1]-L+1))
    Yd = np.diag(poles[:, 0])

    EL = deepcopy(ZL)
    for k in range(1, K):
        EL = np.vstack((EL, np.matmul(ZL, Yd ** k)))

    ER = deepcopy(ZR)
    for m in range(1, N[0]-K+1):
        ER = np.hstack((ER, np.matmul(Yd ** m, ZR)))

    A = np.matmul(pinv(EL), np.matmul(Xe, pinv(ER)))
    alpha = np.diag(A)

    if fprint:
        finish = time.time()
        _print_time(finish-start)

    return alpha


def _negative_damping(para, fprint):
    """
    _negative_damping(para, fprint)

    ———Description—————————————————————————————
    Determines any oscillators with negative damping factors in the
    parameter array, and removes these from the array.

    ———Parameters——————————————————————————————
    para - numpy.ndarray
        Array of signal oscillators, of shape (M, 4) or (M, 6)
    fprint - Bool
        Flag specifiying whether to print infomation to the terminal.

    ———Returns—————————————————————————————————
    alpha - numpy.ndarray
        Array of signal poles, of shape (M,), where M is poles.shape[0]
    """

    M_init = para.shape[0]
    # indices of oscillators with negative damping factors
    neg_damp = np.nonzero(para[:, 3] < 0.0)[0]
    para = np.delete(para, neg_damp, axis=0)
    M = para.shape[0]

    if M < M_init:
        if fprint:
            print(f'{O}WARNING: Oscillations with negative damping'
                  f' factors detected. These have been deleted.\n'
                  f'Corrected number of oscillations: {M}{END}')

    return para


def _enhanced_matrix(data, K, L, fprint):
    """
    _enhanced_matrix(data, K, L, fprint)

    ———Description—————————————————————————————
    Constructs an enhanced matrix Xe from signal data, and pencil
    parameters, with the option of printing information (inc. time taken)
    to the terminal. The final result is a sparse (CSR) matrix.

    ———Parameters——————————————————————————————
    data - numpy.ndarray
        Signal of interest.
    K - int
        Pencil parameter corresponding to dimension 1
    L - int
        Pencil parameter corresponding to dimension 2
    fprint - Bool
        Flag specifiying whether to print infomation (inc. time taken)
        to the terminal.

    ———Returns—————————————————————————————————
    sparse_Xe - scipy.sparse.cst_matrix
        Enhanced matrix with shape (K*L, (N[0]-K+1)*N[1]-L+1)), where N
        is data.shape
    """

    #construct enhanced matrix, Xe
    if fprint:
        start = time.time()
        print('Constructing Enhanced Matrix (Xe)...')

    N = data.shape
    # row of all Hankel blocks, X_0 to X_(M-1)
    # TODO: perhaps list appending would be quicker than h/vstack?
    X = hankel(data[0, 0:L], data[0, L-1:N[1]])
    for m in range(1, N[0]):
        X = np.hstack((X, hankel(data[m, 0:L],data[m, L-1:N[1]])))

    # vertically stack rows of block matricies to get Xe
    Xe = X[:, 0:(N[0]-K+1)*(N[1]-L+1)]
    for k in range(1, K):
        Xe = np.vstack((Xe, X[:, k*(N[1]-L+1):(k+N[0]-K+1)*(N[1]-L+1)]))

    Ne = Xe.shape

    # convert Xe to sparse matrix
    sparse_Xe = sparse.csr_matrix(Xe)

    if fprint:
        finish = time.time()
        print('Enhanced Matrix has been constructed')
        print(f'\tSize:   {Ne[0]} x {Ne[1]}')
        print(f'\tMemory: {round(Xe.nbytes/(2**30), 4)}GiB')
        _print_time(finish-start)

    return sparse_Xe


def _svds(Xe, M, fprint):
    """
    _svd(array, fprint)

    ———Description—————————————————————————————
    Wrapper around scipy.sparse.linalg.svds() Computes svd of a sparse
    matrix, and returns the left singular vectors corresponding to the
    M most significant singular values. Provides option of printing
    information (inc. time taken) to the terminal.

    ———Parameters——————————————————————————————
    Xe - scipy.sparse.cst_matrix
        Enhanced matrix.
    M - int
        Number of singular vectors to obtain.
    fprint - Bool
        Flag specifiying whether to print infomation (inc. time taken)
        to the terminal.

    ———Returns—————————————————————————————————
    U - ndarry
        Array of right singular vectors.
    """

    if fprint:
        start = time.time()
        print('Computing SVD of Xe...')

    U, _, _ = svds(Xe, M)

    if fprint:
        finish = time.time()
        _print_time(finish-start)

    return U

# “Under capitalism, man exploits man.
# Under communism, it's just the opposite.”
# —————————————————John Kenneth Galbraith———
