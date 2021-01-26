# mpm.py
# matrix pencil method for analysis of 1D and 2D time series
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# ==============================================
# SGH 26-1-21
# I have tagged various methods as follows:
#
# TODO: MAKE 2D COMPATIBLE
#
# These only support 1D data treatment currently
# ==============================================

from copy import deepcopy
import functools

import numpy as np
import numpy.linalg as nlinalg
import scipy.linalg as slinalg
from scipy import sparse
import scipy.sparse.linalg as splinalg

import nmrespy._misc as misc
import nmrespy._errors as errors
from ._timing import timer
import nmrespy._cols as cols

if cols.USE_COLORAMA:
    import colorama


def start_end_wrapper(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        inst = args[0]
        if inst.fprint is False:
            return f(*args, **kwargs)

        print(f'{cols.G}===============================\n'
                       'PERFORMING MATRIX PENCIL METHOD\n'
                      f'==============================={cols.END}')

        result = f(*args, **kwargs)

        print(f'{cols.G}=============================\n'
                       'MATRIX PENCIL METHOD COMPLETE\n'
                      f'============================={cols.END}')
        return result
    return wrapper


class MatrixPencil:
    """Class for performing the Matrix Pencil Method with the option of model
    order selection using the Minimum Description Length (MDL). Supports
    analysis of one-dimensional [1]_ [2]_ or two-dimensional data [3]_ [4]_

    Parameters
    ----------
    data : numpy.ndarray
        Signal to be considered (unnormalised).

    sw : [float] or [float, float]
        The experiment sweep width in each dimension in Hz.

    offset : [float] or None, default: None
        The experiment transmitter offset frequency in Hz.

    M : int, default: 0
        The number of oscillators. If ``0``, the number of oscilators will
        be estimated using the MDL.

    fprint : bool
        Flag specifiying whether to print infomation to the terminal as
        the method runs.

    References
    ----------
    .. [1] Yingbo Hua and Tapan K Sarkar. “Matrix pencil method for estimating
       parameters of exponentially damped/undamped sinusoids in noise”. In:
       IEEE Trans. Acoust., Speech, Signal Process. 38.5 (1990), pp. 814–824.

    .. [2] Yung-Ya Lin et al. “A novel detection–estimation scheme for noisy NMR
       signals: applications to delayed acquisition data”. In: J. Magn. Reson.
       128.1 (1997), pp. 30–41.

    .. [3] Yingbo Hua. “Estimating two-dimensional frequencies by matrix
       enhancement and matrix pencil”. In: [Proceedings] ICASSP 91: 1991
       International Conference on Acoustics, Speech, and Signal Processing.
       IEEE. 1991, pp. 3073–3076.

    .. [4] Fang-Jiong Chen et al. “Estimation of two-dimensional frequencies
       using modified matrix pencil method”. In: IEEE Trans. Signal Process.
       55.2 (2007), pp. 718–724.
    """

    def __init__(self, data, sw, offset='zeros', M=0, fprint=True):

        # check data is a NumPy array
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f'{cols.R}data should be a numpy ndarray{cols.END}'
            )

        self.data = data

        # determine data dimension. If greater than 2, return error
        self.dim = self.data.ndim
        if self.dim >= 3:
            raise MoreThanTwoDimError()

        if offset is None:
            offset = [0.0] * self.dim

        for x in (sw, offset):
            if not isinstance(x, list) and len(x) == self.dim:
                raise TypeError(
                    f'{cols.R}sw and offset should be lists with the same'
                    f' number of elements and dimensions in the data{cols.END}'
                )

        self.sw = sw
        self.offset = offset

        if not isinstance(M, int) and M >= 0:
            raise ValueError(
                f'{cols.R}M should be an integer greater than or equal'
                f' to 0{cols.END}'
            )

        self.M_init = M

        if not isinstance(fprint, bool):
            raise TypeError(f'{cols.R}fprint should be an Boolean{cols.END}')

        self.fprint = fprint

        self._mpm()

    def get_parameters(self, unit='ppm'):
        return self.parameters

    def get_poles(self):
        return self.poles

    @timer
    @start_end_wrapper
    def _mpm(self):
        """Performs the appropriate algorithm, based on the data dimension."""

        if self.dim == 1:
            self._mpm_1d()

        elif self.dim == 2:
            self._mpm_2d()

    def _mpm_1d(self):

        # normalise data
        self.norm = nlinalg.norm(self.data)
        self.normed_data = self.data / self.norm

        # data size and pencil parameter
        self.N = self.normed_data.shape
        self._pencil_parameters()

        # Hankel matrix of data, Y, with dimensions (N-L) * L
        self._construct_Y()

        # singular value decomposition of Y
        # returns singular values (M-length vector)
        # and right singular values (LxL size matrix)
        self._svd()

        # number of oscillators
        self._mdl()

        self._signal_poles_1d()

        self._complex_amplitudes_1d()

        # construct Mx4 array of amps, phases, freqs, and damps
        self._construct_parameter_array()

        # removal of terms with negative damping factors
        self._negative_damping()


    @timer
    def _mpm_2d(self):
        pass

    def _pencil_parameters(self):
        """
        Determines the pencil parameter(s) for the 1D or 2D MPM

        Returns
        -------
        L : [int] or [int, int]
            Pencil parameter(s)

        Notes
        -----
        The pencil parameters are set to be :math:`\\lfloor N_d / 3 \\rfloor`,
        where :math:`N_d` is the number of data points in dimension :math:`d`.
        """

        if self.dim == 1:
            # optimal when between N/2 and N/3 (see Lin's paper)
            self.L = [int(np.floor(self.N[0]/3))]

        elif self.dim == 2:
            self.L = [
                int(np.floor((self.N[0] + 1) / 2)),
                int(np.floor((self.N[1] + 1) / 2))
            ]

        if self.fprint:
            msg = '--> Pencil Parameter(s): '
            msg += ' & '.join([str(L) for L in self.L])
            print(msg)

    def _construct_Y(self):
        """
        Wrapper around ``scipy.linalg.hankel()`` [1]_. Constructs Hankel
        data matrix Y.

        References
        ----------
        .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hankel.html
        """

        N = self.N[0]
        L = self.L[0]
        row = self.normed_data[0:N-L]
        column = self.normed_data[N-L-1:N]

        self.Y = slinalg.hankel(column, row)

        if self.fprint:
            print("--> Hankel data matrix constructed:")
            print(f'\tSize:   {self.Y.shape[0]} x {self.Y.shape[1]}')

            gibibytes = self.Y.nbytes / (2**30)

            if round(gibibytes, 4) >= 0.1:
                print(f'\tMemory: {round(gibibytes, 4)}GiB')
            else:
                print(f'\tMemory: {round(gibibytes * (2**10), 4)}MiB')


    @timer
    def _svd(self):
        """
        Wrapper around ``numpy.linalg.svd()`` [1]_. Computes SVD, and gives
        time. Also neglects left singular vectors which are not needed.

        References
        ----------
        .. [1] https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
        """

        if self.fprint:
            print('--> Performing Singular Value Decomposition...')

        _, self.sigma, Vh = nlinalg.svd(self.Y)
        self.V = Vh.T


    def _mdl(self):
        """
        Computes the MDL [1]_, with the option of printing information to the
        terminal.

        References
        ----------
        .. [1] M. Wax, T. Kailath, Detection of signals by information theoretic
           criteria, IEEE Transactions on Acoustics, Speech, and Signal Processing
           33 (2) (1985) 387–392.
        """

        # TODO: MAKE 2D COMPATIBLE

        if self.fprint:
            print('--> Determining number of oscillators...')

        if self.M_init == 0:
            N = self.N[0]
            L = self.L[-1]
            s = self.sigma
            mdl = np.zeros(L)

            if self.fprint:
                print('\tNumber of oscillators will be estimated using MDL')

            for k in range(L):
                mdl[k] = \
                    - N * np.einsum('i->', np.log(s[k:L])) \
                    + N * (L-k) * np.log((np.einsum('i->', s[k:L]) / (L-k))) \
                    + (k * np.log(N) * (2*L-k)) / 2

            self.M = np.argmin(mdl)

        else:
            self.M = self.M_init
            if self.fprint:
                print('\tNumber of oscillations has been pre-defined')

        if self.fprint:
            print(f'\tNumber of oscillations: {self.M}')

    @timer
    def _signal_poles_1d(self):
        """Computes the poles of a 1D signal, given a matrix of singular
        values and the number of poles, with the option of printing information
        to the terminal.
        """

        if self.fprint:
            print('--> Determining signal poles...')

        Vm = self.V[:, :self.M] # retain M first right singular vectors
        V1 = Vm[:-1, :] # remove last column
        V2 = Vm[1:, :] # remove first column

        # determine first M signal poles (others should be 0)
        z, _ = nlinalg.eig(V2 @ nlinalg.pinv(V1))
        self.poles = z[:self.M]

    @timer
    def _complex_amplitudes_1d(self):
        """
        Computes the complex amplitudes of a 1D signal, with the option of printing
        information (inc. time taken) to the terminal."""

        if self.fprint:
            print('--> Determining complex amplitudes...')

        n = np.arange(self.N[0])
        Z = (np.power.outer(self.poles, n)).T # Vandermonde matrix of poles
        self.alpha = nlinalg.pinv(Z) @ self.normed_data

    def _construct_parameter_array(self):

        amp = np.abs(self.alpha) * self.norm
        phase = np.arctan2(np.imag(self.alpha), np.real(self.alpha))
        freq = -(self.sw[0] / (2 * np.pi)) * np.imag(np.log(self.poles)) + self.offset[0]
        damp = - self.sw[0] * np.real(np.log(self.poles))

        self.parameters = (np.vstack((amp, phase, freq, damp))).T
        # order by frequency
        self.parameters = self.parameters[np.argsort(self.parameters[:, 2])]

    def _negative_damping(self):
        """Determines any oscillators with negative damping factors in the
        parameter array, and removes these from the array.
        """

        if self.fprint:
            print('--> Checking for oscillators with negative damping')

        M_before = self.parameters.shape[0]
        # indices of oscillators with negative damping factors
        neg_damp_idx = np.nonzero(self.parameters[:, 3] < 0.0)[0]
        self.parameters = np.delete(self.parameters, neg_damp_idx, axis=0)
        M_after = self.parameters.shape[0]

        if M_after < M_before and self.fprint:
            print(f'\t{cols.O}WARNING: Oscillations with negative damping\n'
                  f'\tfactors detected. These have been deleted.\n'
                  f'\tCorrected number of oscillations: {M_after}{cols.END}')

            self.M = M_after

        elif self.fprint:
            print('\tNone found')


# “Under capitalism, man exploits man.
# Under communism, it's just the opposite.”
# —————————————————John Kenneth Galbraith———
