# mpm.py
# matrix pencil method for analysis of 1D and 2D time series
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Module for computation of signal estimates using the Matrix Pencil Method."""

import copy
import functools

import numpy as np
import numpy.linalg as nlinalg
import scipy.linalg as slinalg
from scipy import sparse
import scipy.sparse.linalg as splinalg

from nmrespy import *
import nmrespy._errors as errors
from ._misc import start_end_wrapper
from ._timing import timer
import nmrespy._cols as cols

if cols.USE_COLORAMA:
    import colorama


class MatrixPencil(FrequencyConverter):
    """Class for performing the Matrix Pencil Method with the option of model
    order selection using the Minimum Description Length (MDL) [#]_. Supports
    analysis of one-dimensional [#]_ [#]_ or two-dimensional data [#]_ [#]_

    Parameters
    ----------
    data : numpy.ndarray
        Signal to be considered (unnormalised).

    sw : [float] or [float, float]
        The experiment sweep width in each dimension in Hz.

    offset : [float], [float, float] or None, default: None
        The experiment transmitter offset frequency in Hz.

    sfo : [float], [float, float] or None, default: None
        The experiment transmitter frequency in each dimension in MHz. This is
        not necessary, however if it set it `None`, no conversion from Hz
        to ppm will be possible!

    m : int, default: 0
        The number of oscillators. If ``0``, the number of oscilators will
        be estimated using the MDL.

    fprint : bool, default: True
        Flag specifiying whether to print infomation to the terminal as
        the method runs.

    References
    ----------
    .. [#] M. Wax, T. Kailath, Detection of signals by information theoretic
       criteria, IEEE Transactions on Acoustics, Speech, and Signal Processing
       33 (2) (1985) 387–392.

    .. [#] Yingbo Hua and Tapan K Sarkar. “Matrix pencil method for estimating
       parameters of exponentially damped/undamped sinusoids in noise”. In:
       IEEE Trans. Acoust., Speech, Signal Process. 38.5 (1990), pp. 814–824.

    .. [#] Yung-Ya Lin et al. “A novel detection–estimation scheme for noisy NMR
       signals: applications to delayed acquisition data”. In: J. Magn. Reson.
       128.1 (1997), pp. 30–41.

    .. [#] Yingbo Hua. “Estimating two-dimensional frequencies by matrix
       enhancement and matrix pencil”. In: [Proceedings] ICASSP 91: 1991
       International Conference on Acoustics, Speech, and Signal Processing.
       IEEE. 1991, pp. 3073–3076.

    .. [#] Fang-Jiong Chen et al. “Estimation of two-dimensional frequencies
       using modified matrix pencil method”. In: IEEE Trans. Signal Process.
       55.2 (2007), pp. 718–724.
    """

    start_txt = 'MATRIX PENCIL METHOD STARTED'
    end_txt = 'MATRIX PENCIL METHOD COMPLETE'

    def __init__(self, data, sw, offset='zeros', sfo=None, m=0, fprint=True):
        """Checks validity of inputs, and if valid, calls :py:meth:`_mpm`"""

        # check data is a NumPy array
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f'{cols.R}data should be a numpy ndarray{cols.END}'
            )

        self.data = data

        # determine data dimension. If greater than 2, return error
        self.dim = self.data.ndim
        if self.dim >= 3:
            raise errors.MoreThanTwoDimError()

        # if offset is None, set it to zero in each dimension
        if offset is None:
            offset = [0.0] * self.dim

        to_check = [sw, offset]

        if sfo != None:
            self.sfo = sfo
            to_check.append(sfo)

        for x in to_check:
            if not isinstance(x, list) and len(x) == self.dim:
                raise TypeError(
                    f'{cols.R}sw and offset should be lists with the same'
                    f' number of elements as dimensions in the data{cols.END}'
                )

        self.n = list(self.data.shape)
        self.sw = sw
        self.offset = offset

        if 'sfo' in self.__dict__:
            self.converter = FrequencyConverter(
                self.n, self.sw, self.offset, self.sfo
            )

        if not isinstance(m, int) and m >= 0:
            raise ValueError(
                f'{cols.R}M should be an integer greater than or equal'
                f' to 0{cols.END}'
            )

        self.m_init = m

        if not isinstance(fprint, bool):
            raise TypeError(f'{cols.R}fprint should be an Boolean{cols.END}')

        self.fprint = fprint

        self._mpm()


    def get_parameters(self, unit='hz'):

        if unit == 'hz':
            return self.parameters

        elif unit == 'ppm':
            # get frequencies in Hz
            hz = [list(self.parameters[:, 2])]
            # convert to ppm
            ppm = np.array(self.converter.convert(hz, conversion='hz->ppm'))
            parameters_ppm = copy.deepcopy(self.parameters)
            parameters_ppm[:, 2] = ppm
            return parameters_ppm

        else:
            raise errors.InvalidUnitError('hz', 'ppm')


    @timer
    @start_end_wrapper(start_txt, end_txt)
    def _mpm(self):
        """Performs the appropriate algorithm, based on the data dimension."""

        if self.dim == 1:
            self._mpm_1d()

        elif self.dim == 2:
            self._mpm_2d()

    def _mpm_1d(self):
        """Performs 1-dimensional Matrix Pencil Method"""

        # normalise data
        self.norm = nlinalg.norm(self.data)
        self.normed_data = self.data / self.norm

        # pencil parameter
        self._pencil_parameters()

        # Hankel matrix of data, Y, with dimensions (N-L) * L
        self._construct_y()

        # singular value decomposition of Y
        # returns singular values: min(N-L, L)-length vector
        # and right singular vectors (LxL size matrix)
        self._svd()

        # determine number of oscillators
        self._mdl()

        self._signal_poles_1d()

        self._complex_amplitudes_1d()

        # construct Mx4 array of amps, phases, freqs, and damps
        self._construct_parameter_array()

        # removal of terms with negative damping factors
        self._negative_damping()

        # at this point, the result is contained in self.parameters
        # this can be access by the self.get_parameters() method


    @timer
    def _mpm_2d(self):
        """Performs 2-dimensional Modified Matrix Enhanced Pencil Method.

        .. todo ::
           To be written
        """

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
            self.l = [int(np.floor(self.n[0]/3))]

        elif self.dim == 2:
            self.l = [
                int(np.floor((self.n[0] + 1) / 2)),
                int(np.floor((self.n[1] + 1) / 2))
            ]

        if self.fprint:
            msg = '--> Pencil Parameter(s): '
            msg += ' & '.join([str(L) for L in self.l])
            print(msg)

    def _construct_y(self):
        """
        Wrapper around `scipy.linalg.hankel <https://docs.scipy.org/doc/\
        scipy/reference/generated/scipy.linalg.hankel.html>`_ Constructs Hankel
        data matrix.
        """

        n = self.n[0]
        l = self.l[0]
        row = self.normed_data[0:n-l]
        column = self.normed_data[n-l-1:n]

        self.y = slinalg.hankel(column, row)

        if self.fprint:
            print("--> Hankel data matrix constructed:")
            print(f'\tSize:   {self.y.shape[0]} x {self.y.shape[1]}')

            gibibytes = self.y.nbytes / (2**30)

            if gibibytes >= 0.1:
                print(f'\tMemory: {round(gibibytes, 4)}GiB')
            else:
                print(f'\tMemory: {round(gibibytes * (2**10), 4)}MiB')


    @timer
    def _svd(self):
        """
        Wrapper around `numpy.linalg.svd <https://numpy.org/doc/stable/\
        reference/generated/numpy.linalg.svd.html>_. Computes SVD, and gives
        time. Also neglects left singular vectors which are not needed.
        """

        if self.fprint:
            print('--> Performing Singular Value Decomposition...')

        _, self.sigma, vh = nlinalg.svd(self.y)
        self.v = vh.T


    def _mdl(self):
        """
        Computes the MDL with the option of printing information to the
        terminal.
        """

        # TODO: MAKE 2D COMPATIBLE

        if self.fprint:
            print('--> Determining number of oscillators...')

        if self.m_init == 0:
            n = self.n[0]
            l = self.l[-1]
            s = self.sigma
            mdl = np.zeros(l)

            if self.fprint:
                print('\tNumber of oscillators will be estimated using MDL')

            for k in range(l):
                mdl[k] = \
                    - n * np.einsum('i->', np.log(s[k:l])) \
                    + n * (l-k) * np.log((np.einsum('i->', s[k:l]) / (l-k))) \
                    + (k * np.log(n) * (2*l-k)) / 2

            self.m = np.argmin(mdl)

        else:
            self.m = self.m_init
            if self.fprint:
                print('\tNumber of oscillations has been pre-defined')

        if self.fprint:
            print(f'\tNumber of oscillations: {self.m}')

    @timer
    def _signal_poles_1d(self):
        """Computes the poles of a 1D signal, given a matrix of singular
        values and the number of poles, with the option of printing information
        to the terminal.
        """

        if self.fprint:
            print('--> Determining signal poles...')

        vm = self.v[:, :self.m] # retain M first right singular vectors
        v1 = vm[:-1, :] # remove last column
        v2 = vm[1:, :] # remove first column

        # determine first M signal poles (others should be 0)
        z, _ = nlinalg.eig(v2 @ nlinalg.pinv(v1))
        self.poles = z[:self.m]

    @timer
    def _complex_amplitudes_1d(self):
        """
        Computes the complex amplitudes of a 1D signal, with the option of printing
        information (inc. time taken) to the terminal."""

        if self.fprint:
            print('--> Determining complex amplitudes...')

        n = np.arange(self.n[0])

        # pseudoinverse of Vandermonde matrix of poles multiplied by
        # vector of complex amplitudes
        self.alpha = \
            nlinalg.pinv((np.power.outer(self.poles, n)).T) @ self.normed_data

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

            self.m = M_after

        elif self.fprint:
            print('\tNone found')



# “Under capitalism, man exploits man.
# Under communism, it's just the opposite.”
# —————————————————John Kenneth Galbraith———
