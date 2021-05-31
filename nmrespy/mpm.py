# mpm.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Computation of signal estimates using the Matrix Pencil Method."""

import copy

import numpy as np
import numpy.linalg as nlinalg
import scipy.linalg as slinalg

import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
    colorama.init()
import nmrespy._errors as errors
from nmrespy._misc import (ArgumentChecker, FrequencyConverter,
                           start_end_wrapper)
from ._timing import timer


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

    M : int, default: 0
        The number of oscillators. If `0`, the number of oscilators will
        be estimated using the MDL.

    fprint : bool, default: True
        Flag specifiying whether to print infomation to the terminal as
        the method runs.

    start_point : int, default: 0
        The first timepoint sampled, in units of
        :math:`\\Delta t = 1 / f_{\\mathrm{sw}}`

    References
    ----------
    .. [#] M. Wax, T. Kailath, Detection of signals by information theoretic
       criteria, IEEE Transactions on Acoustics, Speech, and Signal Processing
       33 (2) (1985) 387–392.

    .. [#] Yingbo Hua and Tapan K Sarkar. “Matrix pencil method for estimating
       parameters of exponentially damped/undamped sinusoids in noise”. In:
       IEEE Trans. Acoust., Speech, Signal Process. 38.5 (1990), pp. 814–824.

    .. [#] Yung-Ya Lin et al. “A novel detection–estimation scheme for noisy
       NMR signals: applications to delayed acquisition data”. In: J. Magn.
       Reson. 128.1 (1997), pp. 30–41.

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

    def __init__(self, data, sw, offset=None, sfo=None, M=0, fprint=True,
                 start_point=0):
        """Checks validity of inputs, and if valid, calls :py:meth:`_mpm`"""

        try:
            self.dim = data.ndim
            if self.dim >= 3:
                raise errors.MoreThanTwoDimError()
        except Exception:
            raise TypeError(
                f'{cols.R}data should be a numpy ndarray{cols.END}'
            )

        if offset is None:
            offset = [0.0] * self.dim

        components = [
            (data, 'data', 'ndarray'),
            (sw, 'sw', 'float_list'),
            (offset, 'offset', 'float_list'),
            (M, 'M', 'positive_int_or_zero'),
            (fprint, 'fprint', 'bool'),
            (start_point, 'start_point', 'positive_int_or_zero'),
        ]

        if sfo is not None:
            components.append((sfo, 'sfo', 'float_list'))

        ArgumentChecker(components, dim=self.dim)

        self.data = data
        self.n = list(self.data.shape)
        self.sw = sw
        self.offset = offset
        self.sfo = sfo
        self.M = M
        self.fprint = fprint
        self.start_point = start_point

        if sfo is not None:
            self.converter = FrequencyConverter(
                self.n, self.sw, self.offset, self.sfo
            )

        if self.dim == 1:
            self._mpm_1d()
        else:
            self._mpm_2d()

    def get_result(self, freq_unit='hz'):
        """Obtain the result of the MPM.

        Parameters
        ----------
        freq_unit : 'hz' or 'ppm', default: 'hz'
            The unit of the oscillator frequencies (corresponding to
            ``result[:, 2]``)

        Returns
        -------
        result : numpy.ndarray
        """

        if freq_unit == 'hz':
            return self.result

        elif freq_unit == 'ppm':
            # Check whether a frequency converter is associated with the
            # class
            if 'converter' not in self.__dict__.keys():
                raise ValueError(
                    f'{cols.R}Insufficient information to determine'
                    f' frequencies in ppm. Did you perhaps forget to specify'
                    f' sfo?{cols.END}'
                )

            result = copy.deepcopy(self.result)

            # Get frequencies in Hz, and format to enable input into
            # the frequency converter.
            # Then convert values to ppm and reconvert back to NumPy array
            ppm = np.array(
                self.converter.convert(
                    [list(result[:, 2])], conversion='hz->ppm',
                )
            )
            result[:, 2] = ppm
            return result

        else:
            raise errors.InvalidUnitError('hz', 'ppm')

    @timer
    @start_end_wrapper(start_txt, end_txt)
    def _mpm_1d(self):
        """Performs 1-dimensional Matrix Pencil Method"""

        # Normalise data
        norm = nlinalg.norm(self.data)
        normed_data = self.data / norm

        # Number of points
        N = self.n[0]

        # Pencil parameter.
        # Optimal when between N/2 and N/3 (see Lin's paper)
        L = int(np.floor(N / 3))
        if self.fprint:
            print(f'--> Pencil Parameter: {L}')

        # Construct Hankel matrix
        row = normed_data[:N - L]
        column = normed_data[N - L - 1:]
        Y = slinalg.hankel(row, column)
        if self.fprint:
            print("--> Hankel data matrix constructed:")
            print(f'\tSize:   {Y.shape[0]} x {Y.shape[1]}')
            gibibytes = Y.nbytes / (2**30)
            if gibibytes >= 0.1:
                print(f'\tMemory: {round(gibibytes, 4)}GiB')
            else:
                print(f'\tMemory: {round(gibibytes * (2**10), 4)}MiB')

        # Singular value decomposition of Y
        # returns singular values: min(N-L, L)-length vector
        # and right singular vectors (LxL size matrix)
        if self.fprint:
            print('--> Performing Singular Value Decomposition...')
        sigma, Vh = nlinalg.svd(Y)[1:]
        V = Vh.T

        # Compute the MDL in order to estimate the number of oscillators
        if self.fprint:
            print('--> Determining number of oscillators...')

        if self.M == 0:
            if self.fprint:
                print('\tNumber of oscillators will be estimated using MDL')

            self.mdl = np.zeros(L)
            for k in range(L):
                self.mdl[k] = \
                    - N * np.einsum('i->', np.log(sigma[k:L])) + \
                    N * (L - k) * \
                    np.log(np.einsum('i->', sigma[k:L]) / (L - k)) + \
                    k * np.log(N) * (2 * L - k) / 2

            self.M = np.argmin(self.mdl)

        else:
            if self.fprint:
                print('\tNumber of oscillations has been pre-defined')

        if self.fprint:
            print(f'\tNumber of oscillations: {self.M}')

        # Determine signal poles
        if self.fprint:
            print('--> Determining signal poles...')

        Vm = V[:, :self.M]  # Retain M first right singular vectors
        V1 = Vm[:-1, :]  # Remove last column
        V2 = Vm[1:, :]  # Remove first column

        # Determine first M signal poles (others should be 0)
        z = nlinalg.eig(V2 @ nlinalg.pinv(V1))[0][:self.M]

        # Compute complex amplitudes
        if self.fprint:
            print('--> Determining complex amplitudes...')

        # Pseudoinverse of Vandermonde matrix of poles multiplied by
        # vector of complex amplitudes
        alpha = nlinalg.pinv(
            np.power.outer(
                z, np.arange(self.start_point, self.start_point + N)
            )
        ).T @ normed_data

        # Extract amplitudes, phases, frequencies and damping factors
        amp = np.abs(alpha) * norm
        phase = np.arctan2(np.imag(alpha), np.real(alpha))
        freq = \
            (self.sw[0] / (2 * np.pi)) * np.imag(np.log(z)) + self.offset[0]
        damp = - self.sw[0] * np.real(np.log(z))

        # Collate into (M x 4) array of parameters
        self.result = (np.vstack((amp, phase, freq, damp))).T
        # Order oscillators by frequency
        self.result = self.result[np.argsort(self.result[:, 2])]

        # Check for oscillators with negative damping
        if self.fprint:
            print('--> Checking for oscillators with negative damping...')

        m_init = self.result.shape[0]
        # Indices of oscillators with negative damping factors
        neg_damp_idx = np.nonzero(self.result[:, 3] < 0.0)[0]
        self.result = np.delete(self.result, neg_damp_idx, axis=0)
        self.M = self.result.shape[0]

        if self.M < m_init and self.fprint:
            print(f'\t{cols.OR}WARNING: Oscillations with negative damping\n'
                  f'\tfactors detected. These have been deleted.\n'
                  f'\tCorrected number of oscillations: {self.M}{cols.END}')

        elif self.fprint:
            print('\tNone found')

    @timer
    def _mpm_2d(self):
        """Performs 2-dimensional Modified Matrix Enhanced Pencil Method.

        .. todo ::
           To be written
        """

        raise errors.TwoDimUnsupportedError()
