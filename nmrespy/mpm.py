# mpm.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Computation of signal estimates using the Matrix Pencil Method."""

import copy
from typing import Iterable, Union

import numpy as np
import numpy.linalg as nlinalg
import scipy.linalg as slinalg
from scipy import sparse
import scipy.sparse.linalg as splinalg

from nmrespy import RED, ORA, END, USE_COLORAMA, ExpInfo
if USE_COLORAMA:
    import colorama
    colorama.init()
import nmrespy._errors as errors
from ._misc import ArgumentChecker, start_end_wrapper
from ._timing import timer


class MatrixPencil:
    """Matrix Pencil Method with model order selection.

    Model order selection achieved using the Minimum Description Length
    (MDL) [#]_. Supports analysis of one-dimensional [#]_ [#]_ or
    two-dimensional data [#]_ [#]_

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

    def __init__(self, data: np.ndarray, expinfo: ExpInfo, *, M: int = 0,
                 start_point: Union[Iterable[int], None] = None,
                 fprint: bool = True) -> None:
        """Initialise the class.

        Check validity of inputs, and runs the MPM.

        Parameters
        ----------
        data
            Signal to be considered (unnormalised).

        expinfo
            Information on the experiment. Used to determine the sweep width,
            transmitter offset, and (optional) transmitter frequency in MHz.
            Transmitter offset is optional, however if it set it `None`, no
            conversion of result frequencies from Hz to ppm will be possible.

        M
            The number of oscillators. If ``0``, the number of oscilators will
            be estimated using the MDL.

        start_point
            For signals that have been truncated at the beginning, this
            specifies the index of the initial point in the full, untruncated
            signal. If ``None``, it will be assumed that the signal is not
            truncated at the beginning (i.e. the first point occurs at time
            zero).

        fprint
            Flag specifiying whether to print infomation to the terminal as
            the method runs.
        """
        self.expinfo = expinfo
        if not isinstance(expinfo, ExpInfo):
            raise TypeError(f'{RED}Check `expinfo` is valid.{END}')
        dim = self.expinfo.unpack('dim')

        try:
            if dim != data.ndim:
                raise ValueError(
                    f'{RED}The dimension of `expinfo` does not agree with the '
                    f'number of dimensions in `data`.{END}'
                )
            elif dim >= 3:
                raise errors.MoreThanTwoDimError()
        except AttributeError:
            # data.ndim raised an attribute error
            raise TypeError(
                f'{RED}`data` should be a numpy ndarray{END}'
            )

        if start_point is None:
            start_point = [0] * dim

        checker = ArgumentChecker(dim=dim)
        checker.stage(
            (data, 'data', 'ndarray'),
            (start_point, 'start_point', 'int_iter'),
            (M, 'M', 'positive_int_or_zero'),
            (fprint, 'fprint', 'bool'),
        )
        checker.check()
        self.__dict__.update(locals())
        self.expinfo.pts = self.data.shape

        if dim == 1:
            self._mpm_1d()
        else:
            self._mpm_2d()

    def get_result(self, freq_unit: str = 'hz') -> np.ndarray:
        """Obtain the result of the MPM.

        Parameters
        ----------
        freq_unit
            The unit of the oscillator frequencies. Should be one of
            ``'hz'``, ``'ppm'``.

        Returns
        -------
        result
            Eetimation result from the MPM.
        """
        if freq_unit == 'hz':
            return self.result

        elif freq_unit == 'ppm':
            if self.expinfo.sfo is None:
                raise ValueError(
                    f'{RED}Insufficient information to determine'
                    f' frequencies in ppm. Did you perhaps forget to specify'
                    f' `sfo` in `expinfo`?{END}'
                )

            result = copy.deepcopy(self.result)

            if self.dim == 1:
                result[:, 2] /= self.expinfo.sfo[0]
            elif self.dim == 2:
                result[:, 2] /= self.expinfo.sfo[0]
                result[:, 3] /= self.expinfo.sfo[1]
            return result

        else:
            raise errors.InvalidUnitError('hz', 'ppm')

    @timer
    @start_end_wrapper('MPM STARTED', 'MPM COMPLETE')
    def _mpm_1d(self) -> None:
        """Perform 1-dimensional Matrix Pencil Method."""
        # Normalise data
        norm = nlinalg.norm(self.data)
        normed_data = self.data / norm

        # Number of points
        N = self.expinfo.unpack('pts')[0]

        # Pencil parameter.
        # Optimal when between N/2 and N/3 (see Lin's paper)
        L = int(np.floor(N / 3))
        if self.fprint:
            print(f'--> Pencil Parameter: {L}')

        # Construct Hankel matrix
        Y = slinalg.hankel(normed_data[:N - L], normed_data[N - L - 1:])

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
            print('--> Computing number of oscillators...')

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
            print('--> Computing signal poles...')

        Vm = V[:, :self.M]  # Retain M first right singular vectors
        V1 = Vm[:-1, :]  # Remove last column
        V2 = Vm[1:, :]  # Remove first column

        # Determine first M signal poles (others should be 0)
        poles = nlinalg.eig(V2 @ nlinalg.pinv(V1))[0][:self.M]

        # Compute complex amplitudes
        if self.fprint:
            print('--> Computing complex amplitudes...')

        # Pseudoinverse of Vandermonde matrix of poles multiplied by
        # vector of complex amplitudes
        sp = self.start_point[0]
        alpha = nlinalg.pinv(
            np.power.outer(poles, np.arange(sp, N + sp))
        ).T @ normed_data

        params = self._generate_params(alpha, poles.reshape((1, self.M)))
        params[:, 0] *= norm
        self.result, self.M = self._remove_negative_damping(params)

    @timer
    @start_end_wrapper('MMEMP STARTED', 'MMEMP COMPLETE')
    def _mpm_2d(self):
        """Perform 2-dimensional Modified Matrix Enhanced Pencil Method."""
        # Normalise data
        norm = nlinalg.norm(self.data)
        normed_data = self.data / norm

        # Number of points
        N1, N2 = self.expinfo.unpack('pts')

        # Pencil parameters
        K, L = tuple([int((n + 1) / 2) for n in (N1, N2)])
        if self.fprint:
            print(f'--> Pencil parameters: {K}, {L}')

        # TODO: MDL for 2D
        if self.M == 0:
            raise ValueError(
                f'{RED}Model order selection is not yet available for 2D '
                f'data. Set `M` as greater than 0.{END}'
            )

        # --- Enhanced Matrix ---
        X = slinalg.hankel(normed_data[0, :L], normed_data[0, L - 1:N2])
        for n1 in range(1, N1):
            blk = slinalg.hankel(
                normed_data[n1, :L], normed_data[n1, L - 1:N2]
            )
            X = np.hstack((X, blk))

        # vertically stack rows of block matricies to get Xe
        Xe = X[:, 0:(N1 - K + 1) * (N2 - L + 1)]

        for k in range(1, K):
            Xe = np.vstack(
                (Xe, X[:, k * (N2 - L + 1):(k + N1 - K + 1) * (N2 - L + 1)])
            )

        if self.fprint:
            print('--> Enhanced Block Hankel matrix constructed:')
            print(f'\tSize: {Xe.shape[0]} x {Xe.shape[1]}')
            gibibytes = Xe.nbytes / (2 ** 30)
            if gibibytes >= 0.1:
                print(f'\tMemory: {round(gibibytes, 4)}GiB')
            else:
                print(f'\tMemory: {round(gibibytes * (2**10), 4)}MiB')

        # convert Xe to sparse matrix
        sparse_Xe = sparse.csr_matrix(Xe)

        if self.fprint:
            print('--> Performing Singular Value Decomposition...')
        U, *_ = splinalg.svds(sparse_Xe, self.M)

        # --- Permutation matrix ---
        if self.fprint:
            print('--> Computing Permutation matrix...')

        # Create first row of matrix: [1, 0, 0, ..., 0]
        fst_row = sparse.lil_matrix((1, K * L))
        fst_row[0, 0] = 1

        # Seed the permutation matrix
        P = copy.deepcopy(fst_row)

        # First block of K rows of permutation matrix
        for k in range(1, K):
            row = sparse.hstack((fst_row[:, -(k * L):], fst_row[:, :-(k * L)]))
            P = sparse.vstack((P, row))

        # first K-sized block of matrix
        fst_blk = copy.deepcopy(P).tolil()

        # Stack K-sized blocks to form (K * L) row matrix
        for el in range(1, L):
            blk = sparse.hstack((fst_blk[:, -el:], fst_blk[:, :-el]))
            P = sparse.vstack((P, blk))

        P = P.todense()

        # --- Signal Poles ---
        # retain only M principle left s. vecs
        if self.fprint:
            print('--> Computing signal poles...')

        Us = U[:, :self.M]
        U1 = Us[:Us.shape[0] - L, :]  # last L rows deleted
        U2 = Us[L:, :]                # first L rows deleted
        eig_y, vec_y = nlinalg.eig(nlinalg.pinv(U1) @ U2)
        Usp = P @ Us
        U1p = Usp[:Usp.shape[0] - K, :]  # last K rows deleted
        U2p = Usp[K:, :]                 # first K rows deleted
        eig_z = np.diag(nlinalg.solve(vec_y, nlinalg.pinv(U1p)) @ U2p @ vec_y)
        poles = np.hstack((eig_y, eig_z)).reshape((2, self.M))

        # --- Complex Amplitudes ---
        if self.fprint:
            print('--> Computing complex amplitudes...')

        ZL = np.power.outer(poles[1], np.arange(L)).T
        ZR = np.power.outer(poles[1], np.arange(N2 - L + 1))
        Yd = np.diag(poles[0])

        EL = copy.deepcopy(ZL)
        for k in range(1, K):
            EL = np.vstack((EL, ZL @ (Yd ** k)))

        ER = copy.deepcopy(ZR)
        for m in range(1, N1 - K + 1):
            ER = np.hstack((ER, (Yd ** m) @ ZR))

        alpha = np.diag(nlinalg.pinv(EL) @ Xe @ nlinalg.pinv(ER))

        params = self._generate_params(alpha, poles)
        params[:, 0] *= norm
        self.result, self.M = self._remove_negative_damping(params)

    def _generate_params(
        self, alpha: np.ndarray, poles: np.ndarray
    ) -> np.ndarray:
        """Convert complex amplitudes and signal poles to parameter array.

        Parameters
        ----------
        alpha
            Complex amplitude array, of shape``(self.M,)``

        poles
            Signal pole array, of shape ``(self.dim, self.M)``

        Returns
        -------
        result
            Parameter array, of shape ``(M, 2 * dim + 2)`` where
            ``M`` is the number of oscillators in the result and ``dim`` is
            the data dimension.
        """
        sw, offset = self.expinfo.unpack('sw', 'offset')
        amp = np.abs(alpha)
        phase = np.arctan2(np.imag(alpha), np.real(alpha))
        freq = np.vstack(
            tuple([(sw_ / (2 * np.pi)) * np.imag(np.log(poles_)) + offset_
                   for sw_, offset_, poles_ in zip(sw, offset, poles)])
        )
        damp = np.vstack(
            tuple([-sw_ * np.real(np.log(poles_))
                   for sw_, poles_ in zip(sw, poles)])
        )

        result = np.vstack((amp, phase, freq, damp)).T
        return result[np.argsort(result[:, 2])]

    def _remove_negative_damping(self, params: np.ndarray) -> np.ndarray:
        """Determine negative amplitude oscillators and remove.

        Parameters
        ----------
        params
            Parameter array, with shape ``(self.M, 2 * self.dim + 2)``

        Returns
        -------
        ud_params
            Updated parameter array, with negative damping oscillators removed,
            with shape ``(M_new, 2 * dim + 2)``, where
            ``M_new <= param.shape[0]`` and ``dim`` is the data dimension.
        """
        if self.fprint:
            print('--> Checking for oscillators with negative damping...')

        M_init = params.shape[0]
        # Indices of oscillators with negative damping factors
        neg_dmp_idx = set()
        for i in range(2 + self.dim, 2 * (self.dim + 1)):
            neg_dmp_idx = neg_dmp_idx | set(np.nonzero(params[:, i] < 0.0)[0])

        ud_params = np.delete(params, list(neg_dmp_idx), axis=0)
        M = ud_params.shape[0]

        if M < M_init and self.fprint:
            print(f'\t{ORA}WARNING: Oscillations with negative damping\n'
                  f'\tfactors detected. These have been deleted.\n'
                  f'\tCorrected number of oscillations: {M}{END}')

        elif self.fprint:
            print('\tNone found')

        return ud_params, M
