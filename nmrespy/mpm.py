# mpm.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 29 Sep 2023 11:07:27 BST

"""Computation of NMR parameter estimates using the Matrix Pencil Method.

MWE
---

.. literalinclude:: examples/mpm_example.py
"""

import copy
import itertools
from typing import Iterable, Union

import numpy as np
import numpy.linalg as nlinalg
import scipy.linalg as slinalg
from scipy import sparse
import scipy.sparse.linalg as splinalg
from scipy.signal import argrelextrema

from nmrespy import ExpInfo
from nmrespy._colors import RED, ORA, END, USE_COLORAMA
import nmrespy._errors as errors
from nmrespy._result_fetcher import ResultFetcher
from nmrespy._sanity import sanity_check, funcs as sfuncs
from ._misc import start_end_wrapper
from ._timing import timer

if USE_COLORAMA:
    import colorama
    colorama.init()


class MatrixPencil(ResultFetcher):
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

    def __init__(
        self,
        expinfo: ExpInfo,
        data: np.ndarray,
        oscillators: int = 0,
        start_point: Union[Iterable[int], None] = None,
        output_mode: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        expinfo
            Experiment information.

        data
            Signal to be considered.

        oscillators
            The number of oscillators. If ``0``, the number of oscilators will
            be estimated using the MDL.

        start_point
            For signals that have been truncated at the beginning, this
            specifies the index of the initial point in the full, untruncated
            signal. If ``None``, it will be assumed that the signal is not
            truncated at the beginning (i.e. the first point occurs at time
            zero).

        output_mode
            Flag specifiying whether to print infomation to the terminal as
            the method runs.
        """
        sanity_check(
            ("expinfo", expinfo, sfuncs.check_expinfo),
            ("oscillators", oscillators, sfuncs.check_int, (), {"min_value": 0}),
            ("output_mode", output_mode, sfuncs.check_bool),
        )
        sanity_check(
            ("data", data, sfuncs.check_ndarray, (expinfo.dim,)),
            (
                "start_point", start_point, sfuncs.check_int_list, (),
                {
                    "length": expinfo.dim,
                    "len_one_can_be_listless": True,
                    "must_be_positive": True,
                },
                True
            ),
        )

        self.dim, self.sw, self.offset, sfo = expinfo.unpack(
            "dim", "sw", "offset", "sfo",
        )
        self.data = data
        self.oscillators = oscillators
        self.start_point = start_point
        if self.start_point is None:
            self.start_point = [0] * self.dim
        self.output_mode = output_mode

        # Init ResultFetcher
        super().__init__(sfo)

        if self.dim > 2:
            raise errors.MoreThanTwoDimError()

        if self.dim == 1:
            self._mpm_1d()
        elif self.dim == 2:
            self._mpm_2d()

    def _mdl_1d(self, sigma: np.ndarray, N: int) -> int:
        """1D MDL.

        Parameters
        ----------
        sigma
            Singular values obtained from Hankel data matrix.

        N
            Number of points the signal is made of
        """
        L = sigma.size
        self.mdl = np.zeros(L)
        for k in range(L):
            self.mdl[k] = (
                -N * np.einsum("i->", np.log(sigma[k:])) +
                N * (L - k) * np.log(np.einsum("i->", sigma[k:]) / (L - k)) +
                k * np.log(N) * (2 * L - k) / 2
            )

        return argrelextrema(self.mdl, np.less)[0][0]

    @timer
    @start_end_wrapper("MPM STARTED", "MPM COMPLETE")
    def _mpm_1d(self) -> None:
        """Perform 1-dimensional Matrix Pencil Method."""
        # Normalise data
        norm = nlinalg.norm(self.data)
        normed_data = self.data / norm

        # Number of points
        N = self.data.size

        # Pencil parameter.
        # Optimal when between N/2 and N/3 (see Lin's paper)
        L = int(np.floor(N / 3))
        if self.output_mode:
            print(f"--> Pencil Parameter: {L}")

        # Construct Hankel matrix
        Y = slinalg.hankel(normed_data[: N - L], normed_data[N - L - 1 :])

        if self.output_mode:
            print("--> Hankel data matrix constructed:")
            print(f"\tSize:   {Y.shape[0]} x {Y.shape[1]}")
            gibibytes = Y.nbytes / (2 ** 30)
            if gibibytes >= 0.1:
                print(f"\tMemory: {round(gibibytes, 4)}GiB")
            else:
                print(f"\tMemory: {round(gibibytes * (2**10), 4)}MiB")

        # Singular value decomposition of Y
        # returns singular values: min(N-L, L)-length vector
        # and right singular vectors (LxL size matrix)
        if self.output_mode:
            print("--> Performing Singular Value Decomposition...")
        _, sigma, Vh = nlinalg.svd(Y)
        V = Vh.T

        # Compute the MDL in order to estimate the number of oscillators
        if self.output_mode:
            print("--> Computing number of oscillators...")

        if self.oscillators == 0:
            if self.output_mode:
                print("\tNumber of oscillators will be estimated using MDL")
            self.oscillators = self._mdl_1d(sigma, N)

        else:
            if self.output_mode:
                print("\tNumber of oscillations has been pre-defined")

        if self.output_mode:
            print(f"\tNumber of oscillations: {self.oscillators}")

        if self.oscillators == 0:
            if self.output_mode:
                print("No oscillators detected!")
                self.params = None
                return

        # Determine signal poles
        if self.output_mode:
            print("--> Computing signal poles...")

        Vm = V[:, : self.oscillators]  # Retain M first right singular vectors
        V1 = Vm[:-1, :]  # Remove last column
        V2 = Vm[1:, :]  # Remove first column

        # Determine M signal poles
        V1inv = nlinalg.pinv(V1)
        V1invV2 = V1inv @ V2
        poles, _ = nlinalg.eig(V1invV2)

        # Compute complex amplitudes
        if self.output_mode:
            print("--> Computing complex amplitudes...")

        # Pseudoinverse of Vandermonde matrix of poles multiplied by
        # vector of complex amplitudes
        sp = self.start_point[0]
        alpha = (
            nlinalg.pinv(np.power.outer(poles, np.arange(sp, N + sp))).T @ normed_data
        )

        params = self._generate_params(alpha, poles.reshape((1, self.oscillators)))
        params[:, 0] *= norm
        self.params, self.oscillators = self._remove_negative_damping(params)

    @timer
    @start_end_wrapper("MMEMP STARTED", "MMEMP COMPLETE")
    def _mpm_2d(self):
        """Perform 2-dimensional Modified Matrix Enhanced Pencil Method."""
        # Number of points
        N1, N2 = self.data.shape
        # Normalise data
        norm = nlinalg.norm(self.data)
        normed_data = self.data / norm

        if self.output_mode:
            print("--> Computing number of oscillators...")

        if self.oscillators == 0:
            if self.output_mode:
                print(
                    "\tNumber of oscillators will be estimated using MDL on first "
                    "t1 increment."
                )
            # Construct Hankel matrix of first t1 increment, and perform MDL.
            L_mdl = int(np.floor(N2 / 3))
            self.oscillators = self._mdl_1d(
                nlinalg.svd(
                    slinalg.hankel(
                        normed_data[0, : N2 - L_mdl],
                        normed_data[0, N2 - L_mdl - 1 :],
                    )
                )[1],
                N2,
            )

        else:
            if self.output_mode:
                print("\tNumber of oscillators has been pre-defined")

        if self.output_mode:
            print(f"\tNumber of oscillators: {self.oscillators}")

        if self.oscillators == 0:
            if self.output_mode:
                print("No oscillators detected!")
                self.params = None
                return

        # Pencil parameters
        L1, L2 = tuple([int((n + 1) / 2) for n in (N1, N2)])
        if self.output_mode:
            print(f"--> Pencil parameters: {L1}, {L2}")

        # === Construct block Hankel EY ===
        row_size = L2
        col_size = N2 - L2 + 1
        EY = np.zeros(
            (L1 * L2, (N1 - L1 + 1) * (N2 - L2 + 1)),
            dtype="complex",
        )
        for n1 in range(N1):
            col = normed_data[n1, :L2]
            row = normed_data[n1, L2 - 1:]
            HYn1 = slinalg.hankel(col, row)
            for n in range(n1 + 1):
                r, c = n, n1 - n
                if r < L1 and c < N1 - L1 + 1:
                    EY[
                        r * row_size : (r + 1) * row_size,
                        c * col_size : (c + 1) * col_size
                    ] = HYn1

        if self.output_mode:
            print("--> Enhanced Block Hankel matrix constructed:")
            print(f"\tSize: {EY.shape[0]} x {EY.shape[1]}")
            gibibytes = EY.nbytes / (2 ** 30)
            if gibibytes >= 0.1:
                print(f"\tMemory: {round(gibibytes, 4)}GiB")
            else:
                print(f"\tMemory: {round(gibibytes * (2**10), 4)}MiB")


        sparse_EY = sparse.csr_array(EY)

        if self.output_mode:
            print("--> Performing Singular Value Decomposition...")
        UM, *_ = sparse.linalg.svds(EY, k=self.oscillators)

        # === Construct permutation matrix ===
        P = np.zeros((L1 * L2, L1 * L2))
        r = 0
        for l2 in range(L2):
            for l1 in range(L1):
                c = l1 * L2 + l2
                P[r, c] = 1
                r += 1
        UM1 = UM[: (L1 - 1) * L2]
        UM2 = UM[L2:]
        z1, W1 = nlinalg.eig(nlinalg.pinv(UM1) @ UM2)

        UMP = P @ UM
        UMP1 = UMP[: L1 * (L2 - 1)]
        UMP2 = UMP[L1:]
        G = nlinalg.inv(W1) @ nlinalg.pinv(UMP1) @ UMP2 @ W1
        z2 = np.diag(G).copy()  # copy needed as slice is readonly

        # === Check for and deal with similar frequencies in f1 ===
        freq1 = (0.5 * self.sw[0] / np.pi) * np.imag(np.log(z1)) + self.offset[0]
        threshold = self.sw[0] / N1
        groupings = {}
        for idx, f1 in enumerate(freq1):
            assigned = False
            for group_f1, indices in groupings.items():
                if np.abs(f1 - group_f1) < threshold:
                    indices.append(idx)
                    n = len(indices)
                    indices = sorted(indices)
                    # Get new mean freq of the group
                    new_group_f1 = (n * group_f1 + f1) / (n + 1)
                    groupings[new_group_f1] = groupings.pop(group_f1)
                    assigned = True
                    break
            if not assigned:
                groupings[f1] = [idx]

        for indices in groupings.values():
            n = len(indices)
            if n != 1:
                Gr_slice = tuple(zip(*itertools.product(indices, repeat=2)))
                Gr = G[Gr_slice].reshape(n, n)
                new_group_z2, _ = np.linalg.eig(Gr)
                z2[indices] = new_group_z2  # $\label{ln:similar-f-end}$

        # === EL and ER ===
        ZL2 = np.power.outer(z2, np.arange(L2)).T
        ZR2 = np.power.outer(z2, np.arange(N2 - L2 + 1))
        Z1D = np.diag(z1)

        EL = np.zeros((L1 * L2, self.oscillators), dtype="complex")
        Z2LZ1D = ZL2
        for i in range(L1):
            EL[i * row_size : (i + 1) * row_size] = Z2LZ1D
            Z2LZ1D = Z2LZ1D @ Z1D

        ER = np.zeros(
            (
                self.oscillators,
                (N1 - L1 + 1) * (N2 - L2 + 1),
            ),
            dtype="complex",
        )
        Z1DZ2R = ZR2
        for i in range(N1 - L1 + 1):
            ER[:, i * col_size : (i + 1) * col_size] = Z1DZ2R
            Z1DZ2R = Z1D @ Z1DZ2R
        alpha = np.diag(np.linalg.pinv(EL) @ EY @ np.linalg.pinv(ER))
        poles = np.hstack((z1, z2)).reshape((2, self.oscillators))
        params = self._generate_params(alpha, poles)
        params[:, 0] *= norm
        self.params, self.oscillators = self._remove_negative_damping(params)

    def _generate_params(self, alpha: np.ndarray, poles: np.ndarray) -> np.ndarray:
        """Convert complex amplitudes and signal poles to parameter array.

        Parameters
        ----------
        alpha
            Complex amplitude array, of shape``(self.oscillators,)``

        poles
            Signal pole array, of shape ``(dim, self.oscillators)``

        Returns
        -------
        result
            Parameter array, of shape ``(M, 2 * dim + 2)`` where
            ``M`` is the number of oscillators in the result and ``dim`` is
            the data dimension.
        """
        amp = np.abs(alpha)
        phase = np.arctan2(np.imag(alpha), np.real(alpha))
        freq = np.vstack(
            tuple(
                [
                    (sw / (2 * np.pi)) * np.imag(np.log(poles_)) + offset
                    for sw, offset, poles_ in zip(self.sw, self.offset, poles)
                ]
            )
        )
        damp = np.vstack(
            tuple(
                [-sw * np.real(np.log(poles_)) for sw, poles_ in zip(self.sw, poles)]
            )
        )

        result = np.vstack((amp, phase, freq, damp)).T
        return result[np.argsort(result[:, 2])]

    def _remove_negative_damping(self, params: np.ndarray) -> np.ndarray:
        """Determine negative amplitude oscillators and remove.

        Parameters
        ----------
        params
            Parameter array, with shape ``(self.oscillators, 2 * dim + 2)``

        Returns
        -------
        ud_params
            Updated parameter array, with negative damping oscillators removed,
            with shape ``(M_new, 2 * dim + 2)``, where
            ``M_new <= param.shape[0]`` and ``dim`` is the data dimension.
        """
        if self.output_mode:
            print("--> Checking for oscillators with negative damping...")

        M_init = params.shape[0]
        # Indices of oscillators with negative damping factors
        neg_damp_idx = list(
            set(np.nonzero(params[:, 2 + self.dim : 2 + (self.dim + 1) + 1] < 0.0)[0])
        )

        ud_params = np.delete(params, neg_damp_idx, axis=0)
        M = ud_params.shape[0]

        if M < M_init and self.output_mode:
            print(
                f"\t{ORA}WARNING: Oscillations with negative damping\n"
                f"\tfactors detected. These have been deleted.\n"
                f"\tCorrected number of oscillations: {M}{END}"
            )

        elif self.output_mode:
            print("\tNone found")

        return ud_params, M

    def _find_similar_frequencies(self, fys: np.ndarray) -> Iterable[Iterable[int]]:
        threshold = self.sw[0] / self.data.shape[0]
        groupings = {}
        for idx, fy in enumerate(fys):
            assigned = False
            for freq, indices in groupings.items():
                n = len(indices)
                if np.abs(fy - freq) < threshold:
                    indices.append(idx)
                    indices = sorted(indices)
                    # Get new mean freq of the group
                    new_freq = (n * freq + fy) / (n + 1)
                    groupings[new_freq] = groupings.pop(freq)
                    assigned = True
                    break
            if not assigned:
                groupings[fy] = [idx]

        return [indices for indices in groupings.values()]
