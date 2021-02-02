# nlp.py
# nonlinear programming for analysis of 1D and 2D time series
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""User API for performing nonlinear programming"""

import copy
import functools
import operator

import numpy as np
from numpy.fft import fft, fftshift
import numpy.linalg as nlinalg
import scipy.optimize as optimize

from nmrespy import *
from nmrespy.fid import get_timepoints
from nmrespy.nlp import _funcs
from nmrespy._misc import start_end_wrapper
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
from nmrespy._timing import timer


class NonlinearProgramming(FrequencyConverter):
    """Class for nonlinear programming for determination of spectral parameter
    estimates.

    Parameters
    ----------
    data : numpy.ndarray
        Signal to be considered (unnormalised).

    theta0 : numpy.ndarray
        Initial parameter guess. The following forms are accepted:

        * **1-dimensional data:**

            1. ::

                  theta0 = numpy.array([
                      [a_1, φ_1, f_1, η_1],
                      [a_2, φ_2, f_2, η_2],
                      ...,
                      [a_m, φ_m, f_m, η_m],
                  ])
            2. ::

                  theta0 = np.array(
                      [a1, a2, ..., am, φ1, ..., φm, f1, ..., fm, η1, ... ηm]
                  )

        * **2-dimensional data:**

            1. ::

                theta0 = numpy.array([
                    [a_1, φ_1, f1_1, f2_1, η1_1, η2_1],
                    [a_2, φ_2, f1_2, f2_2, η1_2, η2_2],
                    ...,
                    [a_m, φ_m, f1_m, f2_m, η1_m, η2_m],
                ])
            2. ::

                theta0 = np.array(
                    [
                        a1, a2, ..., am, φ1, ..., φm, f1_1, ..., f1_m,
                        f2_1, ..., f2_m, η1_1, ... η1_m, η2_1, ... η2_m
                    ]
                )

    sw : [float] or [float, float]
        The experiment sweep width in each dimension in Hz.

    offset : [float] or [float, float] or None, default: None
        The experiment transmitter offset frequency in Hz. If `None`, a list
        satisfying ``len(offset) == data.ndim`` with each element being `0.`
        will be used.

    sfo : [float], [float, float] or None, default: None
        The experiment transmitter frequency in each dimension in MHz. This is
        not necessary, however if it set it to `None`, no conversion of
        frequencies from Hz to ppm will be possible!

    phase_variance : bool, default: True
        Specifies whether or not to include the variance of oscillator
        phases into the NLP routine.

    method : 'trust_region' or 'lbfgs', default: 'trust_region'
        Optimisation algorithm to use. See notes for more details.

    mode : str, default: 'apfd'
        String composed of any combination of characters `'a'`, `'p'`, `'f'`,
        `'d'`. Used to determine which parameter types to optimise, and which to
        remain fixed.

    bound : bool, default: False
        Specifies whether or not to bound the parameters during optimisation.

    max_iterations : int or None, default: None
        A value specifiying the number of iterations the routine may run
        through before it is terminated. If `None`, the default number
        of maximum iterations is set (`100` if `method` is = `'trust_region'`,
        and `500` if `method` is `'lbfgs'`).

    amp_thold : float or None, default: None
        A threshold that imposes a threshold for deleting oscillators of
        negligible ampltiude. If `None`, does nothing. If a float, oscillators
        with amplitudes satisfying :math:`a_m < a_{\\mathrm{thold}}
        \\lVert \\boldsymbol{a} \\rVert`` will be removed from the
        parameter array, where :math:`\\lVert \\boldsymbol{a} \\rVert`
        is the norm of the vector of all the oscillator amplitudes. It is
        advised to set `amp_thold` at least a couple of orders of
        magnitude below 1.

    freq_thold : float or None
        .. warning::

           NOT IMPLEMENTED YET

        If `None`, does nothing. If a float, oscillator pairs with
        frequencies satisfying
        :math:`\\lvert f_m - f_p \\rvert < f_{\\mathrm{thold}}` will be
        removed from the parameter array. A new oscillator will be included
        in the array, with parameters:

        * amplitude: :math:`a = a_m + a_p`
        * phase: :math:`\phi = \\frac{\phi_m + \phi_p}{2}`
        * frequency: :math:`f = \\frac{f_m + f_p}{2}`
        * damping: :math:`\eta = \\frac{\eta_m + \eta_p}{2}`

    negative_amps : 'remove' or 'flip_phase', default: 'remove'
        Indicates how to treat oscillators which have gained negative
        amplitudes during the optimisation. `'remove'` will result
        in such oscillators being purged from the parameter estimate.
        The optimisation routine will the be re-run recursively until
        no oscillators have a negative amplitude. `'flip_phase'` will
        retain oscillators with negative amplitudes, but the the amplitudes
        will be turned positive, and a π radians phase shift will be
        applied to these oscillators.

    fprint : bool, default: True
        If `True`, the method provides information on progress to
        the terminal as it runs. If `False`, the method will run silently.
    """

    start_txt = 'NONLINEAR PROGRAMMING STARTED'
    end_txt = 'NONLINEAR PROGRAMMING COMPLETE'

    def __init__(
        self, data, theta0, sw, sfo=None, offset=None, phase_variance=True,
        method='trust_region', mode='apfd', bound=False, max_iterations=None,
        amp_thold=None, freq_thold=None, negative_amps='remove', fprint=True
    ):
        """Initialise the class instance. Checks that all arguments are valid"""

        # --- Check validity of parameters -------------------------------
        # Data should be a NumPy array.
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f'{cols.R}data should be a numpy ndarray{cols.END}'
            )
        self.data = data

        # Determine data dimension. If greater than 2, return error.
        self.dim = self.data.ndim
        if self.dim >= 3:
            raise errors.MoreThanTwoDimError()

        # Number of points in signal
        self.n = list(self.data.shape)

        # theta0 is should be a NumPy array.
        if not isinstance(theta0, np.ndarray):
            raise TypeError(
                f'{cols.R}theta0 should be a numpy ndarray{cols.END}'
            )

        # Number of "types" or parameters.
        # This will be 4 if the signal is 1D, and 6 if 2D.
        p = 2 * self.dim + 2

        # Check that theta0 is of correct shape
        # Permitted shape 1. (see docstring)
        if theta0.ndim == 2 and theta0.shape[1] == p:
            # Vectorise array: (m, p) -> (p*m,)
            # Column-major (Fortran-style) ordering.
            self.theta0 = theta0.flatten(order='F')

        # permitted shape 2. (see docstring)
        elif theta0.dim == 1 and theta0 % p == 0:
            self.theta0 = theta0

        else:
            raise ValueError(
                f'{cols.R}The shape of theta0 is invalid. It should either'
                f' be of shape (m, {p}) or (m*{p},){cols.END}'
            )
        # Number of oscillators
        self.m = int(self.theta0.size / p)

        # If offset is None, set it to zero in each dimension
        if offset is None:
            offset = [0.0] * self.dim

        to_check = [sw, offset] if sfo == None else [sw, offset, sfo]

        # if transmitter frequency is not NOne, include it in checking
        if sfo != None:
            self.sfo = sfo
            to_check.append(sfo)

        for x in to_check:
            if not isinstance(x, list) and len(x) == self.dim:
                raise TypeError(
                    f'{cols.R}sw and offset (and sfo if specified) should be'
                    f' lists with the same number of elements as dimensions in'
                    f' the data.{cols.END}'
                )

        self.sw = sw
        self.offset = offset
        self.sfo = sfo

        if self.sfo != None:
            self.converter = FrequencyConverter(
                self.n, self.sw, self.offset, self.sfo
            )

        if not isinstance(phase_variance, bool):
            raise TypeError(
                f'{cols.R}phase_variance should be a Boolean{cols.END}'
            )

        self.phase_variance = phase_variance

        if not method in ['trust_region', 'lbfgs']:
            raise ValueError(
                f'{cols.R}method should be \'trust_region\' or \'lbfgs\''
                f'{cols.END}'
            )

        self.method = method

        # check mode is valid
        if not self._check_mode(mode):
            raise ValueError(
                f'{cols.R}mode should be a string containing'
                f' only the characters \'a\', \'p\', \'f\', and'
                f' \'d\'. No character should be repeated{cols.END}'
            )

        # gets upset when phase variance is switched on, but phases
        # are not to be optimised (the user is being unclear about
        # their purpose)
        if self.phase_variance and 'p' not in mode:
            raise PhaseVarianceAmbiguityError(mode)

        self.mode = mode

        if not isinstance(bound, bool):
            raise TypeError(
                f'{cols.R}bound should be a Boolean{cols.END}'
            )

        self.bound = bound

        if max_iterations is None:
            # if 'trust_region', set as 100, if 'lbfgs', set as 500
            max_iterations = 100 if self.method == 'trust_region' else 500

        if not isinstance(max_iterations, int) or max_iterations < 1:
             raise TypeError(
                 f'{cols.R}max_iterations should be None, or an integer'
                 f' greater than 0.{cols.END}'
             )

        self.max_iterations = max_iterations

        # check amplitude and frequency thresholds
        self.amp_thold = self._check_thold(amp_thold)
        self.freq_thold = self._check_thold(amp_thold)

        if not negative_amps in ['remove', 'flip_phase']:
            raise ValueError(
                f'{cols.R}negative_amps should be \'remove\' or \'flip_phase\''
                f'{cols.END}'
            )

        self.negative_amps = negative_amps

        if not isinstance(fprint, bool):
            raise TypeError(
                f'{cols.R}fprint should be a Boolean{cols.END}'
            )

        self.fprint = fprint

        # good to go!
        self._init_nlp()


    @staticmethod
    def _check_mode(mode):
        """Ensures that the optimisation mode is valid. This should be a
        string containing only the characters 'a', 'p', 'f', and 'd', without
        any repetition.
        """
        if not isinstance(mode, str):
            return False

        # check if mode is empty or contains and invalid character
        if any(c not in 'apfd' for c in mode) or mode == '':
            return False

        # check if mode contains a repeated character
        count = {}
        for c in mode:
            if c in count.keys():
                count[c] += 1
            else:
                count[c] = 1

        for key in count:
            if count[key] > 1:
                return False

        return True

    @staticmethod
    def _check_thold(thold):
        """Ensures thresholds (amp_thold and freq_thold) are valid. If valid,
        the value will be returned as a float. If not value, False will be
        returned"""
        if thold is None:
            thold = 0.

        if isinstance(thold, float) and 0. > thold or thold >= 1.:
            # this will lead to an error in __init__
            # if thold is not valid, it will be returned as False
            raise TypeError(
                f'{cols.R}amp_thold and freq_thold should greater'
                f' than 0.{cols.END}'
            )
        return thold

    @timer
    @start_end_wrapper(start_txt, end_txt)
    def _init_nlp(self):
        """Runs nonlinear programming"""

        # normalise data
        self.norm = nlinalg.norm(self.data)
        self.normed_data = self.data / self.norm

        # divide amplitudes by data norm
        theta0_edit = copy.deepcopy(self.theta0)
        theta0_edit[:self.m] = theta0_edit[:self.m] / self.norm

        # shift oscillator frequencies to centre about 0
        theta0_edit = self._shift_offset(theta0_edit, 'center')


        # time points in each dimension
        self.tp = get_timepoints(self.n, self.sw)

        # determine 'active' and 'passive' parameters based on self.mode
        self.active_idx, self.passive_idx = self._get_active_passive_indices()

        # takes the scaled parameter array self.scale_theta0,
        # of shape (self.m, 4) or (self.m, 6), and does the following:
        # 1) vectorises
        # 2) splits up into vector of active parameters and vector of
        #    passive parameters
        self.active, self.passive = \
            self._split_active_passive(theta0_edit)

        # determine cost function, gradient, and hessian
        self.calls = {}
        self.calls['fidelity'] = _funcs.f_1d if self.dim == 1 else _funcs.f_2d
        self.calls['gradient'] = _funcs.g_1d if self.dim == 1 else _funcs.g_2d
        self.calls['hessian'] = _funcs.h_1d if self.dim == 1 else _funcs.h_2d

        self._optimise()


    def _optimise(self):

        self.optimiser_args = (
            self.normed_data, # normalised data
            self.tp, # timepoints for construction of model
            self.m, # number of oscillators
            self.passive, # parameters not to be optimised
            self.active_idx, # indices denoting parameters to be optimised
            self.phase_variance, # include phase variance in function or not
        )

        self.bounds = self._get_bounds()
        self._run_optimiser()

        terminate = self._check_negative_amps()

        if terminate:
            self.errors = self._get_errors()
            self.result = self._merge_active_passive(self.active, self.passive)
            self.result[:self.m] *= self.norm
            self.result = self._shift_offset(self.result, 'displace')

        else:
            self._optimise()


    def _shift_offset(self, params, direction):
        """shifts frequencies to centre to or displace from 0

        Parameters
        ----------
        params : numpy.ndarray
            Full parameter array

        direction : 'center' or 'displace'
            `'center'` shifts frerquencies such that the central frequency
            is set to zero. `'displace'` moves frequencies away from zero,
            to be reflected by offset.
        """

        for i, off in enumerate(self.offset):
            slice = self._get_slice([2+i])

            if direction == 'center':
                params[slice] = params[slice] - off
            elif direction == 'displace':
                params[slice] = params[slice] + off

        return params


    def _get_slice(self, idx, osc_idx='all'):
        """
        Parameters
        ----------
        idx : list
            Parameter types to be targeted. Valid ints are `0` to `3`
            (included) for a 1D signal, and `0` to `5` for a 2D signal

        osc_idx : 'all' or list, default: 'all'
            Oscillators to be targeted. Can be either `'all'`, where all
            oscillators are indexed, or a list of ints, in order to select
            a subset of oscillators. Valid ints are `0` to `self.m - 1`
            (included).

        Returns
        -------
        slice : numpy.ndarray
            Array slice.
        """
        # array of osccilators to index
        if osc_idx == 'all':
            osc_idx = list(range(self.m))

        slice = []
        for i in idx:
            slice += [i*self.m + j for j in osc_idx]

        return slice

    def _get_active_passive_indices(self):

        active_idx = []
        for c in self.mode:
            if c == 'a':
                active_idx.append(0)
            elif c == 'p':
                active_idx.append(1)
            elif c == 'f':
                for i in range(self.dim):
                    active_idx.append(2 + i)
            elif c == 'd':
                for i in range(self.dim):
                    active_idx.append(2 + self.dim + i)

        # initialise passive index array as containing all valid values,
        # and remove all values that are found in active index array
        passive_idx = list(range(2 * (self.dim + 1)))
        for i in active_idx:
            passive_idx.remove(i)

        return active_idx, passive_idx


    def _merge_active_passive(self, active_vec, passive_vec):
        """Given the active and passive parameters in vector form, merge to
        form the complete parameter vector

        Parameters
        ----------
        active_vec : numpy.ndarray
            Active vector.

        passive_vec : numpy,ndarray
            Passive vector.

        Returns
        -------
        merged_vec : numpy.ndarray
            Merged (complete) vector.
        """

        try:
            passive_slice = self._get_slice(self.passive_idx)

        # ValueError is raised if the are no passive parameters
        # simply return the active vector
        except ValueError:
            return active_vec

        active_slice = self._get_slice(self.active_idx)

        merged_vec = np.zeros(self.m * (2 * self.dim + 2))
        merged_vec[active_slice] = active_vec
        merged_vec[passive_slice] = passive_vec

        return merged_vec


    def _split_active_passive(self, merged_vec):
        """Given a full vector of parameters, split to form vectors of active
        and passive parameters.

        Parameters
        ----------
        merged_vec : numpy.ndarray
            Full parameter vector

        Returns
        ----------
        active_vec : numpy.ndarray
            Active vector.

        passive_vec : numpy,ndarray
            Passive vector.
        """

        try:
            passive_slice = self._get_slice(self.passive_idx)

        # ValueError is raised if there are no passive parameters
        # simply return the full vector as the active vector, and an empty
        # vector as the passive vector
        except ValueError:
            return merged_vec, np.array([])

        active_slice = self._get_slice(self.active_idx)

        return merged_vec[active_slice], merged_vec[passive_slice]


    def _get_bounds(self):
        """Constructs a list of bounding constraints to set for each
        parameter. The bounds are as follows:

        * amplitudes: 0 < a < ∞
        * phases: -π < φ < π
        * frequencies: offset - sw/2 < f < offset + sw/2
        * damping: 0 < η < ∞
        """

        if not self.bound:
            return None

        # generate list of all potential bounds
        # amplitude and phase bounds
        all_bounds = [(0, np.inf)] * self.m + [(-np.pi, np.pi)] * self.m
        # frequency bounds
        for sw, offset in zip(self.sw, self.offset):
            all_bounds += [((offset - sw/2), (offset + sw/2))] * self.m
        # damping bounds
        all_bounds += [(0, np.inf)] * (self.dim * self.m)

        # retrieve relevant bounds based on mode
        bounds = []
        for i in self.active_idx:
            bounds += all_bounds[i*self.m : (i+1)*self.m]

        return bounds

    def _run_optimiser(self):

        # value to specify whether to output info to terminal
        fprint = 3 if self.fprint else 0

        if self.method == 'trust_region':
            result = optimize.minimize(
                fun = self.calls['fidelity'],
                x0 = self.active,
                args = self.optimiser_args,
                method = 'trust-constr',
                jac = self.calls['gradient'],
                hess = self.calls['hessian'],
                bounds = self.bounds,
                options = {
                    'maxiter': self.max_iterations,
                    'verbose': fprint,
                },
            )

        elif self.method == 'lbfgs':
            result = optimize.minimize(
                fun = self.calls['fidelity'],
                x0 = self.active,
                args = self.optimiser_args,
                method = 'L-BFGS-B',
                jac = self.calls['gradient'],
                bounds = self.bounds,
                options = {
                    'maxiter': self.max_iterations,
                    'iprint': fprint // 3,
                    'disp': True
                }
            )

        # extract result from optimiser dictionary
        self.active = result['x']


    def _check_negative_amps(self):
        """Determines which oscillators (if any) have negative amplitudes, and
        removes/flips these.

        Returns
        -------
        term : bool
            Used by :py:meth:`_nlp` to decide whether to terminate or re-run
            the optimisation routine.
        """

        if 0 in self.active_idx:
            # generates length-1 tuple (unpack)
            negative_idx = np.nonzero(self.active[:self.m] < 0.0)[0]

            # check if there are any negative amps by determining
            # if negative_idx is empty or not
            if not list(negative_idx):
                return True

            # negative amplitudes exist... deal with these
            if self.negative_amps == 'remove':
                # remove parameters corresponding to negative
                # amplitude oscillators
                self.active = np.delete(
                    self.active,
                    self._get_slice(self.active_idx, osc_idx=negative_idx),
                )

                self.passive = np.delete(
                    self.passive,
                    self._get_slice(self.passive_idx, osc_idx=negative_idx),
                )

                # remove bounds corresponding to negative oscillators
                if self.bounds != None:
                    del self.bounds[_get_slice(
                        self.passive_idx, osc_idx=negative_idx)
                    ]

                # update number of oscillators
                self.m = idx(self.active.size / len(self.active_idx))

                return False

            elif self.negative_amps == 'flip':
                # make negative amplitude oscillators positive and flip
                # phase by 180 degrees

                # amplitude slice
                amp_slice = self._get_slice([0], osc_idx=negative_idx)
                self.active[amp_slice] = - self.active[amp_slice]

                # flip phase
                if 1 in self.active_idx:
                    phase_slice = self._get_slice([1], osc_idx=negative_idx)
                    self.active[phase_slice] = \
                        self._pi_flip(self.active[phase_slice])

                else:
                    phase_slice = self._get_slice([0], osc_idx=negative_idx)
                    self.passive[phase_slice] = \
                        self._pi_flip(self.passive[phase_slice])

        return True

    @staticmethod
    def _pi_flip(arr):
        """flip array of phases by π raidnas, ensuring the phases remain in
        the range (-π, π]"""
        return (arr + 2 * np.pi) % (2 * np.pi) - np.pi

    def _get_errors(self):

        fidelity = self.calls['fidelity'](self.active, *self.optimiser_args)
        hessian = self.calls['hessian'](self.active, *self.optimiser_args)

        return np.sqrt(fidelity) \
             + np.sqrt(np.diag(nlinalg.inv(hessian)) \
             / functools.reduce(operator.mul, self.n))
