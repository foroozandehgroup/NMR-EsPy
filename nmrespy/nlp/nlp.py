# nlp.py
# nonlinear programming for analysis of 1D and 2D time series
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

import copy
import functools

import numpy as np
from numpy.fft import fft, fftshift
import numpy.linalg as nlinalg
import scipy.optimize as optimize

from nmrespy import *
from nmrespy.nlp import _funcs
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
from nmrespy._timing import timer


def start_end_wrapper(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        inst = args[0]
        if inst.fprint is False:
            return f(*args, **kwargs)

        print(f'{cols.G}================================\n'
                       'PERFORMING NONLINEAR PROGRAMMING\n'
                      f'================================{cols.END}')

        result = f(*args, **kwargs)

        print(f'{cols.G}==============================\n'
                       'NONLINEAR PROGRAMMING COMPLETE\n'
                      f'=============================={cols.END}')
        return result
    return wrapper


class NonlinearProgramming(FrequencyConverter):
    """Class for nonlinear programming for determination of spectral parameter
    estimates.

    Parameters
    ----------
    data : numpy.ndarray
        Signal to be considered (unnormalised).

    theta0 : numpy.ndarray
        Initial parameter guess. This should be an array with
        ``theta0.ndim == 2`` and:

        * ``theta0.shape[1] == 4`` for 1D data.
        * ``theta0.shape[1] == 6`` for 2D data.

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
        String composed of any combination of characters 'a', 'p', 'f', 'd'.
        Used to determine which parameter types to optimise, and which to
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
        advised to set ``amp_thold`` at least a couple of orders of
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

    Returns
    -------
    x : numpy.ndarray
        The result of the NLP routine, with shape ``(M_new, 4)`` for 1D data or
        ``(M_new, 6)`` for 2D data, where ``M_new<= M``.

    errors : numpy.ndarray
        An array with the same shape as ``x``, with elements corresponding to
        errors for each parameter
    """


    def __init__(
        self, data, theta0, sw, sfo=None, offset=None, phase_variance=True,
        method='trust_region', mode='apfd', bound=False, max_iterations=None,
        amp_thold=None, freq_thold=None, negative_amps='remove', fprint=True
    ):
        """Initialise the class instance. Checks that all arguments are valid"""

        # check validity of parameters
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f'{cols.R}data should be a numpy ndarray{cols.END}'
            )

        self.data = data

        # determine data dimension. If greater than 2, return error
        self.dim = self.data.ndim
        if self.dim >= 3:
            raise errors.MoreThanTwoDimError()

        # check theta0 is:
        # 1) NumPy array
        # 2) Of the correct shape (M, 4) or (M, 6)
        if not isinstance(theta0, np.ndarray) and theta0.ndim == 2:
            raise TypeError(
                f'{cols.R}theta0 should be a two-dimensional ndarray{cols.END}'
            )

        axis1 = 2 * self.dim + 2
        if not theta0.shape[1] == axis1:
            raise TypeError(
                f'{cols.R}The second axis of theta0 should be of size {axis1}'
                f'{cols.END}'
            )

        self.theta0 = theta0

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

        if self.sfo:
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
        self._nlp()


    @staticmethod
    def _check_mode(mode):
        """Ensures that the optimisation mode is valid. This should be a
        string containing only the characters 'a', 'p', 'f', and 'd', without
        any repetition.
        """
        if not isinstance(mode, str):
            return False

        # could use regex here
        if any(c not in 'apfd' for c in mode):
            return False

        # check each character doesn't appear more than once
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

    @start_end_wrapper
    @timer
    def _nlp(self):
        """Runs nonlinear programming"""

        # normalise data
        self.norm = nlinalg.norm(self.data)
        self.normed_data = self.data / self.norm

        # number of points in signal
        self.n = list(self.normed_data.shape)

        # number of oscillators
        self.m = self.theta0.shape[0]

        # shift frequencies to centre at 0
        self._scale_theta0()

        # time poits in each dimension
        self.tp = [
            np.linspace(0, float(n) / sw, n) for n, sw in zip(self.n, self.sw)
        ]

        # determine 'active' and 'passive' parameters based on self.mode
        self._impl_mode()

        # vectorise active and passive arrays
        act, pas = len(self.active), len(self.passive)
        self.active_theta0 = np.reshape(
            self.active_theta0, (act * self.m), order='F',
        )
        self.passive_theta0 = np.reshape(
            self.passive_theta0, (pas * self.m), order='F',
        )

        if self.bound:
            # generate bounds for bounded optimisation
            self._get_bounds()
        else:
            # unbounded optimisation
            self.bounds = None

        # extra arguments to feed into fidelity, grad and hessian functions
        self.optimiser_args = (
            self.normed_data, # normalised data
            self.tp, # timepoints for construction of model
            self.m, # number of oscillators DO I NEED THIS!?
            self.passive_theta0, # parameters not to be optimised
            self.active, # indices denoting parameters to be optimised
            self.phase_variance, # include phase variance in function or not
        )

        # determine cost function, gradient, and hessian
        self.calls = {}
        self.calls['fidelity'] = _funcs.f_1d if self.dim == 1 else _funcs.f_2d
        self.calls['gradient'] = _funcs.g_1d if self.dim == 1 else _funcs.g_2d
        self.calls['hessian'] = _funcs.h_1d if self.dim == 1 else _funcs.h_2d

        # optimise! generates self.result
        self._run_optimiser()


    def _scale_theta0(self):
        """shifts frequencies to centre at 0, and normalises aplitudes"""
        self.scale_theta0 = copy.deepcopy(self.theta0)

        # divide amps by data norm
        self.scale_theta0[:,0] = self.theta0[:,0] / self.norm

        # shift frequencies
        for i, off in enumerate(self.offset):
            self.scale_theta0[:,2+i] = -self.theta0[:,2+i] + off

    def _impl_mode(self):

        self.active = []
        for c in self.mode:
            if c == 'a':
                self.active.append(0)
            elif c == 'p':
                self.active.append(1)
            elif c == 'f':
                for i in range(self.dim):
                    self.active.append(2 + i)
            elif c == 'd':
                for i in range(self.dim):
                    self.active.append(2 + self.dim + i)

        self.passive = []
        for i in range(2 * (self.dim + 1)):
            if i not in self.active:
                self.passive.append(i)

        self.active_theta0 = self.scale_theta0[:, self.active]
        self.passive_theta0 = self.scale_theta0[:, self.passive]

    def _get_bounds(self):
        """Constructs a list of bounding constraints to set for each
        parameter. The bounds are as follows:

        * amplitudes: 0 < a < ∞
        * phases: -π < φ < π
        * frequencies: offset - sw/2 < f < offset + sw/2
        * damping: 0 < η < ∞
        """

        # generate list of all potential bounds
        # amplitude and phase bounds
        all_bounds = [(0, np.inf)] * self.m + [(-np.pi, np.pi)] * self.m
        # frequency bounds
        for sw, offset in zip(self.sw, self.offset):
            all_bounds += [((offset - sw/2), (offset + sw/2))] * self.m
        # damping bounds
        all_bounds += [(0, np.inf)] * (self.dim * self.m)

        # retrieve relevant bounds based on mode
        self.bounds = []
        for i in self.active:
            self.bounds += all_bounds[i*self.m : (i+1)*self.m]

    def _run_optimiser(self):

        # value to specify whether to output info to terminal
        fprint = 3 if self.fprint else 0

        if self.method == 'trust_region':
            result = optimize.minimize(
                fun = self.calls['fidelity'],
                x0 = self.active_theta0,
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
                x0 = self.active_theta0,
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
        self.active_theta = result['x']
        print(result['x'])
