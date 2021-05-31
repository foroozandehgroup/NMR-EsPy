# nlp.nlp.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Nonlinear programming for generating NMR parameter estiamtes"""

import copy
import functools
import operator

import numpy as np
import numpy.linalg as nlinalg
import scipy.optimize as optimize

from nmrespy import *
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
    colorama.init()
from nmrespy._errors import *
from nmrespy._misc import start_end_wrapper, ArgumentChecker, \
    FrequencyConverter
from nmrespy._timing import timer
import nmrespy.nlp._funcs as funcs
from nmrespy.sig import get_timepoints

# TODO in a later version
# Add support for mode
# Was getting indexing errors inside _check_negative_amps
# when testing using a mode which is 'apfd'
#
# For docs:
#
# mode : str, default: 'apfd'
#     String composed of any combination of characters `'a'`, `'p'`, `'f'`,
#     `'d'`. Used to determine which parameter types to optimise, and which
#     to remain fixed:
#
#     * `'a'`: Amplitudes are optimised
#     * `'p'`: Phases are optimised
#     * `'f'`: Frequencies are optimised
#     * `'d'`: Damping factors are optimised


class NonlinearProgramming(FrequencyConverter):
    """Class for nonlinear programming for determination of spectral parameter
    estimates.

    Parameters
    ----------
    data : numpy.ndarray
        Signal to be considered (unnormalised).

    theta0 : numpy.ndarray
        Initial parameter guess in the following form:

        * **1-dimensional data:**

          .. code-block::

             theta0 = numpy.array([
                 [a_1, φ_1, f_1, η_1],
                 [a_2, φ_2, f_2, η_2],
                 ...,
                 [a_m, φ_m, f_m, η_m],
             ])

        * **2-dimensional data:**

          .. code-block::

             theta0 = numpy.array([
                 [a_1, φ_1, f1_1, f2_1, η1_1, η2_1],
                 [a_2, φ_2, f1_2, f2_2, η1_2, η2_2],
                 ...,
                 [a_m, φ_m, f1_m, f2_m, η1_m, η2_m],
             ])

    sw : [float] or [float, float]
        The experiment sweep width in each dimension in Hz.

    offset : [float] or [float, float] or None, default: None
        The experiment transmitter offset frequency in Hz. If `None`,
        `offset` will be set as ``data.ndim * [0.0]``.

    sfo : [float], [float, float] or None, default: None
        The experiment transmitter frequency in each dimension in MHz.
        This is not necessary, however if it set it to `None`, no conversion
        of frequencies from Hz to ppm will be possible!

    start_point : int, default: 0
        The first timepoint sampled, in units of
        :math:`\\Delta t = 1 / f_{\\mathrm{sw}}`

    phase_variance : bool, default: True
        Specifies whether or not to include the variance of oscillator
        phases into the NLP routine. The fiedlity (cost function) is
        given by:

        * `phase_variance` set to `False`:

          .. math::

             \\mathcal{F}\\left(\\boldsymbol{\\theta}\\right) =
             \\left\\lVert \\boldsymbol{Y} - \\boldsymbol{X} \\right\\rVert_2^2

        * `phase_variance` set to `True`:

          .. math::

             \\mathcal{F}\\left(\\boldsymbol{\\theta}\\right) =
             \\left\\lVert \\boldsymbol{Y} - \\boldsymbol{X} \\right
             \\rVert_2^2 + \\mathrm{Var}\\left(\\boldsymbol{\\phi}\\right)

    method : 'trust_region' or 'lbfgs', default: 'trust_region'
        Optimisation algorithm to use. These utilise
        `scipy.optimise.minimise <https://docs.scipy.org/doc/scipy/\
        reference/generated/scipy.optimize.minimize.html>`_, with
        the method either being `trust-constr <https://docs.scipy.org/doc/\
        scipy/reference/optimize.minimize-trustconstr.html\
        #optimize-minimize-trustconstr>`_, or
        `L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/\
        optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`_.

    bound : bool, default: False
        Specifies whether or not to bound the parameters during optimisation.
        Bounds are given by:

        * :math:`0 \\leq a_m \\leq \\infty`
        * :math:`-\\pi < \\phi_m \\leq \\pi`
        * :math:`-f_{\\mathrm{sw}} / 2 + f_{\\mathrm{off}} \\leq f_m \\leq\
        f_{\\mathrm{sw}} / 2 + f_{\\mathrm{off}}`
        * :math:`0 \\leq \\eta_m \\leq \\infty`

        :math:`(\\forall m \\in \\{1, \\cdots, M\\})`

    max_iterations : int or None, default: None
        A value specifiying the number of iterations the routine may run
        through before it is terminated. If `None`, the default number
        of maximum iterations is set (`100` if `method` is `'trust_region'`,
        and `500` if `method` is `'lbfgs'`).

    amp_thold : float or None, default: None
        A value that imposes a threshold for deleting oscillators of
        negligible ampltiude. If `None`, does nothing. If a float, oscillators
        with amplitudes satisfying :math:`a_m < a_{\\mathrm{thold}}
        \\lVert \\boldsymbol{a} \\rVert_2`` will be removed from the
        parameter array, where :math:`\\lVert \\boldsymbol{a} \\rVert_2`
        is the Euclidian norm of the vector of all the oscillator amplitudes.
        It is advised to set `amp_thold` at least a couple of orders of
        magnitude below 1.

    freq_thold : float or None
        If `None`, does nothing. If a float, oscillator pairs with
        frequencies satisfying
        :math:`\\lvert f_m - f_p \\rvert < f_{\\mathrm{thold}}` will be
        removed from the parameter array. A new oscillator will be included
        in the array, with parameters:

        * amplitude: :math:`a = a_m + a_p`
        * phase: :math:`\\phi = \\left(\\phi_m + \\phi_p\\right) / 2`
        * frequency: :math:`f = \\left(f_m + f_p\\right) / 2`
        * damping: :math:`\\eta = \\left(\\eta_m + \\eta_p\\right) / 2`

        .. warning::

           NOT IMPLEMENTED YET

    negative_amps : 'remove' or 'flip_phase', default: 'remove'
        Indicates how to treat oscillators which have gained negative
        amplitudes during the optimisation.

        * `'remove'` will result in such oscillators being purged from
          the parameter estimate. The optimisation routine will the be
          re-run recursively until no oscillators have a negative amplitude.
        * `'flip_phase'` will retain oscillators with negative amplitudes,
          but the the amplitudes will be multiplied by -1, and a π radians
          phase shift will be applied to these oscillators.

    fprint : bool, default: True
        If `True`, the method provides information on progress to
        the terminal as it runs. If `False`, the method will run silently.

    Notes
    -----

    The two optimisation algorithms (specified by `method`) primarily
    differ in how they treat the calculation of the matrix of cost
    function second derivatives (called the Hessian). `'trust_region'`
    will calculate the Hessian explicitly at every iteration, whilst
    `'lbfgs'` uses an update formula based on gradient information to
    estimate the Hessian. The upshot of this is that the convergence
    rate (the number of iterations needed to reach convergence) is
    typically better for `'trust_region'`, though each iteration
    typically takes longer to generate. By default, it is advised to
    use `'trust_region'`, however if your guess has a large number
    of signals, you may find `'lbfgs'` performs more effectively.
    """

    start_txt = 'NONLINEAR PROGRAMMING STARTED'
    end_txt = 'NONLINEAR PROGRAMMING COMPLETE'

    def __init__(
        self, data, theta0, sw, sfo=None, offset=None, start_point=0,
        phase_variance=True, method='trust_region', bound=False,
        max_iterations=None, amp_thold=None, freq_thold=None,
        negative_amps='remove', fprint=True, mode='apfd',
    ):
        """Initialise the class instance. Checks that all arguments are
        valid"""

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
            raise MoreThanTwoDimError()

        # Number of "types" or parameters.
        # This will be 4 if the signal is 1D, and 6 if 2D.
        self.p = 2 * self.dim + 2

        # If offset is None, set it to zero in each dimension
        if offset is None:
            offset = [0.0] * self.dim

        if max_iterations is None:
            max_iterations = 100

        # Determine validity of other args using ArgumentChecker
        components = [
            (theta0, 'theta0', 'parameter'),
            (sw, 'sw', 'float_list'),
            (offset, 'offset', 'float_list'),
            (start_point, 'start_point', 'positive_int_or_zero'),
            (phase_variance, 'phase_variance', 'bool'),
            (max_iterations, 'max_iterations', 'positive_int'),
            (mode, 'mode', 'optimiser_mode'),  # TODO
            (negative_amps, 'negative_amps', 'negative_amplidue'),
            (fprint, 'fprint', 'bool'),
        ]

        # Certain arguments should be checked only if they are not None...
        if sfo is not None:
            components.append((sfo, 'sfo', 'float_list'))

        if amp_thold is not None:
            components.append((amp_thold, 'amp_thold', 'zero_to_one'))

        if freq_thold is not None:
            components.append((freq_thold, 'freq_thold', 'positive_float'))

        # Check arguments are valid!
        ArgumentChecker(components, self.dim)

        # TODO
        # # Gets upset when phase variance is switched on, but phases
        # # are not to be optimised (the user is being unclear about
        # # their purpose)
        # if phase_variance and 'p' not in mode:
        #     raise PhaseVarianceAmbiguityError(mode)

        # --- Create attributes ------------------------------------------
        # Reshape parameter array to vector:
        # (M, 4) -> (4*M,) or (M, 6) -> (6*M,)
        self.theta0 = theta0

        self.sw = sw
        self.offset = offset
        self.sfo = sfo
        self.start_point = start_point
        self.method = method
        self.phase_variance = phase_variance
        self.mode = mode  # TODO
        self.bound = bound
        self.max_iterations = max_iterations
        self.amp_thold = amp_thold
        self.freq_thold = freq_thold
        self.negative_amps = negative_amps
        self.fprint = fprint

        # Number of oscillators
        self.m = int(self.theta0.size / self.p)
        # Number of points in each dimension
        self.n = list(self.data.shape)

        if self.sfo is not None:
            # If sfo was given an explicit value, create a frequency
            # converter, enabling outputs in ppm.
            self.converter = FrequencyConverter(
                self.n, self.sw, self.offset, self.sfo
            )

        if self.max_iterations is None:
            # If max_iterations is set to None, set it to default value
            # If 'trust_region', set as 100. Need to explicitely compute
            # the Hessian for this alg., so each iteration is typically
            # quite costly. L-BFGS is typically quicker per iteration, so
            # give it more.
            self.max_iterations = 100 if self.method == 'trust_region' else 500

        self.amp_thold = 0. if self.amp_thold is None else self.amp_thold
        # TODO freq-thold?

        self._run_nlp()

    @timer
    @start_end_wrapper(start_txt, end_txt)
    def _run_nlp(self):
        """Runs nonlinear programming"""

        # Normalise data
        self.norm = nlinalg.norm(self.data)
        self.normed_data = self.data / self.norm

        # Vectorise the initial parameter array
        x0 = self.theta0.flatten(order='F')
        # Perform some tweaks to regularise x0:
        # 1. Divide amplitudes by the norm of the data
        x0[:self.m] = x0[:self.m] / self.norm
        # 2. Shift oscillator frequencies to center about 0
        x0 = self._shift_offset(x0, 'center')
        print(self.sw)
        # Time points in each dimension
        self.tp = get_timepoints(
            [n + self.start_point for n in self.n], self.sw,
        )
        self.tp = [t[self.start_point:] for t in self.tp]

        # Determine 'active' and 'passive' parameters based on self.mode
        # generates self.active_idx and self.passive_idx
        #
        # If one wanted to just optimise amplitudes and
        # frequencies, self.active_idx would be [0, 2], and therefore
        # self.passive_idx would be [1, 3]:
        # a1  ...  am  φ1  ...  φm  f1  ...  fm  η1  ...  ηm
        # < idx = 0 >  < idx = 1 >  < idx = 2 >  < idx = 3 >
        self._get_active_passive_indices()

        # Takes the scaled parameter vector x0, with shape
        # (4 * self.m,) or (6 * self.m,), and splits up into vector of
        # active parameters and vector of passive parameters
        # called self.active and self.passive
        #
        # Active parameters: parameters that are going to actually be
        # optimised
        #
        # Passive parameters: parameters that are to be fixed at their
        # original value. These are still required however, in order
        # to compute the fiedlity, its grad and its Hessian.
        self._split_active_passive(x0)

        # Determine cost function, gradient, and hessian based on the data
        # dimension
        self.funcs = {
            'fidelity': funcs.f_1d if self.dim == 1 else funcs.f_2d,
            'gradient': funcs.g_1d if self.dim == 1 else funcs.g_2d,
            'hessian': funcs.h_1d if self.dim == 1 else funcs.h_2d,
        }

        # This method is called recursively until no negative amplitudes
        # are found within the parameter estimate.
        self._recursive_optimise()

        # --- Finishing up -------------------------------------------
        # Merge self.active and self.passive to get the full vector
        # called self.result
        self._merge_active_passive()
        # Remove any oscillators with negligible amplitudes
        self._negligible_amplitudes()

        # Rescale and correct for offset
        self.result[:self.m] *= self.norm
        self.result = self._shift_offset(self.result, 'displace')
        # Get estimate errors (self.errors)
        self._get_errors()

        # Reshape result array back to (M x 4) or (M x 6)
        self.result = np.reshape(self.result, (self.m, self.p), order='F')
        # Order oscillators by frequency
        order = np.argsort(self.result[:, 2])
        self.result = self.result[order]
        self.errors = self.errors[order]

    def _recursive_optimise(self):

        # Extra arguments (other than initial guess, which is self.active)
        # that are needed to compute the fidelity and its derivatives
        self.optimiser_args = (
            self.normed_data,
            self.tp,
            self.m,
            self.passive,
            self.active_idx,
            self.phase_variance,
        )

        # Dermine bounds for optimiser. Could be None (unconstrained), or
        # bounds that are physically reasonable for the system being
        # considered
        self._get_bounds()

        # Calls the desired optimisation routine, updating self.active
        self._run_optimiser()

        # Dermine whether any negative amplitudes are in self.active
        terminate = self._check_negative_amps()

        if not terminate:
            self._recursive_optimise()

    def get_result(self, freq_unit='hz'):
        """Obtain the result of nonlinear programming.

        Parameters
        ----------
        freq_unit : 'hz' or 'ppm', default: 'hz'
            The unit of the oscillator frequencies (corresponding to
            ``result[:, 2]``)

        Returns
        -------
        result : numpy.ndarray
        """
        return self._get_array('result', freq_unit)

    def get_errors(self, freq_unit='hz'):
        """Obtain errors of parameters estimates.

        Parameters
        ----------
        freq_unit : 'hz' or 'ppm', default: 'hz'
            The unit of the oscillator frequencies (corresponding to
            ``result[:, 2]``)

        Returns
        -------
        result : numpy.ndarray
        """
        return self._get_array('errors', freq_unit)

    def _get_array(self, name, freq_unit):

        if freq_unit == 'hz':
            return self.__dict__[name]

        elif freq_unit == 'ppm':
            # Check whether a frequency converter is associated with the
            # class
            if 'converter' not in self.__dict__.keys():
                raise ValueError(
                    f'{cols.R}Insufficient information to determine'
                    f' frequencies in ppm. Did you perhaps forget to specify'
                    f' sfo?{cols.END}'
                )

            result = copy.deepcopy(self.__dict__[name])

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
            raise InvalidUnitError('hz', 'ppm')

    def _shift_offset(self, params, direction):
        """Shifts frequencies to centre to or displace from 0

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
            # Dimension (i+1)'s frequency parameters are given by this slice
            slice = self._get_slice([2 + i])
            # Take frequencies from offset values to be centred at zero
            # i.e.
            # | 10 9 8 7 6 5 4 3 2 1 0 | -> | 5 4 3 2 1 0 -1 -2 -3 -4 -5 |
            if direction == 'center':
                params[slice] = params[slice] - off
            # Do the reverse of the above (take away from being centered at
            # zero)
            # i.e.
            # | 5 4 3 2 1 0 -1 -2 -3 -4 -5 | -> | 10 9 8 7 6 5 4 3 2 1 0 |
            elif direction == 'displace':
                params[slice] = params[slice] + off

        return params

    def _get_slice(self, idx, osc_idx=None):
        """
        Parameters
        ----------
        idx : list
            Parameter types to be targeted. Valid ints are `0` to `3`
            (included) for a 1D signal, and `0` to `5` for a 2D signal

        osc_idx : list or None default: None
            Oscillators to be targeted. Can be either `None`, where all
            oscillators are indexed, or a list of ints, in order to select
            a subset of oscillators. Valid ints are `0` to `self.m - 1`
            (included).

        Returns
        -------
        slice : numpy.ndarray
            Array slice.
        """
        # Array of osccilators to index
        if osc_idx is None:
            osc_idx = list(range(self.m))

        slice = []
        for i in idx:
            # Note that parameters are arranged as:
            # a1  ...  am  φ1  ...  φm  f1  ...  fm  η1  ...  ηm (1D case)
            # ∴ stride length of m to go to the next "type" of parameter
            # and stride length of 1 to go to the next oscillator.
            slice += [i * self.m + j for j in osc_idx]

        return np.s_[slice]

    def _get_active_passive_indices(self):
        """Determine the indices of blocks of the parameter vector that
        contain the active and passive parameters"""
        # Recall, for the 1D case, the indices correspond to the following
        # blocks in the vector:
        # a1  ...  am  φ1  ...  φm  f1  ...  fm  η1  ...  ηm
        # < idx = 0 >  < idx = 1 >  < idx = 2 >  < idx = 3 >
        self.active_idx = []
        for c in self.mode:
            if c == 'a':  # Amplitude
                self.active_idx.append(0)
            elif c == 'p':  # Phase
                self.active_idx.append(1)
            elif c == 'f':  # Frequecy (add indices for each dim)
                for i in range(self.dim):
                    self.active_idx.append(2 + i)
            elif c == 'd':  # Damping (add indices for each dim)
                for i in range(self.dim):
                    self.active_idx.append(2 + self.dim + i)

        # Initialise passive index array as containing all valid values,
        # and remove all values that are found in active index array
        self.passive_idx = list(range(2 * (self.dim + 1)))
        for i in self.active_idx:
            self.passive_idx.remove(i)

    def _merge_active_passive(self):
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
            # Determine indices in merged_vec that will relate to passive
            # parameters
            passive_slice = self._get_slice(self.passive_idx)

        # ValueError is raised if the are no passive parameters,
        # as an empty list is not iterable!
        # In this case, the active vector is equivalent to the full
        # vector, so just return it.
        except ValueError:
            self.result = self.active

        # Determine indices in merged_vec that will relate to active
        # parameters
        active_slice = self._get_slice(self.active_idx)
        # Construct the merged vector
        self.result = np.zeros(self.m * (2 * self.dim + 2))
        self.result[active_slice] = self.active
        self.result[passive_slice] = self.passive

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
        # Determine indices in the merged vector that correspond to
        # values for the passive vector
        try:
            passive_slice = self._get_slice(self.passive_idx)

        # ValueError is raised if there are no passive parameters
        # simply return the full vector as the active vector, and an empty
        # vector as the passive vector
        except ValueError:
            return merged_vec, np.array([])

        # Determine indices in the merged vector that correspond to
        # values for the active vector
        active_slice = self._get_slice(self.active_idx)

        self.active, self.passive = \
            merged_vec[active_slice], merged_vec[passive_slice]

    def _get_bounds(self):
        """Constructs a list of bounding constraints to set for each
        parameter. The bounds are as follows:

        * amplitudes: 0 < a < ∞
        * phases: -π < φ < π
        * frequencies: offset - sw/2 < f < offset + sw/2
        * damping: 0 < η < ∞
        """

        if not self.bound:
            # Unconstrained optimisation selected
            self.bounds = None

        else:
            self.bounds = []
            # Amplitude
            if 0 in self.active_idx:
                self.bounds += [(0, np.inf)] * self.m
            # Phase
            if 1 in self.active_idx:
                self.bounds += [(-np.pi, np.pi)] * self.m
            # Frequency (iterate over each dimension)
            if 2 in self.active_idx:
                for sw in self.sw:
                    # N.B. as the frequencies are centred about zero
                    # the valid frequency range is:
                    # -sw / 2 -> sw / 2
                    # NOT -sw / 2 + offset -> sw / 2 + offset
                    self.bounds += [(-sw / 2, sw / 2)] * self.m
            # Damping (iterate over each dimension)
            # 2 + self.dim = 3 for 1D and 4 for 2D
            if 2 + self.dim in self.active_idx:
                self.bounds += [(0, np.inf)] * (self.dim * self.m)

    def _run_optimiser(self):
        fprint = 3 if self.fprint else 0
        # Trust-Region
        if self.method == 'trust_region':
            result = optimize.minimize(
                fun=self.funcs['fidelity'],
                x0=self.active,
                args=self.optimiser_args,
                method='trust-constr',
                jac=self.funcs['gradient'],
                hess=self.funcs['hessian'],
                bounds=self.bounds,
                options={
                    'maxiter': self.max_iterations,
                    'verbose': fprint,
                },
            )
        # L-BFGS
        elif self.method == 'lbfgs':
            result = optimize.minimize(
                fun=self.funcs['fidelity'],
                x0=self.active,
                args=self.optimiser_args,
                method='L-BFGS-B',
                jac=self.funcs['gradient'],
                bounds=self.bounds,
                options={
                    'maxiter': self.max_iterations,
                    'iprint': fprint // 3,
                    'disp': True
                }
            )

        # Extract result from optimiser dictionary
        self.active = result['x']

    def _check_negative_amps(self):
        """Determines which oscillators (if any) have negative amplitudes, and
        removes them, or recasts them with positive amplitude and a 180° phase
        shift.

        Returns
        -------
        term : bool
            Used by :py:meth:`_optimise` to decide whether to terminate
            or re-run the optimisation routine.
        """

        if 0 in self.active_idx:
            # Generates length-1 tuple (unpack)
            negative_idx = list(np.nonzero(self.active[:self.m] < 0.0)[0])

            # Check if there are any negative amps by determining
            # if negative_idx is empty or not
            if not negative_idx:
                return True

            # Negative amplitudes exist... deal with these
            if self.negative_amps == 'remove':
                # Remove oscillators with negative amplitudes
                self.active = np.delete(
                    self.active,
                    self._get_slice(self.active_idx, osc_idx=negative_idx),
                )
                self.passive = np.delete(
                    self.passive,
                    self._get_slice(self.passive_idx, osc_idx=negative_idx),
                )

                # Update the number of oscillators
                self.m = int(self.active.size / len(self.active_idx))

                if self.fprint:
                    print(
                        f'{cols.OR}Negative amplitudes detected. These'
                        f' oscillators will be removed\n'
                        f'Updated number of oscillators: {self.m}{cols.END}'
                    )
                # Returning False means the optimisiser will be re-run
                return False

            elif self.negative_amps == 'flip_phase':
                # Make negative amplitude oscillators positive and flip
                # phase by 180°

                # Amplitudes
                amp_slice = self._get_slice([0], osc_idx=negative_idx)
                self.active[amp_slice] *= -1

                # Phase flip
                if 1 in self.active:
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
        """Determine the errors of the estimation result"""

        # Set phase_variance to False
        args = list(copy.deepcopy(self.optimiser_args))
        args[-1] = False
        args = tuple(args)

        # Compute fidelity and hessian for error
        fidelity = self.funcs['fidelity'](self.active, *args)
        hessian = self.funcs['hessian'](self.active, *args)

        # See newton_meets_ockham, Eq. (22)
        self.errors = np.sqrt(
            fidelity * np.abs(np.diag(nlinalg.inv(hessian))) /
            functools.reduce(operator.mul, [n - 1 for n in self.n])
        )

        # Re-scale amplitude errors
        self.errors[:self.m] = self.errors[:self.m] * self.norm
        self.errors = np.reshape(
            self.errors, (int(self.errors.size / 4), 4), order='F',
        )

    def _negligible_amplitudes(self):

        # Threshold
        thold = self.amp_thold * nlinalg.norm(self.result[:self.m])
        # Indices of negligible amplitude oscillators
        negligible_idx = list(np.nonzero(self.result[:self.m] < thold)[0])
        # Remove negligible oscillators
        slice = self._get_slice(list(range(self.p)), osc_idx=negligible_idx)
        self.result = np.delete(self.result, slice)
        # Update number of oscillators
        self.m = int(self.result.size / self.p)

        if negligible_idx:
            print(
                f'{cols.OR}Oscillations with negligible amplitude removed.'
                f' \nUpdated number of oscillators: {self.m}{cols.END}'
            )
