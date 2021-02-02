# load.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

import numpy as np
import numpy.random as nrandom

import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama

"""Provides functionality for constructing synthetic FIDs"""

def make_fid(parameters, n, sw, offset=None, snr=None, decibels=True):
    """Constructs a discrete time-domain signal (FID), as a summation of
    exponentially damped complex sinusoids.

    Parameters
    ----------
    parameters : numpy.ndarray
        Parameter array with the following structure:

        * **1-dimensional data:**

          .. code:: python

             parameters = numpy.array([
                [a_1, φ_1, f_1, η_1],
                [a_2, φ_2, f_2, η_2],
                ...,
                [a_m, φ_m, f_m, η_m],
             ])

        * **2-dimensional data:**

          .. code:: python

             parameters = numpy.array([
                [a_1, φ_1, f1_1, f2_1, η1_1, η2_1],
                [a_2, φ_2, f1_2, f2_2, η1_2, η2_2],
                ...,
                [a_m, φ_m, f1_m, f2_m, η1_m, η2_m],
             ])

    n : [int], [int, int]
        Number of points to construct signal from in each dimension.

    sw : [float], [float, float]
        Sweep width in each dimension, in Hz.

    offset : [float], [float, float], or None, default: None
        Transmitter offset frequency in each dimension, in Hz. If set to
        `None`, the offset frequency will be set to 0Hz in each dimension.

    snr : float or None, default: None
        The signal-to-noise ratio. If `None` then no noise will be added
        to the FID.

    decibels : bool, default: True
        If `True`, the snr is taken to be in units of decibels. If `False`,
        it is taken to be simply the ratio of the singal power over the
        noise power.

    Returns
    -------
    fid : numpy.ndarray
        The synthetic FID.

    tp : [numpy.ndarray], [numpy.ndarray, numpy.ndarray]
        The time points the FID is sampled at in each dimension.
    """
    # --- Check validity of inputs ---------------------------------------
    # Parameters should be a 2-dimnesional numpy array
    if not isinstance(parameters, np.ndarray) or parameters.ndim != 2:
        raise TypeError(
            f'{cols.R}parameters should be a numpy ndarray with 2'
            f' dimesions.{cols.END}'
        )
    # Parameters should have shape (m, 4) for 1D data or (m, 6) for 2D data
    if not parameters.shape[1] in [4, 6]:
        raise ValueError(
            f'{cols.R}parameters should statisfy parameters.shape[1] == 4'
            f' (1D FID) or parameters.shape[1] == 6 (2D FID).{cols.END}'
        )

    # FID dimensionality
    dim = int(parameters.shape[1] / 2) - 1

    # If offset is None, assume it is 0Hz in each dimension.
    if offset is None:
        offset = [0.] * dim

    # Check number of points, sweeep width and offset are of the required
    # type
    for obj, name, type_ in zip((n, offset, sw), ('n', 'offset', 'sw'), (int, float, float)):
        _check_valid_arg(obj, name, type_, dim)

    if not isinstance(snr, float) and snr != None:
        raise TypeError(f'{cols.R}snr should be a float or None{cols.END}')

    if not isinstance(decibels, bool):
        raise TypeError(f'{cols.R}decibels should be a boolean{cols.END}')

    # --- Extract amplitudes, phases, frequencies and damping ------------
    amp = parameters[:, 0]
    phase = parameters[:, 1]
    # Center frequencies at 0 based on offset
    freq = [parameters[:, 2+i] + offset[i] for i in range(dim)]
    damp = [parameters[:, dim+2+i] for i in range(dim)]

    # Time points in each dimension
    tp = get_timepoints(n, sw)

    # --- Generate noiseless FID -----------------------------------------
    if dim == 1:
        # Vandermonde matrix of poles
        Z = np.exp(np.outer(tp[0], (1j*2*np.pi*freq[0] - damp[0])))
        # Vector of complex ampltiudes
        alpha = amp * np.exp(1j * phase)
        # Compute FID!
        fid = Z @ alpha

    if dim == 2:
        # Vandermonde matrices
        Z1 = np.exp(np.outer(tp[0], (1j*2*np.pi*freq[0] - damp[0])))
        Z2t = np.exp(np.outer((1j*2*np.pi*freq[1] - damp[1]), tp[1]))
        # Diagonal matrix of complex amplitudes
        A = np.diag(amp * np.exp(1j * phase))
        # Compute FID!
        fid = Z1 @ A @ Z2t

    # --- Add noise to FID -----------------------------------------------
    if snr == None:
        return fid, tp
    else:
        return fid + _make_noise(fid, snr, decibels), tp


def get_timepoints(n, sw):
    """Generates the timepoints at which an FID is sampled at, given
    its sweep-width, and the number of points.

    Parameters
    ----------
    n : [int] or [int, int]
        The number of points in each dimension.

    sw : [float] or [float, float]
        THe sweep width in each dimension (Hz).

    Returns
    -------
    tp : [numpy.ndarray] or [numpy.ndarray, numpy.ndarray]
        The time points sampled in each dimension
    """
    # Determine the desired dim based on the length on n
    try:
        dim = len(n)
    except:
        raise TypeError(f'{cols.R}\'n\' needs to be a list!{cols.END}')

    # check n and sw are both ogf the correct type
    for obj, name, type_ in zip((n, sw), ('n', 'sw'), (int, float)):
        _check_valid_arg(obj, name, type_, dim)

    return [np.linspace(0, float(n_) / sw_, n_) for n_, sw_ in zip(n, sw)]


def get_shifts(n, sw):
    """Generates the frequencies that the FT of the FID is sampled at, given
    its sweep-width, and the number of points.

    Parameters
    ----------
    n : [int] or [int, int]
        The number of points in each dimension.

    sw : [float] or [float, float]
        THe sweep width in each dimension (Hz).

    Returns
    -------
    tp : [numpy.ndarray] or [numpy.ndarray, numpy.ndarray]
        The time points sampled in each dimension
    """
    # Determine the desired dim based on the length on n
    try:
        dim = len(n)
    except:
        raise TypeError(f'{cols.R}\'n\' needs to be a list!{cols.END}')

    # check n and sw are both ogf the correct type
    for obj, name, type_ in zip((n, sw), ('n', 'sw'), (int, float)):
        _check_valid_arg(obj, name, type_, dim)

    return [np.linspace(0, float(n_) / sw_, n_) for n_, sw_ in zip(n, sw)]



def _make_noise(fid, snr, decibels=True):
    """Given a synthetic FID, generate an array of normally distributed
    noise with zero mean and a variance that abides by the desired SNR.

    Parameters
    ----------
    fid : numpy.ndarray
        Noiseless FID.

    snr : float
        The signal-to-noise ratio.

    decibels : bool, default: True
        If `True`, the snr is taken to be in units of decibels. If `False`,
        it is taken to be simply the ratio of the singal power over the
        noise power.

    Returns
    _______
    nfid : numpy.ndarray
        Noisy FID
    """

    size = fid.size
    shape = fid.shape

    # Compute the variance of the noise
    if decibels:
        var = np.real((np.sum(np.abs(fid) ** 2)) / (size * (20 ** (snr / 10))))
    else:
        var = np.real((np.sum(np.abs(fid) ** 2)) / (2 * size * snr))

    # Make a number of noise instances and check which two are closest
    # to the desired variance.
    # These two are then taken as the real and imaginary noise components
    instances = []
    variances = []
    for _ in range(100):
        instance = nrandom.normal(loc=0, scale=np.sqrt(var), size=shape)
        instances.append(instance)
        variances.append(np.var(instance))

    # determine which instance's variance is the closest to the desired
    # variance
    differences = np.abs((np.array(variances) - var))
    first, second, *_ = np.argpartition(differences, 1)

    # The noise is constructed from the two closest arrays in a variance-sense
    # to the desired SNR
    return instances[first] + 1j * instances[second]


def _check_valid_arg(obj, name, type_, dim):
    """Ensures `obj` is of the correct format.

    Checks the following things:

    1. `obj` is a `list`
    2. The length of `obj` matches `dim`
    3. Each element of `obj` is an instance of `type_`

    Parameters
    ----------
    obj : misc
        The object to be tested

    name : str
        The name of the variable. Used in error message if `obj` doesn't
        pass the checks.

    type_ : class
        The type that the list elements should be.

    dim : 1, 2
        The experiment dimension that `obj` relates to.

    Raises
    ------
    TypeError if any of the checks fail.
    """

    # Message for the error if checking fails
    errmsg = (
        f'{cols.R}{name} should be a list of length {dim} with values of'
        f' type {type_.__name__}{cols.END}'
    )
    if isinstance(obj, list) and len(obj) == dim:
        for elem in obj:
            if not isinstance(elem, type_):
                raise TypeError(errmsg)
        return
    else:
        raise TypeError(errmsg)
