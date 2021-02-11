# signal.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Constructing and processing NMR signals"""

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
import numpy.random as nrandom

import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
from nmrespy._misc import ArgumentChecker

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
    try:
        dim = len(n)
    except:
        raise TypeError(f'{cols.R}n should be iterable.{cols.END}')

    if offset == None:
        offset = [0.0] * dim

    components = [
        (parameters, 'parameters', 'parameter'),
        (n, 'n', 'int_list'),
        (sw, 'sw', 'float_list'),
        (offset, 'offset', 'float_list'),
        (decibels, 'decibels', 'bool'),
    ]

    if snr != None:
        components.append((snr, 'snr', 'float'))

    ArgumentChecker(components, dim)

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
        return fid + make_noise(fid, snr, decibels), tp


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

    try:
        dim = len(n)
    except:
        raise TypeError(f'{cols.R}n should be iterable.{cols.END}')

    ArgumentChecker([(n, 'n', 'int_list'), (sw, 'sw', 'float_list')], dim)

    return [np.linspace(0, float(n_) / sw_, n_) for n_, sw_ in zip(n, sw)]


def get_shifts(n, sw, offset):
    """Generates the frequencies that the FT of the FID is sampled at, given
    its sweep-width, and the number of points.

    Parameters
    ----------
    n : [int] or [int, int]
        The number of points in each dimension.

    sw : [float] or [float, float]
        The sweep width in each dimension (Hz).

    offset : [float] or [float, float]
        The transmitter offset in each dimension (Hz).

    Returns
    -------
    shifts : [numpy.ndarray] or [numpy.ndarray, numpy.ndarray]
        The chemical shift values sampled in each dimension (Hz).
    """

    try:
        dim = len(n)
    except:
        raise TypeError(f'{cols.R}n should be iterable.{cols.END}')

    ArgumentChecker(
        [(n, 'n', 'int_list'),
         (sw, 'sw', 'float_list'),
         (offset, 'offset', 'float_list'),
        ], dim=dim
    )

    shifts = []
    for n_, sw_, off in zip(n, sw, offset):
        shifts.append(
            np.linspace((-sw_ / 2) + off, (sw_ / 2) + off, n_)
        )

    return shifts

def ft(fid):
    """Performs Fourier transformation and flips the resulting spectrum
    to satisfy NMR convention.

    Parameters
    ----------
    fid : numpy.ndarray
        Time-domain data

    Returns
    -------
    spectrum : numpy.ndarray
        Fourier transform of the data, flipped in each dimension.
    """

    try:
        dim = fid.ndim
    except:
        raise TypeError(f'{cols.R}fid should be a numpy ndarray{cols.END}')

    ArgumentChecker([(fid, 'fid', 'ndarray')])

    for axis in range(dim):
        try:
            spectrum = fft(spectrum, axis=axis)
        except NameError:
            spectrum = fft(fid, axis=axis)

    return np.flip(fftshift(spectrum))

def ift(spectrum):
    """Flips spectral data in each dimension, and then inverse Fourier
    transforms.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum

    Returns
    -------
    fid : numpy.ndarray
        Inverse Fourier transform of the spectrum.
    """

    ArgumentChecker([(spectrum, 'spectrum', 'ndarray')])

    spectrum = ifftshift(np.flip(spectrum))
    for axis in range(spectrum.ndim):
        try:
            fid = ifft(fid, axis=axis)
        except NameError:
            fid = ifft(spectrum, axis=axis)

    return fid


def phase_spectrum(spectrum, p0, p1):
    """Applies a linear phase correction to `spectrum`.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum

    p0 : [float] or [float, float]
        Zero-order phase correction in each dimension, in radians.

    p1 : [float] or [float, float]
        First-order phase correction in each dimension, in radians.

    Returns
    -------
    phased_spectrum : numpy.ndarray
    """

    try:
        dim = len(p0)
    except:
        raise TypeError(f'{cols.R}p0 should be iterable.{cols.END}')

    components = [
        (spectrum, 'spectrum', 'ndarray'),
        (p0, 'p0', 'float_list'),
        (p1, 'p1', 'float_list'),
    ]

    ArgumentChecker(components, dim)

    # Indices for einsum
    # For 1D: 'i'
    # For 2D: 'ij'
    idx = ''.join([chr(i + 105) for i in range(dim)])

    for axis, (p0_, p1_) in enumerate(zip(p0, p1)):
        n = spectrum.shape[axis]
        # Determine axis for einsum (i or j)
        axis = chr(axis + 105)
        p = np.exp(1j * (p0_ + p1_ * np.arange(n) / n))
        phased_spectrum = np.einsum(f'{idx},{axis}->{idx}', spectrum, p)

    return phased_spectrum


def make_noise(fid, snr, decibels=True):
    """Given a synthetic FID, generate an array of normally distributed
    complex noise with zero mean and a variance that abides by the desired
    SNR.

    Parameters
    ----------
    fid : numpy.ndarray
        Noiseless FID.

    snr : float
        The signal-to-noise ratio.

    decibels : bool, default: True
        If `True`, the snr is taken to be in units of decibels. If `False`,
        it is taken to be simply the ratio of the singal power and noise
        power.

    Returns
    _______
    noise : numpy.ndarray
    """

    components = [
        (fid, 'fid', 'ndarray'),
        (snr, 'snr', 'float'),
        (decibels, 'decibels', 'bool'),
    ]

    ArgumentChecker(components)

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
    var_discrepancies = []
    for _ in range(100):
        instance = nrandom.normal(loc=0, scale=np.sqrt(var), size=shape)
        instances.append(instance)
        var_discrepancies.append(np.abs(np.var(instances) - var))

    # Determine which instance's variance is the closest to the desired
    # variance
    first, second, *_ = np.argpartition(var_discrepancies, 1)

    # The noise is constructed from the two closest arrays in a variance-sense
    # to the desired SNR
    return instances[first] + 1j * instances[second]


def generate_random_signal(m, n, sw, snr=None):
    """A convienince function to generate a synthetic FID with random
    parameters for testing purposes.

    Parameters
    ----------
    m : int
        Number of oscillators

    n : [int] or [int, int]
        Number of points in each dimension

    sw : [float] or [float, float]
        Sweep width in each dimension

    snr : float or None, default: None
        Signal-to-noise ratio (dB)
    """

    try:
        dim = len(n)
    except:
        raise TypeError(f'{cols.R}n should be an iterable{cols.END}')
