# ve.py
# a series of functions to aid virtual echo construction
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

import itertools

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
from numpy.random import normal


def phase(spec, p0, p1):
    """
    First order phase correction applied to Fourier transformed data.

    Parameters
    ----------
    spec : numpy.ndarray
        Spectrum (FT of time-domain signal)

    p0 : float
        Zero-order phase correction (radians)
    
    p1 : float
        First-order phase correction (radians)

    Returns
    spec_phased : numpy.ndarray
        Phased spectrum
    """

    n = spec.shape[0]
    p = np.exp(1j * p0) * np.exp(1j * p1 * np.arange(n) / n)
    spec_phased = spec * p
    return spec_phased


def super_gaussian(n, region, p=40):
    """
    Generates a super-Gaussian for filtration of
    frequency-domian data.

    Parameters
    ----------
    n : tuple
        Number of points the function is composed of in each dimension.

    region : ((int, int),) or ((int, int), (int, int),)
        Cut-off points of region in each dimensions, in array indices.
        Ordered from lower index to higher

    p : int
        Power of the super-Gaussian. Defaults to 40

    Returns
    -------
    superg : numpy.ndarray
        super-Gaussian"""

    # determine center and bandwidth of super gaussian
    center = ()
    bw = ()

    # loop over each dimension
    # determine center of region and bandwidth
    for bounds in region:
        center += ((bounds[0] + bounds[1]) / 2),
        bw += (bounds[1] - bounds[0]),

    # construct super gaussian
    for i, (n_, c, b) in enumerate(zip(n, center, bw)):

        sg = np.exp(-2 ** (p+1) * ((np.arange(1, n_+1) - c) / b) ** p)
        if i == 0:
            superg = sg

        # None creates a new dimension (see numpy newaxis)
        else:
            superg = superg[..., None] * sg

    return superg


def sg_noise(sg, var):
    """
    Creates an array of noise whose variance at each point reflects
    the super-Gaussian amplitude

    Parameters
    ----------
    sg : numpy.ndarray
        super-Gaussian filter, construted using :func:`_super_gaussian`

    var : float
        Noise variance of the signal

    Returns
    -------
    noise : ndarray
        Noise array, with equivalent shape to sg
    """

    n = sg.shape
    noise = np.zeros(n)

    # generate a nested loop covering all points in the array
    indices = []
    for n_ in n:
        indices.append(range(n_))
    for idx in itertools.product(*indices):
        noise[idx] = normal(0, np.sqrt(var*(1-sg[idx])))
    return noise
