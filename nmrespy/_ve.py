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
    nmrespy.ve.phase(spec, p0, p1)

    ───Description─────────────────────────────
    First order phase correction applied to Fourier transformed data.
    *** ONLY 1D DATA CURRENTLY SUPPORTED ***

    ───Parameters──────────────────────────────
    spec - ndarray
        Spectrum (FT of time-domain signal)
    p0 - float
        Zero-order phase correction (radians)
    p1 - float
        First-order phase correction (radians)

    ───Returns─────────────────────────────────
    spec_phased - ndarray
        Phased version of spec
    """

    n = spec.shape[0]
    p = np.exp(1j * p0) * np.exp(1j * p1 * np.arange(n) / n)
    spec_phased = spec * p
    return spec_phased


def super_gaussian(n, region, p=40):
    """
    nmrespy.ve.super_gaussian(n, highs, lows, p=40)

    ───Description─────────────────────────────
    Generates a super-Gaussian for filtration of
    frequency-domian data.

    ───Parameters──────────────────────────────
    n - tuple
        Number of points the function is composed of in each dimension.
    region - ((int, int),) or ((int, int), (int, int),)
        Cut-off points of region in each dimensions, in array indices.
        Ordered from lower index to higher
    p - int
        Power of the super-Gaussian. Defaults to 40

    ───Returns─────────────────────────────────
    superg - ndarray
        super-Gaussian function"""

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
    nmrespy.ve.sg_noise(sg, var)

    ───Description─────────────────────────────
    Creates an array of noise whose value at the point with coordinates
    [idx] is drawn from a normal distribution with mean 0 and variance
    var*(1-sg[idx]).

    ───Parameters──────────────────────────────
    sg - ndarray
        Super-Gaussian filter, construted using ve._super_gaussian
    var - float
        Noise variance of the signal the Super-gaussian will be
        applied to

    ───Returns─────────────────────────────────
    noise - ndarray
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
