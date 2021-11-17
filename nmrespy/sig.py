# sig.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 18 Oct 2021 11:17:37 BST

"""Constructing and processing NMR signals."""

import copy
import re
import tkinter as tk
from typing import Dict, Iterable, Tuple, Union

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
import numpy.random as nrandom
import scipy.integrate as integrate

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from nmrespy import RED, ORA, END, USE_COLORAMA, ExpInfo

if USE_COLORAMA:
    import colorama

    colorama.init()
import nmrespy._errors as errors
from nmrespy._misc import ArgumentChecker

"""Provides functionality for constructing synthetic FIDs"""


return_type = Tuple[
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    Union[Tuple[np.ndarray], Tuple[np.ndarray, np.ndarray]],
]


def make_fid(
    params: np.ndarray,
    expinfo: ExpInfo,
    *,
    snr: Union[float, None] = None,
    decibels: bool = True,
    modulation: str = "none",
) -> return_type:
    r"""Construct a FID, as a summation of damped complex sinusoids.

    Parameters
    ----------
    params
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

    expinfo
        Information on the experiment. Used to determine the number of points,
        sweep width and transmitter offset.

    snr
        The signal-to-noise ratio. If `None` then no noise will be added
        to the FID.

    decibels
        If `True`, the snr is taken to be in units of decibels. If `False`,
        it is taken to be simply the ratio of the singal power over the
        noise power.

    modulation
        The type of modulation present in the indirect dimension, if the data
        is 2D. `In the expressions below, a it is assumed a single oscillator
        has been provided for simplicity`.

        * `'none'`: Returns a single signal of the form:

          .. math::

             y(t_1, t_2) = a \exp(\mathrm{i} \phi)
             \exp \left[ \left( 2 \pi \mathrm{i} f_1 - \eta_1 \right)
             t_1 \right]
             \exp \left[ \left( 2 \pi \mathrm{i} f_2 - \eta_2 \right)
             t_2 \right]

        * `'amp'`: Returns an amplitude-modulated pair of signals of the form:

          .. math::

             y_{\mathrm{cos}}(t_1, t_2) = a \exp(\mathrm{i} \phi)
             \cos \left( 2 \pi f_1 t_1 \right)
             \exp \left( - \eta_1 t_1 \right)
             \exp \left[ \left( 2 \pi \mathrm{i} f_2 - \eta_2 \right)
             t_2 \right]

             y_{\mathrm{sin}}(t_1, t_2) = a \exp(\mathrm{i} \phi)
             \sin \left( 2 \pi f_1 t_1 \right)
             \exp \left( - \eta_1 t_1 \right)
             \exp \left[ \left( 2 \pi \mathrm{i} f_2 - \eta_2 \right)
             t_2 \right]

        * `'phase'`: Returns an phase-modulated pair of signals of the form:

          .. math::

             y_{\mathrm{P}}(t_1, t_2) = a \exp(\mathrm{i} \phi)
             \exp \left[ \left( 2 \pi \mathrm{i} f_1 - \eta_1 \right)
             t_1 \right]
             \exp \left[ \left( 2 \pi \mathrm{i} f_2 - \eta_2 \right)
             t_2 \right]

             y_{\mathrm{N}}(t_1, t_2) = a \exp(\mathrm{i} \phi)
             \exp \left[ \left( - 2 \pi \mathrm{i} f_1 - \eta_1 \right)
             t_1 \right]
             \exp \left[ \left( 2 \pi \mathrm{i} f_2 - \eta_2 \right)
             t_2 \right]

    Returns
    -------
    fid
        The synthetic signal generated.

        + If the data to be constructed is 1D or 2D with `modulation` set to
          `'none'`, the result will be a NumPy array.
        + If the data is 2D with `modulation` set to `'amp'`, or `'phase'`
          the result will be a length-2 list with signals of the forms
          indicated above (See `modulation`).

    tp
        The time points the FID is sampled at in each dimension.

    Notes
    -----
    The resulting `fid` is given by

    .. math::

       y[n_1, \cdots, n_D] =
       \sum_{m=1}^{M} a_m \exp\left(\mathrm{i} \phi_m\right)
       \prod_{d=1}^{D}
       \exp\left[\left(2 \pi \mathrm{i} f_m - \eta_m\right)
       n_d \Delta t_d\right]

    where :math:`d` is either 1 or 2, :math:`M` is the number of
    oscillators, and :math:`\Delta t_d = 1 / f_{\mathrm{sw}, d}`.
    """
    # --- Check validity of inputs ---------------------------------------
    try:
        dim, offset = expinfo.unpack("dim", "offset")
    except Exception:
        raise TypeError(
            f"{RED}Check `expinfo` is an instance of " f"nmrespy.ExpInfo{END}"
        )

    checker = ArgumentChecker(dim=dim)
    checker.stage(
        (params, "params", "parameter"),
        (offset, "offset", "float_iter"),
        (decibels, "decibels", "bool"),
        (modulation, "modulation", "modulation"),
        (snr, "snr", "float", True),
    )
    checker.check()

    # --- Extract amplitudes, phases, frequencies and damping ------------
    amp = params[:, 0]
    phase = params[:, 1]
    # Center frequencies at 0 based on offset
    freq = [params[:, 2 + i] - offset[i] for i in range(dim)]
    damp = [params[:, dim + 2 + i] for i in range(dim)]

    # Time points in each dimension
    tp = get_timepoints(expinfo)

    # --- Generate noiseless FID -----------------------------------------
    if dim == 1:
        # Vandermonde matrix of poles
        Z = np.exp(np.outer(tp[0], (1j * 2 * np.pi * freq[0] - damp[0])))
        # Vector of complex ampltiudes
        alpha = amp * np.exp(1j * phase)
        # Compute FID!
        fid = Z @ alpha

    if dim == 2:
        if modulation in ["none", "amp"]:
            Z1 = np.exp(np.outer(tp[0], (1j * 2 * np.pi * freq[0] - damp[0])))
            if modulation == "none":
                Z1 = [Z1]

            elif modulation == "amp":
                Z1 = [np.real(Z1), np.imag(Z1)]
        elif modulation == "phase":
            Z1 = [
                np.exp(np.outer(tp[0], (1j * 2 * np.pi * freq[0] - damp[0]))),
                np.exp(np.outer(tp[0], (-1j * 2 * np.pi * freq[0] - damp[0]))),
            ]

        Z2T = np.exp(np.outer((1j * 2 * np.pi * freq[1] - damp[1]), tp[1]))
        # TODO: Support for constructing negative time signals
        # rev_poles = np.outer(1j * 2 * np.pi * freq[1], -tp[1][::-1]) + \
        #             np.outer(-damp[1], tp[1][::-1])
        # Z2revT = np.exp(rev_poles)
        # Z2fullT = np.hstack((Z2revT, Z2T))
        # print(Z2fullT.shape)
        # Diagonal matrix of complex amplitudes
        A = np.diag(amp * np.exp(1j * phase))

        fid = []
        for z1 in Z1:
            fid.append(z1 @ A @ Z2T)
            # fid.append(z1 @ A @ Z2fullT)

        if len(fid) == 1:
            fid = fid[0]

    # --- Add noise to FID -----------------------------------------------
    if snr is None:
        return fid, tp
    else:
        if isinstance(fid, np.ndarray):
            return fid + _make_noise(fid, snr, decibels), tp
        elif isinstance(fid, list):
            for i, f in enumerate(fid):
                fid[i] = f + _make_noise(f, snr, decibels)
            return fid, tp


def make_virtual_echo(
    data: Iterable[np.ndarray], modulation: str = "amp"
) -> np.ndarray:
    """Generate a virtual echo [#]_ from a time-domain signal.

    A vitrual echo is a signal with a purely real Fourier-Tranform and
    absorption mode line shape if the data is phased.

    Parameters
    ----------
    data
        The data to construct the virtual echo from. This should be a list of
        NumPy arrays, with ``len(data) == d`` where ``d`` is the dimension
        of the signal.

    modulation
        If the data is 2D, this parameter specifies the type of modulation
        present in the indirect dimension of the dataset.

        * If set to `'amp'`, the two signals in the `data` should be an
          amplitude modulated pair.
        * If set to `'phase'`, the two signals in the `data` should be a phase
          modulated pair.

        See the docs for :py:func:`make_fid` for more info on `modulation`.

    Returns
    -------
    virtual_echo
        The virtual echo signal assocaited with `data`.

    References
    ----------
    .. [#] M. Mayzel, K. Kazimierczuk, V. Y. Orekhov, The causality principle
           in the reconstruction of sparse nmr spectra, Chem. Commun. 50 (64)
           (2014) 8947–8950.
    """
    try:
        dim = len(data)
    except Exception:
        raise TypeError(f"{RED}n should be iterable.{END}")

    if dim > 2:
        raise errors.MoreThanTwoDimError()

    checker = ArgumentChecker(dim=dim)
    checker.stage(
        (data, "data", "array_iter"), (modulation, "modulation", "modulation")
    )
    checker.check()

    if dim == 1:
        data = data[0]
        pts = data.size
        ve = np.zeros((2 * pts - 1), dtype="complex")
        ve[0] = np.real(data[0])
        ve[1:pts] = data[1:]
        ve[pts:] = data[1:][::-1].conj()

    elif dim == 2:
        # TODO NEEDS FIXING
        if modulation == "amp":
            c = data[0]
            s = data[1]
        else:
            # Phase modulated (P and N) signals
            c = 0.5 * (data[0] + data[1])
            s = -1j * 0.5 * (data[0] - data[1])

        # S±± = (R₁ ± iI₁)(R₂ ± iI₂)
        # where: Re(c) -> R₁R₂, Im(c) -> R₁I₂, Re(s) -> I₁R₂, Im(s) -> I₁I₂
        r1r2 = np.real(c)
        r1i2 = np.imag(c)
        i1r2 = np.real(s)
        i1i2 = np.imag(s)

        # S++ = R₁R₂ - I₁I₂ + i(R₁I₂ + I₁R₂)
        pp = r1r2 - i1i2 + 1j * (r1i2 + i1r2)
        # S+- = R₁R₂ + I₁I₂ + i(I₁R₂ - R₁I₂)
        pm = r1r2 + i1i2 + 1j * (i1r2 - r1i2)
        # S-+ = R₁R₂ + I₁I₂ + i(R₁I₂ - I₁R₂)
        mp = r1r2 + i1i2 + 1j * (r1i2 - i1r2)
        # S-- = R₁R₂ - I₁I₂ - i(R₁I₂ + I₁R₂)
        mm = r1r2 - i1i2 - 1j * (r1i2 + i1r2)

        pts = s.shape

        tmp1 = np.zeros(tuple(2 * p - 1 for p in pts), dtype="complex")
        tmp1[: pts[0], : pts[1]] = pp
        tmp1[0] /= 2
        tmp1[:, 0] /= 2

        tmp2 = np.zeros(tuple(2 * p - 1 for p in pts), dtype="complex")
        tmp2[: pts[0], pts[1] - 1 :] = pm[:, ::-1]
        tmp2[0] /= 2
        tmp2[:, -1] /= 2
        tmp2 = np.roll(tmp2, 1, axis=1)

        tmp3 = np.zeros(tuple(2 * p - 1 for p in pts), dtype="complex")
        tmp3[pts[0] - 1 :, : pts[1]] = mp[::-1]
        tmp3[-1] /= 2
        tmp3[:, 0] /= 2
        tmp3 = np.roll(tmp3, 1, axis=0)

        tmp4 = np.zeros(tuple(2 * p - 1 for p in pts), dtype="complex")
        tmp4[pts[0] - 1 :, pts[1] - 1 :] = mm[::-1, ::-1]
        tmp4[-1] /= 2
        tmp4[:, -1] /= 2
        tmp4 = np.roll(tmp4, 1, axis=(0, 1))

        ve = tmp1 + tmp2 + tmp3 + tmp4

    return ve


def zf(data: np.ndarray) -> np.ndarray:
    """Zero-fill data to the next power of 2 in each dimension.

    Parameters
    ----------
    data
        Signal to zero-fill.

    Returns
    -------
    zf_data: numpy.ndarray
        Zero-filled data.
    """
    zf_data = copy.deepcopy(data)
    # Zero-fill to nearest power of 2
    for i, n in enumerate(zf_data.shape):
        if n & (n - 1) == 0:
            pass
        else:
            nearest_pow_2 = int(2 ** np.ceil(np.log2(n)))
            pts_to_append = nearest_pow_2 - n
            shape_to_add = list(zf_data.shape)
            shape_to_add[i] = pts_to_append
            zeros = np.zeros(shape_to_add, dtype="complex")
            zf_data = np.concatenate((zf_data, zeros), axis=i)

    return zf_data


def get_timepoints(
    expinfo: ExpInfo,
    *,
    start_time: Union[Iterable[Union[float, str]], None] = None,
    meshgrid_2d: bool = False,
) -> Iterable[np.ndarray]:
    r"""Generate the timepoints at which an FID was sampled at.

    Parameters
    ----------
    expinfo
        Information on the experiment. Used to determine the number of points,
        and sweep width.

    start_time
        The start time in each dimension. If set to `None`, the initial
        point in each dimension with be ``0.0``. To set non-zero start times,
        a list of floats or strings can be used. If floats are used, they
        specify the first value in each dimension in seconds. Alternatively,
        strings of the form ``f'{N}dt'``, where ``N`` is an integer, may be
        used, which indicates a cetain multiple of the difference in time
        between two adjacent points.

    meshgrid_2d
        If time-points are being derived for a two-dimensional signal, setting
        this argument to ``True`` will return two two-dimensional arrays
        corresponding to all pairs of x and y values to construct a 3D
        plot/contour plot.

    Returns
    -------
    tp: Iterable[numpy.ndarray]
        The time points sampled in each dimension.

    Notes
    -----
    If strings are used in the ``start_time`` argument, they must match the
    following regular expression: ``r'^-?\d+dt$'``
    """
    try:
        dim, pts, sw = expinfo.unpack("dim", "pts", "sw")
    except Exception:
        raise TypeError(
            f"{RED}Check `expinfo` is an instance of " f"nmrespy.ExpInfo{END}"
        )

    if start_time is None:
        start_time = [0.0] * dim

    checker = ArgumentChecker(dim=dim)
    checker.stage(
        (pts, "pts", "int_iter"),
        (sw, "sw", "float_iter"),
        (start_time, "start_time", "start_time"),
        (meshgrid_2d, "meshgrid_2d", "bool"),
    )
    checker.check()

    for i, (st, sw_) in enumerate(zip(start_time, sw)):
        if isinstance(st, str):
            start_time[i] = float(re.match(r"^(-?\d+)dt$", st).group(1)) / sw_

    tp = tuple(
        [
            np.linspace(0, float(pts_ - 1) / sw_, pts_) + st
            for pts_, sw_, st in zip(pts, sw, start_time)
        ]
    )

    if dim == 2 and meshgrid_2d:
        tp = tuple(np.meshgrid(*tp, indexing="ij"))

    return tp


def get_shifts(
    expinfo: ExpInfo, *, unit: str = "hz", flip: bool = True
) -> Iterable[np.ndarray]:
    """Generate the frequencies a spectrum is sampled at.

    Parameters
    ----------
    expinfo
        Information on the experiment. Used to determine the number of points,
        sweep width, offset, and transmitter frequency. Note that if
        ``expinfo.sfo`` is ``None``, shifts can only be obtained in Hz.

    unit
        The unit of the chemical shifts. One of ``'hz'``, ``'ppm'``.

    flip
        If `True`, the shifts will be returned in descending order, as is
        conventional in NMR. If `False`, the shifts will be in ascending order.

    Returns
    -------
    shifts: Iterable[numpy.ndarray]
        The chemical shift values sampled in each dimension.
    """
    try:
        pts, sw, offset, sfo, dim = expinfo.unpack("pts", "sw", "offset", "sfo", "dim")
    except Exception:
        raise TypeError(
            f"{RED}Check `expinfo` is an instance of " f"nmrespy.ExpInfo{END}"
        )

    if unit not in ["hz", "ppm"]:
        raise ValueError(f"{RED}`unit` should be either 'hz' or " f"'ppm'.{END}")

    if sfo is None and unit == "ppm":
        print(
            f"{ORA}You need to specify `sfo` if you want chemical"
            f" shifts in ppm! Falling back to Hz...{END}"
        )
        unit = "hz"

    checker = ArgumentChecker(dim=dim)
    checker.stage(
        (pts, "pts", "int_iter"),
        (sw, "sw", "float_iter"),
        (offset, "offset", "float_iter"),
        (sfo, "sfo", "float_iter", True),
        (flip, "flip", "bool"),
    )
    checker.check()

    shifts = [
        np.linspace((-sw_ / 2) + offset_, (sw_ / 2) + offset_, pts_)
        for pts_, sw_, offset_ in zip(pts, sw, offset)
    ]
    if unit == "ppm":
        shifts = [s / sfo_ for s, sfo_ in zip(shifts, sfo)]

    return tuple([np.flip(s) for s in shifts]) if flip else tuple(shifts)


def ft(fid: np.ndarray, *, flip: bool = True) -> np.ndarray:
    """Fourier transformation with optional spectrum flipping.

    It is conventional in NMR to plot spectra from high to low going
    left to right/down to up. This function utilises the
    `numpy.fft <https://numpy.org/doc/stable/reference/routines.fft.html>`_
    module to carry out the Fourier Transformation.

    Parameters
    ----------
    fid
        Time-domain data.

    flip
        Whether or not to flip the Fourier Trnasform of `fid` in each
        dimension.

    Returns
    -------
    spectrum: numpy.ndarray
        Fourier transform of the data, (optionally) flipped in each
        dimension.
    """
    checker = ArgumentChecker()
    checker.stage(
        (fid, "fid", "ndarray"),
        (flip, "flip", "bool"),
    )
    checker.check()

    spectrum = copy.deepcopy(fid)
    for axis in range(fid.ndim):
        spectrum = fftshift(fft(spectrum, axis=axis), axes=axis)

    return np.flip(spectrum) if flip else spectrum


def ift(spectrum, flip=True):
    """Optionally Flip and Inverse Fourier Transform a spectrum.

    This function utilises the
    `numpy.fft <https://numpy.org/doc/stable/reference/routines.fft.html>`_
    module to carry out the Fourier Transformation.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum

    flip : bool, default: True
        Whether or not to flip `spectrum` in each dimension prior to Inverse
        Fourier Transform.

    Returns
    -------
    fid : numpy.ndarray
        Inverse Fourier transform of the spectrum.
    """
    checker = ArgumentChecker()
    checker.stage(
        (spectrum, "spectrum", "ndarray"),
        (flip, "flip", "bool"),
    )
    checker.check()

    spectrum = ifftshift(np.flip(spectrum) if flip else spectrum)
    fid = copy.deepcopy(spectrum)

    for axis in range(spectrum.ndim):
        fid = ifft(fid, axis=axis)

    return fid


def proc_amp_modulated(data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Generate a frequency-dscrimiated signal from amp-modulated 2D FIDs.

    Parameters
    ----------
    data
        cos-modulated signal and sin-modulated signal

    Returns
    -------
    spectrum: np.ndarray
        Frequency-dsicrimiated spectrum.
    """
    checker = ArgumentChecker(dim=2)
    checker.stage((data, "data", "array_iter"))
    checker.check()

    c = data[0]
    s = data[1]

    c_t1_f2 = fftshift(fft(c, axis=1), axes=1)
    s_t1_f2 = fftshift(fft(s, axis=1), axes=1)

    return fftshift(fft(np.real(c_t1_f2) + 1j * np.real(s_t1_f2), axis=0), axes=0)


def proc_phase_modulated(data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
    """Process phase modulated 2D FIDs.

    This function generates the set of spectra corresponding to the
    processing protocol outlined in [#]_.

    Parameters
    ----------
    data
        P-type signal and N-type signal

    Returns
    -------
    spectra: Dict[str, numpy.ndarray]
        Dictionary of four elements: ``rr``, ``ri``, ``ir``, and ``ii``.

    References
    ----------
    .. [#] A. L. Davis, J. Keeler, E. D. Laue, and D. Moskau, “Experiments for
           recording pure-absorption heteronuclear correlation spectra using
           pulsed field gradients,” Journal of Magnetic Resonance (1969),
           vol. 98, no. 1, pp. 207–216, 1992.
    """
    checker = ArgumentChecker(dim=2)
    checker.stage((data, "data", "array_iter"))
    checker.check()

    p = data[0]
    n = data[1]

    p_t1_f2 = fftshift(fft(p, axis=1), axes=1)  # Sₚ(t₁,f₂)
    n_t1_f2 = fftshift(fft(n, axis=1), axes=1)  # Sₙ(t₁,f₂)

    # Generating rr and ir
    plus_t1_f2 = 0.5 * (p_t1_f2 + n_t1_f2.conj())  # S⁺(t₁,f₂)
    plus_f1_f2 = fftshift(fft(plus_t1_f2, axis=0), axes=0)  # S⁺(f₁,f₂)
    rr = np.real(plus_f1_f2)  # A₁A₂
    ir = np.imag(plus_f1_f2)  # D₁A₂

    # Generating ri and ii
    minus_t1_f2 = -0.5 * 1j * (p_t1_f2 - n_t1_f2.conj())  # S⁻(t₁,f₂)
    minus_f1_f2 = fftshift(fft(minus_t1_f2, axis=0), axes=0)  # S⁻(f₁,f₂)
    ri = np.real(minus_f1_f2)  # A₁D₂
    ii = np.imag(minus_f1_f2)  # D₁D₂

    return {
        "rr": rr,
        "ri": ri,
        "ir": ir,
        "ii": ii,
    }


def phase(
    data: np.ndarray,
    p0: Iterable[float],
    p1: Iterable[float],
    pivot: Union[Iterable[float], None] = None,
) -> np.ndarray:
    """Apply a linear phase correction to a signal.

    Parameters
    ----------
    data
        Data to be phased.

    p0
        Zero-order phase correction in each dimension, in radians.

    p1
        First-order phase correction in each dimension, in radians.

    pivot
        Index of the pivot in each dimension. If None, the pivot will be `0`
        in each dimension.

    Returns
    -------
    phased_data : numpy.ndarray
    """
    try:
        dim = data.ndim
    except AttributeError:
        raise TypeError(f"{RED}`data` should be a NumPy array.{END}")

    if pivot is None:
        pivot = dim * [0]

    checker = ArgumentChecker(dim=dim)
    checker.stage(
        (data, "data", "ndarray"),
        (p0, "p0", "float_iter"),
        (p1, "p1", "float_iter"),
        (pivot, "pivot", "int_iter"),
    )
    checker.check()

    # Indices for einsum
    # For 1D: 'i'
    # For 2D: 'ij'
    idx = "".join([chr(i + 105) for i in range(dim)])

    for axis, (piv, p0_, p1_) in enumerate(zip(pivot, p0, p1)):
        n = data.shape[axis]
        # Determine axis for einsum (i or j)
        axis = chr(axis + 105)
        p = np.exp(1j * (p0_ + p1_ * np.arange(-piv, -piv + n) / n))
        phased_data = np.einsum(f"{idx},{axis}->{idx}", data, p)

    return phased_data


def manual_phase_data(
    spectrum: np.ndarray, *, max_p1: Union[Iterable[float], None] = None
) -> Tuple[Union[Iterable[float], None], Union[Iterable[float], None]]:
    """Manual phase correction using a Graphical User Interface.

    .. warning::
       Only 1D spectral data is currently supported.

    Parameters
    ----------
    spectrum
        Spectral data of interest.

    max_p1
        Specifies the range of first-order phases permitted.
        Bounds are set as ``[-max_p1, max_p1]``.

    Returns
    -------
    p0: Union[Iterable[float], None]
        Zero-order phase correction in each dimension, in radians. If the
        user chooses to cancel rather than save, this is set to ``None``.

    p1: Union[Iterable[float], None]
        First-order phase correction in each dimension, in radians. If the
        user chooses to cancel rather than save, this is set to ``None``.
    """
    try:
        dim = spectrum.ndim
    except Exception:
        raise TypeError(f"{RED}spectrum should be a numpy array{END}")
    if dim != 1:
        raise errors.TwoDimUnsupportedError()
    if max_p1 is None:
        max_p1 = tuple(dim * [10 * np.pi])

    init_spectrum = copy.deepcopy(spectrum)

    app = PhaseApp(init_spectrum, max_p1)
    app.mainloop()
    print(app.p0, app.p1)
    return (app.p0,), (app.p1,)


class PhaseApp(tk.Tk):
    """Tkinter application for manual phase correction.

    Notes
    -----
    This is invoked when :py:func:`manual_phase_data` is called.
    """

    def __init__(self, spectrum: np.ndarray, max_p1: Iterable[float]) -> None:
        """Construct the GUI.

        Parameters
        ----------
        spectrum
            Spectral data of interest.

        max_p1
            Specifies the range of first-order phases permitted.
            Bounds are set as ``[-max_p1, max_p1]`` in each dimension.
        """
        super().__init__()
        self.p0 = 0.0
        self.p1 = 0.0
        self.n = spectrum.size
        self.pivot = 0
        self.init_spectrum = copy.deepcopy(spectrum)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 4), dpi=160)
        # Set colour of figure frame
        r, g, b = [x >> 8 for x in self.winfo_rgb(self.cget("bg"))]
        color = f"#{r:02x}{g:02x}{b:02x}"
        if not re.match(r"^#[0-9a-f]{6}$", color):
            color = "#d9d9d9"

        self.fig.patch.set_facecolor(color)
        self.ax = self.fig.add_axes([0.03, 0.1, 0.94, 0.87])
        self.ax.set_yticks([])
        self.specline = self.ax.plot(np.real(spectrum), color="k")[0]

        ylim = self.ax.get_ylim()

        mx = max(
            np.amax(np.real(spectrum)),
            np.abs(np.amin(np.real(spectrum))),
        )
        self.pivotline = self.ax.plot(
            2 * [self.pivot],
            [-10 * mx, 10 * mx],
            color="r",
        )[0]

        self.ax.set_ylim(ylim)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(
            row=0,
            column=0,
            padx=10,
            pady=10,
            sticky="nsew",
        )

        self.toolbar = NavigationToolbar2Tk(
            self.canvas,
            self,
            pack_toolbar=False,
        )
        self.toolbar.grid(row=1, column=0, pady=(0, 10), sticky="w")

        self.scale_frame = tk.Frame(self)
        self.scale_frame.grid(
            row=2,
            column=0,
            padx=10,
            pady=(0, 10),
            sticky="nsew",
        )
        self.scale_frame.columnconfigure(1, weight=1)
        self.scale_frame.rowconfigure(0, weight=1)

        items = [
            (self.pivot, self.p0, self.p1),
            ("pivot", "p0", "p1"),
            (0, -np.pi, -max_p1[0]),
            (self.n, np.pi, max_p1[0]),
        ]

        for i, (init, name, mn, mx) in enumerate(zip(*items)):
            lab = tk.Label(self.scale_frame, text=name)
            pady = (0, 10) if i != 2 else 0
            lab.grid(row=i, column=0, sticky="w", padx=(0, 5), pady=pady)

            self.__dict__[f"{name}_scale"] = scale = tk.Scale(
                self.scale_frame,
                from_=mn,
                to=mx,
                resolution=0.001,
                orient=tk.HORIZONTAL,
                showvalue=0,
                sliderlength=15,
                bd=0,
                highlightthickness=1,
                highlightbackground="black",
                relief="flat",
                command=lambda value, name=name: self.update_phase(name),
            )
            scale.set(init)
            scale.grid(row=i, column=1, sticky="ew", pady=pady)

            self.__dict__[f"{name}_label"] = label = tk.Label(
                self.scale_frame,
                text=f"{self.__dict__[f'{name}']:.3f}"
                if i != 0
                else str(self.__dict__[f"{name}"]),
            )
            label.grid(row=i, column=2, padx=5, pady=pady, sticky="w")

        self.button_frame = tk.Frame(self)
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.save_button = tk.Button(
            self.button_frame,
            width=8,
            highlightbackground="black",
            text="Save",
            bg="#77dd77",
            command=self.save,
        )
        self.save_button.grid(row=1, column=0, sticky="e")

        self.cancel_button = tk.Button(
            self.button_frame,
            width=8,
            highlightbackground="black",
            text="Cancel",
            bg="#ff5566",
            command=self.cancel,
        )
        self.cancel_button.grid(row=1, column=1, sticky="e", padx=(10, 0))

    def update_phase(self, name: str) -> None:
        """Command run whenever a parameter is altered.

        Parameters
        ----------
        name
            Name of quantity that was adjusted. One of ``'p0'``, ``'p1'``,
            and ``'pivot'``.
        """
        value = self.__dict__[f"{name}_scale"].get()

        if name == "pivot":
            self.pivot = int(value)
            self.pivot_label["text"] = str(self.pivot)
            self.pivotline.set_xdata([self.pivot, self.pivot])

        else:
            self.__dict__[name] = float(value)
            self.__dict__[f"{name}_label"]["text"] = f"{self.__dict__[name]:.3f}"

        spectrum = phase(
            self.init_spectrum,
            [self.p0],
            [self.p1],
            [self.pivot],
        )
        self.specline.set_ydata(np.real(spectrum))
        self.canvas.draw_idle()

    def save(self) -> None:
        """Kill the application and update p0 based on pivot and p1."""
        self.p0 = self.p0 - self.p1 * (self.pivot / self.n)
        self.destroy()

    def cancel(self) -> None:
        """Kill the application and set phases to None."""
        self.p0 = None
        self.p1 = None
        self.destroy()


def _make_noise(fid: np.ndarray, snr: float, decibels: bool = True) -> np.ndarray:
    r"""Generate an array of white Guassian complex noise.

    The noise will be created with zero mean and a variance that abides by
    the desired SNR, in accordance with the FID provided.

    Parameters
    ----------
    fid
        Noiseless FID.

    snr
        The signal-to-noise ratio.

    decibels
        If `True`, the snr is taken to be in units of decibels. If `False`,
        it is taken to be simply the ratio of the singal power and noise
        power.

    Returns
    -------
    noise

    Notes
    -----
    Noise variance is given by:

    .. math::

       \rho = \frac{\sum_{n=0}^{N-1} \left(x_n - \mu_x\right)^2}
       {N \cdot 20 \log_10 \left(\mathrm{SNR}_{\mathrm{dB}}\right)}
    """
    checker = ArgumentChecker()
    checker.stage(
        (fid, "fid", "ndarray"), (snr, "snr", "float"), (decibels, "decibels", "bool")
    )
    checker.check()

    shape = fid.shape

    # Compute the variance of the noise
    if decibels:
        snr = 10 ** (snr / 20)

    std = np.std(np.abs(fid)) / snr

    # Make a number of noise instances and check which two are closest
    # to the desired stdev.
    # These two are then taken as the real and imaginary noise components
    instances = []
    std_discrepancies = []
    for _ in range(100):
        instance = nrandom.normal(loc=0, scale=std, size=shape)
        instances.append(instance)
        std_discrepancies.append(np.std(np.abs(instance)) - std)

    # Determine which instance's stdev is the closest to the desired
    # variance
    first, second, *_ = np.argpartition(std_discrepancies, 1)

    # The noise is constructed from the two closest arrays
    # to the desired SNR
    return instances[first] + 1j * instances[second]


def _generate_random_signal(
    m: int, expinfo: ExpInfo, *, snr: Union[float, None] = None
) -> Tuple[np.ndarray, Iterable[np.ndarray], np.ndarray]:
    """Convienince function to generate a random synthetic FID.

    Parameters
    ----------
    m
        Number of oscillators.

    expinfo
        Information on the experiment. Used to determine the number of points,
        sweep width, and transmitter offset.

    snr
        Signal-to-noise ratio (dB).

    Returns
    -------
    fid
        The synthetic FID.

    tp
        The time points the FID is sampled at in each dimension.

    parameters
        Parameters used to construct the signal
    """
    try:
        pts, sw, offset, dim = expinfo.unpack("pts", "sw", "offset", "dim")
    except Exception:
        raise TypeError(
            f"{RED}Check `expinfo` is an instance of " f"nmrespy.ExpInfo{END}"
        )

    checker = ArgumentChecker(dim=dim)
    checker.stage(
        (m, "m", "positive_int"),
        (pts, "pts", "int_iter"),
        (offset, "offset", "float_iter"),
        (sw, "sw", "float_iter"),
        (snr, "snr", "positive_float", True),
    )

    # low: 0.0, high: 1.0
    # amplitdues
    para = nrandom.uniform(size=m)
    # phases
    para = np.hstack((para, nrandom.uniform(low=-np.pi, high=np.pi, size=m)))
    # frequencies
    f = [
        nrandom.uniform(low=-s / 2 + o, high=s / 2 + o, size=m)
        for s, o in zip(sw, offset)
    ]
    para = np.hstack((para, *f))
    # damping
    eta = [nrandom.uniform(low=0.1, high=0.3, size=m) for _ in range(dim)]
    para = np.hstack((para, *eta))
    para = para.reshape((m, 2 * (dim + 1)), order="F")

    return (*make_fid(para, expinfo, snr=snr), para)


def oscillator_integral(
    params: np.ndarray, expinfo: ExpInfo, *, abs_: bool = True
) -> float:
    """Determine the integral of the FT of an oscillator.

    Parameters
    ----------
    params
        Oscillator parameters of the following form:

        * **1-dimensional data:**

          .. code:: python

             parameters = numpy.array([a, φ, f, η])

        * **2-dimensional data:**

          .. code:: python

             parameters = numpy.array([a, φ, f1, f2, η1, η2])

    expinfo
        Information on the experiment. Used to determine the number of points,
        sweep width, and transmitter offset.

    abs_
        Whether or not to take the absolute value of the spectrum before
        integrating.

    Returns
    -------
    integral: float
        Oscillator integral.

    Notes
    -----
    The integration is performed using the composite Simpsons rule, provided
    by `scipy.integrate.simps <https://docs.scipy.org/doc/scipy-1.5.4/\
    reference/generated/scipy.integrate.simps.html>`_

    Spacing of points along the frequency axes is set a `1` (i.e. `dx = 1`).
    """
    # integral is the spectrum initally. It is mutated and converted
    # into the integral during the for loop.
    integral = np.real(ft(make_fid(np.expand_dims(params, axis=0), expinfo)[0]))
    integral = np.absolute(integral) if abs_ else integral

    for axis in reversed(range(integral.ndim)):
        integral = integrate.simps(integral, axis=axis)

    return integral
