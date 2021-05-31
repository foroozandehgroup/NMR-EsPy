# sig.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Constructing and processing NMR signals"""

import copy
import tkinter as tk

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
import numpy.random as nrandom
import scipy.integrate as integrate

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
    colorama.init()
import nmrespy._errors as errors
from nmrespy._misc import ArgumentChecker

"""Provides functionality for constructing synthetic FIDs"""


def make_fid(parameters, n, sw, offset=None, snr=None, decibels=True,
             modulation='none'):
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

    modulation : {'none', 'amp', 'phase'}, default: 'none'
        The type of modulation present in the indirect dimension, if the data
        is 2D. `In the expressions below, a it is assumed a single oscillator
        has been provided for simplicity`.

        * `'none'`: Returns a single signal of the form:

          .. math::

             y(t_1, t_2) = a \\exp(\\mathrm{i} \\phi)
             \\exp \\left[ \\left( 2 \\pi \\mathrm{i} f_1 - \\eta_1 \\right)
             t_1 \\right]
             \\exp \\left[ \\left( 2 \\pi \\mathrm{i} f_2 - \\eta_2 \\right)
             t_2 \\right]

        * `'amp'`: Returns an amplitude-modulated pair of signals of the form:

          .. math::

             y_{\\mathrm{cos}}(t_1, t_2) = a \\exp(\\mathrm{i} \\phi)
             \\cos \\left( 2 \\pi f_1 t_1 \\right)
             \\exp \\left( - \\eta_1 t_1 \\right)
             \\exp \\left[ \\left( 2 \\pi \\mathrm{i} f_2 - \\eta_2 \\right)
             t_2 \\right]

             y_{\\mathrm{sin}}(t_1, t_2) = a \\exp(\\mathrm{i} \\phi)
             \\sin \\left( 2 \\pi f_1 t_1 \\right)
             \\exp \\left( - \\eta_1 t_1 \\right)
             \\exp \\left[ \\left( 2 \\pi \\mathrm{i} f_2 - \\eta_2 \\right)
             t_2 \\right]

        * `'phase'`: Returns an phase-modulated pair of signals of the form:

          .. math::

             y_{\\mathrm{P}}(t_1, t_2) = a \\exp(\\mathrm{i} \\phi)
             \\exp \\left[ \\left( 2 \\pi \\mathrm{i} f_1 - \\eta_1 \\right)
             t_1 \\right]
             \\exp \\left[ \\left( 2 \\pi \\mathrm{i} f_2 - \\eta_2 \\right)
             t_2 \\right]

             y_{\\mathrm{N}}(t_1, t_2) = a \\exp(\\mathrm{i} \\phi)
             \\exp \\left[ \\left( - 2 \\pi \\mathrm{i} f_1 - \\eta_1 \\right)
             t_1 \\right]
             \\exp \\left[ \\left( 2 \\pi \\mathrm{i} f_2 - \\eta_2 \\right)
             t_2 \\right]

    Returns
    -------
    fid : numpy.ndarray, [numpy.ndarray, numpy.ndarray]
        The synthetic signal generated.

        + If the data to be constructed is 1D or 2D with `modulation` set to
          `'none'`, the result will be a NumPy array.
        + If the data is 2D with `modulation` set to `'amp'`, or `'phase'`
          the result will be a length-2 list with signals of the forms
          indicated above (See `modulation`).

    tp : [numpy.ndarray], [numpy.ndarray, numpy.ndarray]
        The time points the FID is sampled at in each dimension.

    Notes
    -----
    The resulting `fid` is given by

    .. math::

       y[n_1, \\cdots, n_D] =
       \\sum_{m=1}^{M} a_m \\exp\\left(\\mathrm{i} \\phi_m\\right)
       \\prod_{d=1}^{D}
       \\exp\\left[\\left(2 \\pi \\mathrm{i} f_m - \\eta_m\\right)
       n_d \\Delta t_d\\right]

    where :math:`d` is either 1 or 2, :math:`M` is the number of
    oscillators, and :math:`\\Delta t_d = 1 / f_{\\mathrm{sw}, d}`.
    """

    # --- Check validity of inputs ---------------------------------------
    try:
        dim = len(n)
    except Exception:
        raise TypeError(f'{cols.R}n should be iterable.{cols.END}')

    if offset is None:
        offset = [0.0] * dim

    components = [
        (parameters, 'parameters', 'parameter'),
        (n, 'n', 'int_list'),
        (sw, 'sw', 'float_list'),
        (offset, 'offset', 'float_list'),
        (decibels, 'decibels', 'bool'),
        (modulation, 'modulation', 'modulation'),
    ]

    if snr is not None:
        components.append((snr, 'snr', 'float'))

    ArgumentChecker(components, dim)

    # --- Extract amplitudes, phases, frequencies and damping ------------
    amp = parameters[:, 0]
    phase = parameters[:, 1]
    # Center frequencies at 0 based on offset
    freq = [parameters[:, 2 + i] - offset[i] for i in range(dim)]
    damp = [parameters[:, dim + 2 + i] for i in range(dim)]

    # Time points in each dimension
    tp = get_timepoints(n, sw)

    # --- Generate noiseless FID -----------------------------------------
    if dim == 1:
        # Vandermonde matrix of poles
        Z = np.exp(np.outer(tp[0], (1j * 2 * np.pi * freq[0] - damp[0])))
        # Vector of complex ampltiudes
        alpha = amp * np.exp(1j * phase)
        # Compute FID!
        fid = Z @ alpha

    if dim == 2:
        if modulation in ['none', 'amp']:
            Z1 = np.exp(np.outer(tp[0], (1j * 2 * np.pi * freq[0] - damp[0])))
            if modulation == 'amp':
                Z1 = [np.real(Z1), np.imag(Z1)]
        elif modulation == 'phase':
            Z1 = [
                np.exp(np.outer(tp[0], (1j * 2 * np.pi * freq[0] - damp[0]))),
                np.exp(np.outer(tp[0], (-1j * 2 * np.pi * freq[0] - damp[0])))
            ]

        Z2t = np.exp(np.outer((1j * 2 * np.pi * freq[1] - damp[1]), tp[1]))
        # Diagonal matrix of complex amplitudes
        A = np.diag(amp * np.exp(1j * phase))

        fid = []
        for z1 in Z1:
            fid.append(z1 @ A @ Z2t)

        if len(fid) == 1:
            fid = fid[0]

    # --- Add noise to FID -----------------------------------------------
    if snr is None:
        return fid, tp
    else:
        if isinstance(fid, np.ndarray):
            return fid + make_noise(fid, snr, decibels), tp
        elif isinstance(fid, list):
            for i, f in enumerate(fid):
                fid[i] = f + make_noise(f, snr, decibels)
            return fid, tp


def make_virtual_echo(data, modulation='amp'):
    """Given the time-domain signal `data`, generates the corresponding
    virtual echo [#]_, a signal with a purely real Fourier-Tranform and
    absorption mode line shape if the data is phased.

    Parameters
    ----------
    data : [numpy.ndarray] or [numpy.ndarray, numpy.ndarray]
        The data to construct the virtual echo from. This should be a list of
        NumPy arrays, with ``len(data) == d`` where ``d`` is the dimension
        of the signal.

    modulation : {'amp' or 'phase'}, default: 'amp'
        If the data is 2D, this parameter specifies the type of modulation
        present in the indirect dimension of the dataset.

        * If set to `'amp'`, the two signals in the `data` should be an
          amplitude modulated pair.
        * If set to `'phase'`, the two signals in the `data` should be a phase
          modulated pair.

        See the docs for :py:func:`make_fid` for more info on `modulation`.

    Returns
    -------
    virtual_echo : numpy.ndarray
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
        raise TypeError(f'{cols.R}n should be iterable.{cols.END}')

    if dim > 2:
        raise errors.MoreThanTwoDimError()

    ArgumentChecker(
        [(data, 'data', 'array_list'),
         (modulation, 'modulation', 'modulation')],
        dim,
    )

    if dim == 1:
        data = data[0]
        pts = data.size
        ve = np.zeros((2 * pts - 1), dtype='complex')
        ve[:pts] = data
        ve[0] = np.real(ve[0])
        ve[pts:] = data.conj()[1:][::-1]
        return ve

    elif dim == 2:
        # TODO NEEDS FIXING
        if modulation == 'amp':
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

        tmp1 = np.zeros(tuple(2 * p - 1 for p in pts), dtype='complex')
        tmp1[:pts[0], :pts[1]] = pp
        tmp1[0] /= 2
        tmp1[:, 0] /= 2

        tmp2 = np.zeros(tuple(2 * p - 1 for p in pts), dtype='complex')
        tmp2[:pts[0], pts[1] - 1:] = pm[:, ::-1]
        tmp2[0] /= 2
        tmp2[:, -1] /= 2
        tmp2 = np.roll(tmp2, 1, axis=1)

        tmp3 = np.zeros(tuple(2 * p - 1 for p in pts), dtype='complex')
        tmp3[pts[0] - 1:, :pts[1]] = mp[::-1]
        tmp3[-1] /= 2
        tmp3[:, 0] /= 2
        tmp3 = np.roll(tmp3, 1, axis=0)

        tmp4 = np.zeros(tuple(2 * p - 1 for p in pts), dtype='complex')
        tmp4[pts[0] - 1:, pts[1] - 1:] = mm[::-1, ::-1]
        tmp4[-1] /= 2
        tmp4[:, -1] /= 2
        tmp4 = np.roll(tmp4, 1, axis=(0, 1))

        return tmp1 + tmp2 + tmp3 + tmp4


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
    except Exception:
        raise TypeError(f'{cols.R}n should be iterable.{cols.END}')

    ArgumentChecker([(n, 'n', 'int_list'), (sw, 'sw', 'float_list')], dim)

    return [np.linspace(0, float(n_ - 1) / sw_, n_) for n_, sw_ in zip(n, sw)]


def get_shifts(n, sw, offset=None, flip=True):
    """Generates the frequencies that the FT of the FID is sampled at, given
    its sweep-width, the transmitter offset, and the number of points.

    Parameters
    ----------
    n : [int] or [int, int]
        The number of points in each dimension.

    sw : [float] or [float, float]
        The sweep width in each dimension.

    offset : [float], [float, float], or None, default: None
        The transmitter offset in each dimension. If `None`, the
        offset will be set to zero in each dimension.

    flip : bool, default: True
        If `True`, the shifts will be returned in descending order, as is
        conventional in NMR. If `False`, the shifts will be in ascending order.

    Returns
    -------
    shifts : [numpy.ndarray] or [numpy.ndarray, numpy.ndarray]
        The chemical shift values sampled in each dimension."""

    try:
        dim = len(n)
    except Exception:
        raise TypeError(f'{cols.R}n should be iterable.{cols.END}')

    if offset is None:
        offset = dim * [0.]

    ArgumentChecker(
        [
            (n, 'n', 'int_list'),
            (sw, 'sw', 'float_list'),
            (offset, 'offset', 'float_list'),
        ],
        dim=dim
    )

    shifts = []
    for n_, sw_, off in zip(n, sw, offset):
        shifts.append(
            np.linspace((-sw_ / 2) + off, (sw_ / 2) + off, n_)
        )

    return np.flip(shifts) if flip else shifts


def ft(fid, flip=True):
    """Performs Fourier transformation and (optionally) flips the resulting
    spectrum to satisfy NMR convention.

    Parameters
    ----------
    fid : numpy.ndarray
        Time-domain data.

    flip : bool, default: True
        Whether or not to flip the Fourier Trnasform of `fid` in each
        dimension.

    Returns
    -------
    spectrum : numpy.ndarray
        Fourier transform of the data, flipped in each dimension.
    """

    ArgumentChecker([(fid, 'fid', 'ndarray'), (flip, 'flip', 'bool')])

    spectrum = fft(fid, axis=0)

    for axis in range(1, fid.ndim):
        spectrum = fft(spectrum, axis=axis)

    return np.flip(fftshift(spectrum)) if flip else fftshift(spectrum)


def proc_amp_modulated(data):
    """Takes a pair of 2D amplitude-modulated signals, and generates the
    frequency-discriminated spectrum.

    Parameters
    ----------
    data : [numpy.ndarray, numpy.ndarray]
        cos-modulated signal and sin-modulated signal

    Returns
    -------
    spectrum : dict
        Dictionary of four elements: ``rr``, ``ri``, ``ir``, and ``ii``."""

    ArgumentChecker([(data, 'data', 'array_list')], dim=2)

    c = data[0]
    s = data[1]

    c_t1_f2 = fftshift(fft(c, axis=1), axes=1)
    s_t1_f2 = fftshift(fft(s, axis=1), axes=1)

    return fftshift(
        fft(np.real(c_t1_f2) + 1j * np.real(s_t1_f2), axis=0), axes=0
    )


def proc_phase_modulated(data):
    """Takes a pair of 2D phase-modulated signals, and generates the
    set of spectra corresponding to the processing protocol outlined in [#]_.

    Parameters
    ----------
    data : [numpy.ndarray, numpy.ndarray]
        P-type signal and N-type signal

    Returns
    -------
    spectra : dict
        Dictionary of four elements: ``rr``, ``ri``, ``ir``, and ``ii``.

    References
    ----------
    .. [#] A. L. Davis, J. Keeler, E. D. Laue, and D. Moskau, “Experiments for
           recording pure-absorption heteronuclear correlation spectra using
           pulsed field gradients,” Journal of Magnetic Resonance (1969),
           vol. 98, no. 1, pp. 207–216, 1992."""

    ArgumentChecker([(data, 'data', 'array_list')], dim=2)

    p = data[0]
    n = data[1]

    p_t1_f2 = fftshift(fft(p, axis=1), axes=1)  # Sₚ(t₁,f₂)
    n_t1_f2 = fftshift(fft(n, axis=1), axes=1)  # Sₙ(t₁,f₂)

    # Generating rr and ir
    plus_t1_f2 = 0.5 * (p_t1_f2 + n_t1_f2.conj())           # S⁺(t₁,f₂)
    plus_f1_f2 = fftshift(fft(plus_t1_f2, axis=0), axes=0)  # S⁺(f₁,f₂)
    rr = np.real(plus_f1_f2)  # A₁A₂
    ir = np.imag(plus_f1_f2)  # D₁A₂

    # Generating ri and ii
    minus_t1_f2 = - 0.5 * 1j * (p_t1_f2 - n_t1_f2.conj())     # S⁻(t₁,f₂)
    minus_f1_f2 = fftshift(fft(minus_t1_f2, axis=0), axes=0)  # S⁻(f₁,f₂)
    ri = np.real(minus_f1_f2)  # A₁D₂
    ii = np.imag(minus_f1_f2)  # D₁D₂

    return {
        'rr': rr,
        'ri': ri,
        'ir': ir,
        'ii': ii,
    }


def ift(spectrum, flip=True):
    """Flips spectral data in each dimension, and then inverse Fourier
    transforms.

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

    ArgumentChecker(
        [
            (spectrum, 'spectrum', 'ndarray'),
            (flip, 'flip', 'bool'),
        ],
    )

    spectrum = ifftshift(np.flip(spectrum)) if flip else ifftshift(spectrum)

    fid = ifft(spectrum, axis=0)

    for axis in range(1, spectrum.ndim):
        fid = ifft(fid, axis=axis)

    return fid


def phase(data, p0, p1, pivot=None):
    """Applies a linear phase correction to `data`.

    Parameters
    ----------
    data : numpy.ndarray
        Data to be phased.

    p0 : [float] or [float, float]
        Zero-order phase correction in each dimension, in radians.

    p1 : [float] or [float, float]
        First-order phase correction in each dimension, in radians.

    pivot : [int], [int, int] or None
        Index of the pivot in each dimension. If None, the pivot will be `0`
        in each dimension.

    Returns
    -------
    phased_data : numpy.ndarray
    """

    try:
        dim = len(p0)
    except Exception:
        raise TypeError(f'{cols.R}p0 should be iterable.{cols.END}')

    if pivot is None:
        pivot = [0] * dim

    components = [
        (data, 'data', 'ndarray'),
        (p0, 'p0', 'float_list'),
        (p1, 'p1', 'float_list'),
        (pivot, 'pivot', 'int_list')
    ]

    ArgumentChecker(components, dim)

    # Indices for einsum
    # For 1D: 'i'
    # For 2D: 'ij'
    idx = ''.join([chr(i + 105) for i in range(dim)])

    for axis, (piv, p0_, p1_) in enumerate(zip(pivot, p0, p1)):
        n = data.shape[axis]
        # Determine axis for einsum (i or j)
        axis = chr(axis + 105)
        p = np.exp(1j * (p0_ + p1_ * np.arange(-piv, -piv + n) / n))
        phased_data = np.einsum(f'{idx},{axis}->{idx}', data, p)

    return phased_data


def manual_phase_spectrum(spectrum, max_p1=None):
    """Generates a GUI, enabling manual phase correction, with the zero- and
    first-order phases returned.

    .. warning::
       Only 1D spectral data is currently supported.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectral data of interest.

    max_p1 : float or None, default: None
        Specifies the range of first-order phases permitted. For each
        dimension, the user will be allowed to choose a value of `p1`
        within [`-max_p1`, `max_p1`]. By default, `max_p1` will be
        ``10 * numpy.pi``.

    Returns
    -------
    p0 : [float] or None
        Zero-order phase correction in each dimension, in radians. If the
        user chooses to cancel rather than save, this is set to `None`.

    p1 : [float] or None
        First-order phase correction in each dimension, in radians. If the
        user chooses to cancel rather than save, this is set to `None`.
    """
    try:
        dim = spectrum.ndim
    except Exception:
        raise TypeError(f'{cols.R}soectrum should be a numpy array{cols.END}')

    # Only valid for 1D so far
    if dim != 1:
        raise errors.TwoDimUnsupportedError()

    init_spectrum = copy.deepcopy(spectrum)

    if max_p1 is None:
        max_p1 = 10 * np.pi
    app = PhaseApp(init_spectrum, max_p1)
    app.mainloop()

    return app.p0, app.p1


class PhaseApp(tk.Tk):
    """Tkinter application for manual phase correction.

    See Also
    --------
    :py:func:`manual_phase_spectrum`
    """

    def __init__(self, spectrum, max_p1):
        super().__init__()
        self.p0 = 0.0
        self.p1 = 0.0
        self.n = spectrum.size
        self.pivot = int(self.n // 2)
        self.init_spectrum = copy.deepcopy(spectrum)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.fig = Figure(figsize=(6, 4), dpi=160)
        # Set colour of figure frame
        color = self.cget('bg')
        self.fig.patch.set_facecolor(color)
        self.ax = self.fig.add_axes([0.03, 0.1, 0.94, 0.87])
        self.ax.set_yticks([])
        self.specline = self.ax.plot(np.real(spectrum), color='k')[0]

        ylim = self.ax.get_ylim()

        mx = max(
            np.amax(np.real(spectrum)),
            np.abs(np.amin(np.real(spectrum))),
        )
        self.pivotline = self.ax.plot(
            2 * [self.pivot], [-10 * mx, 10 * mx], color='r',
        )[0]

        self.ax.set_ylim(ylim)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(
            row=0, column=0, padx=10, pady=10, sticky='nsew',
        )

        self.toolbar = NavigationToolbar2Tk(
            self.canvas, self, pack_toolbar=False,
        )
        self.toolbar.grid(row=1, column=0, pady=(0, 10), sticky='w')
        self.scale_frame = tk.Frame(self)
        self.scale_frame.grid(
            row=2, column=0, padx=10, pady=(0, 10), sticky='nsew',
        )
        self.scale_frame.columnconfigure(1, weight=1)
        self.scale_frame.rowconfigure(0, weight=1)

        items = [
            (self.pivot, self.p0, self.p1),
            ('pivot', 'p0', 'p1'),
            (0, -np.pi, -max_p1),
            (self.n, np.pi, max_p1),
        ]

        for i, (init, name, mn, mx) in enumerate(zip(*items)):
            lab = tk.Label(self.scale_frame, text=name)
            pady = (0, 10) if i != 2 else 0
            lab.grid(row=i, column=0, sticky='w', padx=(0, 5), pady=pady)

            self.__dict__[f'{name}_scale'] = scale = tk.Scale(
                self.scale_frame,
                from_=mn,
                to=mx,
                resolution=0.001,
                orient=tk.HORIZONTAL,
                showvalue=0,
                sliderlength=15,
                bd=0,
                highlightthickness=1,
                highlightbackground='black',
                relief='flat',
                command=lambda value, name=name: self.update_phase(name),
            )
            scale.set(init)
            scale.grid(row=i, column=1, sticky='ew', pady=pady)

            self.__dict__[f'{name}_label'] = label = tk.Label(
                self.scale_frame,
                text=f"{self.__dict__[f'{name}']:.3f}"
                     if i != 0 else str(self.__dict__[f'{name}'])
            )
            label.grid(row=i, column=2, padx=5, pady=pady, sticky='w')

        self.button_frame = tk.Frame(self)
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.grid(row=3, column=0, padx=10, pady=10, sticky='ew')

        self.save_button = tk.Button(
            self.button_frame,
            width=8,
            highlightbackground='black',
            text='Save',
            bg='#77dd77',
            command=self.save
        )
        self.save_button.grid(row=1, column=0, sticky='e')

        self.cancel_button = tk.Button(
            self.button_frame,
            width=8,
            highlightbackground='black',
            text='Cancel',
            bg='#ff5566',
            command=self.cancel
        )
        self.cancel_button.grid(row=1, column=1, sticky='e', padx=(10, 0))

    def update_phase(self, name):
        value = self.__dict__[f'{name}_scale'].get()

        if name == 'pivot':
            self.pivot = int(value)
            self.pivot_label['text'] = str(self.pivot)
            self.pivotline.set_xdata([self.pivot, self.pivot])

        else:
            self.__dict__[name] = float(value)
            self.__dict__[f'{name}_label']['text'] = \
                f"{self.__dict__[name]:.3f}"

        spectrum = phase(
            self.init_spectrum, [self.p0], [self.p1], [self.pivot],
        )
        self.specline.set_ydata(np.real(spectrum))
        self.canvas.draw_idle()

    def save(self):
        self.p0 = self.p0 - self.p1 * (self.pivot / self.n)
        self.destroy()

    def cancel(self):
        self.destroy()
        self.p0 = None
        self.p1 = None


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


def generate_random_signal(m, n, sw, offset=None, snr=None):
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

    offset : [float], [float, float] or None, deafult: None
        Transmitter offset in each dimension

    snr : float or None, default: None
        Signal-to-noise ratio (dB)

    fid : numpy.ndarray
        The synthetic FID.

    tp : [numpy.ndarray], [numpy.ndarray, numpy.ndarray]
        The time points the FID is sampled at in each dimension.

    parameters : numpy.ndarray
        Parameters used to construct the signal
    """

    try:
        dim = len(n)
    except Exception:
        raise TypeError(f'{cols.R}n should be an iterable{cols.END}')

    if offset is None:
        offset = [0.0] * dim

    components = [
        (m, 'm', 'positive_int'),
        (n, 'n', 'int_list'),
        (sw, 'sw', 'float_list'),
    ]

    if snr is not None:
        components.append((snr, 'snr', 'positive_float'))

    ArgumentChecker(components, dim)

    # low: 0.0, high: 1.0
    # amplitdues
    para = nrandom.uniform(size=m)
    # phases
    para = np.hstack((para, nrandom.uniform(low=-np.pi, high=np.pi, size=m)))
    # frequencies
    f = [nrandom.uniform(low=-s / 2 + o, high=s / 2 + o, size=m)
         for s, o in zip(sw, offset)]
    para = np.hstack((para, *f))
    # damping
    eta = [nrandom.uniform(low=0.1, high=0.3, size=m) for _ in range(dim)]
    para = np.hstack((para, *eta))
    para = para.reshape((m, 2 * (dim + 1)), order='F')

    return *make_fid(para, n, sw, offset, snr), para


def oscillator_integral(parameters, n, sw, offset=None):
    """Determines the (absolute) integral of the Fourier transform of
    an oscillator.

    Parameters
    ----------
    parameters : numpy.ndarray
        Oscillator parameters of the following form:

        * **1-dimensional data:**

          .. code:: python

             parameters = numpy.array([a, φ, f, η])

        * **2-dimensional data:**

          .. code:: python

             parameters = numpy.array([a, φ, f1, f2, η1, η2])

    n : [int], [int, int]
        Number of points to construct signal from in each dimension.

    sw : [float], [float, float]
        Sweep width in each dimension, in Hz.

    offset : [float], [float, float], or None, default: None
        Transmitter offset frequency in each dimension, in Hz. If set to
        `None`, the offset frequency will be set to 0Hz in each dimension.

    Returns
    -------
    integral :

    Notes
    -----
    The integration is performed using the composite Simpsons rule, provided
    by `scipy.integrate.simpson <https://docs.scipy.org/doc/scipy/reference/\
    generated/scipy.integrate.simpson.html#scipy.integrate.simpson>`_

    Spacing of points along the frequency axes is set a `1` (i.e. `dx = 1`).
    """

    fid, _ = make_fid(np.expand_dims(parameters, axis=0), n, sw, offset)
    spectrum = np.absolute(ft(fid))

    for axis in range(spectrum.ndim):
        try:
            integral = integrate.simpson(integral, axis=axis)
        except NameError:
            integral = integrate.simpson(spectrum, axis=axis)

    return integral
