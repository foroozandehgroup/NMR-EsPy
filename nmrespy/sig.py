# sig.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sun 15 May 2022 12:05:04 BST

"""Manipulating and processing NMR signals."""

import copy
import re
import tkinter as tk
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
import numpy.random as nrandom

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from nmrespy._sanity import sanity_check, funcs as sfuncs


def make_virtual_echo(
    data: np.ndarray,
    twodim_dtype: Optional[str] = None,
) -> np.ndarray:
    """Generate a virtual echo [#]_ from a time-domain signal.

    A vitrual echo is a signal with a purely real Fourier-Tranform and
    absorption mode line shape if the data is phased.

    Parameters
    ----------
    data
        The data to construct the virtual echo from. If the data comprises a pair
        of amplitude/phase modulated signals, these should be stored in a single
        3D array with ``shape[2] == 2``, such that ``data[:, :, 0]`` if the cos/p
        signal, and ``data[:, :, 1]`` is the sin/n signal.

    twodim_dtype
        If the data is 2D, this parameter specifies the way to process the data.
        Allowed options are:

        * ``"jres"``: The data should be derived from a J-Resolved (2DJ) experiment.
        * ``"amp"``: the two signals in axis 2 of ``data`` should be an
          amplitude modulated pair.
        * ``"phase"``: the two signals in axis 2 of ``data`` should be a phase
          modulated pair.

    References
    ----------
    .. [#] M. Mayzel, K. Kazimierczuk, V. Y. Orekhov, The causality principle
           in the reconstruction of sparse nmr spectra, Chem. Commun. 50 (64)
           (2014) 8947–8950.
    """
    sanity_check(("data", data, sfuncs.check_ndarray))
    if data.ndim == 2:
        sanity_check(
            ("twodim_dtype", twodim_dtype, sfuncs.check_one_of, ("jres",))
        )
    elif data.ndim == 3:
        sanity_check(
            ("twodim_dtype", twodim_dtype, sfuncs.check_one_of, ("amp", "phase"))
        )

    if data.ndim == 1:
        pts = data.size
        ve = np.zeros((2 * pts - 1), dtype="complex")
        ve[0] = np.real(data[0])
        ve[1:pts] = data[1:]
        ve[pts:] = data[1:][::-1].conj()
        return ve

    if twodim_dtype == "jres":
        pts = data.shape
        ve = np.zeros((pts[0], 2 * pts[1] - 1), dtype="complex")
        ve[:, 0] = np.real(data[:, 0])
        ve[:, 1 : pts[1]] = data[:, 1:]
        ve[:, pts[1]:] = data[:, 1:][:, ::-1].conj()
        return ve

    if twodim_dtype == "amp":
        # TODO NEEDS FIXING
        cos = data[:, :, 0]
        sin = data[:, :, 1]

    elif twodim_dtype == "phase":
        cos = 0.5 * (data[:, :, 0] + data[:, :, 1])
        sin = -1j * 0.5 * (data[:, :, 0] - data[:, :, 1])

    # S±± = (R₁ ± iI₁)(R₂ ± iI₂)
    # where: Re(cos) -> R₁R₂, Im(cos) -> R₁I₂, Re(sin) -> I₁R₂, Im(sin) -> I₁I₂
    r1r2 = np.real(cos)
    r1i2 = np.imag(cos)
    i1r2 = np.real(sin)
    i1i2 = np.imag(sin)

    # S++ = R₁R₂ - I₁I₂ + i(R₁I₂ + I₁R₂)
    pp = r1r2 - i1i2 + 1j * (r1i2 + i1r2)
    # S+- = R₁R₂ + I₁I₂ + i(I₁R₂ - R₁I₂)
    pm = r1r2 + i1i2 + 1j * (i1r2 - r1i2)
    # S-+ = R₁R₂ + I₁I₂ + i(R₁I₂ - I₁R₂)
    mp = r1r2 + i1i2 + 1j * (r1i2 - i1r2)
    # S-- = R₁R₂ - I₁I₂ - i(R₁I₂ + I₁R₂)
    mm = r1r2 - i1i2 - 1j * (r1i2 + i1r2)

    pts = data.shape[:2]

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
    """
    zf_data = copy.deepcopy(data)
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


def ft(
    fid: np.ndarray,
    axes: Optional[Union[Iterable[int], int]] = None,
    flip: bool = True,
) -> np.ndarray:
    """Fourier transformation with optional spectrum flipping.

    It is conventional in NMR to plot spectra from high to low going
    left to right/down to up. This function utilises the
    `numpy.fft <https://numpy.org/doc/stable/reference/routines.fft.html>`_
    module to carry out the Fourier Transformation.

    Parameters
    ----------
    fid
        Time-domain data.

    axes
        The axes to apply Fourier Transformation to. By default (``None``), FT is
        applied to all axes. If an int, FT will only be applied to the relevant axis.
        If a list of ints, FT will be applied to this subset of axes.

    flip
        Whether or not to flip the Fourier Transform of `fid` in each
        dimension.
    """
    sanity_check(
        ("fid", fid, sfuncs.check_ndarray),
        ("flip", flip, sfuncs.check_bool),
    )
    dim = fid.ndim
    sanity_check(
        (
            "axes", axes, sfuncs.check_int_list, (),
            {
                "len_one_can_be_listless": True,
                "must_be_positive": True,
                "max_value": dim - 1,
            },
            True,
        )
    )

    if axes is None:
        axes = list(range(dim))
    if isinstance(axes, int):
        axes = [axes]

    spectrum = copy.deepcopy(fid)
    for axis in axes:
        spectrum = fftshift(fft(spectrum, axis=axis), axes=axis)

    if flip:
        spectrum = np.flip(spectrum, axis=axes)

    return spectrum


def ift(
    spectrum: np.ndarray,
    axes: Optional[Union[Iterable[int], int]] = None,
    flip: bool = True
) -> np.ndarray:
    """Inverse Fourier Transform a spectrum.

    This function utilises the
    `numpy.fft <https://numpy.org/doc/stable/reference/routines.fft.html>`_
    module to carry out the Fourier Transformation.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum

    axes
        The axes to apply IFT to. By default (``None``), IFT is
        applied to all axes. If an int, IFT will only be applied to the relevant axis.
        If a list of ints, IFT will be applied to this subset of axes.

    flip : bool, default: True
        Whether or not to flip ``spectrum`` in each dimension prior to Inverse
        Fourier Transform.
    """
    sanity_check(
        ("spectrum", spectrum, sfuncs.check_ndarray),
        ("flip", flip, sfuncs.check_bool),
    )
    dim = spectrum.ndim
    sanity_check(
        (
            "axes", axes, sfuncs.check_int_list, (),
            {
                "len_one_can_be_listless": True,
                "must_be_positive": True,
                "max_value": dim - 1,
            },
            True,
        )
    )
    if axes is None:
        axes = list(range(dim))
    if isinstance(axes, int):
        axes = [axes]

    fid = copy.deepcopy(spectrum)
    fid = np.flip(fid, axis=axes) if flip else fid
    for axis in axes:
        fid = ifft(ifftshift(fid, axes=axis), axis=axis)

    return fid


def proc_amp_modulated(data: np.ndarray) -> np.ndarray:
    """Generate a frequency-discrimiated signal from amplitude-modulated 2D FIDs.

    Parameters
    ----------
    data
        cos-modulated signal and sin-modulated signal, stored in a 3D numpy array,
        such that ``data[0]`` is the the cos signal and ``data[1]`` is the sin
        signal.

    Returns
    -------
    spectrum: np.ndarray
        Frequency-discrimiated spectrum.
    """
    sanity_check(
        ("data", data, sfuncs.check_ndarray, (), {"dim": 3, "shape": [(0, 2)]}),
    )
    cos_t1_f2, sin_t1_f2 = [ft(x, axes=1).real for x in (data[0], data[1])]
    return ft(cos_t1_f2 + 1j * sin_t1_f2, axes=0)


def proc_phase_modulated(data: np.ndarray) -> np.ndarray:
    """Process phase modulated 2D FIDs.

    This function generates the set of spectra corresponding to the
    processing protocol outlined in [#]_.

    Parameters
    ----------
    data
        P-type signal and N-type signal, stored in a 3D numpy array, such that
        ``data[0]`` is the the P signal and ``data[1]`` is the N signal.

    Returns
    -------
    spectra
        3D array with ``spectra.shape[0] == 4``. The sub-arrays in axis 0 correspond
        to the following signals:

        * ``spectra[0]``: RR
        * ``spectra[1]``: RI
        * ``spectra[2]``: IR
        * ``spectra[3]``: II

    References
    ----------
    .. [#] A. L. Davis, J. Keeler, E. D. Laue, and D. Moskau, “Experiments for
           recording pure-absorption heteronuclear correlation spectra using
           pulsed field gradients,” Journal of Magnetic Resonance (1969),
           vol. 98, no. 1, pp. 207–216, 1992.
    """
    sanity_check(
        ("data", data, sfuncs.check_ndarray, (), {"dim": 3, "shape": [(0, 2)]}),
    )
    p_t1_f2, n_t1_f2 = [ft(x, axes=1) for x in (data[0], data[1])]

    spectra = np.zeros((4, *data.shape[1:]))

    # Generating RR and IR
    plus_f1_f2 = ft(0.5 * (p_t1_f2 + n_t1_f2.conj()), axes=0)  # S⁺(f₁,f₂)
    spectra[0], spectra[2] = plus_f1_f2.real, plus_f1_f2.imag

    # Generating RI and II
    minus_f1_f2 = ft(-0.5j * (p_t1_f2 - n_t1_f2.conj()), axes=0)  # S⁻(f₁,f₂)
    spectra[1], spectra[3] = minus_f1_f2.real, minus_f1_f2.imag

    return spectra


def phase(
    data: np.ndarray,
    p0: Iterable[float],
    p1: Iterable[float],
    pivot: Optional[Iterable[Union[float, int]]] = None,
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
        Index of the pivot in each dimension. If ``None``, the pivot will be ``0``
        in each dimension.
    """
    sanity_check(("data", data, sfuncs.check_ndarray))
    dim = data.ndim
    sanity_check(
        ("p0", p0, sfuncs.check_float_list, (), {"length": dim}),
        ("p1", p1, sfuncs.check_float_list, (), {"length": dim}),
    )

    if pivot is None:
        pivot = dim * [0]

    # Indices for einsum... For 1D: 'i', For 2D: 'ij'
    idx = "".join([chr(i + 105) for i in range(dim)])

    for axis, (piv, p0_, p1_) in enumerate(zip(pivot, p0, p1)):
        n = data.shape[axis]
        # Determine axis for einsum (i or j)
        axis = chr(axis + 105)
        p = np.exp(1j * (p0_ + p1_ * np.arange(-piv, -piv + n) / n))
        phased_data = np.einsum(f"{idx},{axis}->{idx}", data, p)

    return phased_data


def manual_phase_data(
    spectrum: np.ndarray,
    max_p1: Optional[Iterable[float]] = None,
) -> Tuple[Optional[Iterable[float]], Optional[Iterable[float]]]:
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
    p0
        Zero-order phase correction in each dimension, in radians. If the
        user chooses to cancel rather than save, this is set to ``None``.

    p1
        First-order phase correction in each dimension, in radians. If the
        user chooses to cancel rather than save, this is set to ``None``.
    """
    sanity_check(("spectrum", spectrum, sfuncs.check_ndarray, (), {"dim": 1}))
    dim = spectrum.ndim
    sanity_check(
        (
            "max_p1", max_p1, sfuncs.check_float_list,
            {
                "length": dim,
                "len_one_can_be_listless": True,
                "must_be_positive": True,
            },
            True,
        )
    )

    if isinstance(max_p1, float):
        max_p1 = (max_p1,)
    if max_p1 is None:
        max_p1 = tuple(dim * [10 * np.pi])

    init_spectrum = copy.deepcopy(spectrum)

    app = PhaseApp(init_spectrum, max_p1)
    app.mainloop()

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
    sanity_check(
        ("fid", fid, sfuncs.check_ndarray),
        ("snr", snr, sfuncs.check_float),
        ("decibels", decibels, sfuncs.check_bool),
    )

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
        instance = nrandom.normal(loc=0, scale=std, size=fid.shape)
        instances.append(instance)
        std_discrepancies.append(np.std(np.abs(instance)) - std)

    # Determine which instance's stdev is the closest to the desired
    # variance
    first, second, *_ = np.argpartition(std_discrepancies, 1)

    # The noise is constructed from the two closest arrays
    # to the desired SNR
    return instances[first] + 1j * instances[second]
