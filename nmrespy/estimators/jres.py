# jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 02 Nov 2022 17:08:29 GMT

from __future__ import annotations
import copy
import io
import os
from pathlib import Path
import re
import tkinter as tk
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)

from nmrespy import MATLAB_AVAILABLE, ExpInfo, sig
from nmrespy.app.custom_widgets import MyEntry
from nmrespy.plot import make_color_cycle
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._files import cd, check_existent_dir
from nmrespy._paths_and_links import SPINACHPATH
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)
from nmrespy.estimators import logger, _Estimator1DProc
from nmrespy.load import load_bruker


if USE_COLORAMA:
    import colorama
    colorama.init()

if MATLAB_AVAILABLE:
    import matlab
    import matlab.engine


class Estimator2DJ(_Estimator1DProc):

    default_mpm_trim = 256
    default_nlp_trim = 1024
    default_max_iterations_exact_hessian = 40
    default_max_iterations_gn_hessian = 80

    @classmethod
    def new_bruker(
        cls,
        directory: Union[str, Path],
        convdta: bool = True,
    ) -> Estimator2DJ:
        """Create a new instance from Bruker-formatted data.

        Parameters
        ----------
        directory
            Absolute path to data directory.

        convdta
            If ``True``, removal of the FID's digital filter will be carried out.

        Notes
        -----
        **Directory Requirements**

        There are certain file paths expected to be found relative to ``directory``
        which contain the data and parameter files:

        * ``directory/ser``
        * ``directory/acqus``
        * ``directory/acqu2s``
        """
        sanity_check(
            ("directory", directory, check_existent_dir),
            ("convdta", convdta, sfuncs.check_bool),
        )

        directory = Path(directory).expanduser()
        data, expinfo = load_bruker(directory)

        if data.ndim != 2:
            raise ValueError(f"{RED}Data dimension should be 2.{END}")

        if directory.parent.name == "pdata":
            raise ValueError(f"{RED}Importing pdata is not permitted.{END}")

        if convdta:
            grpdly = expinfo.parameters["acqus"]["GRPDLY"]
            data = sig.convdta(data, grpdly)

        expinfo._offset = (0., expinfo.offset()[1])
        expinfo._sfo = (None, expinfo.sfo[1])
        expinfo._default_pts = data.shape

        return cls(data, expinfo, directory)

    @classmethod
    def new_spinach(
        cls,
        shifts: Iterable[float],
        couplings: Optional[Iterable[Tuple(int, int, float)]],
        pts: Tuple[int, int],
        sw: Tuple[float, float],
        offset: float,
        sfo: float = 500.,
        nucleus: str = "1H",
        snr: Optional[float] = 20.,
        lb: Optional[Tuple[float, float]] = (6.91, 6.91),
    ) -> None:
        r"""Create a new instance from a 2DJ Spinach simulation.

        Parameters
        ----------
        shifts
            A list of tuple of chemical shift values for each spin.

        couplings
            The scalar couplings present in the spin system. Given ``shifts`` is of
            length ``n``, couplings should be an iterable with entries of the form
            ``(i1, i2, coupling)``, where ``1 <= i1, i2 <= n`` are the indices of
            the two spins involved in the coupling, and ``coupling`` is the value
            of the scalar coupling in Hz. ``None`` will set all spins to be
            uncoupled.

        pts
            The number of points the signal comprises.

        sw
            The sweep width of the signal (Hz).

        offset
            The transmitter offset (Hz).

        sfo
            The transmitter frequency (MHz).

        nucleus
            The identity of the nucleus targeted in the pulse sequence.

        snr
            The signal-to-noise ratio of the resulting signal, in decibels. ``None``
            produces a noiseless signal.

        lb
            Line broadening (exponential damping) to apply to the signal.
            The first point will be unaffected by damping, and the final point will
            be multiplied by ``np.exp(-lb)``. The default results in the final
            point being decreased in value by a factor of roughly 1000.
        """
        if not MATLAB_AVAILABLE:
            raise NotImplementedError(
                f"{RED}MATLAB isn't accessible to Python. To get up and running, "
                "take at look here:\n"
                "https://www.mathworks.com/help/matlab/matlab_external/"
                f"install-the-matlab-engine-for-python.html{END}"
            )

        sanity_check(
            ("shifts", shifts, sfuncs.check_float_list),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}),
            ("sw", sw, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("offset", offset, sfuncs.check_float),
            ("sfo", sfo, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("nucleus", nucleus, sfuncs.check_nucleus),
            ("snr", snr, sfuncs.check_float),
            ("lb", lb, sfuncs.check_float, (), {"greater_than_zero": True})
        )
        nspins = len(shifts)
        sanity_check(
            ("couplings", couplings, sfuncs.check_spinach_couplings, (nspins,)),
        )

        if couplings is None:
            couplings = []

        with cd(SPINACHPATH):
            devnull = io.StringIO(str(os.devnull))
            try:
                eng = matlab.engine.start_matlab()
                fid = eng.jres_sim(
                    shifts, couplings, pts, matlab.double(sw), matlab.double(offset),
                    sfo, nucleus, stdout=devnull, stderr=devnull,
                )
            except matlab.engine.MatlabExecutionError:
                raise ValueError(
                    f"{RED}Something went wrong in trying to run Spinach. This "
                    "is likely due to one of two things:\n"
                    "1. An inappropriate argument was given which was not noticed by "
                    "sanity checks. For example, you provided an isotope of the "
                    "correct format but which is unknown\n"
                    "2. You have not correctly configured Spinach.\n"
                    "Read what is stated below the line "
                    "\"matlab.engine.MatlabExecutionError:\" "
                    f"for more details on the error raised.{END}"
                )

        fid = sig.phase(np.array(fid), (0., np.pi / 2), (0., 0.))

        # Apply exponential damping
        for i, k in enumerate(lb):
            fid = sig.exp_apodisation(fid, k, axes=[i])

        if snr is not None:
            fid = sig.add_noise(fid, snr)

        expinfo = ExpInfo(
            dim=2,
            sw=sw,
            offset=(0., offset),
            sfo=(None, sfo),
            nuclei=(None, nucleus),
            default_pts=fid.shape,
        )

        return cls(fid, expinfo)

    def view_data(
        self,
        domain: str = "freq",
        abs_: bool = False,
    ) -> None:
        """View the data.

        Parameters
        ----------
        domain
            Must be ``"freq"`` or ``"time"``.

        abs_
            Whether or not to display frequency-domain data in absolute-value mode.
        """
        sanity_check(
            ("domain", domain, sfuncs.check_one_of, ("freq", "time")),
            ("abs_", abs_, sfuncs.check_bool),
        )

        if domain == "freq":
            spectrum = np.abs(self.spectrum) if abs_ else self.spectrum
            app = ContourApp(spectrum, self.expinfo)
            app.mainloop()

        elif domain == "time":
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            x, y = self.get_timepoints()
            xlabel, ylabel = [f"$t_{i}$ (s)" for i in range(1, 3)]

            ax.plot_wireframe(x, y, self.data)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(reversed(ax.get_xlim()))
            ax.set_ylim(reversed(ax.get_ylim()))
            ax.set_zticks([])

            plt.show()

    @property
    def direct_expinfo(self) -> ExpInfo:
        """Return :py:meth:`~nmrespy.ExpInfo` for the direct dimension."""
        return ExpInfo(
            dim=1,
            sw=self.sw()[1],
            offset=self.offset()[1],
            sfo=self.sfo[1],
            nuclei=self.nuclei[1],
            default_pts=self.default_pts[1],
        )

    @property
    def spectrum_zero_t1(self) -> np.ndarray:
        """Generate a 1D spectrum of the first time-slice in the indirect dimension."""
        data = copy.deepcopy(self.data[0])
        data[0] *= 0.5
        return sig.ft(data)

    @property
    def spectrum(self) -> np.ndarray:
        data = copy.deepcopy(self.data)
        data[0, 0] *= 0.5
        return sig.ft(data)

    @property
    def default_multiplet_thold(self) -> float:
        """The default margin for error when determining oscillators which belong to
        the same multiplet.

        Given by ``0.5 * self.sw()[0] / self.default_pts[0]`` (i.e. half the
        spetral resolution in the indirect dimension).
        """
        return 0.5 * (self.sw()[0] / self.default_pts[0])

    @logger
    def negative_45_signal(
        self,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[int] = None,
        _log: bool = True,
    ) -> np.ndarray:
        r"""Generate the synthetic signal :math:`y_{-45^{\circ}}(t)`, where
        :math:`t \geq 0`:

        .. math::

            y_{-45^{\circ}}(t) = \sum_{m=1}^M a_m \exp\left( \mathrm{i} \phi_m \right)
            \exp\left( 2 \mathrm{i} \pi f_{1,m} t \right)
            \exp\left( -t \left[2 \mathrm{i} \pi f_{2,m} + \eta_{2,m} \right] \right)

        Producing this signal from parameters derived from estimation of a 2DJ dataset
        should generate an absorption-mode 1D homodecoupled spectrum.

        Parameters
        ----------
        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        pts
            The number of points to construct the signal from. If ``None``,
            ``self.default_pts`` will be used.
        """
        self._check_results_exist()
        sanity_check(
            (
                "indices", indices, sfuncs.check_int_list, (),
                {
                    "len_one_can_be_listless": True,
                    "min_value": -len(self._results),
                    "max_value": len(self._results) - 1,
                },
                True,

            ),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}, True),
        )

        params = self.get_params(indices)
        offset = self.offset()[1]
        if pts is None:
            pts = self.default_pts[1]
        tp = self.get_timepoints(pts=(1, pts), meshgrid=False)[1]
        f1 = params[:, 2]
        f2 = params[:, 3]
        signal = np.einsum(
            "ij,j->i",
            np.exp(
                np.outer(
                    tp,
                    2j * np.pi * (f2 - f1 - offset) - params[:, 5],
                )
            ),
            params[:, 0] * np.exp(1j * params[:, 1])
        )

        return signal

    @logger
    def predict_multiplets(
        self,
        indices: Optional[Iterable[int]] = None,
        thold: Optional[float] = None,
        freq_unit: str = "hz",
        _log: bool = True,
    ) -> Dict[float, Iterable[int]]:
        """Predict the estimated oscillators which correspond to each multiplet
        in the signal.

        Parameters
        ----------
        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        thold
            Frequency threshold. All oscillators that make up a multiplet are assumed
            to obey the following expression:

            .. math::
                f_c - f_t < f_{2,m} - f_{1,m} < f_c + f_t

            where :math:`f_c` is the central frequency of the multiplet, and `f_t` is
            ``thold``

        freq_unit
            Must be ``"hz"`` or ``"ppm"``.

        _log
            Ignore me!

        Returns
        -------
        multiplets
            A dictionary with keys as the multiplet's central frequency, and values
            as a list of oscillator indices which make up the multiplet.
        """
        self._check_results_exist()
        sanity_check(
            self._indices_check(indices),
            ("thold", thold, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            ("freq_unit", freq_unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
        )
        if thold is None:
            thold = self.default_multiplet_thold

        params = self.get_params(indices)
        multiplets = {}
        in_range = lambda f, g: (g - thold < f < g + thold)
        for i, osc in enumerate(params):
            centre_freq = osc[3] - osc[2]
            assigned = False
            for freq in multiplets:
                if in_range(centre_freq, freq):
                    multiplets[freq].append(i)
                    assigned = True
                    break
            if not assigned:
                multiplets[centre_freq] = [i]

        for old_freq, mp_indices in list(multiplets.items()):
            new_freq = np.mean(params[mp_indices, 3] - params[mp_indices, 2])
            multiplets[new_freq] = multiplets.pop(old_freq)

        factor = 1. if freq_unit == "hz" else self.sfo[-1]
        multiplets = {
            freq / factor: indices
            for freq, indices in sorted(multiplets.items(), key=lambda item: item[0])
        }

        return multiplets

    def get_multiplet_integrals(
        self,
        indices: Optional[Iterable[int]] = None,
        thold: Optional[float] = None,
        freq_unit: str = "hz",
        scale: bool = True,
    ) -> Dict[float, float]:
        """
        """
        self._check_results_exist()
        sanity_check(
            self._indices_check(indices),
            ("thold", thold, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            ("freq_unit", freq_unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
            ("scale", scale, sfuncs.check_bool),
        )

        multiplets = self.predict_multiplets(
            indices=indices, thold=thold, freq_unit=freq_unit,
        )
        params = self.get_params(indices)
        integrals = {
            freq: sum(self.oscillator_integrals(params[mp]))
            for freq, mp in list(multiplets.items())
        }

        if scale:
            min_integral = min(list(integrals.values()))
            integrals = {
                freq: integral / min_integral
                for freq, integral in list(integrals.items())
            }

        return integrals

    @logger
    def find_spurious_oscillators(
        self,
        thold: Optional[float] = None,
    ) -> Dict[int, Iterable[int]]:
        r"""Predict which oscillators are spurious.

        This predicts the multiplet structures in the estimationm result, and then
        purges all oscillators which fall into the following criteria:

        * The oscillator is the only one in the multiplet.
        * The frequency in F1 is greater than ``thold``.

        Parameters
        ----------
        thold
            Frequency threshold within which :math:`f_2 - f_1` of the oscillators
            in a multiplet should agree. If ``None``, this is set to be
            :math:`N_1 / 2 f_{\mathrm{sw}, 1}``

        Returns
        -------
        A dictionary with int keys corresponding to result indices, and list
        values corresponding to oscillators which are deemed spurious.
        """
        self._check_results_exist()
        sanity_check(
            ("thold", thold, sfuncs.check_float, (), {"greater_than_zero": True}, True),
        )
        if thold is None:
            thold = self.default_multiplet_thold

        params = self.get_params()
        multiplets = self.predict_multiplets(thold=thold)
        spurious = {}
        for cfreq, oscs in multiplets.items:
            if len(oscs) == 1 and abs(cfreq) > thold:
                # osc_loc is a tuple of the form (result_index, osc_index)
                osc_loc = self.find_osc(params[oscs[0]])
                if osc_loc[0] in spurious:
                    spurious[osc_loc[0]].append(osc_loc[1])
                else:
                    spurious[osc_loc[0]] = [osc_loc[1]]

        return spurious

    @logger
    def remove_spurious_oscillators(
        self,
        thold: Optional[float] = None,
        **estimate_kwargs,
    ) -> None:
        r"""Attempt to remove spurious oscillators from the estimation result.

        See :py:meth:`find_spurious_oscillators` for information on how spurious
        oscillators are predicted.

        Oscillators deemed spurious are removed using :py:meth:`remove_oscillators`.

        Parameters
        ----------
        thold
            Frequency threshold within which :math:`f_2 - f_1` of the oscillators
            in a multiplet should agree. If ``None``, this is set to be
            :math:`N_1 / 2 f_{\mathrm{sw}, 1}``

        estimate_kwargs
            Keyword arguments to provide to :py:meth:`remove_oscillators`. Note
            that ``"initial_guess"`` and ``"region_unit"`` are set internally and
            will be ignored if given.
        """
        self._check_results_exist()
        sanity_check(
            ("thold", thold, sfuncs.check_float, (), {"greater_than_zero": True}, True),
        )
        spurious = self.find_spurious_oscillators(thold)
        for res_idx, osc_idx in spurious.items():
            self.remove_oscillators(osc_idx, res_idx, **estimate_kwargs)

    @logger
    def sheared_signal(
        self,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[Tuple[int, int]] = None,
        indirect_modulation: Optional[str] = None,
    ) -> np.ndarray:
        r"""Return an FID where direct dimension frequencies are perturbed such that:

        .. math::

            f_{2, m} = f_{2, m} - f_{1, m}\ \forall\ m \in \{1, \cdots, M\}

        This should yeild a signal where all components in a multiplet are centered
        at the spin's chemical shift in the direct dimenion, akin to "shearing" 2DJ
        data.

        Parameters
        ----------
        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        pts
            The number of points to construct the signal from. If ``None``,
            ``self.default_pts`` will be used.

        indirect_modulation
            Acquisition mode in indirect dimension of a 2D experiment. If the
            data is not 1-dimensional, this should be one of:

            * ``None`` - :math:`y \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              e^{\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
            * ``"amp"`` - amplitude modulated pair:
              :math:`y_{\mathrm{cos}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              \cos\left(\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1\right)
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
              :math:`y_{\mathrm{sin}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              \sin\left(\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1\right)
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
            * ``"phase"`` - phase-modulated pair:
              :math:`y_{\mathrm{P}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              e^{\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
              :math:`y_{\mathrm{N}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              e^{\left(-2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`

            ``None`` will lead to an array of shape ``(*pts)``. ``amp`` and ``phase``
            will lead to an array of shape ``(2, *pts)``.
        """
        self._check_results_exist()
        sanity_check(
            (
                "indices", indices, sfuncs.check_index,
                (len(self._results),), {}, True,
            ),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}, True),
        )

        edited_params = copy.deepcopy(self.get_params(indices))
        edited_params[:, 3] -= edited_params[:, 2]

        return self.make_fid(
            edited_params, pts=pts, indirect_modulation=indirect_modulation,
        )

    @logger
    def plot_result(
        self,
        indices: Optional[Iterable[int]] = None,
        multiplet_thold: Optional[float] = None,
        high_resolution_pts: Optional[int] = None,
        ratio_1d_2d: Tuple[float, float] = (2., 1.),
        region_unit: str = "hz",
        axes_left: float = 0.07,
        axes_right: float = 0.96,
        axes_bottom: float = 0.08,
        axes_top: float = 0.96,
        axes_region_separation: float = 0.05,
        xaxis_label_height: float = 0.02,
        xaxis_ticks: Optional[Iterable[Tuple[int, Iterable[float]]]] = None,
        contour_base: Optional[float] = None,
        contour_nlevels: Optional[int] = None,
        contour_factor: Optional[float] = None,
        contour_lw: float = 0.5,
        contour_color: Any = "k",
        multiplet_colors: Any = None,
        multiplet_lw: float = 1.,
        multiplet_vertical_shift: float = 0.,
        multiplet_show_center_freq: bool = True,
        multiplet_show_45: bool = True,
        marker_size: float = 3.,
        marker_shape: str = "o",
        label_peaks: bool = False,
        denote_regions: bool = False,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, np.ndarray[mpl.axes.Axes]]:
        """Generate a figure of the estimation result.

        The figure includes a contour plot of the 2DJ spectrum, a 1D plot of the
        first slice through the indirect dimension, plots of estimated multiplets,
        and a plot of the spectrum generated from :py:meth:`negative_45_signal`.

        Parameters
        ----------
        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        multiplet_thold
            Frequency threshold for multiplet prediction. All oscillators that make
            up a multiplet are assumed to obey the following expression:

            .. math::
                f_c - f_t < f^{(2)} - f^{(1)} < f_c + f_t

            where :math:`f_c` is the central frequency of the multiplet, and `f_t` is
            ``multiplet_thold``

        high_resolution_pts
            Indicates the number of points used to generate the multiplet structures
            and :py:meth:`negative_45_signal` spectrum. Should be greater than or
            equal to ``self.default_pts[1]``.

        ratio_1d_2d
            The relative heights of the regions containing the 1D spectra and the
            2DJ spectrum.

        axes_left
            The position of the left edge of the axes, in figure coordinates. Should
            be between ``0.`` and ``1.``.

        axes_right
            The position of the right edge of the axes, in figure coordinates. Should
            be between ``0.`` and ``1.``.

        axes_top
            The position of the top edge of the axes, in figure coordinates. Should
            be between ``0.`` and ``1.``.

        axes_bottom
            The position of the bottom edge of the axes, in figure coordinates. Should
            be between ``0.`` and ``1.``.

        axes_region_separation
            The extent by which adjacent regions are separated in the figure.

        xaxis_label_height
            The vertical location of the x-axis label, in figure coordinates. Should
            be between ``0.`` and ``1.``, though you are likely to want this to be
            slightly larger than ``0.`` typically.

        xaxis_ticks
            Specifies custom x-axis ticks for each region, overwriting the default
            ticks. Should be of the form: ``[(i, (a, b, ...)), (j, (c, d, ...)), ...]``
            where ``i`` and ``j`` are ints indicating the region under consideration,
            and ``a``-``d`` are floats indicating the tick values.

        contour_base
            The lowest level for the contour levels in the 2DJ spectrum plot.

        contour_nlevels
            The number of contour levels in the 2DJ spectrum plot.

        contour_factor
            The geometric scaling factor for adjacent contours in the 2DJ spectrum
            plot.

        contour_lw
            The linewidth of contours in the 2DJ spectrum plot.

        contour_color
            The color of the 2DJ spectrum plot.

        multiplet_colors
            **TODO**

        multiplet_lw
            Line width of multiplet plots

        multiplet_vertical_shift
            The vertical displacement of adjacent mutliplets, as a multiple of
            ``mutliplet_lw``. Set to ``0.`` if you want all mutliplets to lie on the
            same line.

        multiplet_show_center_freq
            If ``True``, lines are plotted on the 2DJ spectrum indicating the central
            frequency of each mutliplet.

        multiplet_show_45
            If ``True``, lines are plotted on the 2DJ spectrum indicating the 45° line
            along which peaks lie in ech multiplet.

        marker_size
            The size of markers indicating positions of peaks on the 2DJ contour plot.

        marker_shape
            The shape of markers indicating positions of peaks on the 2DJ contour plot.

        Returns
        -------
        fig
            The result figure

        axs
            A ``(2, N)`` NumPy array of the axes used for plotting.

        Notes
        -----
        **Figure coordinates** are a system in which ``0.`` indicates the left/bottom
        edge of the figure, and ``1.`` indicates the right/top.
        """
        sanity_check(
            (
                "indices", indices, sfuncs.check_int_list, (),
                {
                    "len_one_can_be_listless": True,
                    "min_value": -len(self._results),
                    "max_value": len(self._results) - 1,
                },
                True,
            ),
            (
                "multiplet_thold", multiplet_thold, sfuncs.check_float, (),
                {"greater_than_zero": True}, True,
            ),
            (
                "high_resolution_pts", high_resolution_pts, sfuncs.check_int, (),
                {"min_value": self.default_pts[1]}, True,
            ),
            (
                "ratio_1d_2d", ratio_1d_2d, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True},
            ),
            self._funit_check(region_unit, "region_unit"),
            (
                "axes_left", axes_left, sfuncs.check_float, (),
                {"min_value": 0., "max_value": 1.},
            ),
            (
                "axes_right", axes_right, sfuncs.check_float, (),
                {"min_value": 0., "max_value": 1.},
            ),
            (
                "axes_bottom", axes_bottom, sfuncs.check_float, (),
                {"min_value": 0., "max_value": 1.},
            ),
            (
                "axes_top", axes_top, sfuncs.check_float, (),
                {"min_value": 0., "max_value": 1.},
            ),
            (
                "axes_region_separation", axes_region_separation, sfuncs.check_float,
                (), {"min_value": 0., "max_value": 1.},
            ),
            (
                "xaxis_label_height", xaxis_label_height, sfuncs.check_float, (),
                {"min_value": 0., "max_value": 1.},
            ),
            (
                "contour_base", contour_base, sfuncs.check_float, (),
                {"min_value": 0.}, True,
            ),
            (
                "contour_nlevels", contour_nlevels, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            (
                "contour_factor", contour_factor, sfuncs.check_float, (),
                {"min_value": 1.}, True,
            ),
            ("contour_lw", contour_lw, sfuncs.check_float, (), {"min_value": 0.}),
            ("marker_size", marker_size, sfuncs.check_float, (), {"min_value": 0.}),
            (
                "multiplet_colors", multiplet_colors, sfuncs.check_oscillator_colors,
                (), {}, True,
            ),
            ("multiplet_lw", multiplet_lw, sfuncs.check_float, (), {"min_value": 0.}),
            (
                "multiplet_vertical_shift", multiplet_vertical_shift,
                sfuncs.check_float, (), {"min_value": 0.},
            ),
            (
                "multiplet_show_center_freq", multiplet_show_center_freq,
                sfuncs.check_bool,
            ),
            ("multiplet_show_45", multiplet_show_45, sfuncs.check_bool),
            ("denote_regions", denote_regions, sfuncs.check_bool),
        )
        # TODO
        # contour_color
        # linewidth
        # marker_shape: str = "o",

        indices = self._process_indices(indices)
        regions = sorted(
            [
                (i, result.get_region(unit=region_unit)[1])
                for i, result in enumerate(self.get_results())
                if i in indices
            ],
            key=lambda x: x[1][0],
            reverse=True,
        )

        # Megre overlapping/bordering regions
        merge_indices = []
        merge_regions = []
        for idx, region in regions:
            assigned = False
            for i, reg in enumerate(merge_regions):
                if max(region) >= min(reg):
                    merge_regions[i] = (max(reg), min(region))
                    assigned = True
                elif min(region) >= max(reg):
                    merge_regions[i] = (max(region), min(reg))
                    assigned = True

                if assigned:
                    merge_indices[i].append(idx)
                    break

            if not assigned:
                merge_indices.append([idx])
                merge_regions.append(region)

        n_regions = len(merge_regions)

        fig, axs = plt.subplots(
            nrows=2,
            ncols=n_regions,
            gridspec_kw={
                "left": axes_left,
                "right": axes_right,
                "bottom": axes_bottom,
                "top": axes_top,
                "wspace": axes_region_separation,
                "hspace": 0.,
                "width_ratios": [r[0] - r[1] for r in merge_regions],
                "height_ratios": ratio_1d_2d,
            },
            **kwargs,
        )
        if n_regions == 1:
            axs = axs.reshape(2, 1)

        if all(
            [isinstance(x, (float, int))
             for x in (contour_base, contour_nlevels, contour_factor)]
        ):
            contour_levels = [
                contour_base * contour_factor ** i
                for i in range(contour_nlevels)
            ]
        else:
            contour_levels = None

        if high_resolution_pts is None:
            high_resolution_pts = self.default_pts[1]

        expinfo_1d = self.direct_expinfo
        expinfo_1d_highres = copy.deepcopy(expinfo_1d)
        expinfo_1d_highres.default_pts = (high_resolution_pts,)
        full_shifts_1d, = expinfo_1d.get_shifts(unit=region_unit)
        full_shifts_1d_highres, = expinfo_1d_highres.get_shifts(unit=region_unit)
        full_shifts_2d_y, full_shifts_2d_x = self.get_shifts(unit=region_unit)
        sfo = self.sfo[1]

        shifts_2d = []
        shifts_1d = []
        shifts_1d_highres = []
        spectra_2d = []
        spectra_1d = []
        neg_45_spectra = []
        f1_f2 = []
        center_freqs = []
        multiplet_spectra = []
        multiplet_indices = []

        conv = f"{region_unit}->idx"
        for idx, region in zip(merge_indices, merge_regions):
            slice_ = slice(*expinfo_1d.convert([region], conv)[0])
            highres_slice = slice(*expinfo_1d_highres.convert([region], conv)[0])

            shifts_2d.append(
                (full_shifts_2d_x[:, slice_], full_shifts_2d_y[:, slice_])
            )
            shifts_1d.append(full_shifts_1d[slice_])
            shifts_1d_highres.append(full_shifts_1d_highres[highres_slice])

            spectra_2d.append(np.abs(self.spectrum).real[:, slice_])
            spectra_1d.append(self.spectrum_zero_t1.real[slice_])
            neg_45_spectra.append(
                sig.ft(
                    self.negative_45_signal(
                        indices=idx, pts=high_resolution_pts, _log=False,
                    )
                ).real[highres_slice]
            )

            params = self.get_params(indices=idx)
            multiplet_indices.append(
                list(
                    reversed(
                        self.predict_multiplets(
                            indices=idx, thold=multiplet_thold, _log=False,
                        ).values()
                    )
                )
            )

            multiplet_params = [params[i] for i in multiplet_indices[-1]]
            f1_f2_region = []
            center_freq = []
            for multiplet_param in multiplet_params:
                f1, f2 = multiplet_param[:, [2, 3]].T
                cf = np.mean(f2 - f1)
                f2 = f2 / sfo if region_unit == "ppm" else f2
                cf = cf / sfo if region_unit == "ppm" else cf
                center_freq.append(cf)
                f1_f2_region.append((f1, f2))

                multiplet = expinfo_1d.make_fid(
                    multiplet_param[:, [0, 1, 3, 5]],
                    pts=high_resolution_pts,
                )
                multiplet[0] *= 0.5
                multiplet_spectra.append(sig.ft(multiplet).real)

            f1_f2.append(f1_f2_region)
            center_freqs.append(center_freq)

        n_multiplets = len(multiplet_spectra)

        # Plot individual mutliplets
        for ax in axs[0]:
            colors = make_color_cycle(multiplet_colors, n_multiplets)
            ymax = -np.inf
            for i, mp_spectrum in enumerate(multiplet_spectra):
                color = next(colors)
                x = n_multiplets - 1 - i
                line = ax.plot(
                    full_shifts_1d_highres,
                    mp_spectrum + multiplet_vertical_shift * x,
                    color=color,
                    lw=multiplet_lw,
                    zorder=i,
                )[0]
                line_max = np.amax(line.get_ydata())
                if line_max > ymax:
                    ymax = line_max
                i += 1

        # Plot 1D spectrum
        spec_1d_low_pt = min([np.amin(spec) for spec in spectra_1d])
        shift = 1.03 * (ymax - spec_1d_low_pt)
        ymax = -np.inf
        for ax, shifts, spectrum in zip(axs[0], shifts_1d, spectra_1d):
            line = ax.plot(shifts, spectrum + shift, color="k")[0]
            line_max = np.amax(line.get_ydata())
            if line_max > ymax:
                ymax = line_max

        # Plot homodecoupled spectrum
        homo_spec_low_pt = min([np.amin(spec) for spec in neg_45_spectra])
        shift = 1.03 * (ymax - homo_spec_low_pt)
        for ax, shifts, spectrum in zip(axs[0], shifts_1d_highres, neg_45_spectra):
            ax.plot(shifts, spectrum + shift, color="k")

        # Plot 2DJ contour
        for ax, shifts, spectrum in zip(axs[1], shifts_2d, spectra_2d):
            ax.contour(
                *shifts,
                spectrum,
                colors=contour_color,
                linewidths=contour_lw,
                levels=contour_levels,
                zorder=0,
            )

        # Plot peak positions onto 2DJ
        colors = make_color_cycle(multiplet_colors, n_multiplets)
        for ax, f1f2, mp_idxs in zip(axs[1], f1_f2, multiplet_indices):
            for mp_f1f2, mp_idx in zip(f1f2, mp_idxs):
                color = next(colors)
                f1, f2 = mp_f1f2
                ax.scatter(
                    x=f2,
                    y=f1,
                    s=marker_size,
                    marker=marker_shape,
                    color=color,
                    zorder=100,
                )
                if label_peaks:
                    for f1_, f2_, idx in zip(f1, f2, mp_idx):
                        ax.text(
                            x=f2_,
                            y=f1_,
                            s=str(idx),
                            color=color,
                            fontsize=8,
                            clip_on=True,
                        )

        ylim1 = (shifts_2d[0][1][0, 0], shifts_2d[0][1][-1, 0])
        # Plot multiplet central frequencies
        if multiplet_show_center_freq:
            colors = make_color_cycle(multiplet_colors, n_multiplets)
            for ax, center_freq in zip(axs[1], center_freqs):
                for cf in center_freq:
                    color = next(colors)
                    ax.plot(
                        [cf, cf],
                        ylim1,
                        color=color,
                        lw=0.8,
                        zorder=2,
                    )

        # Plot 45 lines that multiplets lie along
        if multiplet_show_45:
            colors = make_color_cycle(multiplet_colors, n_multiplets)
            for ax, center_freq in zip(axs[1], center_freqs):
                for cf in center_freq:
                    color = next(colors)
                    ax.plot(
                        [cf + lim / (sfo if region_unit == "ppm" else 1.)
                         for lim in ylim1],
                        ylim1,
                        color=color,
                        lw=0.8,
                        zorder=2,
                        ls=":",
                    )

        # Configure axis appearance
        ylim0 = (
            min([ax.get_ylim()[0] for ax in axs[0]]),
            max([ax.get_ylim()[1] for ax in axs[0]]),
        )

        if denote_regions:
            for i, mi in enumerate(merge_indices):
                if len(mi) > 1:
                    locs_to_plot = [reg[1][0] for reg in regions if reg[0] in mi[1:]]
                    for loc in locs_to_plot:
                        for j, y in enumerate((ylim0, ylim1)):
                            axs[j, i].plot(
                                [loc, loc],
                                y,
                                color="#808080",
                                ls=":",
                            )

        axs[0, 0].spines["left"].set_zorder(1000)
        axs[0, -1].spines["right"].set_zorder(1000)

        self._configure_axes(
            fig,
            axs,
            merge_regions,
            xaxis_ticks,
            axes_left,
            axes_right,
            xaxis_label_height,
            region_unit,
        )
        for ax in axs[0]:
            ax.set_ylim(ylim0)
        for ax in axs[1]:
            ax.set_ylim(ylim1)
        axs[1, 0].set_ylabel("Hz")

        return fig, axs


class ContourApp(tk.Tk):
    """Tk app for viewing 2D spectra as contour plots."""

    def __init__(self, data: np.ndarray, expinfo) -> None:
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.shifts = list(reversed(
            [s.T for s in expinfo.get_shifts(data.shape, unit="ppm")]
        ))
        nuclei = expinfo.nuclei
        units = ["ppm" if sfo is not None else "Hz" for sfo in expinfo.sfo]
        self.f1_label, self.f2_label = [
            f"{nuc} ({unit})" if nuc is not None
            else unit
            for nuc, unit in zip(nuclei, units)
        ]

        self.data = data.T.real

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.fig = plt.figure(dpi=160, frameon=True)
        self._color_fig_frame()

        self.ax = self.fig.add_axes([0.1, 0.1, 0.87, 0.87])
        self.ax.set_xlim(self.shifts[0][0][0], self.shifts[0][-1][0])
        self.ax.set_ylim(self.shifts[1][0][0], self.shifts[1][0][-1])

        self.cmap = tk.StringVar(self, "bwr")
        self.nlevels = 10
        self.factor = 1.3
        self.base = np.amax(np.abs(self.data)) / 10
        self.update_plot()

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

        self.widget_frame = tk.Frame(self)
        self._add_widgets()
        self.widget_frame.grid(
            row=2,
            column=0,
            padx=10,
            pady=(0, 10),
            sticky="nsew",
        )
        self.close_button = tk.Button(
            self, text="Close", command=self.quit,
        )
        self.close_button.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="w")

    def _color_fig_frame(self) -> None:
        r, g, b = [x >> 8 for x in self.winfo_rgb(self.cget("bg"))]
        color = f"#{r:02x}{g:02x}{b:02x}"
        if not re.match(r"^#[0-9a-f]{6}$", color):
            color = "#d9d9d9"

        self.fig.patch.set_facecolor(color)

    def _add_widgets(self) -> None:
        # Colormap selection
        self.cmap_label = tk.Label(self.widget_frame, text="Colormap:")
        self.cmap_label.grid(row=0, column=0, padx=(0, 10))
        self.cmap_widget = tk.OptionMenu(
            self.widget_frame,
            self.cmap,
            self.cmap.get(),
            "PiYG",
            "PRGn",
            "BrBG",
            "PuOr",
            "RdGy",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "coolwarm",
            "bwr",
            "seismic",
            command=lambda x: self.update_plot(),
        )
        self.cmap_widget.grid(row=0, column=1)

        # Number of contour levels
        self.nlevels_label = tk.Label(self.widget_frame, text="levels")
        self.nlevels_label.grid(row=0, column=2, padx=(0, 10))
        self.nlevels_box = MyEntry(
            self.widget_frame,
            return_command=self.change_levels,
            return_args=("nlevels",),
        )
        self.nlevels_box.insert(0, str(self.nlevels))
        self.nlevels_box.grid(row=0, column=3)

        # Base contour level
        self.base_label = tk.Label(self.widget_frame, text="base")
        self.base_label.grid(row=0, column=4, padx=(0, 10))
        self.base_box = MyEntry(
            self.widget_frame,
            return_command=self.change_levels,
            return_args=("base",),
        )
        self.base_box.insert(0, f"{self.base:.2f}")
        self.base_box.grid(row=0, column=5)

        # Contour level scaling factor
        self.factor_label = tk.Label(self.widget_frame, text="factor")
        self.factor_label.grid(row=0, column=6, padx=(0, 10))
        self.factor_box = MyEntry(
            self.widget_frame,
            return_command=self.change_levels,
            return_args=("factor",),
        )
        self.factor_box.insert(0, f"{self.factor:.2f}")
        self.factor_box.grid(row=0, column=7)

    def change_levels(self, var: str) -> None:
        input_ = self.__dict__[f"{var}_box"].get()
        try:
            if var == "nlevels":
                value = int(input_)
                if value <= 0.:
                    raise ValueError
            else:
                value = float(input_)
                if (
                    value <= 1. and var == "factor" or
                    value <= 0. and var == "base"
                ):
                    raise ValueError

            self.__dict__[var] = value
            self.update_plot()

        except ValueError:
            box = self.__dict__[f"{var}_box"]
            box.delete(0, "end")
            box.insert(0, str(self.__dict__[var]))

    def make_levels(self) -> Iterable[float]:
        levels = [self.base * self.factor ** i
                  for i in range(self.nlevels)]
        return [-x for x in reversed(levels)] + levels

    def update_plot(self) -> None:
        levels = self.make_levels()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        self.ax.contour(
            *self.shifts, self.data, cmap=self.cmap.get(), levels=levels,
        )
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel(self.f2_label)
        self.ax.set_ylabel(self.f1_label)
        self.fig.canvas.draw_idle()
