# onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 24 Feb 2023 15:12:53 GMT

from __future__ import annotations
import copy
import io
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from nmrespy import MATLAB_AVAILABLE, ExpInfo, sig
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._files import cd, check_existent_dir, check_saveable_dir
from nmrespy._paths_and_links import SPINACHPATH
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)
from nmrespy.load import load_bruker
from nmrespy.plot import make_color_cycle

from . import logger, _Estimator1DProc


if USE_COLORAMA:
    import colorama
    colorama.init()

if MATLAB_AVAILABLE:
    import matlab
    import matlab.engine


class Estimator1D(_Estimator1DProc):
    """Estimator class for 1D data. For a tutorial on the basic functionailty
    this provides, see :ref:`ESTIMATOR1D`.

    .. note::

        To create an instance of ``Estimator1D``, you are advised to use one of
        the following methods if any are appropriate:

        * :py:meth:`new_bruker`
        * :py:meth:`new_from_parameters`
        * :py:meth:`new_spinach`
        * :py:meth:`from_pickle` (re-loads a previously saved estimator).
    """

    default_mpm_trim = 4096
    default_nlp_trim = None
    default_max_iterations_exact_hessian = 100
    default_max_iterations_gn_hessian = 200

    @classmethod
    def new_bruker(
        cls,
        directory: Union[str, Path],
        convdta: bool = True,
    ) -> Estimator1D:
        """Create a new instance from Bruker-formatted data.

        Parameters
        ----------
        directory
            Absolute path to data directory.

        convdta
            If ``True`` and the data is derived from an ``fid`` file, removal of
            the FID's digital filter will be carried out.

        Notes
        -----
        There are certain file paths expected to be found relative to ``directory``
        which contain the data and parameter files. Here is an extensive list of
        the paths expected to exist, for different data types:

        * Raw FID

          + ``directory/fid``
          + ``directory/acqus``

        * Processed data

          + ``directory/1r``
          + ``directory/../../acqus``
          + ``directory/procs``
        """
        sanity_check(
            ("directory", directory, check_existent_dir),
            ("convdta", convdta, sfuncs.check_bool),
        )

        directory = Path(directory).expanduser()
        data, expinfo = load_bruker(directory)

        if data.ndim != 1:
            raise ValueError(f"{RED}Data dimension should be 1.{END}")

        if directory.parent.name == "pdata":
            slice_ = slice(0, data.shape[0] // 2)
            data = (2 * sig.ift(data))[slice_]

        elif convdta:
            grpdly = expinfo.parameters["acqus"]["GRPDLY"]
            data = sig.convdta(data, grpdly)

        return cls(data, expinfo, directory)

    @classmethod
    def new_spinach(
        cls,
        shifts: Iterable[float],
        couplings: Optional[Iterable[Tuple(int, int, float)]],
        pts: int,
        sw: float,
        offset: float = 0.,
        sfo: float = 500.,
        nucleus: str = "1H",
        snr: Optional[float] = 20.,
        lb: float = 6.91,
    ) -> Estimator1D:
        r"""Create a new instance from a pulse-acquire Spinach simulation.

        See :ref:`SPINACH_INSTALL` for requirments to use this method.

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
            The identity of the nucleus. Should be of the form ``"<mass><sym>"``
            where ``<mass>`` is the atomic mass and ``<sym>`` is the element symbol.
            Examples:

            * ``"1H"``
            * ``"13C"``
            * ``"195Pt"``

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
                fid = eng.onedim_sim(
                    shifts, couplings, pts, sw, offset, sfo, nucleus,
                    stdout=devnull, stderr=devnull,
                )
            except matlab.engine.MatlabExecutionError:
                raise ValueError(
                    f"{RED}Something went wrong in trying to run Spinach.\n"
                    "Read what is stated below the line "
                    "\"matlab.engine.MatlabExecutionError:\" "
                    f"for more details on the error raised.{END}"
                )

        fid = sig.exp_apodisation(
            sig.add_noise(
                np.array(fid).flatten(),
                snr,
            ),
            lb,
        )

        expinfo = ExpInfo(
            dim=1,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nucleus,
            default_pts=fid.shape,
        )

        return cls(fid, expinfo)

    @classmethod
    def new_from_parameters(
        cls,
        params: np.ndarray,
        pts: int,
        sw: float,
        offset: float,
        sfo: float = 500.,
        nucleus: str = "1H",
        snr: Optional[float] = 20.,
    ) -> Estimator1D:
        """Generate an estimator instance with sythetic data created from an
        array of oscillator parameters.

        Parameters
        ----------
        params
            Parameter array with the following structure:

              .. code:: python

                 params = numpy.array([
                    [a_1, φ_1, f_1, η_1],
                    [a_2, φ_2, f_2, η_2],
                    ...,
                    [a_m, φ_m, f_m, η_m],
                 ])

        pts
            The number of points the signal comprises.

        sw
            The sweep width of the signal (Hz).

        offset
            The transmitter offset (Hz).

        sfo
            The transmitter frequency (MHz).

        nucleus
            The identity of the nucleus. Should be of the form ``"<mass><sym>"``
            where ``<mass>`` is the atomic mass and ``<sym>`` is the element symbol.
            Examples: ``"1H"``, ``"13C"``, ``"195Pt"``

        snr
            The signal-to-noise ratio (dB). If ``None`` then no noise will be added
            to the FID.
        """
        sanity_check(
            ("params", params, sfuncs.check_parameter_array, (1,)),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}),
            ("sw", sw, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("offset", offset, sfuncs.check_float, (), {}, True),
            ("nucleus", nucleus, sfuncs.check_nucleus),
            ("sfo", sfo, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            ("snr", snr, sfuncs.check_float, (), {"greater_than_zero": True}, True),
        )

        expinfo = ExpInfo(
            dim=1,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nucleus,
            default_pts=pts,
        )

        data = expinfo.make_fid(params, snr=snr)
        return cls(data, expinfo)

    @property
    def spectrum(self) -> np.ndarray:
        """Return the spectrum corresponding to ``self.data``

        The spectrum is generated by halving the initial point in the data, and
        Fourier Transforming.
        """
        data = copy.deepcopy(self.data)
        data[0] *= 0.5
        return sig.ft(data)

    def view_data(
        self,
        domain: str = "freq",
        components: str = "real",
        freq_unit: str = "hz",
    ) -> None:
        """View the data (FID or spectrum) with an interactive matplotlib plot.

        Parameters
        ----------
        domain
            Must be ``"freq"`` or ``"time"``.

        components
            Must be ``"real"``, ``"imag"``, or ``"both"``.

        freq_unit
            Must be ``"hz"`` or ``"ppm"``. If ``domain`` is ``freq``, this
            determines which unit to set chemical shifts to.
        """
        sanity_check(
            ("domain", domain, sfuncs.check_one_of, ("freq", "time")),
            ("components", components, sfuncs.check_one_of, ("real", "imag", "both")),
            ("freq_unit", freq_unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
        )

        fig = plt.figure()
        ax = fig.add_subplot()
        y = copy.deepcopy(self._data)

        if domain == "freq":
            x = self.get_shifts(unit=freq_unit)[0]
            y[0] /= 2
            y = sig.ft(y)
            label = f"$\\omega$ ({freq_unit.replace('h', 'H')})"
        elif domain == "time":
            x = self.get_timepoints()[0]
            label = "$t$ (s)"

        if components in ["real", "both"]:
            ax.plot(x, y.real, color="k")
        if components in ["imag", "both"]:
            ax.plot(x, y.imag, color="#808080")

        ax.set_xlabel(label)
        ax.set_xlim((x[0], x[-1]))

        plt.show()

    def write_to_bruker(
        self,
        path: Union[str, Path],
        indices: Optional[Iterable[int]] = None,
        pts: Optional[Iterable[int]] = None,
        expno: Optional[int] = None,
        procno: Optional[int] = None,
        force_overwrite: bool = False,
    ) -> None:
        """Write a signal generated with estimated parameters to Bruker format.

        * ``<path>/<expno>/`` will contain the time-domain data and information
          (``fid``, ``acqus``, ...)
        * ``<path>/<expno>/pdata/1/`` will contain the processed data and
          information (``pdata``, ``procs``, ...)

        .. note::

             There is a known problem that the spectral data has timepoints along
             the x-axis rather than chemical shifts. I will try to figure out why
             and fix this in due course!

        Parameters
        ----------
        path
            The path to the root directory to store the data in.

        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        pts
            The number of points to construct the signal from.

        expno
            The experiment number. If ``None``, the smallest int ``x`` for which the
            directory ``<path>/<x>/`` doesn't exist will be used.

        force_overwrite
            If ``False`` and the directory ``<path>/<expno>/`` already exists,
            the user will be prompted to confirm whether they are happy to
            overwrite it. If ``True``, said directory will be overwritten.
        """
        # TODO: figure out x-axis issue (see warning above).
        self._check_results_exist()
        sanity_check(
            ("path", path, check_saveable_dir, (True,)),
            self._indices_check(indices),
            self._pts_check(pts),
            ("expno", expno, sfuncs.check_int, (), {"min_value": 1}, True),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
        )
        fid = self.make_fid_from_result(indices=indices, pts=pts)
        # calls ExpInfo.write_to_bruker()
        super().write_to_bruker(fid, path, expno, 1, force_overwrite)

    @logger
    def plot_result(
        self,
        indices: Optional[Iterable[int]] = None,
        high_resolution_pts: Optional[int] = None,
        region_unit: str = "hz",
        axes_left: float = 0.07,
        axes_right: float = 0.96,
        axes_bottom: float = 0.08,
        axes_top: float = 0.96,
        axes_region_separation: float = 0.05,
        xaxis_unit: str = "hz",
        xaxis_label_height: float = 0.02,
        xaxis_ticks: Optional[Iterable[Tuple[int, Iterable[float]]]] = None,
        oscillator_colors: Any = None,
        plot_model: bool = True,
        plot_residual: bool = True,
        model_shift: Optional[float] = None,
        label_peaks: bool = True,
        denote_regions: bool = False,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, np.ndarray[mpl.axes.Axes]]:
        """Generate a figure of the estimation result.

        Parameters
        ----------
        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        high_resolution_pts
            Indicates the number of points used to generate the oscillators and model.
            Should be greater than or equal to ``self.default_pts[0]``. If ``None``,
            ``self.default_pts[0]`` will be used.

        axes_left
            The position of the left edge of the axes, in `figure coordinates
            <https://matplotlib.org/stable/tutorials/advanced/\
            transforms_tutorial.html>`_\. Should be between ``0.`` and ``1.``.

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
            The extent by which adjacent regions are separated in the figure,
            in figure coordinates.

        xaxis_unit
            The unit to express chemical shifts in. Should be ``"hz"`` or ``"ppm"``.

        xaxis_label_height
            The vertical location of the x-axis label, in figure coordinates. Should
            be between ``0.`` and ``1.``, though you are likely to want this to be
            only slightly larger than ``0.``.

        xaxis_ticks
            Specifies custom x-axis ticks for each region, overwriting the default
            ticks. Should be of the form: ``[(i, (a, b, ...)), (j, (c, d, ...)), ...]``
            where ``i`` and ``j`` are ints indicating the region under consideration,
            and ``a``-``d`` are floats indicating the tick values.

        oscillator_colors
            Describes how to color individual oscillators. The following
            is a complete list of options:

            * If a `valid matplotlib colour
              <https://matplotlib.org/stable/tutorials/colors/colors.html>`_ is
              given, all oscillators will be given this color.
            * If a string corresponding to a `matplotlib colormap
              <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_
              is given, the oscillators will be consecutively shaded by linear
              increments of this colormap.
            * If an iterable object containing valid matplotlib colors is
              given, these colors will be cycled.
              For example, if ``oscillator_colors = ['r', 'g', 'b']``:

              + Oscillators 0, 3, 6, ... would be :red:`red (#FF0000)`
              + Oscillators 1, 4, 7, ... would be :green:`green (#008000)`
              + Oscillators 2, 5, 8, ... would be :blue:`blue (#0000FF)`

            * If ``None``, the default colouring method will be applied, which
              involves cycling through the following colors:

              + :oscblue:`#1063E0`
              + :oscorange:`#EB9310`
              + :oscgreen:`#2BB539`
              + :oscred:`#D4200C`

        plot_model
            .. todo::

                Add description

        plot_residual
            .. todo::

                Add description

        model_shift
            The vertical displacement of the model relative to the data.

        label_peaks
            If True, label peaks according to their index. The parameters of a peak
            denoted with the label ``i`` in the figure can be accessed with
            ``self.get_results(indices)[i]``.

        denote_regions
            If ``True``, and there are regions which share a boundary, a
            vertical line will be plotted to show the boundary.

        kwargs
            Keyword arguments provided to `matplotlib.pyplot.figure
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html\
            #matplotlib.pyplot.figure>`_\. Allowed arguments include
            ``figsize``, ``facecolor``, ``edgecolor``, etc.

        Returns
        -------
        fig
            The result figure. This can be saved to various formats using the
            `savefig <https://matplotlib.org/stable/api/figure_api.html\
            #matplotlib.figure.Figure.savefig>`_ method.

        axs
            A ``(1, N)`` NumPy array of the axes generated.
        """
        sanity_check(
            self._indices_check(indices),
            (
                "high_resolution_pts", high_resolution_pts, sfuncs.check_int, (),
                {"min_value": self.default_pts[-1]}, True,
            ),
            # (
            #     "figure_size", figure_size, sfuncs.check_float_list, (),
            #     {"length": 2, "must_be_positive": True},
            # ),
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
            ("plot_model", plot_model, sfuncs.check_bool),
            ("plot_residual", plot_residual, sfuncs.check_bool),
            (
                "model_shift", model_shift, sfuncs.check_float, (),
                {"min_value": 0.}, True,
            ),
            (
                "oscillator_colors", oscillator_colors, sfuncs.check_oscillator_colors,
                (), {}, True,
            ),
            ("label_peaks", label_peaks, sfuncs.check_bool),
            ("denote_regions", denote_regions, sfuncs.check_bool),
        )

        indices = self._process_indices(indices)
        merge_indices, merge_regions = self._plot_regions(indices, region_unit)
        n_regions = len(merge_regions)

        fig, axs = plt.subplots(
            nrows=1,
            ncols=n_regions,
            gridspec_kw={
                "left": axes_left,
                "right": axes_right,
                "bottom": axes_bottom,
                "top": axes_top,
                "wspace": axes_region_separation,
                "width_ratios": [r[0] - r[1] for r in merge_regions],
            },
            **kwargs,
        )
        if n_regions == 1:
            axs = np.array([axs])
        axs = np.expand_dims(axs, axis=0)

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

        data = self._make_plot_data(
            indices,
            high_resolution_pts,
            region_unit,
            model_shift,
        )

        self._plot_data(
            axs[0],
            data,
            merge_indices,
            oscillator_colors,
            label_peaks,
            plot_model,
            plot_residual,
        )
        self._set_ylim(axs[0], data, plot_model, plot_residual)

        return fig, axs

    def _make_plot_data(
        self,
        indices: Iterable[int],
        high_resolution_pts: Optional[int],
        region_unit: str,
        model_shift: Optional[float],
    ) -> Dict:
        merge_indices, merge_regions = self._plot_regions(indices, region_unit)

        if high_resolution_pts is None:
            high_resolution_pts = self.default_pts[-1]

        highres_expinfo = self.expinfo
        highres_expinfo.default_pts = (high_resolution_pts,)

        full_shifts_highres, = highres_expinfo.get_shifts(unit=region_unit)
        full_shifts, = self.get_shifts(unit=region_unit)
        full_model = self.make_fid_from_result(indices)
        full_model[0] *= 0.5
        full_residual = self.spectrum.real - sig.ft(full_model).real
        full_model_highres = self.make_fid_from_result(indices, pts=high_resolution_pts)
        full_model_highres[0] *= 0.5
        full_model_highres = sig.ft(full_model_highres).real

        data = {
            "spectra": [],
            "models": [],
            "residuals": [],
            "oscillators": [],
            "shifts": [],
            "shifts_highres": [],
        }
        # Get all the data
        params = self.get_params(indices=indices)
        for idx, region in zip(merge_indices, merge_regions):
            slice_ = slice(
                *self.convert([region], f"{region_unit}->idx")[0]
            )
            highres_slice = slice(
                *highres_expinfo.convert([region], f"{region_unit}->idx")[0]
            )

            data["shifts"].append(full_shifts[slice_])
            data["shifts_highres"].append(full_shifts_highres[highres_slice])
            data["spectra"].append(self.spectrum.real[slice_])
            oscs = []
            for p in self.get_params(indices=idx):
                p = np.expand_dims(p, axis=0)
                osc = self.make_fid(p, pts=high_resolution_pts)
                osc[0] *= 0.5
                label = int(np.where((params == p).all(axis=-1))[0][0])
                oscs.append((label, sig.ft(osc).real[highres_slice]))
            data["oscillators"].append(oscs)
            data["residuals"].append(full_residual[slice_])
            data["models"].append(full_model_highres[highres_slice])

        if model_shift is None:
            model_shift = 0.1 * max([np.amax(spectrum) for spectrum in data["spectra"]])
        data["models"] = [(model + model_shift) for model in data["models"]]

        resid_span = self._get_data_span(data["residuals"])

        rest_lines = (
            [osc[1] for oscs in data["oscillators"] for osc in oscs] +
            [model for model in data["models"]] +
            [spectrum for spectrum in data["spectra"]]
        )

        rest_span = self._get_data_span(rest_lines)

        t = ((resid_span[1] - resid_span[0]) + (rest_span[1] - rest_span[0])) / 0.91
        rest_shift = resid_span[1] - rest_span[0] + (0.03 * t)
        model_shift += rest_shift

        for i, oscs in enumerate(data["oscillators"]):
            data["oscillators"][i] = [
                (label, osc + rest_shift)
                for (label, osc) in oscs
            ]
        data["spectra"] = [(spectrum + rest_shift) for spectrum in data["spectra"]]
        data["models"] = [(model + rest_shift) for model in data["models"]]

        return data

    @staticmethod
    def _plot_data(
        axs: Iterable[mpl.axes.Axes],
        data: Dict,
        merge_indices: Iterable[Iterable[int]],
        oscillator_colors: Any,
        label_peaks: bool,
        plot_model: bool,
        plot_residual: bool,
    ) -> None:
        if plot_residual:
            for ax, shifts_, residual in zip(axs, data["shifts"], data["residuals"]):
                ax.plot(shifts_, residual, color="#808080")

        for ax, shifts_, spectrum in zip(axs, data["shifts"], data["spectra"]):
            ax.plot(shifts_, spectrum, color="#000000")

        if plot_model:
            for ax, shifts_hr, model in zip(axs, data["shifts_highres"], data["models"]):  # noqa: E501
                ax.plot(shifts_hr, model, color="#808080")

        noscs = sum(len(oscs) for oscs in data["oscillators"])
        colors = make_color_cycle(oscillator_colors, noscs)
        for ax, shifts_hr, oscs, merge_idxs in zip(
            axs[::-1],
            data["shifts_highres"][::-1],
            data["oscillators"][::-1],
            merge_indices[::-1],
        ):
            for i, (label, osc) in enumerate(oscs):
                color = next(colors)
                ax.plot(shifts_hr, osc, color=color)
                if label_peaks:
                    label_idx = np.argmax(np.abs(osc))
                    ax.annotate(
                        str(label),
                        xy=(shifts_hr[label_idx], osc[label_idx]),
                        color=color,
                    )

    @staticmethod
    def _get_data_span(data: Iterable[np.ndarray]) -> Tuple[float, float]:
        return (
            min([np.amin(datum) for datum in data]),
            max([np.amax(datum) for datum in data]),
        )

    def _set_ylim(
        self,
        axs: Iterable[mpl.axes.Axes],
        data: Dict,
        plot_model: bool,
        plot_residual: bool,
    ) -> None:
        all_lines = (
            [osc[1] for oscs in data["oscillators"] for osc in oscs] +
            [spectrum for spectrum in data["spectra"]]
        )
        if plot_model:
            all_lines += [model for model in data["models"]]
        if plot_residual:
            all_lines += [residual for residual in data["residuals"]]

        data_span = self._get_data_span(all_lines)
        h = data_span[1] - data_span[0]
        bottom = data_span[0] - (0.03 * h)
        top = data_span[1] + (0.03 * h)

        for ax in axs:
            ax.set_yticks([])
            ax.set_ylim(bottom, top)
