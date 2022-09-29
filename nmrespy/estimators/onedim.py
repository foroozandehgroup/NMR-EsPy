# onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 29 Sep 2022 15:56:30 BST

from __future__ import annotations
import copy
import io
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from nmrespy import MATLAB_AVAILABLE, ExpInfo, sig
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._files import cd, check_existent_dir
from nmrespy._paths_and_links import NMRESPYPATH, SPINACHPATH
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)
from nmrespy.freqfilter import Filter
from nmrespy.load import load_bruker
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import NonlinearProgramming
from nmrespy.plot import ResultPlotter, make_color_cycle

from . import logger, _Estimator1DProc, Result


if USE_COLORAMA:
    import colorama
    colorama.init()

if MATLAB_AVAILABLE:
    import matlab
    import matlab.engine


class Estimator1D(_Estimator1DProc):
    """Estimator class for 1D data.

    .. note::

        To create an instance of ``Estimator1D``, you should use one of the following
        methods:

        * :py:meth:`new_bruker`
        * :py:meth:`new_synthetic_from_parameters`
        * :py:meth:`new_synthetic_from_simulation`
        * :py:meth:`from_pickle` (re-loads a previously saved estimator).
    """

    default_mpm_trim = 4096
    default_nlp_trim = None

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
        **Directory Requirements**

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
    ) -> None:
        r"""Create a new instance from a pulse-acquire Spinach simulation.

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
    def new_synthetic_from_parameters(
        cls,
        params: np.ndarray,
        pts: int,
        sw: float,
        offset: float,
        sfo: float = 500.,
        nucleus: str = "1H",
        snr: Optional[float] = 20.,
        lb: float = 6.91,
    ) -> Estimator1D:
        """Generate an estimator instance from an array of oscillator parameters.

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

        snr
            The signal-to-noise ratio. If ``None`` then no noise will be added
            to the FID.
        """
        sanity_check(
            ("params", params, sfuncs.check_parameter_array, (1,)),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}),
            ("sw", sw, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("offset", offset, sfuncs.check_float, (), {}, True),
            ("sfo", sfo, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            ("snr", snr, sfuncs.check_float, (), {"greater_than_zero": True}, True),
        )

        expinfo = ExpInfo(
            dim=1,
            sw=sw,
            offset=offset,
            sfo=sfo,
            default_pts=pts,
        )

        data = expinfo.make_fid(params, snr=snr)
        return cls(data, expinfo)

    @property
    def spectrum(self) -> np.ndarray:
        """Return the spectrum corresponding to ``self.data``"""
        data = copy.deepcopy(self.data)
        data[0] *= 0.5
        return sig.ft(data)

    def view_data(
        self,
        domain: str = "freq",
        components: str = "real",
        freq_unit: str = "hz",
    ) -> None:
        """View the data.

        Parameters
        ----------
        domain
            Must be ``"freq"`` or ``"time"``.

        components
            Must be ``"real"``, ``"imag"``, or ``"both"``.

        freq_unit
            Must be ``"hz"`` or ``"ppm"``.
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

    def write_to_topspin(
        self,
        path: Union[str, Path],
        expno: int,
        force_overwrite: bool = False,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[Iterable[int]] = None,
    ) -> None:
        # TODO: Work in progress
        res_len = len(self._results)
        sanity_check(
            ("expno", expno, sfuncs.check_int, (), {"min_value": 1}),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            (
                "indices", indices, sfuncs.check_int_list, (),
                {"min_value": -res_len, "max_value": res_len - 1}, True,
            ),
            (
                "pts", pts, sfuncs.check_int_list, (),
                {
                    "length": self.dim,
                    "len_one_can_be_listless": True,
                    "min_value": 1,
                },
                True,
            ),
        )

        fid = self.make_fid(indices, pts)
        fid_uncomplex = np.zeros((2 * fid.size,), dtype="float64")
        fid_uncomplex[::2] = fid.real
        fid_uncomplex[1::2] = fid.imag

        with open(NMRESPYPATH / "ts_templates/acqus", "r") as fh:
            text = fh.read()

        int_regex = r"-?\d+"
        float_regex = r"-?\d+(\.\d+)?"
        text = re.sub(
            r"\$BF1= " + float_regex,
            "$BF1= " +
            str(self.bf[0] if self.bf is not None else 500. - 1e-6 * self.offset()[0]),
            text,
        )
        text = re.sub(r"\$BYTORDA= " + int_regex, "$BYTORDA= 0", text)
        text = re.sub(r"\$DTYPA= " + int_regex, "$DTYPA= 2", text)
        text = re.sub(r"\$GRPDLY= " + float_regex, "$GRPDLY= 0", text)
        text = re.sub(
            r"\$NUC1= <\d+[a-zA-Z]+>",
            "$NUC1= <" +
            (self.nuclei[0] if self.nuclei is not None else "1H") +
            ">",
            text,
        )
        text = re.sub(r"\$O1= " + float_regex, f"$O1= {self.offset()[0]}", text)
        text = re.sub(
            r"\$SFO1= " + float_regex,
            "$SFO1= " +
            str(self.sfo[0] if self.sfo is not None else 500.0),
            text,
        )
        text = re.sub(
            r"\$SW= " + float_regex,
            "$SW= " +
            str(self.sw(unit="ppm")[0] if self.hz_ppm_valid else self.sw()[0] / 500.0),
            text,
        )
        text = re.sub(r"\$SW_h= " + float_regex, f"$SW_h= {self.sw()[0]}", text)
        text = re.sub(r"\$TD= " + int_regex, f"$TD= {fid.size}", text)

        path = Path(path).expanduser()
        if not path.is_dir():
            path.mkdir()
        if (expdir := path / str(expno)).is_dir():
            shutil.rmtree(expdir)
        expdir.mkdir()

        with open(expdir / "acqus", "w") as fh:
            fh.write(text)
        with open(expdir / "acqu", "w") as fh:
            fh.write(text)
        with open(expdir / "fid", "wb") as fh:
            fh.write(fid_uncomplex.astype("<f8").tobytes())

    # @logger
    # def plot_result(
    #     self,
    #     indices: Optional[Iterable[int]] = None,
    #     *,
    #     plot_residual: bool = True,
    #     plot_model: bool = False,
    #     residual_shift: Optional[Iterable[float]] = None,
    #     model_shift: Optional[float] = None,
    #     shifts_unit: str = "ppm",
    #     data_color: Any = "#000000",
    #     residual_color: Any = "#808080",
    #     model_color: Any = "#808080",
    #     oscillator_colors: Optional[Any] = None,
    #     show_labels: bool = False,
    #     stylesheet: Optional[Union[str, Path]] = None,
    # ) -> Iterable[ResultPlotter]:
    #     """Write estimation results to text and PDF files.

    #     Parameters
    #     ----------
    #     indices
    #         The indices of results to include. Index ``0`` corresponds to the first
    #         result obtained using the estimator, ``1`` corresponds to the next, etc.
    #         If ``None``, all results will be included.

    #     plot_model
    #         If ``True``, plot the model generated using ``result``. This model is
    #         a summation of all oscillator present in the result.

    #     plot_residual
    #         If ``True``, plot the difference between the data and the model
    #         generated using ``result``.

    #     residual_shift
    #         Specifies a translation of the residual plot along the y-axis. If
    #         ``None``, a default shift will be applied.

    #     model_shift
    #         Specifies a translation of the residual plot along the y-axis. If
    #         ``None``, a default shift will be applied.

    #     shifts_unit
    #         Units to display chemical shifts in. Must be either ``'ppm'`` or
    #         ``'hz'``.

    #     data_color
    #         The colour used to plot the data. Any value that is recognised by
    #         matplotlib as a color is permitted. See `here
    #         <https://matplotlib.org/stable/tutorials/colors/colors.html>`_ for
    #         a full description of valid values.

    #     residual_color
    #         # The colour used to plot the residual. See ``data_color`` for a
    #         # description of valid colors.

    #     model_color
    #         The colour used to plot the model. See ``data_color`` for a
    #         description of valid colors.

    #     oscillator_colors
    #         Describes how to color individual oscillators. The following
    #         is a complete list of options:

    #         * If a valid matplotlib color is given, all oscillators will
    #           be given this color.
    #         * If a string corresponding to a matplotlib colormap is given,
    #           the oscillators will be consecutively shaded by linear increments
    #           of this colormap. For all valid colormaps, see
    #           `here <https://matplotlib.org/stable/tutorials/colors/\
    #           colormaps.html>`__
    #         * If an iterable object containing valid matplotlib colors is
    #           given, these colors will be cycled.
    #           For example, if ``oscillator_colors = ['r', 'g', 'b']``:

    #           + Oscillators 1, 4, 7, ... would be :red:`red (#FF0000)`
    #           + Oscillators 2, 5, 8, ... would be :green:`green (#008000)`
    #           + Oscillators 3, 6, 9, ... would be :blue:`blue (#0000FF)`

    #         * If ``None``, the default colouring method will be applied, which
    #           involves cycling through the following colors:

    #             - :oscblue:`#1063E0`
    #             - :oscorange:`#EB9310`
    #             - :oscgreen:`#2BB539`
    #             - :oscred:`#D4200C`

    #     show_labels
    #         If ``True``, each oscillator will be given a numerical label
    #         in the plot, if ``False``, the labels will be hidden.

    #     stylesheet
    #         The name of/path to a matplotlib stylesheet for further
    #         customaisation of the plot. See `here <https://matplotlib.org/\
    #         stable/tutorials/introductory/customizing.html>`__ for more
    #         information on stylesheets.
    #     """
    #     self._check_results_exist()
    #     sanity_check(
    #         (
    #             "indices", indices, sfuncs.check_int_list, (),
    #             {
    #                 "must_be_positive": True,
    #                 "max_value": len(self._results) - 1,
    #             },
    #             True,
    #         ),
    #         ("plot_residual", plot_residual, sfuncs.check_bool),
    #         ("plot_model", plot_model, sfuncs.check_bool),
    #         ("residual_shift", residual_shift, sfuncs.check_float, (), {}, True),
    #         ("model_shift", model_shift, sfuncs.check_float, (), {}, True),
    #         (
    #             "shifts_unit", shifts_unit, sfuncs.check_frequency_unit,
    #             (self.hz_ppm_valid,),
    #         ),
    #         ("data_color", data_color, sfuncs.check_mpl_color),
    #         ("residual_color", residual_color, sfuncs.check_mpl_color),
    #         ("model_color", model_color, sfuncs.check_mpl_color),
    #         (
    #             "oscillator_colors", oscillator_colors, sfuncs.check_oscillator_colors,
    #             (), {}, True,
    #         ),
    #         ("show_labels", show_labels, sfuncs.check_bool),
    #         ("stylesheet", stylesheet, sfuncs.check_str, (), {}, True),
    #     )
    #     results = self.get_results(indices)

    #     expinfo = ExpInfo(
    #         1,
    #         sw=self.sw(),
    #         offset=self.offset(),
    #         sfo=self.sfo,
    #         nuclei=self.nuclei,
    #         default_pts=self.default_pts,
    #     )

    #     return [
    #         ResultPlotter(
    #             self._data,
    #             result.get_params(funit="hz"),
    #             expinfo,
    #             region=result.get_region(unit=shifts_unit),
    #             shifts_unit=shifts_unit,
    #             plot_residual=plot_residual,
    #             plot_model=plot_model,
    #             residual_shift=residual_shift,
    #             model_shift=model_shift,
    #             data_color=data_color,
    #             residual_color=residual_color,
    #             model_color=model_color,
    #             oscillator_colors=oscillator_colors,
    #             show_labels=show_labels,
    #             stylesheet=stylesheet,
    #         )
    #         for result in results

    @logger
    def plot_result(
        self,
        indices: Optional[Iterable[int]] = None,
        high_resolution_pts: Optional[int] = None,
        figure_size: Tuple[float, float] = (8., 6.),
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
        model_shift: Optional[float] = None,
        data_shift: float = 0.,
        label_peaks: bool = False,
        denote_regions: bool = False,
    ) -> Tuple[mpl.figure.Figure, np.ndarray[mpl.axes.Axes]]:
        sanity_check(
            self._indices_check(indices),
            (
                "high_resolution_pts", high_resolution_pts, sfuncs.check_int, (),
                {"min_value": self.default_pts[-1]}, True,
            ),
            (
                "figure_size", figure_size, sfuncs.check_float_list, (),
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
        n_oscs = self.get_params(indices=indices).shape[0]

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

        # Get all the ydata
        spectra = []
        models = []
        residuals = []
        oscillators = []
        shifts = []
        shifts_highres = []
        for idx, region in zip(merge_regions, merge_regions):
            slice_ = slice(
                *self.convert([region], f"{region_unit}->idx")[0]
            )
            highres_slice = slice(
                *highres_expinfo.convert([region], f"{region_unit}->idx")[0]
            )

            shifts.append(full_shifts[slice_])
            shifts_highres.append(full_shifts_highres[highres_slice])
            spectra.append(self.spectrum.real[slice_])
            oscs = []
            for i in range(n_oscs):
                osc = self.make_fid_from_result(
                    indices=indices, osc_indices=i, pts=high_resolution_pts,
                )
                osc[0] *= 0.5
                oscs.append(sig.ft(osc).real[highres_slice])
            oscillators.append(oscs)
            residuals.append(full_residual[slice_])
            models.append(full_model_highres[highres_slice])

        resid_span = (
            min([np.amin(residual) for residual in residuals]),
            max([np.amax(residual) for residual in residuals]),
        )
        r = resid_span[1] - resid_span[0]

        if model_shift is None:
            model_shift = 0.1 * max([np.amax(spectrum) for spectrum in spectra])
        models = [(model + model_shift) for model in models]

        rest_lines = (
            [osc for oscs in oscillators for osc in oscs] +
            [spectrum for spectrum in spectra] +
            [model for model in models]
        )
        rest_span = (
            min([np.amin(line) for line in rest_lines]),
            max([np.amax(line) for line in rest_lines]),
        )
        s = rest_span[1] - rest_span[0]

        t = (r + s) / 0.91
        bottom = resid_span[0] - 0.03 * t
        top = bottom + t

        rest_shift = resid_span[1] - rest_span[0] + (0.03 * t)

        model_shift += rest_shift

        for i, oscs in enumerate(oscillators):
            oscillators[i] = [osc + rest_shift for osc in oscs]
        spectra = [(spectrum + rest_shift) for spectrum in spectra]
        models = [(model + rest_shift) for model in models]

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
            figsize=figure_size,
        )
        axs = np.expand_dims(axs, axis=0)

        for ax, shifts_, residual in zip(axs[0], shifts, residuals):
            ax.plot(shifts_, residual, color="#808080")

        for ax, shifts_, spectrum in zip(axs[0], shifts, spectra):
            ax.plot(shifts_, spectrum, color="#000000")

        for ax, shifts_hr, model in zip(axs[0], shifts_highres, models):
            ax.plot(shifts_hr, model, color="#808080")

        noscs = len(oscillators[0])
        for ax, shifts_hr, oscs in zip(axs[0], shifts_highres, oscillators):
            colors = make_color_cycle(oscillator_colors, noscs)
            for osc in oscs:
                color = next(colors)
                ax.plot(shifts_hr, osc, color=color)

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
            ax.set_yticks([])
            ax.set_ylim(bottom, top)

        return fig, axs
