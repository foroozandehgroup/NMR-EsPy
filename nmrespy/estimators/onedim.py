# onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 24 May 2023 10:59:32 BST

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import nmrespy as ne
from nmrespy.load import load_bruker
from nmrespy.plot import make_color_cycle

from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._files import check_existent_dir, check_saveable_dir
from nmrespy._misc import proc_kwargs_dict
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)

from nmrespy.estimators import logger
from nmrespy.estimators._proc_onedim import _Estimator1DProc


if USE_COLORAMA:
    import colorama
    colorama.init()


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

    dim = 1
    twodim_dtype = None
    proc_dims = [0]
    ft_dims = [0]
    default_mpm_trim = [4096]
    default_nlp_trim = [None]
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
            data = (2 * ne.sig.ift(data))[slice_]

        elif convdta:
            grpdly = expinfo.parameters["acqus"]["GRPDLY"]
            data = ne.sig.convdta(data, grpdly)

        return cls(data, expinfo, directory)

    @classmethod
    def new_spinach(
        cls,
        shifts: Iterable[float],
        couplings: Optional[Iterable[Tuple(int, int, float)]],
        pts: int,
        sw: float,
        offset: float = 0.,
        field: float = 11.74,
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
            The magnetic field strength (T).

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
        sanity_check(
            ("shifts", shifts, sfuncs.check_float_list),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}),
            ("sw", sw, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("offset", offset, sfuncs.check_float),
            ("field", field, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("nucleus", nucleus, sfuncs.check_nucleus),
            ("snr", snr, sfuncs.check_float, (), {}, True),
            ("lb", lb, sfuncs.check_float, (), {"greater_than_zero": True})
        )
        nspins = len(shifts)
        sanity_check(
            ("couplings", couplings, sfuncs.check_spinach_couplings, (nspins,), {}, True),  # noqa: E501
        )

        if couplings is None:
            couplings = []

        fid, sfo = cls._run_spinach(
            "onedim_sim", shifts, couplings, pts, sw, offset, field, nucleus,
        )
        fid = np.array(fid).flatten()

        if snr is not None:
            fid = ne.sig.add_noise(fid, snr)

        fid = ne.sig.exp_apodisation(fid, lb)

        expinfo = ne.ExpInfo(
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

        expinfo = ne.ExpInfo(
            dim=1,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nucleus,
            default_pts=pts,
        )

        data = expinfo.make_fid(params, snr=snr)
        return cls(data, expinfo)

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
            y = self.spectrum
            label, = self._axis_freq_labels(freq_unit)
        elif domain == "time":
            x, = self.get_timepoints()
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
            See :ref:`INDICES`.

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
        # calls ne.ExpInfo.write_to_bruker()
        super().write_to_bruker(fid, path, expno, 1, force_overwrite)

    @logger
    def plot_result(
        self,
        indices: Optional[Iterable[int]] = None,
        high_resolution_pts: Optional[int] = None,
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
        residual_shift: Optional[float] = None,
        label_peaks: bool = True,
        denote_regions: bool = False,
        spectrum_line_kwargs: Optional[Dict] = None,
        oscillator_line_kwargs: Optional[Dict] = None,
        residual_line_kwargs: Optional[Dict] = None,
        model_line_kwargs: Optional[Dict] = None,
        label_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, np.ndarray[mpl.axes.Axes]]:
        r"""Generate a figure of the estimation result.

        Parameters
        ----------
        indices
            See :ref:`INDICES`.

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
            Describes how to color individual oscillators. See :ref:`COLOR_CYCLE`
            for details.

        plot_model
            .. todo::

                Add description

        plot_residual
            .. todo::

                Add description

        model_shift
            The vertical displacement of the model relative to the data.

        residual_shift
            The vertical displacement of the residaul relative to the data.

        label_peaks
            If True, label peaks according to their index. The parameters of a peak
            denoted with the label ``i`` in the figure can be accessed with
            ``self.get_results(indices)[i]``.

        denote_regions
            If ``True``, and there are regions which share a boundary, a
            vertical line will be plotted to show the boundary.

        spectrum_line_kwargs
            Keyword arguments for the spectrum line. All keys should be valid
            arguments for `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.

        oscillator_line_kwargs
            Keyword arguments for the oscillator lines. All keys should be valid
            arguments for `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.
            If ``"color"`` is included, it is ignored (colors are processed
            based on the ``oscillator_colors`` argument.

        residual_line_kwargs
            Keyword arguments for the residual line (if included). All keys
            should be valid arguments for `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.

        model_line_kwargs
            Keyword arguments for the model line (if included). All keys should
            be valid arguments for `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.

        label_kwargs
            Keyword arguments for oscillator labels. All keys should be valid
            arguments for
            `matplotlib.text.Text
            <https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text>`_
            If ``"color"`` is included, it is ignored (colors are procecessed
            based on the ``oscillator_colors`` argument.
            ``"horizontalalignment"``, ``"ha"``, ``"verticalalignment"``, and
            ``"va"`` are also ignored, as these are determined internally.

        kwargs
            Keyword arguments provided to `matplotlib.pyplot.figure
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html\
            #matplotlib.pyplot.figure>`_\.

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
            self._funit_check(xaxis_unit, "xaxis_unit"),
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

        spectrum_line_kwargs = proc_kwargs_dict(
            spectrum_line_kwargs,
            default={"color": "#000000"},
        )
        oscillator_line_kwargs = proc_kwargs_dict(
            oscillator_line_kwargs,
            to_pop=("color",)
        )
        if plot_residual:
            residual_line_kwargs = proc_kwargs_dict(
                residual_line_kwargs,
                default={"color": "#808080"},
            )
        if plot_model:
            model_line_kwargs = proc_kwargs_dict(
                model_line_kwargs,
                default={"color": "#808080"},
            )
        if label_peaks:
            label_kwargs = proc_kwargs_dict(
                label_kwargs,
                to_pop=("ha", "horizontalalignment", "va", "verticalalignment", "color"),  # noqa: E501
            )

        indices = self._process_indices(indices)
        merge_indices, merge_regions = self._plot_regions(indices, xaxis_unit)
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
            xaxis_unit,
        )

        # Configure high-resolutions points for oscillator and model plots
        if high_resolution_pts is None:
            high_resolution_pts = self.data.size
        highres_expinfo = self.expinfo
        highres_expinfo.default_pts = (high_resolution_pts,)

        # Get data which spans full spectral width.
        # These will be sliced for each region.
        full_spectrum = self.spectrum.real
        full_shifts_highres, = highres_expinfo.get_shifts(unit=xaxis_unit)
        full_shifts, = self.get_shifts(unit=xaxis_unit)
        full_model = self.make_fid_from_result(indices)
        full_model[0] *= 0.5
        full_model = ne.sig.ft(full_model).real
        full_residual = full_spectrum - full_model
        full_model_highres = self.make_fid_from_result(indices, pts=high_resolution_pts)
        full_model_highres[0] *= 0.5
        full_model_highres = ne.sig.ft(full_model_highres).real

        slices = [
            slice(
                *self.convert([region], f"{xaxis_unit}->idx")[0]
            ) for region in merge_regions
        ]
        highres_slices = [
            slice(
                *highres_expinfo.convert([region], f"{xaxis_unit}->idx")[0]
            ) for region in merge_regions
        ]

        params = self.get_params(indices=indices)
        label_ax_idxs = []
        for idx in merge_indices:
            vals = []
            ps = self.get_params(indices=idx)
            for i, p in enumerate(params):
                if len(np.where((ps == p).all(axis=-1))[0]) == 1:
                    vals.append(i)
            label_ax_idxs.append(vals)

        n_oscs = params.shape[0]

        # Store line and text objects.
        # Will be shifting these vertically later on
        spectra = []
        oscs = []

        if label_peaks:
            labels = []

        if plot_model:
            models = []

        if plot_residual:
            residuals = []

        for ax, slce, highres_slice, ax_labels in zip(axs[0], slices, highres_slices, label_ax_idxs):  # noqa: E501
            shifts = full_shifts[slce]
            shifts_highres = full_shifts_highres[highres_slice]
            spectrum = full_spectrum[slce]
            spectra.append(ax.plot(shifts, spectrum, **spectrum_line_kwargs)[0])

            if plot_residual:
                residual = full_residual[slce]
                residuals.append(ax.plot(shifts, residual, **residual_line_kwargs)[0])

            if plot_model:
                model = full_model[slce]
                models.append(ax.plot(shifts, model, **model_line_kwargs)[0])

            colorcycle = make_color_cycle(oscillator_colors, n_oscs)
            for i, p in enumerate(params):
                color = next(colorcycle)
                p = np.expand_dims(p, axis=0)
                osc = self.make_fid(p, pts=high_resolution_pts)
                osc[0] *= 0.5
                spec = ne.sig.ft(osc).real[highres_slice]
                oscs.append(ax.plot(shifts_highres, spec, color=color, **oscillator_line_kwargs)[0])  # noqa: E501

                if label_peaks and (i in ax_labels):
                    label_idx = np.argmax(np.abs(spec))
                    label_x = shifts_highres[label_idx]
                    label_y = spec[label_idx]
                    label_va, label_ha = (
                        ("bottom", "left") if spec[label_idx] >= 0
                        else ("top", "right")
                    )
                    labels.append(
                        ax.text(
                            label_x, label_y, str(i), color=color, va=label_va,
                            ha=label_ha, **label_kwargs,
                        )
                    )

        # Vertical shifting of plot lines and labels
        if plot_model and (model_shift is None):
            model_shift = 0.1 * max(
                [np.amax(spectrum.get_ydata()) for spectrum in spectra]
            )

        if plot_residual:
            residual_span = self._get_line_span(residuals)

            lines_to_shift = oscs + spectra
            if plot_model:
                lines_to_shift.extend(models)

            lines_to_shift_span = self._get_line_span(lines_to_shift)
            if residual_shift is None:
                top = (
                    (residual_span[1] - residual_span[0]) +
                    (lines_to_shift_span[1] - lines_to_shift_span[0])
                ) / 0.91
                line_shift = residual_span[1] - lines_to_shift_span[0] + (0.03 * top)

            else:
                line_shift = residual_shift

            for line in lines_to_shift:
                line.set_ydata(line.get_ydata() + line_shift)

            if label_peaks:
                for label in labels:
                    old_pos = label.get_position()
                    new_pos = (old_pos[0], old_pos[1] + line_shift)
                    label.set_position(new_pos)

            if plot_model:
                for model in models:
                    model.set_ydata(model.get_ydata() + model_shift)

        # Set y-limit
        all_lines = oscs + spectra
        if plot_model:
            all_lines.extend(models)
        if plot_residual:
            all_lines.extend(residuals)
        all_lines_span = self._get_line_span(all_lines)
        height = all_lines_span[1] - all_lines_span[0]
        bottom = all_lines_span[0] - (0.03 * height)
        top = all_lines_span[1] + (0.03 * height)

        for ax in axs[0]:
            ax.set_ylim(bottom, top)
            ax.set_yticks([])

        return fig, axs

    @staticmethod
    def _get_line_span(lines: Iterable[mpl.lines.Line2D]) -> Tuple[float, float]:
        return (
            min([np.amin(line.get_ydata()) for line in lines]),
            max([np.amax(line.get_ydata()) for line in lines]),
        )
