# plot.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 03 May 2022 15:42:56 BST

"""Module for plotting estimation results."""

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.rcsetup import cycler

import numpy as np

from nmrespy import ExpInfo, sig
from nmrespy._colors import END, GRE, RED, USE_COLORAMA
from nmrespy._files import (
    check_saveable_path,
    configure_path,
)
from nmrespy._paths_and_links import STYLESHEETPATH
import nmrespy._errors as errors
from nmrespy._sanity import sanity_check, funcs as sfuncs

if USE_COLORAMA:
    import colorama
    colorama.init()


def check_axes(obj: Any) -> Optional[str]:
    if not isinstance(obj, mpl.axes.Axes):
        return "Should be an instance of matplotlib.axes.Axes or one of its children."


def _to_hex(color: Any) -> Optional[str]:
    r"""Attempt to convert color into a hexadecimal RGBA string.

    If an invalid RGBA argument is given, ``None`` is returned.

    Parameters
    ----------
    color
        Object to attempt to convert to a color.
    """
    try:
        return mcolors.to_hex(color).lower()
    except ValueError:
        return None


class ResultPlotter(ExpInfo):
    """Class for generating figures of estimation results."""

    def __init__(
        self,
        data: np.ndarray,
        result: np.ndarray,
        expinfo: ExpInfo,
        *,
        plot_residual: bool = True,
        plot_model: bool = False,
        residual_shift: Union[float, None] = None,
        model_shift: Union[float, None] = None,
        shifts_unit: str = "ppm",
        region: Union[Iterable[Tuple[float, float]], None] = None,
        data_color: Any = "#000000",
        residual_color: Any = "#808080",
        model_color: Any = "#808080",
        oscillator_colors: Any = None,
        show_labels: bool = True,
        stylesheet: Union[str, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        data
            Time-domain data.

        result
            Parameter estimate, of form:

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
            Experiment information.

        shifts_unit
            Units to display chemical shifts in. Must be either ``'ppm'`` or
            ``'hz'``.

        region
            Boundaries specifying the region to show. **N.B. the units**
            ``region`` **is given in should match** ``shifts_unit`` **.**

        plot_model
            If ``True``, plot the model generated using ``result``. This model is
            a summation of all oscillator present in the result.

        plot_residual
            If ``True``, plot the difference between the data and the model
            generated using ``result``.

        model_shift
            Specifies a translation of the residual plot along the y-axis. If
            ``None``, a default shift will be applied.

        residual_shift
            Specifies a translation of the residual plot along the y-axis. If
            ``None``, a default shift will be applied.

        data_color
            The colour used to plot the data. Any value that is recognised by
            matplotlib as a color is permitted. See `here
            <https://matplotlib.org/stable/tutorials/colors/colors.html>`_ for
            a full description of valid values.

        residual_color
            The colour used to plot the residual. See ``data_color`` for a
            description of valid colors.

        model_color
            The colour used to plot the model. See ``data_color`` for a
            description of valid colors.

        oscillator_colors
            Describes how to color individual oscillators. The following
            is a complete list of options:

            * If a valid matplotlib color is given, all oscillators will
              be given this color.
            * If a string corresponding to a matplotlib colormap is given,
              the oscillators will be consecutively shaded by linear increments
              of this colormap. For all valid colormaps, see
              `here <https://matplotlib.org/stable/tutorials/colors/\
              colormaps.html>`__
            * If an iterable object containing valid matplotlib colors is
              given, these colors will be cycled.
              For example, if ``oscillator_colors = ['r', 'g', 'b']``:

              + Oscillators 1, 4, 7, ... would be :red:`red (#FF0000)`
              + Oscillators 2, 5, 8, ... would be :green:`green (#008000)`
              + Oscillators 3, 6, 9, ... would be :blue:`blue (#0000FF)`

            * If ``None``, the default colouring method will be applied, which
              involves cycling through the following colors:

                - :oscblue:`#1063E0`
                - :oscorange:`#EB9310`
                - :oscgreen:`#2BB539`
                - :oscred:`#D4200C`

        show_labels
            If ``True``, each oscillator will be given a numerical label
            in the plot, if ``False``, the labels will be hidden.

        stylesheet
            The name of/path to a matplotlib stylesheet for further
            customaisation of the plot. See `here <https://matplotlib.org/\
            stable/tutorials/introductory/customizing.html>`__ for more
            information on stylesheets.
        """
        sanity_check(
            ("expinfo", expinfo, sfuncs.check_expinfo),
            ("plot_residual", plot_residual, sfuncs.check_bool),
            ("plot_model", plot_model, sfuncs.check_bool),
            ("residual_shift", residual_shift, sfuncs.check_float, (), {}, True),
            ("model_shift", model_shift, sfuncs.check_float, (), {}, True),
            ("shifts_unit", shifts_unit, sfuncs.check_frequency_unit, (expinfo.sfo is not None,)),  # noqa: E501
            ("data_color", data_color, sfuncs.check_mpl_color),
            ("residual_color", residual_color, sfuncs.check_mpl_color),
            ("model_color", model_color, sfuncs.check_mpl_color),
            ("oscillator_colors", oscillator_colors, sfuncs.check_oscillator_colors, (), {}, True),  # noqa: E501
            ("show_labels", show_labels, sfuncs.check_bool),
            ("stylesheet", stylesheet, sfuncs.check_stylesheet, (), {}, True),
        )
        sanity_check(
            ("data", data, sfuncs.check_ndarray, (expinfo.dim,)),
            ("result", result, sfuncs.check_parameter_array, (expinfo.dim,)),
        )

        sanity_check(
            (
                "region", region, sfuncs.check_region,
                (expinfo.sw(shifts_unit), expinfo.offset(shifts_unit)), {}, True,
            ),
        )

        if data.ndim == 2:
            raise errors.TwoDimUnsupportedError()
        elif data.ndim >= 3:
            raise errors.MoreThanTwoDimError()

        super().__init__(
            dim=data.ndim,
            sw=expinfo.sw(),
            offset=expinfo.offset(),
            sfo=expinfo.sfo,
            nuclei=expinfo.nuclei,
            default_pts=data.shape,
        )

        self.result = result
        self.data = data
        self.plot_residual = plot_residual
        self.plot_model = plot_model
        self.data_color = data_color
        self.model_color = model_color
        self.residual_color = residual_color
        self.model_shift = model_shift
        self.residual_shift = residual_shift

        if stylesheet is None:
            stylesheet = STYLESHEETPATH
        self._update_rc(stylesheet, oscillator_colors)

        if region is None:
            region_idx = tuple([(0, s - 1) for s in data.shape])
        else:
            region_idx = self.convert(region, f"{shifts_unit}->idx")

        self.slice_ = tuple([slice(r[0], r[1] + 1) for r in region_idx])

        self.shifts = self.get_shifts(unit=shifts_unit)[0][self.slice_]
        self._make_ydata()

        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0.02, 0.15, 0.96, 0.83])
        self._create_artists()

        label = (
            f"{self.unicode_nuclei[0]} ({shifts_unit.replace('h', 'H')})"
            if self.unicode_nuclei is not None
            else shifts_unit.replace('h', 'H')
        )
        self.ax.set_xlabel(label)
        self.configure_lims()

    def _make_ydata(self) -> None:
        datacopy = copy.deepcopy(self.data)
        datacopy[0] /= 2
        self.spectrum = sig.ft(datacopy)[self.slice_].real

        self.peaks = []
        for osc in self.result:
            fid = self.make_fid(np.expand_dims(osc, axis=0))
            fid[0] /= 2
            self.peaks.append(sig.ft(fid).real[self.slice_])

        self.model = sum(self.peaks, np.zeros(self.spectrum.shape)) + (
            0.1 * np.amax(self.spectrum) if self.model_shift is None
            else self.model_shift
        )
        self.residual = self.spectrum - self.model
        self.residual += (
            -1.5 * np.amax(np.abs(self.residual)) if self.residual_shift is None
            else self.residual_shift
        )

    def _create_artists(self) -> None:
        # Clear out any children already present
        for child in self.ax.get_children():
            if (
                isinstance(child, mpl.lines.Line2D) or
                isinstance(child, mpl.text.Text) and child.get_text() != ""
            ):
                child.remove()

        self.data_plot = self.ax.plot(
            self.shifts, self.spectrum, self.data_color,
        )[0]

        self.model_plot = self.ax.plot(
            self.shifts,
            self.model,
            color=self.model_color,
            alpha=1 if self.plot_model else 0
        )[0]

        self.residual_plot = self.ax.plot(
            self.shifts,
            self.residual,
            self.residual_color,
            alpha=1 if self.plot_residual else 0
        )[0]

        self.oscillator_plots = [
            {
                "label": self.ax.text(
                    self.shifts[np.argmax(peak)],
                    peak[np.argmax(peak)],
                    f"{i + 1}",
                    verticalalignment="center",
                ),
                "line": self.ax.plot(
                    self.shifts,
                    peak,
                )[0],
            } for i, peak in enumerate(self.peaks)
        ]

    def _update_rc(
        self,
        stylesheet: Union[str, Path],
        oscillator_colors: Any,
    ) -> Dict:
        """Construct rc for the result plot.

        Parameters
        ----------
        stylesheet
            Specification of stylesheet to extract the rc from.

        oscillator_colors
            Specification of how to color oscillator peaks.
        """
        paths = [
            stylesheet,
            Path(mpl.__file__).resolve().parent /
            f"mpl-data/stylelib/{stylesheet}.mplstyle",
        ]
        for path in paths:
            try:
                mpl.rc_file(path, use_default_template=True)
                break
            except FileNotFoundError:
                continue

        osc_cols = self._process_oscillator_colors(oscillator_colors)
        mpl.rcParams["axes.prop_cycle"] = cycler("color", osc_cols)

    def _process_oscillator_colors(self, oscillator_colors: Any) -> Iterable[str]:
        """Attempt to convert oscillator color input into a string of hex values.

        Parameters
        ----------
        oscillator_colors
            Input to convert to a list of hexadecimal colors.
        """
        if oscillator_colors is None:
            return ["#1063e0", "#eb9310", "#2bb539", "#d4200c"]
        if oscillator_colors in plt.colormaps():
            return [
                _to_hex(c) for c in
                cm.get_cmap(oscillator_colors)(np.linspace(0, 1, self.result.shape[0]))
            ]
        if isinstance(oscillator_colors, str):
            oscillator_colors = [oscillator_colors]
        return [_to_hex(c) for c in oscillator_colors]

    def configure_lims(self, xpad: float = 0.0, ypad: float = 0.03) -> None:
        xpad = xpad * (self.shifts[0] - self.shifts[1])
        self.ax.set_xlim((self.shifts[0] + xpad, self.shifts[-1] - xpad))

        lines = [line.get_ydata() for line in self.lines if line.get_alpha() != 0]
        ymin = min([np.amin(line) for line in lines])
        ymax = max([np.amax(line) for line in lines])
        ypad = ypad * (ymax - ymin)
        self.ax.set_ylim((ymin - ypad, ymax + ypad))

    def save(
        self,
        path: Union[str, Path],
        fmt: str = "png",
        dpi: int = 400,
        force_overwrite: bool = False,
        fprint: bool = True,
    ) -> None:
        """Save the result figure.

        Parameters
        ----------
        path
            Path to save the figure to. Suffix does not need to be included.

        fmt
            File format. Must be one of
            ``"svgz"``, ``"ps"``, ``"emf"``, ``"rgba"``, ``"raw"``, ``"pdf"``,
            ``"svg"``, ``"eps"``, ``"png"``

        dpi
            Dots per inch.

        force_overwrite
            If ``False``, and the path provided to save the file to already exists,
            the user is prompted to indicate whether they are happy to overwrite.
            If ``True``, the file will always be overwritten.

        fprint
            Whether or not to output information to the terminal.
        """
        sanity_check(
            (
                "fmt", fmt, sfuncs.check_one_of,
                ("svgz", "ps", "emf", "rgba", "raw", "pdf", "svg", "eps", "png"),
            ),
            ("dpi", dpi, sfuncs.check_int, (), {"min_value": 10}),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            ("fprint", fprint, sfuncs.check_bool),
        )
        sanity_check(
            ("path", path, check_saveable_path, (fmt, force_overwrite)),
        )

        path = configure_path(path, fmt)
        self.fig.savefig(path, dpi=dpi)

        if fprint:
            print(f"{GRE}Saved file {path}{END}")

    @property
    def lines(self) -> Iterable[mpl.lines.Line2D]:
        return (
            [self.data_plot, self.model_plot, self.residual_plot] +
            [osc["line"] for osc in self.oscillator_plots]
        )

    @property
    def labels(self) -> Iterable[mpl.text.Text]:
        return [osc["label"] for osc in self.oscillator_plots]

    def show_labels(self) -> None:
        """Make the oscillator labels visible"""
        for label in self.labels:
            label.set_alpha(1)

    def hide_labels(self) -> None:
        """Make the oscillator labels visible"""
        for label in self.labels:
            label.set_alpha(0)

    def displace_labels(
        self,
        labels: Iterable[int],
        displacement: Tuple[float, float],
    ):
        """Displace labels.

        Often the default location of the labels can be undesirable. This method
        aims to make it simple to re-position the labels.

        Parameters
        ----------
        labels
            The identities of the labels to displace.

        displacement
            The amount to displace the labels relative to their current
            positions in the x- and y-directions, respectively. The displacement
            uses the `axes co-ordinate system <https://matplotlib.org/stable/\
            tutorials/advanced/transforms_tutorial.html#axes-coordinates>`_.
            Both values provided should be less than ``1.0``.  """
        sanity_check(
            (
                "labels", labels, sfuncs.check_int_list,
                (), {"must_be_positive": True, "max_value": len(self.labels)},
            ),
            (
                "displacement", displacement, sfuncs.check_float_list, (),
                {"length": 2},
            ),
        )

        for label in labels:
            # Get initial position (this is in data coordinates)
            init_pos = self.labels[label - 1].get_position()
            # Converter from data coordinate system to axis coordinate system
            axis_to_data = self.ax.transAxes + self.ax.transData.inverted()
            # Converter from axis coordinate system to data coordinate system
            data_to_axis = axis_to_data.inverted()
            # Convert initial position to axis coordinates
            init_pos = data_to_axis.transform(init_pos)

            # Add displacement
            new_pos = tuple(init + disp for init, disp in zip(init_pos, displacement))
            if not all([0.0 <= coord <= 1.0 for coord in new_pos]):
                raise ValueError(
                    f"{RED}The specified displacement for label {label} "
                    "places it outside the axes! You may want to reduce the "
                    "magnitude of displacement in one or both dimesions to "
                    f"ensure this does not occur.{END}"
                )

            # Transform new position to data coordinates
            new_pos = axis_to_data.transform(new_pos)
            # Update position
            self.labels[label - 1].set_position(new_pos)

    def transfer_to_axes(self, ax: mpl.axes.Axes) -> None:
        """Reproduces the plot in ``self.ax`` in another axes object.

        Parameters
        ----------
        ax
            The axes object to construct the result plot onto.

        Warning
        -------
        Everything present in ``ax`` before calling the method will be
        removed. If you want to add further things to ``ax``, do it after calling
        this method.
        """
        sanity_check(("ax", ax, check_axes))
        ax.clear()

        for child in self.ax.get_children():
            if isinstance(child, mpl.lines.Line2D):
                ax.plot(
                    child.get_xdata(),
                    child.get_ydata(),
                    color=child.get_color(),
                    lw=child.get_lw(),
                    alpha=child.get_alpha(),
                )
            elif isinstance(child, mpl.text.Text):
                ax.text(
                    *child.get_position(),
                    child.get_text(),
                    fontproperties=child.get_fontproperties(),
                    color=child.get_color(),
                    alpha=child.get_alpha(),
                )

        # Set ticks
        ax.set_xticks(self.ax.get_xticks())
        ax.set_yticks(self.ax.get_yticks())

        # Set correct x- and y-limits
        ax.set_xlim(self.ax.get_xlim())
        ax.set_ylim(self.ax.get_ylim())

        # Set x-label
        ax.set_xlabel(self.ax.get_xlabel())
