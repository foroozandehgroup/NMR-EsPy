# plot.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Support for plotting estimation results"""

from collections.abc import Iterable
from pathlib import Path
import re
import tempfile

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import numpy as np

from nmrespy import *
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
    colorama.init()
import nmrespy._errors as errors
from nmrespy._misc import *
from nmrespy import sig


def _to_hex(color):
    """Attempts to convert color into a hex. If an invalid RGBA argument is
    given, None is returned, rather then an error call"""
    try:
        return mcolors.to_hex(color)
    except ValueError:
        return None


def _configure_oscillator_colors(oscillator_colors, m):
    # Determine oscillator colours to use
    if oscillator_colors is None:
        return ['#1063e0', '#eb9310', '#2bb539', '#d4200c']

    if oscillator_colors in plt.colormaps():
        return [_to_hex(c) for c in
                cm.get_cmap(oscillator_colors)(np.linspace(0, 1, m))]

    if isinstance(oscillator_colors, str):
        oscillator_colors = [oscillator_colors]

    osc_cols = [_to_hex(c) for c in oscillator_colors]
    nones = [i for i, c in enumerate(osc_cols) if c is None]
    if nones:
        msg = (
            f'{cols.R}The following entries in `oscillator_colors` could '
            f'not be recognised as valid colours in matplotlib:\n'
            + '\n'.join([f'--> repr({oscillator_colors[i]})' for i in nones])
            + cols.END
        )
        raise ValueError(msg)

    return osc_cols


def _get_rc_from_file(path):
    try:
        rc = str(mpl.rc_params_from_file(
            path, fail_on_error=True, use_default_template=False
        ))
        # If the file exists, but no lines can be parsed, an empty
        # string is returned.
        return rc if rc else None

    except FileNotFoundError:
        return None


def _extract_rc(stylesheet):
    # Default style sheet if one isn't explicitly given
    if stylesheet is None:
        stylesheet = NMRESPYPATH / 'config/nmrespy_custom.mplstyle'

    # Check two possible paths.
    # First one is simply the user input:
    # This will be valid if a full path to a stylesheet has been given.
    # Second one is to check whether the user has given a name for one of
    # the provided stylesheets that ship with matplotlib.
    paths = [
        stylesheet,
        (Path(mpl.__file__).resolve().parent / "mpl-data/stylelib" /
         f"{stylesheet}.mplstyle")
    ]

    for path in paths:
        rc = _get_rc_from_file(path)
        if rc:
            return rc

    raise ValueError(
        f'{cols.R}Error in loading the stylesheet. Check you gave '
        'a valid path or name for the stylesheet, and that the '
        f'stylesheet is formatted correctly.{cols.END}'
    )


def _create_rc(stylesheet, oscillator_colors, m):
    rc = _extract_rc(stylesheet)
    osc_cols = _configure_oscillator_colors(oscillator_colors, m)
    col_txt = ', '.join([f'\'{c.lower()}\'' for c in osc_cols])
    # Overwrite the line containing axes.prop_cycle
    rc = '\n'.join(
        filter(lambda ln: 'axes.prop_cycle' not in ln, rc.split('\n'))
    ) + f'\naxes.prop_cycle: cycler(\'color\', [{col_txt}])\n'
    # Seem to be getting bugs when using stylesheet with any hex colours
    # that have a # in front. Remove these.
    rc = re.sub(r'#([0-9a-fA-F]{6})', r'\1', rc)
    return rc


def _create_stylesheet(rc):
    # Temporary path to save stylesheet to
    tmp_path = Path(tempfile.gettempdir()).resolve() / 'stylesheet.mplstyle'
    with open(tmp_path, 'w') as fh:
        fh.write(rc)
    return tmp_path


def _configure_shifts_unit(shifts_unit, sfo):
    if shifts_unit not in ['hz', 'ppm']:
        raise errors.InvalidUnitError('hz', 'ppm')
    if shifts_unit == 'ppm' and sfo is None:
        shifts_unit = 'hz'
        print(
            f'{cols.OR}You need to specify `sfo` if you want chemical'
            f' shifts in ppm! Falling back to Hz...{cols.END}'
        )
    return shifts_unit


def _get_region_slice(shifts_unit, region, n, sw, offset, sfo):
    if region is None:
        return tuple(slice(0, n_, None) for n_ in n)

    limits = []
    if isinstance(sfo, Iterable):
        for (bound, n_, sw_, offset_, sfo_) in zip(region, n, sw, offset, sfo):
            converter = FrequencyConverter([n_], [sw_], [offset_], sfo=[sfo_])
            bound = converter.convert([bound], f'{shifts_unit}->idx')[0]
            limits.append([min(bound), max(bound)])

    else:
        for (bounds, n_, sw_, offset_) in zip(region, n, sw, offset):
            converter = FrequencyConverter([n_], [sw_], [offset_])
            bounds = converter.convert([bounds], f'{shifts_unit}->idx')[0]
            limits.append([min(bounds), max(bounds)])

    return tuple(slice(x[0], x[1] + 1, None) for x in limits)


def _generate_peaks(result, n, sw, offset, region_slice):
    return [np.real(sig.ft(sig.make_fid(
            np.expand_dims(oscillator, axis=0), n, sw, offset=offset
            )[0]))[region_slice]
            for oscillator in result]


def _generate_model_and_residual(peaks, spectrum):
    model = sum(peaks)
    return model, spectrum - model


def _plot_oscillators(lines, labels, ax, shifts, peaks, show_labels):
    label_alpha = int(show_labels)
    for m, peak in enumerate(peaks, start=1):
        _plot_spectrum(lines, m, ax, shifts, peak)
        idx = np.argmax(np.absolute(peak))
        x, y = shifts[idx], peak[idx]
        labels[m] = ax.text(x, y, str(m), fontsize=8, alpha=label_alpha)


def _plot_spectrum(lines, name, ax, shifts, spectrum, color=None, show=True):
    lines[name] = ax.plot(shifts, spectrum, color=color, alpha=int(show))[0]


def _process_yshift(data, yshift, scale):
    if yshift:
        return yshift
    else:
        return scale * np.max(np.absolute(data))


def _set_axis_limits(ax, lines):
    # Flip the x-axis
    ax.set_xlim(reversed(ax.get_xlim()))
    # y-limits
    ydatas = [ln.get_ydata() for ln in lines.values() if ln.get_alpha() != 0]
    maxi = max([np.amax(ydata) for ydata in ydatas])
    mini = min([np.amin(ydata) for ydata in ydatas])
    vertical_span = maxi - mini
    bottom = mini - (0.03 * vertical_span)
    top = maxi + (0.03 * vertical_span)
    ax.set_ylim(bottom, top)


def _set_xaxis_label(ax, nucleus, shifts_unit):
    # TODO Only works for 1D data ATM
    # Produces a label of form ¹H (Hz) or ¹³C (ppm) etc.
    nuc = latex_nucleus(nucleus[0]) if nucleus else 'chemical shift'
    unit = '(Hz)' if shifts_unit == 'hz' else '(ppm)'
    ax.set_xlabel(f'{nuc} {unit}')


def plot_result(
    data, result, sw, offset, plot_residual=True, plot_model=False,
    residual_shift=None, model_shift=None, sfo=None, shifts_unit='ppm',
    nucleus=None, region=None, data_color='#000000', residual_color='#808080',
    model_color='#808080', oscillator_colors=None,
    show_labels=True, stylesheet=None,
):
    """
    Produces a figure of an estimation result.

    The figure consists of the original data, in the Fourier domain, along
    with each oscillator.

    Parameters
    ----------
    data : numpy.ndarray
        Data of interest (in the time-domain).

    result : numpy.ndarray
        Parameter estimate, of form:

        * **1-dimensional data:**

          .. code:: python3

             parameters = numpy.array([
                [a_1, φ_1, f_1, η_1],
                [a_2, φ_2, f_2, η_2],
                ...,
                [a_m, φ_m, f_m, η_m],
             ])

        * **2-dimensional data:**

          .. code:: python3

             parameters = numpy.array([
                [a_1, φ_1, f1_1, f2_1, η1_1, η2_1],
                [a_2, φ_2, f1_2, f2_2, η1_2, η2_2],
                ...,
                [a_m, φ_m, f1_m, f2_m, η1_m, η2_m],
             ])

    sw : [float] or [float, float]
        Sweep width in each dimension (Hz).

    offset : [float] or [float, float]
        Transmitter offset in each dimension (Hz).

    sfo : [float, [float, float] or None, default: None
        Transmitter frequency in each dimnesion (MHz). Needed to plot the
        chemical shift axis in ppm. If `None`, chemical shifts will be plotted
        in Hz.

    shifts_unit : {'ppm', 'hz'}, default: 'ppm'
        Units to display chemical shifts in. If this is set to `'ppm'` but
        `sfo` is not specified, it will revert to `'hz'`.

    nucleus : [str], [str, str] or None, default: None
        The nucleus in each dimension.

    region : [[int, int]], [[float, float]], [[int, int], [int, int]] or \
    [[float, float], [float, float]]
        Boundaries specifying the region to show. See also
        :py:class:`nmrespy.freqfilter.FrequencyFilter`. **N.B. the units
        `region` is given in should match `shifts_unit`.**

    plot_residual : bool, default: True
        If `True`, plot a difference between the FT of `data` and the FT of
        the model generated using `result`. NB the residual is plotted
        regardless of `plot_residual`. `plot_residual` specifies the alpha
        transparency of the plot line (1 for `True`, 0 for `False`)

    residual_shift : float or None, default: None
        Specifies a translation of the residual plot along the y-axis. If
        `None`, the default shift will be applied.

    plot_model : bool, default: False
        If `True`, plot the FT of the model generated using `result`.
        NB the residual is plotted regardless of `plot_model`. `plot_model`
        specifies the alpha transparency of the plot line (1 for `True`,
        0 for `False`)

    model_shift : float or None, default: None
        Specifies a translation of the residual plot along the y-axis. If
        `None`, the default shift will be applied.

    data_color : matplotlib color, default: '#000000'
        The colour used to plot the original data. Any value that is
        recognised by matplotlib as a color is permitted. See
        `<here https://matplotlib.org/3.1.0/tutorials/colors/\
        colors.html>`_ for a full description of valid values.

    residual_color : matplotlib color, default: '#808080'
        The colour used to plot the residual.

    model_color : matplotlib color, default: '#808080'
        The colour used to plot the model.

    oscillator_colors : {matplotlib color, matplotlib colormap name, \
    list, numpy.ndarray, None}, default: None
        Describes how to color individual oscillators. The following
        is a complete list of options:

        * If a valid matplotlib color is given, all oscillators will
          be given this color.
        * If a string corresponding to a matplotlib colormap is given,
          the oscillators will be consecutively shaded by linear increments
          of this colormap. For all valid colormaps, see
          `<here https://matplotlib.org/stable/tutorials/colors/\
          colormaps.html>`__
        * If a list or NumPy array containing valid matplotlib colors is
          given, these colors will be cycled.
          For example, if ``oscillator_colors = ['r', 'g', 'b']``:

          + Oscillators 1, 4, 7, ... would be :red:`red (#FF0000)`
          + Oscillators 2, 5, 8, ... would be :green:`green (#008000)`
          + Oscillators 3, 6, 9, ... would be :blue:`blue (#0000FF)`

        * If `None`:

          + If a stylesheet is specified, with the attribute
            ``axes.prop_cycle`` provided, this colour cycle will be used.
          + Otherwise, the default colouring method will be applied,
            which involves cycling through the following colors:

            - :oscblue:`#1063E0`
            - :oscorange:`#EB9310`
            - :oscgreen:`#2BB539`
            - :oscred:`#D4200C`

    show_labels : Bool, default: True
        If `True`, each oscillator will be given a numerical label
        in the plot, if `False`, the labels will be hidden.

    stylesheet : str or None, default: None
        The name of/path to a matplotlib stylesheet for further
        customaisation of the plot. See `<here https://matplotlib.org/\
        stable/tutorials/introductory/customizing.html>`__ for more
        information on stylesheets.

    Returns
    -------
    plot : :py:class:`NmrespyPlot`
        A class instance with the following attributes:

        * fig : `matplotlib.figure.Figure <https://matplotlib.org/3.3.1/\
        api/_as_gen/matplotlib.figure.Figure.html>`_
            The resulting figure.
        * ax : `matplotlib.axes.Axes <https://matplotlib.org/3.3.1/api/\
        axes_api.html#matplotlib.axes.Axes>`_
            The resulting set of axes.
        * lines : dict
            A dictionary containing a series of
            `matplotlib.lines.Line2D <https://matplotlib.org/3.3.1/\
            api/_as_gen/matplotlib.lines.Line2D.html>`_
            instances. The data plot is given the key `0`, and the
            individual oscillator plots are given the keys `1`,
            `2`, `3`, ..., `<M>` where `<M>` is the number of
            oscillators in the parameter estimate.
        * labels : dict
            A dictionary containing a series of
            of `matplotlib.text.Text <https://matplotlib.org/3.1.1/\
            api/text_api.html#matplotlib.text.Text>`_ instances, with the
            keys `1`, `2`, etc. The Boolean argument `labels` affects the
            alpha transparency of the labels:

            + `True` sets alpha to 1 (making the labels visible)
            + `False` sets alpha to 0 (making the labels invisible)
    """

    # --- Check arguments ------------------------------------------------
    try:
        dim = data.ndim
    except Exception:
        raise TypeError(f'{cols.R}`data` should be a NumPy array.{cols.END}')
    if dim == 2:
        raise TwoDimUnsupportedError()
    elif dim >= 3:
        raise MoreThanTwoDimError()

    checker = ArgumentChecker(dim=dim)
    checker.stage(
        (data, 'data', 'ndarray'),
        (result, 'result', 'parameter'),
        (sw, 'sw', 'float_list'),
        (offset, 'offset', 'float_list'),
        (data_color, 'data_color', 'mpl_color'),
        (residual_color, 'residual_color', 'mpl_color'),
        (model_color, 'model_color', 'mpl_color'),
        (show_labels, 'labels', 'bool'),
        (plot_residual, 'plot_residual', 'bool'),
        (plot_model, 'plot_model', 'bool'),
        (sfo, 'sfo', 'float_list', True),
        (nucleus, 'nucleus', 'str_list', True),
        (region, 'region', 'region_float', True),
        (residual_shift, 'residual_shift', 'float', True),
        (model_shift, 'model_shift', 'float', True),
        (oscillator_colors, 'oscillator_colors', 'osc_cols', True),
        (stylesheet, 'stylesheet', 'str', True)
    )
    checker.check()

    # Setup the stylesheet
    rc = _create_rc(stylesheet, oscillator_colors, result.shape[0])
    plt.style.use(_create_stylesheet(rc))

    n = list(data.shape)
    shifts_unit = _configure_shifts_unit(shifts_unit, sfo)
    region_slice = _get_region_slice(shifts_unit, region, n, sw, offset, sfo)
    # Generate data: chemical shifts, data spectrum, oscillator peaks
    shifts = [shifts[slice_] for shifts, slice_ in
              zip(sig.get_shifts(n, sw, offset, sfo=sfo, unit=shifts_unit),
                  region_slice)][0]
    # TODO should replicate that way the spectrum was created (ve for example)
    spectrum = np.real(sig.ft(data))[region_slice]
    peaks = _generate_peaks(result, n, sw, offset, region_slice)
    model, residual = _generate_model_and_residual(peaks, spectrum)

    # Generate figure and axis
    fig = plt.figure()
    ax = fig.add_axes([0.02, 0.15, 0.96, 0.83])
    lines, labels = {}, {}
    # Plot oscillator peaks
    _plot_oscillators(lines, labels, ax, shifts, peaks, show_labels)
    # Plot data
    _plot_spectrum(lines, 'data', ax, shifts, spectrum, color=data_color)
    # Plot model
    model += _process_yshift(model, model_shift, 0.1)
    _plot_spectrum(
        lines, 'model', ax, shifts, model, color=model_color, show=plot_model
    )
    # Plot residual
    residual += _process_yshift(residual, residual_shift, -1.5)
    _plot_spectrum(
        lines, 'residual', ax, shifts, residual, color=residual_color,
        show=plot_residual
    )

    _set_axis_limits(ax, lines)
    _set_xaxis_label(ax, nucleus, shifts_unit)

    return NmrespyPlot(fig, ax, lines, labels)


# TODO: Grow this class: provide functionality for easy tweaking
# of figure
class NmrespyPlot:
    """Plot result class

    .. note::
       The class is very minimal at the moment. I plan to expand its
       functionality in later versions.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure <https://matplotlib.org/3.3.1/\
    api/_as_gen/matplotlib.figure.Figure.html>`_
        Figure.

    ax : `matplotlib.axes._subplots.AxesSubplot <https://matplotlib.org/\
    3.3.1/api/axes_api.html#the-axes-class>`_
        Axes.

    lines : dict
        Lines dictionary.

    labels : dict
        Labels dictionary.

    Notes
    -----
    To save the figure, simply access the `fig` attribute and use the
    `savefig <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot\
    .savefig.html>`_ method:

    .. code :: python3

       >>> plot.fig.savefig(f'example.{ext}', ...)
    """
    def __init__(self, fig, ax, lines, labels):
        self.fig = fig
        self.ax = ax
        self.lines = lines
        self.labels = labels

    def show_labels(self):
        """Make the oscillator labels visible"""
        for k in self.labels.keys():
            self.labels[k].set_alpha(1)

    def hide_labels(self):
        """Make the oscillator labels visible"""
        for k in self.labels.keys():
            self.labels[k].set_alpha(0)

    def displace_labels(self, values, displacement):
        """Displace labels with values given by `values`

        Parameters
        ----------
        values : list
            The value(s) of the label(s) to displace. All elements should be
            ints.

        displacement : (float, float)
            The amount to displace the labels relative to their current
            positions. The displacement uses the
            `axes co-ordinate system <https://matplotlib.org/stable/\
            tutorials/advanced/transforms_tutorial.html#axes-coordinates>`_.
            Both values provided should be less than `1.0`.
        """

        checker = ArgumentChecker()
        checker.stage(
            (values, 'values', 'generic_int_list'),
            (displacement, 'displacement', 'displacement')
        )
        checker.check()

        # Check that all values given are between 1 and number of oscillators
        max_value = max(self.labels.keys())
        if not all(0 < value <= max_value for value in values):
            raise ValueError(f'{cols.R}At least one element in `values` is '
                             'invalid. Ensure that all elements are ints '
                             f'between 1 and {max_value}.{cols.END}')

        for value in values:
            # Get initial position (this is in data coordinates)
            init_pos = self.labels[value].get_position()
            # Converter from data coordinate system to axis coordinate system
            axis_to_data = self.ax.transAxes + self.ax.transData.inverted()
            # Converter from axis coordinate system to data coordinate system
            data_to_axis = axis_to_data.inverted()
            # Convert initial position to axis coordinates
            init_pos = data_to_axis.transform(init_pos)

            # Add displacement
            new_pos = tuple(
                init + disp for init, disp in zip(init_pos, displacement)
            )
            if not all(0. <= coord <= 1. for coord in new_pos):
                raise ValueError(
                    f'{cols.R}The specified displacement for label {value} '
                    'places it outside the axes! You may want to reduce the '
                    'magnitude of displacement in one or both dimesions to '
                    f'ensure this does not occur.{cols.END}'
                )

            # Transform new position to data coordinates
            new_pos = axis_to_data.transform(new_pos)
            # Update position
            self.labels[value].set_position(new_pos)

    def transfer_to_axes(self, ax):
        """Reproduces the plot in `self.ax` in another axes object.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes <https://matplotlib.org/3.3.1/api/\
        axes_api.html#matplotlib.axes.Axes>`_
            The axes object to construct the result plot onto.

        Warning
        -------
        Everything present in `ax` before calling the method will be
        removed. If you want to add further things to `ax`, do it after calling
        this method.
        """

        # Remove the contents of `ax`
        try:
            ax.clear()
        except AttributeError:
            raise ValueError(f"{cols.R}`ax` is not a valid matplotlib axes "
                             f"object, and instead is:\n{type(ax)}.{cols.END}")

        # Transfer line objects
        for line in self.ax.__dict__['lines']:
            x = line.get_xdata()
            y = line.get_ydata()
            color = line.get_color()
            lw = line.get_lw()
            ax.plot(x, y, color=color, lw=lw)

        # Transfer text objects
        for text in self.ax.__dict__['texts']:
            x, y = text.get_position()
            txt = text.get_text()
            fontprops = text.get_fontproperties()
            color = text.get_color()
            ax.text(x, y, txt, fontproperties=fontprops, color=color)

        # Set ticks
        ax.set_xticks(self.ax.get_xticks())
        ax.set_yticks(self.ax.get_yticks())

        # Set correct x- and y-limits
        ax.set_xlim(self.ax.get_xlim())
        ax.set_ylim(self.ax.get_ylim())

        # Set x-label
        ax.set_xlabel(self.ax.get_xlabel())
