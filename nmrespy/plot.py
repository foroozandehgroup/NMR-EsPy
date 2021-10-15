# plot.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 15 Oct 2021 11:28:46 BST

"""Support for plotting estimation results"""

from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterable, List, Tuple, Union

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import numpy as np

from nmrespy import RED, END, ORA, USE_COLORAMA, NMRESPYPATH, ExpInfo
import nmrespy._errors as errors
from nmrespy._misc import ArgumentChecker, FrequencyConverter, latex_nucleus
from nmrespy import sig

if USE_COLORAMA:
    import colorama
    colorama.init()


def _to_hex(color: Any) -> Union[str, None]:
    r"""Attempt to convert color into a hexadecimal RGBA string.

    If an invalid RGBA argument is given, ``None`` is returned.

    Parameters
    ----------
    color
        Object to attempt to convert to a color.

    Returns
    -------
    hex_color: Union[str, None]
        A string matching the regular expression ``r'^#[0-9a-f]{8}$'``, if
        ``color`` could be converted. Otherwise, ``None``.
    """
    try:
        return mcolors.to_hex(color).lower()
    except ValueError:
        return None


def _configure_oscillator_colors(oscillator_colors: Any, m: int) -> str:
    """Attempt to convert oscillator color input into a string of hex values.

    Parameters
    ----------
    oscillator_colors
        Input to convert to a list of hexadecimal colors.

    m
        Number of oscillators in the result. This is used if
        ``oscillator_colors`` is a valid
        `mpl colormap <https://matplotlib.org/stable/gallery/color/
        colormap_reference.html>`_, to create a list of the correct length.

    Returns
    -------
    color_list
        A list of hexadecimal colors, if  ``oscillator_colors`` is a valid
        input, otherwise ``None``.

    Raises
    ------
    TypeError
        If ``oscillator_colors`` is not ``None``, a ``str`` or an iterable
        object.

    ValueError
        If an iterable argument is given for `oscillator_colors`, and at least
        one of the inputs cannot be converted to a hex color.
    """
    if oscillator_colors is None:
        return ['#1063e0', '#eb9310', '#2bb539', '#d4200c']
    if oscillator_colors in plt.colormaps():
        return [_to_hex(c) for c in
                cm.get_cmap(oscillator_colors)(np.linspace(0, 1, m))]
    if isinstance(oscillator_colors, str):
        oscillator_colors = [oscillator_colors]

    try:
        color_list = [_to_hex(c) for c in oscillator_colors]
    except TypeError:
        raise TypeError(f'{RED}`oscillator_colors` has an invalid type.{END}')

    nones = [i for i, c in enumerate(color_list) if c is None]
    if nones:
        msg = (
            f'{RED}The following entries in `oscillator_colors` could '
            f'not be recognised as valid colours in matplotlib:\n'
            + '\n'.join([f'--> {repr(oscillator_colors[i])}' for i in nones])
            + END
        )
        raise ValueError(msg)

    return color_list


def _get_rc_from_file(path: Path) -> Union[str, None]:
    """Extract rc object from a filepath.

    If the file is not a valid matplotlib stylesheet, ``None`` is returned.

    Parameters
    ----------
    path
        Path to extract rc from.

    Returns
    -------
    rc: Union[str, None]
        The stylesheet's rc, if the sheet exists and is formatted correctly.
        Otherwise, ``None``.
    """
    try:
        rc = str(mpl.rc_params_from_file(
            path, fail_on_error=True, use_default_template=False
        ))
        # If the file exists, but no lines can be parsed, an empty
        # string is returned.
        return rc if rc else None

    except FileNotFoundError:
        return None


def _extract_rc(stylesheet: Union[Path, None]) -> str:
    """Extract matplotlib rc from ``stylesheet``.

    Parameters
    ----------
    stylesheet
        Specification of the stylesheet. Both the stylesheet as a complete
        path, and in the form
        ``matplotlib/mpl-data/stylelib/{stylesheet}.mplstyle`` are tested.
        If given ``None``, then the default nmrespy stylesheet will be used.

    Returns
    -------
    rc: str
        The rc found within the specified stylesheet.

    Raises
    ------
    ValueError
        If an appropriate file could not be found.
    """
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
        f'{RED}Error in loading the stylesheet. Check you gave '
        'a valid path or name for the stylesheet, and that the '
        f'stylesheet is formatted correctly.{END}'
    )


def _create_rc(stylesheet: Union[Path, None], oscillator_colors: Any,
               m: int) -> str:
    """Construct rc for the result plot.

    The folowing things are done:

    * The rc is extracted based on ``stylesheet``.
    * The oscillator colors are processed nased on ``oscillator_colors`` and
      ``m``.
    * The ``axes.prop_cycle`` parameter is overwritten from the rc, if it
      exists.
    * A new ``axes.prop_cycle`` parameter is appended to the rc based on the
      specified oscillator colors.

    Parameters
    ----------
    stylesheet
        Specification of stylesheet to extract the rc from.

    oscillator_colors
        Specification of how to color oscillator peaks.

    m
        Number of oscillators. This is only required in the case that the
        user has specified a matplotlib stylesheet.

    Returns
    -------
    rc
        Parameter specification.

    Raises
    ------
    ValueError
        If ``stylesheet`` or ``oscillator_colors`` are invalid arguments.
        See :py:func:`_configure_oscillator_colors` and
        :py:func:`_extract_rc` for more details.

    TypeError
        If ``oscillator_colors`` is an invalid type.
    """
    rc = _extract_rc(stylesheet)
    osc_cols = _configure_oscillator_colors(oscillator_colors, m)
    rc = _update_prop_cycle(rc, osc_cols)
    # Seem to be getting bugs when using stylesheet with any hex colours
    # that have a # in front. Remove these.
    return re.sub(r'#([0-9a-fA-F]{6})', r'\1', rc)


def _update_prop_cycle(rc: str, osc_cols: List) -> str:
    """Overwrite color cycle in rc.

    Parameters
    ----------
    rc
        Rc.

    osc_cols
        List of colors to make up the color cycle.

    Returns
    -------
    new_rc
        Overwritten rc.
    """
    lines = rc.split('\n')
    lines = list(filter(lambda ln: 'axes.prop_cycle' not in ln, lines))
    color_strs = [f'\'{col}\'' for col in osc_cols]
    lines.append(
        f"axes.prop_cycle: cycler(\'color\', [{', '.join(color_strs)}])"
    )
    return '\n'.join(lines)


def _create_stylesheet(rc: str) -> Path:
    """Create a temportary stylesheet.

    Parameters
    ----------
    rc
        Content to insert into stylesheet.

    Returns
    -------
    path: pathlib.Path
        The path to the stylesheet.
    """
    tmp_path = Path(tempfile.gettempdir()).resolve() / 'stylesheet.mplstyle'
    with open(tmp_path, 'w') as fh:
        fh.write(rc)
    return tmp_path


def _configure_shifts_unit(shifts_unit: str, expinfo: ExpInfo):
    """Dermine the unit to set chemical shifts as.

    Parameters
    ----------
    shifts_unit
        The shift unit specification.

    expinfo
        Experiment information.

    Returns
    -------
    ud_shifts_unit: str
        Configured shifts unit.

    Raises
    ------
    InvalidUnitError
        If ``shifts_unit`` is not ``'hz'`` or ``'ppm'``.
    """
    if shifts_unit not in ['hz', 'ppm']:
        raise errors.InvalidUnitError('hz', 'ppm')
    # If user specifies ppm, but sfo isn;t specified, fall back to Hz
    if shifts_unit == 'ppm' and expinfo.sfo is None:
        shifts_unit = 'hz'
        print(
            f'{ORA}You need to specify `sfo` in `expinfo` if you want chemical'
            f' shifts in ppm! Falling back to Hz...{END}'
        )
    return shifts_unit


def _get_region_slice(
    shifts_unit: str, region: Union[Iterable[Tuple[float, float]], None],
    expinfo: ExpInfo
) -> Iterable[slice]:
    """Determine the slice for the specified spectral region.

    Parameters
    ----------
    shifts_unit
        Unit the region is expressed in.

    region
        The desired region.

    expinfo
        Experiment information.

    Returns
    -------
    region_slice: Iterable[slice]
        Slice for the region of interest.
    """
    if region is None:
        return tuple(slice(0, p, None) for p in expinfo.pts)

    converter = FrequencyConverter(expinfo)
    int_region = converter.convert(region, f'{shifts_unit}->idx')

    return tuple(slice(x[0], x[1] + 1, None) for x in int_region)


def _generate_peaks(result: np.ndarray, region_slice: Iterable[slice],
                    expinfo: ExpInfo) -> Iterable[np.ndarray]:
    """Create a list of peaks for inidivdual oscillators.

    Parameters
    ----------
    result
        Estimation result.

    region_slice
        Spectral region of interest.

    expinfo
        Experiment information.

    Returns
    -------
    peaks: Iterable[numpy.ndarray]
        Iterable of peaks.
    """
    return [
        np.real(
            sig.ft(
                sig.make_fid(
                    np.expand_dims(oscillator, axis=0), expinfo,
                )[0]
            )
        )[region_slice]
        for oscillator in result
    ]


def _generate_model_and_residual(
    peaks: Iterable[np.ndarray], spectrum: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the model and residual of the estimation result.

    Parameters
    ----------
    peaks
        Iterable of individual peaks which make up the estimation model.

    spectrum
        Fourier transform of estimated data.

    Returns
    -------
    model: np.ndarray
        Summation of all individual peaks.

    residual: np.ndarray
        Difference between ``spectrum`` and ``model``.
    """
    model = sum(peaks)
    return model, spectrum - model


def _plot_oscillators(
    lines: List[mpl.lines.Line2D], labels: List[mpl.text.Text], ax: plt.Axes,
    shifts: np.ndarray, peaks: Iterable[np.ndarray], show_labels: bool
) -> None:
    """Plot each oscillator peak onto the specifed axes.

    Nothing is returned from this function, however ``lines`` is mutated to
    include a collection of
    `matplotlib.lines.Line2D
    <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html>`_
    objects. ``ax`` is also mutated to include these lines.

    Parameters
    ----------
    lines
        List to append line objects to.

    labels
        List to append text objects to.

    ax
        Axes to plot lines and labels onto.

    shifts
        Chemical shifts.

    peaks
        Iterable of indiivdual oscillator peaks. Note that each item in this
        should have the same length as ``shifts``.

    show_labels
        Whether or not to show oscillator labels on the plot.

    Notes
    -----
    If ``show_labels`` is ``False``, text objects will still be created,
    and appended to ``labels``, but their alpha transparency will be set to
    ``0``.
    """
    label_alpha = int(show_labels)
    for m, peak in enumerate(peaks, start=1):
        _plot_spectrum(lines, m, ax, shifts, peak)
        idx = np.argmax(np.absolute(peak))
        x, y = shifts[idx], peak[idx]
        labels[m] = ax.text(x, y, str(m), fontsize=8, alpha=label_alpha)


def _plot_spectrum(
    lines: Dict[str, mpl.lines.Line2D], name: str, ax: plt.Axes,
    shifts: np.ndarray, spectrum: np.ndarray, color: Union[str, None] = None,
    show: bool = True
) -> None:
    """Plot a spectrum to axes, and append to ``lines``.

    Parameters
    ----------
    lines
        Dictionary of line objects already present in the plot.

    name
        Name of key of newly added line.

    ax
        Axes to plot the line on.

    shifts
        Chemical shifts.

    spectrum
        Spectrum. Should have the same length as ``shifts``.

    color
        Color of plot line. If ``None``, the line's color will be dictated
        by ``axes.prop_color`` is the stylesheet.

    show
        Whether or not to show the line. This parameter dictates the alpha
        transparency of the plotline (``True -> 1``, ``False -> 0``).
    """
    lines[name] = ax.plot(shifts, spectrum, color=color, alpha=int(show))[0]


def _process_yshift(spectrum: np.ndarray, yshift: Union[float, None],
                    scale: Union[float, None]) -> float:
    """Determine the extent to shift spectrum in the y-axis.

    Parameters
    ----------
    spectrum
        Data of interest.

    yshift
        Either an explicit numerical value inidcating how much to shift the
        spectrum, or ``None``. If ``None``, the shift is computed based on
        ``scale`` and the maximum value in ``spectrum``.

    scale
        Proportionality quantity to determine the shift if ``yshift`` is
        ``None``.

    Returns
    -------
    ud_yshift: float
        Equivalent to ``yshift`` if ``yshift`` is a ``float``, otherwise a
        computed value based on the maximum point in the spectrum, and
        ``scale``.
    """
    if yshift:
        return yshift
    else:
        return scale * np.max(np.absolute(spectrum))


def _set_axis_limits(ax: plt.Axes) -> None:
    """Configure the x- and y-limits of the plot axes.

    Parameters
    ----------
    ax
        Axes to manipulate.
    """
    # Flip the x-axis
    ax.set_xlim(reversed(ax.get_xlim()))
    # y-limits
    ydatas = [ln.get_ydata() for ln in ax.lines if ln.get_alpha() != 0]
    maxi = max([np.amax(ydata) for ydata in ydatas])
    mini = min([np.amin(ydata) for ydata in ydatas])
    vertical_span = maxi - mini
    bottom = mini - (0.03 * vertical_span)
    top = maxi + (0.03 * vertical_span)
    ax.set_ylim(bottom, top)


def _set_xaxis_label(ax: plt.Axes, expinfo: ExpInfo, shifts_unit: str) -> None:
    """Construct axis label for the shifts axis.

    Label will be of the form ``f'$^{mass}${symbol} ({unit})'``, where
    ``mass`` is the isotope mass, ``symbol`` is the element symbol, and unit
    is the frequency unit (``'hz'`` or ``'ppm'``).

    Parameters
    ----------
    ax
        Axes to append the label to.

    expinfo
        Experiment information. The nucleus is extracted from this.

    shifts_unit
        The units of the frequency axis (``'hz'`` or ``'ppm'``).
    """
    # Produces a label of form ¹H (Hz) or ¹³C (ppm) etc.
    nuc = expinfo.nuclei
    nuc = latex_nucleus(nuc[0]) if nuc else 'chemical shift'
    unit = '(Hz)' if shifts_unit == 'hz' else '(ppm)'
    ax.set_xlabel(f'{nuc} {unit}')


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
            raise ValueError(f'{RED}At least one element in `values` is '
                             'invalid. Ensure that all elements are ints '
                             f'between 1 and {max_value}.{END}')

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
                    f'{RED}The specified displacement for label {value} '
                    'places it outside the axes! You may want to reduce the '
                    'magnitude of displacement in one or both dimesions to '
                    f'ensure this does not occur.{END}'
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
            raise ValueError(f"{RED}`ax` is not a valid matplotlib axes "
                             f"object, and instead is:\n{type(ax)}.{END}")

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


def plot_result(
    spectrum: np.ndarray, result: np.ndarray, expinfo: ExpInfo, *,
    plot_residual: bool = True, plot_model: bool = False,
    residual_shift: Union[float, None] = None,
    model_shift: Union[float, None] = None, shifts_unit: str = 'ppm',
    region: Union[Iterable[Tuple[float, float]], None] = None,
    data_color: Any = '#000000', residual_color: Any = '#808080',
    model_color: Any = '#808080', oscillator_colors: Any = None,
    show_labels: bool = True, stylesheet: Union[str, None] = None,
) -> NmrespyPlot:
    """Produce a figure of an estimation result.

    The figure consists of the spectral data, along
    with each oscillator. Optionally, a plot of the complete model, and
    the residual between the data amd the model can be plotted.

    .. note::

        Currently, only 1D data is supported.

    Parameters
    ----------
    spectrum
        Spectral data of interest.

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
        Units to display chemical shifts in. Can be either ``'ppm'`` or
        ``'hz'``. If this is set to ``'ppm'`` but ``sfo`` is not specified in
        ``expinfo``, it will revert to ``'hz'``.

    region
        Boundaries specifying the region to show. **N.B. the units
        ``region`` is given in should match ``shifts_unit``.**

    plot_residual
        If ``True``, plot a difference between the FT of ``spectrum`` and the
        FT of the model generated using ``result``. NB the residual is plotted
        regardless of ``plot_residual``. ``plot_residual`` specifies the alpha
        transparency of the plot line (1 for ``True``, 0 for ``False``)

    residual_shift
        Specifies a translation of the residual plot along the y-axis. If
        ``None``, a default shift will be applied.

    plot_model
        If ``True``, plot the FT of the model generated using ``result``.
        NB the residual is plotted regardless of ``plot_model``. ``plot_model``
        specifies the alpha transparency of the plot line (1 for ``True``,
        0 for ``False``).

    model_shift
        Specifies a translation of the residual plot along the y-axis. If
        ``None``, a default shift will be applied.

    data_color
        The colour used to plot the original spectrum. See `Notes` for a
        discussion of valid colors.
        Any value that is
        recognised by matplotlib as a color is permitted. See
        `<here https://matplotlib.org/3.1.0/tutorials/colors/\
        colors.html>`_ for a full description of valid values.

    residual_color
        The colour used to plot the residual. See `Notes` for a discussion of
        valid colors.

    model_color
        The colour used to plot the model. See `Notes` for a discussion of
        valid colors.

    oscillator_colors
        Describes how to color individual oscillators. The following
        is a complete list of options:

        * If a valid matplotlib color is given, all oscillators will
          be given this color.
        * If a string corresponding to a matplotlib colormap is given,
          the oscillators will be consecutively shaded by linear increments
          of this colormap. For all valid colormaps, see
          `<here https://matplotlib.org/stable/tutorials/colors/\
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
        customaisation of the plot. See `<here https://matplotlib.org/\
        stable/tutorials/introductory/customizing.html>`__ for more
        information on stylesheets.

    Returns
    -------
    plot: :py:class:`NmrespyPlot`
        The result plot.

    Notes
    -----
    **Valid matplotlib colors**

    Any argument which can be converted to a hexadecimal RGB value by
    matpotlib is be specified as a color. A complete list of all valid
    color arguments can be found `here <https://matplotlib.org/stable/
    tutorials/colors/colors.html#sphx-glr-tutorials-colors-colors-py>`_
    """
    if not isinstance(expinfo, ExpInfo):
        raise TypeError(f'{RED}Check `expinfo` is valid.{END}')
    dim = expinfo.unpack('dim')

    try:
        if dim != spectrum.ndim:
            raise ValueError(
                f'{RED}The dimension of `expinfo` does not agree with the '
                f'number of dimensions in `spectrum`.{END}'
            )
        elif dim == 2:
            raise errors.TwoDimUnsupportedError()
        elif dim >= 3:
            raise errors.MoreThanTwoDimError()
    except AttributeError:
        # spectrum.ndim raised an attribute error
        raise TypeError(
            f'{RED}`spectrum` should be a numpy array{END}'
        )

    checker = ArgumentChecker(dim=dim)
    checker.stage(
        (spectrum, 'spectrum', 'ndarray'),
        (result, 'result', 'parameter'),
        (data_color, 'data_color', 'mpl_color'),
        (residual_color, 'residual_color', 'mpl_color'),
        (model_color, 'model_color', 'mpl_color'),
        (show_labels, 'labels', 'bool'),
        (plot_residual, 'plot_residual', 'bool'),
        (plot_model, 'plot_model', 'bool'),
        (region, 'region', 'region_float', True),
        (residual_shift, 'residual_shift', 'float', True),
        (model_shift, 'model_shift', 'float', True),
        (oscillator_colors, 'oscillator_colors', 'osc_cols', True),
        (stylesheet, 'stylesheet', 'str', True)
    )
    checker.check()

    # Setup the stylesheet
    rc = _create_rc(stylesheet, oscillator_colors, result.shape[0])
    stylepath = _create_stylesheet(rc)
    plt.style.use(stylepath)

    expinfo._pts = spectrum.shape
    shifts_unit = _configure_shifts_unit(shifts_unit, expinfo)
    region_slice = _get_region_slice(shifts_unit, region, expinfo)
    shifts = [
        shifts[slice_] for shifts, slice_ in
        zip(
            sig.get_shifts(expinfo, unit=shifts_unit),
            region_slice
        )
    ][0]
    # TODO should replicate that way the spectrum was created (ve for example)
    spectrum = spectrum[region_slice]
    peaks = _generate_peaks(result, region_slice, expinfo)
    model, residual = _generate_model_and_residual(peaks, spectrum)

    # Generate figure and axis
    fig = plt.figure()
    ax = fig.add_axes([0.02, 0.15, 0.96, 0.83])
    lines, labels = {}, {}
    # Plot oscillator peaks
    _plot_oscillators(lines, labels, ax, shifts, peaks, show_labels)
    # Plot spectrum
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

    _set_axis_limits(ax)
    _set_xaxis_label(ax, expinfo, shifts_unit)

    return NmrespyPlot(fig, ax, lines, labels)
