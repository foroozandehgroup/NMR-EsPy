# plot.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Support for plotting estimation results"""

from itertools import cycle
import os
import re
import tempfile

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftshift

from nmrespy import *
import nmrespy._errors as errors
from nmrespy._misc import *
from nmrespy import signal

def plot_result(
    data, result, sw, offset, sfo=None, shifts_unit='ppm', nucleus=None,
    region=None, data_color='#808080', oscillator_colors=None, labels=True,
    stylesheet=None,
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

    nucleus : [str], [str, str] or None, default: None
        The nucleus in each dimension.

    region : [[int, int]], [[float, float]], [[int, int], [int, int]] or \
    [[float, float], [float, float]]
        Boundaries specifying the region to show. See also
        :py:class:`nmrespy.filter.FrequencyFilter`.

    data_color : matplotlib color, default: '#808080'
        The color used to plot the original data. Any value that is
        recognised by matplotlib as a color is permitted. See
        `this link <https://matplotlib.org/3.1.0/tutorials/colors/\
        colors.html>`_ for a full description of valid values.

    oscillator_colors : {matplotlib color, matplotlib colormap name, \
    list, numpy.ndarray, None}, default: None
        Describes how to color individual oscillators. The following
        is a complete list of options:

        * If the a valid matplotlib color is given, all oscillators will
          be given this color.
        * If a string corresponding to a matplotlib colormap is given,
          the oscillators will be consecutively shaded by linear increments
          of this colormap. For all valid colormaps, see
          `this link <https://matplotlib.org/3.3.1/tutorials/colors/\
          colormaps.html>`_
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

    labels : Bool, default: True
        If `True`, each oscillator will be given a numerical label
        in the plot, if `False`, no labels will be produced.

    stylesheet : str or None, default: None
        The name of/path to a matplotlib stylesheet for further
        customaisation of the plot. See `this link <https://matplotlib.org/\
        stable/tutorials/introductory/customizing.html>`_ for more
        information on stylesheets.

    Returns
    -------
    fig : `matplotlib.figure.Figure <https://matplotlib.org/3.3.1/\
    api/_as_gen/matplotlib.figure.Figure.html>`_
        The resulting figure.

    ax : `matplotlib.axes._subplots.AxesSubplot <https://matplotlib.org/\
    3.3.1/api/axes_api.html#the-axes-class>`_
        The resulting set of axes.

    lines : dict
        A dictionary containing a series of
        `matplotlib.lines.Line2D <https://matplotlib.org/3.3.1/\
        api/_as_gen/matplotlib.lines.Line2D.html>`_
        instances. The data plot is given the key ``'data'``, and the
        individual oscillator plots are given the keys ``'osc1'``,
        ``'osc2'``, ``'osc3'``, ..., ``'osc<M>'`` where ``<M>`` is the number of
        oscillators in the parameter estimate.

    labs : dict
        If ``labels`` is ``True``, this dictionary will contain a series
        of `matplotlib.text.Text <https://matplotlib.org/3.1.1/\
        api/text_api.html#matplotlib.text.Text>`_ instances, with the
        keys ``'osc1'``, ``'osc2'``, etc. If ``labels`` is ``False``, the dict
        will be empty.
    """


    # --- Check arguments ------------------------------------------------
    try:
        dim = data.ndim
    except:
        raise TypeError(f'{cols.R}data should be a NUmPu array.{cols.END}')

    if dim == 2:
        raise TwoDimUnsupportedError()

    if dim >= 3:
        raise MoreThanTwoDimError()

    components = [
        (data, 'data', 'ndarray'),
        (result, 'result', 'parameter'),
        (sw, 'sw', 'float_list'),
        (offset, 'offset', 'float_list'),
        (data_color, 'data_color', 'mpl_color'),
        (labels, 'labels', 'bool'),
    ]

    if sfo is not None:
        components.append((sfo, 'sfo', 'float_list'))
    if nucleus is not None:
        components.append((nucleus, 'nucleus', 'str_list'))
    if region is not None:
        components.append((region, 'region', 'region_float'))
    if oscillator_colors is not None:
        components.append((oscillator_colors, 'oscillator_colors', 'osc_cols'))
    if stylesheet is not None:
        components.append((stylesheet, 'stylesheet', 'str'))

    ArgumentChecker(components, dim=dim)

    # Number of oscillators
    m = result.shape[0]


    def get_rc(stylesheet):
        try:
            return str(mpl.rc_params_from_file(
                stylesheet, fail_on_error=True, use_default_template=False,
            ))

        except:
            raise ValueError(
                f'{cols.R}Error in loading the stylesheet. Check you gave'
                ' a valid path, and that the stylesheet is formatted'
                f' correctly{cols.END}'
            )

    # Default style sheet if one isn't explicitly given
    dft_stylesheet = NMRESPYPATH / 'config/nmrespy_custom.mplstyle'
    # Load text from style sheet
    rc = get_rc(dft_stylesheet) if stylesheet is None else get_rc(stylesheet)

    # Seem to be getting bugs when using stylesheet with any hex colours
    # that have a # in front. Remove these.
    rc = re.sub(r'#([0-9a-fA-F]{6})', r'\1', rc)

    # Determine oscillator colours to use
    if oscillator_colors is None:
        # Check if axes.prop_cycle is in the stylesheet.
        # If not, add the default colour cycle to it
        if 'axes.prop_cycle' not in rc:
            rc += (
                '\naxes.prop_cycle: cycler(\'color\', [\'1063e0\', \'eb9310\','
                ' \'2bb539\', \'d4200c\'])'
            )

    else:
        # --- Get colors in list form ------------------------------------
        # Check for single mpl colour
        try:
            oscillator_colors = [mcolors.to_hex(oscillator_colors)]
        except ValueError:
            pass
        # Check for colormap, and construct linearly sampled array
        # of colors from it
        if oscillator_colors in plt.colormaps():
            oscillator_colors = \
                cm.get_cmap(oscillator_colors)(np.linspace(0, 1, m))
        # Covers both a colormap specification or a list/numpy array input
        oscillator_colors = [mcolors.to_hex(col) for col in oscillator_colors]

        # Remove the line containing axes.prop_cycle (if it exists),
        # as it is going to be overwritten by the custom colorcycle
        rc = '\n'.join(
            filter(lambda ln: 'axes.prop_cycle' not in ln, rc.split('\n'))
        )
        # Append the custom colorcycle to the rc text
        col_txt = ', '.join([f'\'{c[1:].lower()}\'' for c in oscillator_colors])
        rc += f'\naxes.prop_cycle: cycler(\'color\', [{col_txt}])\n'

    # Temporary path to save stylesheet to
    tmp_path = Path(tempfile.gettempdir()) / 'stylesheet.mplstyle'
    with open(tmp_path, 'w') as fh:
        print(rc)
        fh.write(rc)
    # Invoke the stylesheet!
    plt.style.use(tmp_path)
    # Delete the stylesheet
    os.remove(tmp_path)

    # Generate data: chemical shifts, data spectrum, oscillator spectra
    n = list(data.shape)
    shifts = signal.get_shifts(n, sw, offset)[0]

    if shifts_unit not in ['hz', 'ppm']:
        raise errors.InvalidUnitError('hz', 'ppm')
    if shifts_unit == 'ppm':
        try:
            shifts /= sfo[0]
        except:
            raise TypeError(
                f'{cols.R}You need to specify sfo if you want chemical'
                f' shifts in ppm!{cols.END}'
            )

    spectrum = np.real(signal.ft(data, flip=False))
    peaks = []
    for osc in result:
        print(np.expand_dims(osc, axis=0))
        peaks.append(
            np.real(
                signal.ft(
                    signal.make_fid(
                        np.expand_dims(osc, axis=0), n, sw, offset=offset,
                    )[0]
                , flip=False)
            )
        )

    # Generate figure and axis
    fig = plt.figure()
    ax = fig.add_axes([0.02, 0.15, 0.96, 0.83])

    # To store each plotline (mpl.lines.Line2D objects)
    lines = {}
    # To store each oscillator label (mpl.text.Text objects)
    labs = {}

    # Plot original data (Given the key 0)
    lines[0] = ax.plot(shifts, spectrum, color=data_color)[0]

    # Plot oscillators and label
    for m, peak in enumerate(peaks, start=1):
        lines[m] = ax.plot(shifts, peak)[0]
        # Give oscillators numerical labels
        # x-value of peak maximum (in ppm)
        x = shifts[np.argmax(peak)]
        # y-value of peak maximum
        y = peak[np.argmax(np.absolute(peak))]
        alpha = 1 if labels else 0
        labs[m] = ax.text(x, y, str(m), fontsize=8, alpha=alpha)

    # Change x-axis limits if a specific region was studied
    if region is not None:
        ax.set_xlim(max(region[0]), min(region[0]))

        # Determine highest/lowest values points in region,
        # and set ylims to accommodate these.
        # TODO: Needs some rethinking...
        converter = FrequencyConverter(n, sw, offset, sfo=sfo)
        region_idx = converter.convert(region, f'{shifts_unit}->idx')

        mx, mn = _get_ymaxmin(lines, min(region_idx[0]), max(region_idx[0]))
        ax.set_ylim(mn, mx)

    # x-axis label, of form ¹H or ¹³C etc.
    xlab = 'chemical shifts' if nucleus is None else latex_nucleus(nucleus[0])
    ax.set_xlabel(xlab)
    plt.show()
    return NmrespyFigure(fig, ax, lines, labs)

# TODO: Grow this class: provide functionality for easy tweaking
# of figure
class NmrespyFigure:

    def __init__(self, fig, ax, lines, labels):
        self.fig = fig
        self.ax = ax
        self.lines = lines
        self.labels = labels


def _get_ymaxmin(lines, left, right):
    """
    Out of original data plot, and oscillator plots, determine largest
    and smallest values. Used for determine the y-axis limits

    Parameters
    ----------
    lines : dict
        Dictionary of plots.

    left : int
        Index of leftmost point in the region of interest.

    right : int
        Index of rightmost point in the region of interest.

    Returns
    -------
    max : float
        Higest value in the region of interest, amongst all plotlines.

    min : float
        Lowest value in the region of interest, amongst all plotlines.
    """
    # initialise max and min to be values that are certain to be
    # overwritten (anything is bigger than -∞, everything is smaller
    # than +∞)

    max = -np.inf
    min = np.inf

    # for each plot, get max and min values
    for line in list(lines.values()):
        data = line.get_ydata()[left:right]
        line_max = np.amax(data)
        line_min = np.amin(data)
        # check if plot's max value is larger than current max
        if line_max > max:
            max = line_max
        # check if plot's min value is larger than current min
        if line_min < min:
            min = line_min
    # if min is +ve, give it a small negative value so that 0 features
    if min >= 0.0:
        min = -0.01 * max
    return max, min



def _get_osc_cols(inp, M):
    """
    Constructs and iterator for coloring individual oscillator plots.

    Parameters
    ----------
    inp : any type
        An input which describes how to color the oscillators. See
        :py:meth:`~nmrespy.core.NMREsPyBruker.plot_result` for details.

    M : int
        The number of oscillators.

    Returns
    -------
    osc_cols : itertools.cycle
        A cyclic iterator of color arguments.
    """
    if inp is None:
        # default: cycle through blue, orange, green, red
        return cycle(['#1063e0', '#eb9310', '#2bb539', '#d4200c'])

    # check if inp can be interpreted as a valid individual colour
    singlecol = _check_valid_mpl_color(inp, kill=False)
    if singlecol:
        return cycle([singlecol])

    # check is inp refers to a mpl colormap
    if isinstance(inp, str):
        if inp in plt.colormaps():
            # create colormap
            return cycle(vars(cm)[inp](np.linspace(0, 1, M)))

    # check if inp is a list/numpy array of valid mpl colours
    if isinstance(inp, np.ndarray):
        inp = inp.tolist()
    if isinstance(inp, list):
        for elem in inp:
            c = _check_valid_mpl_color(elem, True)
        return cycle(inp)

    else:
        raise TypeError(f'\n{R}osc_cols could not be understood.{END}')


def _generate_xlabel(nuc):
    """
    Generates an xlabel for the plot, of the form: ``'$^{M}$E'`` (ppm), where M
    is mass number of nucleus, and E is the element symbol.

    Parameters
    ----------
    nuc : str
        Identity of the nucleus, in the form ``'<M><E>'``, where ``<M>`` is the
        mass number, and ``<E>`` is the element symbol.

    Returns
    -------
    xlab : str
        Formatted string for x-axis of figure.
    """
    # comps: [mass number, element symbol]
    # seem to get empty string as first arg, so filter any NoneTypes
    comps = filter(None, re.split(r'(\d+)', nuc))
    return r'$^{' + next(comps) + r'}$' + next(comps) + r' (ppm)'
