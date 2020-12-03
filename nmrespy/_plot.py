#!/usr/bin/python3
# nlp.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# plotting result of estimation routine

from itertools import cycle
import os
import re

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftshift

import nmrespy as espy
from ._misc import *

def plotres_1d(data, peaks, shifts, region, nuc, data_col,
               osc_col, labels, stylesheet):
    """
    Produces a figure of an estimation result (1D data).

    Parameters
    ----------
    data : numpy.ndarray
        Fourier transform of estimated signal.

    peaks : list
        List with each element being a `numpy.ndarray`. These arrays
        correspond to Fourier transforms of synthetic signals assocaited
        with each oscillator.

    shifts : numpy.ndarray
        Chemical shifts sampled.

    region : (float, float),
        The highest and lowest ppm values of the region considered in
        the estimation routine.

    nuc : str
        Identity of the nucleus, in the form ``'<M><E>'``, where ``<M>`` is the
        elements mass number, and ``<E>`` is the element symbol.

    data_col : matplotlib color or None
        The color used to plot the original data. See
        :py:meth:`~nmrespy.core.NMREsPyBruker.plot_result` for details.

    osc_col : matplotlib color, matplotlib colormap, list, numpy.ndarray, or None
        Describes how to colour individual oscillators. See
        :py:meth:`~nmrespy.core.NMREsPyBruker.plot_result` for details.

    labels : Bool
        If `True`, each oscillator will be given a numerical label
        in the plot, if `False`, no labels will be produced.

    stylesheet : None or str
        The name of/path to a matplotlib stylesheet for further
        customaisation of the plot. See
        :py:meth:`~nmrespy.core.NMREsPyBruker.plot_result` for details.

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

    # get mpl stylesheet
    if stylesheet is None:
        espypath = os.path.dirname(espy.__file__)
        stylesheet = os.path.join(espypath, 'config/nmrespy_custom.mplstyle')

    plt.style.use(stylesheet)

    # number of oscillators
    M = len(peaks)
    shifts = shifts[0] # unpack from tuple

    # colour determination - save to itertools.cycle object
    data_col = _color_checker(data_col, default='#808080')
    osc_cols = _get_osc_cols(osc_col, M)
    # generate figure and axis attributes
    fig = plt.figure()
    ax = fig.add_axes([0.02, 0.15, 0.96, 0.83])

    # store each plot (mpl.lines.Line2D objects)
    lines = {}
    # store each oscillator label (mpl.text.Text objects)
    labs = {}

    # plot original data
    # N.B. |plt.plot| always produces a list, so need to access the
    # 0 element
    lines['data'] = ax.plot(shifts, np.real(data), color=data_col)[0]

    # plot oscillators and label
    for m, peak in enumerate(peaks):
        # generate next color in cycle
        c = next(osc_cols)
        lines[f'osc{m+1}'] = ax.plot(shifts, peak, color=c)[0]

        # give oscillators numerical labels
        if labels is True:
            # x-value of peak maximum (in ppm)
            x = shifts[np.argmax(peak)]
            # y-value of peak maximum
            y = peak[np.argmax(np.absolute(peak))]
            labs[f'osc{m+1}'] = ax.text(x, y, f'{m+1}', fontsize=8)

    # change x-axis limits if a specific region was studied
    if region:
        # left and right boundaries of region, in ppm
        left = region[0][0]
        right = region[0][1]

        # set new x-axis limits to region studied
        # TODO generalise when writing for 2D as well
        ax.set_xlim(shifts[left], shifts[right])

        # determine highest/lowest values points in region,
        # and set ylims to accommodate these.
        max, min = _get_ymaxmin(lines, left, right)
        ax.set_ylim(1.1*min, 1.1*max)

    # x-axis label, of form $^{1}H$ or $^{13}C$ etc.
    # depending on nucleus
    xlab = _generate_xlabel(nuc[0])
    ax.set_xlabel(xlab)

    return fig, ax, lines, labs


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


def _color_checker(inp, default='k', kill=True):
    """
    Determines whether a given input can is recognised as a valid color in
    matplotlib, and returns the color.

    Parameters
    ----------
    inp : any type
        Input to check.
    default : matplotlib color, default: 'k'
        The default color, returned if ``inp`` is ``None``
    kill : Bool, default: True
        Determines what to do if ``inp`` is not a valid matplotlib color, or
        ``None``. If ``True``, a ValueError is raised. If ``False``, ``None``
        is returned.

    Returns
    -------
    col - matplotlib color or None
        A valid color, recognised by matplotlib, or None, if ``inp`` wasn't
        recognised as a maplotlib color, and ``kill`` is set to ``False``.
    """
    if inp is None:
        return default
    return _check_valid_mpl_color(inp, kill)


def _check_valid_mpl_color(col, kill):
    try:
        col = mcolors.to_hex(col)
        return col
    except ValueError:
        if kill is True:
            msg = f'\n{R}The following colour input is invalid: {col}. To see' \
                  + f' all valid colour arguments, refer to:\n' \
                  + f'{C}https://matplotlib.org/3.1.0/tutorials/colors/colors.html{END}'
            raise ValueError(msg)
        else:
            return None


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
