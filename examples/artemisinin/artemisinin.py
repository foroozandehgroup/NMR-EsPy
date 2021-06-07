#!/usr/bin/python3

import os
import pathlib

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
plt.style.use('../stylesheet.mplstyle')

from nmrespy.core import Estimator

# ------------------------------------------------------------------------
# Set to True to carry out estimation.
# Set to False to reload already obtained results and simply plot figure
ESTIMATE = False

B = 0.05 # Bottom of lowest axes
T = 0.98 # Top of highest axes
R = 0.98 # Rightmost position of axes
L = 0.02 # Leftmost position of axes
H_SEP = 0.01 # Horizontal separation of figs a) -> c) and d) -> f)
INSET_RECT = [0.05, 0.5, 0.27, 0.4] # Location of inset axes in panels e) and f)
FIGSIZE = (3.5, 7) # Figure size (inches)
# x-ticks of panels a) -> f)
XTICKS = 4 * [[1.74 - i * (0.03) for i in range(5)]]
YLIMS = [ # y-limits of panels a) -> f)
    (-1.5E4, 1.3E5),
    (-4E4, 1.55E5),
    (-4E4, 1.5E5),
]
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Need at least 5 colors

DISPS = [
    # b)
    iter([
         (0.003, 3000), # 1
         (-0.0025, 1000),
         (0.006, 1000),
         (0.001, 5000),
         (0.002, 3000),
         (-0.0025, 1000), # 6
         (0.006, 1000),
         (0.001, 3000),
         (-0.0035, 2000),
         (-0.0015, 2000),
         (0.009, 1000), # 11
         (0.003, 3000),
         (0.005, 3000),
         (-0.002, 2000),
         (0.01, 2000),
         (0.002, 2000), # 16
    ]),
    # c)
    iter([
        (0.003, 3000), # 1
        (-0.0025, 1000),
        (0.006, 1000),
        (0.001, 5000),
        (0.002, 3000),
        (-0.0025, 1000), # 6
        (0.006, 1000),
        (0.001, 3000),
        (0.0035, 2000),
        (-0.0015, 2000),
        (0.009, 1000), # 11
        (0.003, 3000),
        (0.005, 3000),
        (-0.002, 2000),
        (0.009, 2000),
        (0.0027, 2000), # 16
    ]),
]


LABEL_FS = 8 # Fontsize of oscillator labels
MODEL_SHIFTS = iter([2.5E4, 2E4]) # Upward displacement of model plots (+ve)
RESID_SHIFTS = iter([2E4, 2E4]) # Downward displacement of residual plots (+ve)
# ------------------------------------------------------------------------

def estimate():
    pwd = pathlib.Path.cwd()
    if not (relpath := pwd / 'results').is_dir():
        os.mkdir(relpath)

    datapath = pwd / '../data/2/pdata/1' 

    estimator.frequency_filter([[1.76, 1.6]], [[-4.6, -5.2]])
    estimator.matrix_pencil(M=16)
    estimator.to_pickle(path="result/mpm", force_overwrite=True)
    estimator.nonlinear_programming(phase_variance=True, max_iterations=400, fprint=False)
    estimator.to_pickle(path="result/nlp", force_overwrite=True)

    desc = "1mM artemisinin in DMSO-d6"
    for fmt in ["txt", "pdf", "csv"]:
        if fmt == "pdf":
            desc.replace("1mM", "$1$m\\textsc{M}")
            desc.replace("d6", "\\emph{d}$_6$")
        estimator.write_result(
            path="result/artemisinin_result", fmt=fmt, description=desc,
            force_overwrite=True,
        )

    np.savetxt("result/errors.txt", estimator.get_errors())
    np.savetxt('result/parameters.txt', estimator.get_result())


def plot():

    try:
        estimators = [
            Estimator.from_pickle(path="result/mpm"),
            Estimator.from_pickle(path="result/nlp"),
        ]

    except:
        raise IOError("Couldn't find pickled estimator files")

    # Colors of each oscillator
    col_indices = 2 * [4 * [0] + 4 * [1] + 4 * [2] + 4 * [3]]
    # List of plots
    # Info will be extracted from these to construct customised plots
    plots = []
    for inds, est in zip(col_indices, estimators):
        cols = [COLORS[i] for i in inds]
        # Prevent error for estimators which have only had MPM run
        est._saveable = True
        plots.append(est.plot_result(data_color='k', oscillator_colors=cols))

    shifts = plots[0].lines['data'].get_xdata()
    spectrum = plots[0].lines['data'].get_ydata()

    # --- Construct figure ---------------------------------------------------
    fig = plt.figure(figsize=FIGSIZE)
    xlims = 3 * [(shifts[0], shifts[-1])]

    # Determine axes geometries
    lefts = 3 * [L]
    widths = 3 * [(R - L)]
    heights = []
    bottoms = []
    spans = [abs(YLIMS[i][1] - YLIMS[i][0]) for i in range(0, 3)]
    heights = [s / sum(spans) * (T - B) for s in spans]
    bottoms = [B + sum(heights[i:]) for i in range(1, 4)]
    dims = [[l, b, w, h] for l, b, w, h in zip(lefts, bottoms, widths, heights)]

    # Create axes a), b) and c)
    axs = []
    for i, (dim, xl, yl, xtks) in enumerate(zip(dims, xlims, YLIMS, XTICKS)):
        axs.append(fig.add_axes(dim))
        ax = axs[-1]
        # labels for each panel
        # `trans` ensures x is in axes coords and y is in figure coords
        trans = transforms.blended_transform_factory(
            axs[-1].transAxes, fig.transFigure,
        )
        ax.text(
            0.01, dim[1] + dim[3] - 0.025, f'{chr(i + 97)})',
            transform=trans, fontsize=10, weight='bold'
        )
        # Set limits
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_xticks(xtks)
        if i == 2:
            ax.set_xlabel('$^1$H (ppm)')
        else:
            ax.set_xticklabels([])


    # Plot original data in panel a)
    axs[0].plot(shifts, spectrum, color='k')

    # Extract oscillator lines and labels
    lines = []
    labels = []
    for p in plots:
        # Strip data plot residual plots from lines (first and last elements)
        lines.append([line for line in p.lines.values()][1:-1])
        labels.append([lab for lab in p.labels.values()])


    # Loop over b), c)
    # These axes will show individual oscillators
    osc_axes = axs[1:]
    for i, (ax, lns, lbs, dsps) in enumerate(zip(osc_axes, lines, labels, DISPS)):
        model = np.zeros(lns[0].get_xdata().shape)
        for j, (ln, lb) in enumerate(zip(lns, lbs)):
            # Plot oscillator line
            color = ln.get_color()
            x_ln = ln.get_xdata()
            y_ln = ln.get_ydata()
            ax.plot(x_ln, y_ln, color=color)

            model += y_ln
            # Add oscillator text label
            x_lb, y_lb = lb.get_position()
            txt = lb.get_text()
            d_x, d_y = next(dsps)
            ax.text(x_lb + d_x, y_lb + d_y, txt, color=color,
                    fontsize=LABEL_FS, zorder=200)

        # Plot model and residual
        residual = spectrum - model
        ax.plot(shifts, model + next(MODEL_SHIFTS), color='k')
        ax.plot(shifts, residual - next(RESID_SHIFTS), color='k')

    fig.savefig(
        'artemisinin.png', transparent=False, facecolor='#ffffff',
        dpi=200,
    )

if __name__ == '__main__':
    if ESTIMATE:
        estimate()
    plot()
