import os
from pathlib import Path
import subprocess

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
plt.style.use('./../stylesheet.mplstyle')

from nmrespy.core import Estimator

# ------------------------------------------------------------------------
# Parameters to customise features of the result plot.
# Set to True to carry out estimation.
# Set to False to reload already obtained results and simply plot figure
ESTIMATE = False

B = 0.05 # Bottom of lowest axes
T = 0.98 # Top of highest axes
R = 0.98 # Rightmost position of axes
L = 0.05 # Leftmost position of axes
H_SEP = 0.01 # Horizontal separation of figs a) -> c) and d) -> f)
INSET_RECT = [0.05, 0.5, 0.27, 0.4] # Location of inset axes in panels e) and f)
FIGSIZE = (7, 5) # Figure size
# x-ticks of panels a) -> f)
XTICKS = 3 * [[5.53 - i * (0.02) for i in range(6)]] + \
         3 * [[5.28 - i * (0.02) for i in range(5)]]
YTICKS = [[0, 2.5E7], [0, 2.8E7]] # y-ticks of panels a) and d)
YLIMS = [ # y-limits of panels a) -> f)
    (-1E6, 2.9E7),
    (-5E6, 3E7),
    (-5E6, 3E7),
    (-1E6, 3.2E7),
    (-5E6, 3.3E7),
    (-5E6, 3.3E7),
]
INSET_XLIM = (5.27, 5.23) # x-limits of inset axes in panels e) & f)
INSET_YLIM = (-5.5E5, 1.2E6) # y-limits of inset axes in panels e) & f)
LW = 1 # linewidths
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Need at least 5 colors
DISPS = [ # Oscillator label displacements
    # b)
    iter([(-0.004, -1.4E6), # 1
    (-0.002, 1E5), # 2
    (0.001, 4E5), # 3
    (0.001, 4E5), # 4
    (0.004, 1E5), # 5
    (-0.005, -3.5E6), # 6
    (-0.002, 1E5), # 7
    (-0.002, 1E5), # 8
    (0.0035, 1E5), # 9
    (0.0055, 1E5), # 10
    (-0.0015, 1E5), # 11
    (-0.001, -2E5)]), # 12

    # c)
    iter([(-0.002, 1E5), # 1
    (0.001, 4E5), # 2
    (0.001, 4E5), # 3
    (0.004, 1E5), # 4
    (-0.002, 1E5), # 5
    (-0.002, 1E5), # 6
    (0.0035, 1E5), # 7
    (0.00355, 1E5), # 8
    (0, 3E5)]), # 9

    # e)
    iter([(-0.006, -1.8E6), # 1
    (-0.0015, 1E5), # 2
    (-0.0015, 1E5), # 3
    (0.001, 8E5), # 4
    (0.001, 8E5), # 5
    (0.0035, -1E5), # 6
    (-0.002, 1E5), # 8
    (-0.002, 5E5), # 10
    (0.003, 5E5)]), # 11

    # f)
    iter([(-0.0015, 1E5), # 1
    (-0.0015, 1E5), # 2
    (0.001, 8E5), # 3
    (0.001, 8E5), # 4
    (0.0035, -1E5), # 5
    (-0.002, 0), # 7
    (-0.003, 1000)]), # 9

    # e) (inset)
    iter([(0.002, 0), # 7
    (-0.002, -1E5)]), # 9

    # f) (inset)
    iter([(-0.0015, 0), # 6
    (-0.001, 0)]), # 8
]

LABEL_FS = 7 # Fontsize of oscillator labels
MODEL_SHIFTS = iter(2 * [1E6] + 2 * [1E6]) # Upward displacement of model plots (+ve)
RESID_SHIFTS = iter(2 * [3E6] + 2 * [3E6]) # Downward displacement of residual plots (+ve)
DOT_LS = (2.5, (3, 2)) # Linestyle of dashed lines
# ------------------------------------------------------------------------

def estimate():
    """Performs estimation routine on two different spectral regions
    using NMR-EsPy"""

    pwd = Path.cwd()

    relpaths = ['results'] + [f'results/{i}' for i in (1, 2)]

    for relpath in relpaths:
        if not (rp := pwd / relpath).is_dir():
            os.mkdir(rp)

    datapath = pwd / '../../data/cyclosporin/1/pdata/1'

    estimator = Estimator.new_bruker(datapath)
    regions = ([[5.54, 5.42]], [[5.287, 5.185]])

    for i, region in enumerate(regions, start=1):
        estimator.frequency_filter(region, noise_region=[[6.48, 6.38]])
        estimator.matrix_pencil()
        estimator.to_pickle(path=f"results/{i}/mpm", force_overwrite=True)
        estimator.nonlinear_programming(phase_variance=True, max_iterations=500)
        estimator.to_pickle(path=f"results/{i}/nlp", force_overwrite=True)

        desc = f"50mM cyclosporin in Benzene-d6 (Example {i})"
        for fmt in ["txt", "pdf", "csv"]:
            if fmt == "pdf":
                desc.replace("50mM", "$50$m\\textsc{M}")
                desc.replace("d6", "d\\textsubscript{6}")

            estimator.write_result(
                path=f"results/{i}/cyclosporin_result", fmt=fmt, description=desc,
                force_overwrite=True,
            )

        np.savetxt(f'results/{i}/errors.txt', estimator.get_errors())
        np.savetxt(f'results/{i}/parameters.txt', estimator.get_result())


def plot():

    plt.rcParams['lines.linewidth'] = LW

    try:
        estimators = [
            Estimator.from_pickle(path="results/1/mpm"),
            Estimator.from_pickle(path="results/1/nlp"),
            Estimator.from_pickle(path="results/2/mpm"),
            Estimator.from_pickle(path="results/2/nlp"),
        ]

    except:
        raise IOError("Couldn't find pickled estimator files")

    # Colors of each oscillator
    col_indices = [
        [4, 0, 0, 0, 0, 4, 1, 1, 1, 1, 2, 4], # b)
        [0, 0, 0, 0, 1, 1, 1, 1, 2], # c)
        [4, 0, 0, 1, 0, 1, 2, 1, 2, 3, 4], # e)
        [0, 0, 1, 0, 1, 2, 1, 2, 3], # f)
    ]
    # List of plots
    # Info will be extracted from these to construct customised plots
    plots = []
    for inds, est in zip(col_indices, estimators):
        cols = [COLORS[i] for i in inds]
        # Prevent error for estimators which have only had MPM run
        est._saveable = True
        plots.append(est.plot_result(data_color='k', oscillator_colors=cols))


    # Duplicate the last two plots (panels e) and f)):
    # One set for the main axes, one set for inset axes
    plots = plots + plots[-2:]

    # Extract chemical shifts and original spectra
    shifts = [plots[i].lines['data'].get_xdata() for i in (0, 2)]
    spectra = [plots[i].lines['data'].get_ydata() for i in (0, 2)]

    # Construct figure
    fig = plt.figure(figsize=FIGSIZE)

    xlims = 3 * [(shifts[0][0], shifts[0][-1])] + \
            3 * [(shifts[1][0], shifts[1][-1])]

    # Determine axes geometries
    lefts = 3 * [L] + 3 * [L + H_SEP + (R - L) / 2]
    widths = 6 * [(R - L - H_SEP) / 2]
    heights = []
    bottoms = []
    for start in (0, 3):
        spans = [abs(YLIMS[i][1] - YLIMS[i][0]) for i in range(start, start+3)]
        heights += [s / sum(spans) * (T - B) for s in spans]
        bottoms += [B + sum(heights[-3:][i:]) for i in range(1, 4)]
    dims = [[l, b, w, h] for l, b, w, h in zip(lefts, bottoms, widths, heights)]

    # Create main axes ( a) -> f) )
    axs = []
    for i, (dim, xl, yl, xtks) in enumerate(zip(dims, xlims, YLIMS, XTICKS)):
        axs.append(fig.add_axes(dim))
        ax = axs[-1]
        # panel labels ( a), b), ..., f) )
        # `trans` ensures x is in axes coords and y is in figure coords
        trans = transforms.blended_transform_factory(
            axs[-1].transAxes, fig.transFigure,
        )
        ax.text(
            0.01, dim[1] + dim[3] - 0.03, f'{chr(i + 97)})',
            transform=trans, fontsize=9, weight='bold',
        )
        # Set limits
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_xticks(xtks)
        if i in [2, 5]:
            ax.set_xlabel('$^1$H (ppm)')
        else:
            ax.set_xticklabels([])

    # Create inset axes for panels e) and f)
    axins = []
    for ax in axs[4:]:
        axins.append(ax.inset_axes(INSET_RECT, zorder=200))
        axin = axins[-1]
        axin.set_xlim(INSET_XLIM)
        axin.set_ylim(INSET_YLIM)
        axin.set_xticks([])
        for pos in ['top', 'bottom', 'left', 'right']:
            axin.spines[pos].set_linewidth(LW)

    # Plot original data in panels a) and d)
    axs[0].plot(shifts[0], spectra[0], color='k')
    axs[3].plot(shifts[1], spectra[1], color='k')

    # Extract oscillator lines and labels
    lines = []
    labels = []
    for p in plots:
        # Strip data plot residual plots from lines (first and last elements)
        lines.append([line for line in p.lines.values()][1:-1])
        labels.append([lab for lab in p.labels.values()])

    # Indices of labels to show for each panel
    show_labs = [
        range(len(labels[0])), # b) (show all labels)
        range(len(labels[1])), # c) (show all labels)
        [0, 1, 2, 3, 4, 5, 7, 9, 10], # e)
        [0, 1, 2, 3, 4, 6, 8], # f)
        [6, 8], # e) (inset)
        [5, 7], # f) (inset)
    ]

    # Loop over b), c), e), f), e) (inset), and f) (inset)
    # These axes will show individual oscillators
    osc_axes = [axs[i] for i in (1,2,4,5)] + axins
    for i, (ax, lns, lbs, sl, dsps) in enumerate(zip(osc_axes, lines, labels, show_labs, DISPS)):
        if i < 4:
            # Panels b), c), e), f)
            # Will plot the sum of oscillators (model), and residual in
            # these panels
            model = np.zeros(lns[0].get_xdata().shape)
        for j, (ln, lb) in enumerate(zip(lns, lbs)):
            # Plot oscillator line
            color = ln.get_color()
            x_ln = ln.get_xdata()
            y_ln = ln.get_ydata()
            ax.plot(x_ln, y_ln, color=color)

            if i < 4:
                # Add oscillator to model
                model += y_ln

            # Add oscillator text label
            x_lb, y_lb = lb.get_position()
            txt = lb.get_text()
            if j in sl:
                d_x, d_y = next(dsps)
                ax.text(x_lb + d_x, y_lb + d_y, txt, color=color,
                        fontsize=LABEL_FS, zorder=200)
        if i < 4:
            # Plot model and residual
            idx = i // 2
            residual = spectra[idx] - model
            ax.plot(shifts[idx], model + next(MODEL_SHIFTS), color='k')
            ax.plot(shifts[idx], residual - next(RESID_SHIFTS), color='k')

    # White rectangles to make certain oscillator labels clearer (overlap with
    # plot lines)
    r = Rectangle((5.2412, 6.5E5), -0.0035, 2.5E5, facecolor='w', zorder=50)
    axins[1].add_patch(r)
    r = Rectangle((5.2825, 7E5), -0.0033, 1.5E6, facecolor='w', zorder=50)
    axs[4].add_patch(r)

    # Create rectangle highlighting region illustrated by inset axes in
    # panels e) and f
    for ax, inset_ax, in zip(axs[4:], axins):
        xlim = inset_ax.get_xlim()
        ylim = inset_ax.get_ylim()
        # Get figure -> data transformation matrix
        figure_to_data = fig.transFigure + ax.transData.inverted()
        # Get points (in figure coords) of inset axis vertices
        pts = ax.__dict__['child_axes'][0].__dict__['_position'].__dict__['_points']
        # Transform inset axis vertices to data coords
        inset_vertices = [list(figure_to_data.transform(pt)) for pt in pts]

        # Create rectangle for region illustrated, and lines to link
        # region and inset axis.
        # Two lines:
        # 1. Thinner dotted black line
        # 2. Thicker white line below the black dotted line to provide
        # some padding from the rest of the figure.
        for i, (col, lw, ls) in enumerate(zip(('w', 'k'), (2.4 * LW, LW), ('-', DOT_LS))):
            # Zorder used ensures white line is above rest of plot, but below
            # black dotted line
            ax.add_patch(
                Rectangle(
                    (xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                    linewidth=lw, edgecolor=col, facecolor='none',
                    zorder=100+i*10,
                )
            )
            # Lines linking region of interest and insert axes
            ax.plot(
                [xlim[0], inset_vertices[0][0]], [ylim[0],
                inset_vertices[0][1]], solid_capstyle='round', linewidth=lw,
                color=col, zorder=100+i*10, linestyle=ls,
                dash_capstyle='round',
            )
            ax.plot(
                [xlim[1], inset_vertices[1][0]], [ylim[1],
                inset_vertices[1][1]], solid_capstyle='round', linewidth=lw,
                color=col, zorder=100+i*10, linestyle=ls,
                dash_capstyle='round',
            )

        # Lines linking xticks with rectangle
        ax.plot(
            [xlim[0], xlim[0]], [ax.get_ylim()[0], ylim[0]],
            color='#808080', dash_capstyle='round', lw=0.6, zorder=0,
        )
        ax.plot(
            [xlim[1], xlim[1]], [ax.get_ylim()[0], ylim[0]],
            color='#808080', dash_capstyle='round', lw=0.6, zorder=0,
        )

    fig.savefig(
        f"cyclosporin.png", transparent=False, facecolor='#ffffff',
        dpi=200,
    )

if __name__ == '__main__':
    if ESTIMATE:
        estimate()
    plot()
