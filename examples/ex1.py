# ex1.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 09 Jun 2022 23:41:52 BST

# EXAMPLE 1:
# Estimation applied to 3 selected regions of a 1D andrographolide proton spectrum

from pathlib import Path
import matplotlib as mpl
import nmrespy as ne

mpl.rcdefaults()

BESPOKE_PLOT = True

directory = Path("ex1_output")
if not directory.is_dir():
    directory.mkdir()

regions = ((6.665, 6.59), (2.38, 2.28), (1.43, 1.29))
if not (directory / "andro.pkl").is_file():

    estimator = ne.Estimator1D.new_bruker("data/1/pdata/1")
    noise_region = (8.01, 7.99)
    for region in regions:
        estimator.estimate(region, noise_region, region_unit="ppm")
    if estimator.get_params([0]).shape[0] == 7:
        # One peak in dt in 6.665-6.59 ppm region is fit by two peaks sadly.
        # We can semi-manually fix this be using `merge_oscillators`
        # This will merge the culprit oscillators, and re-run the optimiser on the
        # edited estimation result.
        estimator.merge_oscillators([3, 4] , 0)

    estimator.to_pickle(directory / "andro")

else:
    estimator = ne.Estimator1D.from_pickle(directory / "andro")

# Generate result files
for fmt in ("txt", "pdf"):
    estimator.write_result(
        path=directory / "result",
        fmt=fmt,
        description="Andrographolide dataset",
        force_overwrite=True,
    )

# Generate result figures
plots = estimator.plot_result()
for region, plot in zip(regions, plots):
    plot.save(
        directory / f"{region[0]}-{region[1]}".replace(".", "_"),
        fmt="png",
        force_overwrite=True,
    )

# END OF "STANDARD" NMR-EsPy USAGE

# ===============================================================================

# Below, I generate a more sopisticated "publication-worthy" figure.
# If you wish to run this code, change line 14 to `BESPOKE_PLOT = True`.

if not BESPOKE_PLOT:
    exit()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
mpl.rc_file_defaults()

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
FIGWIDTH = 7.22  # inches
FIGHEIGHT = 0.4 * FIGWIDTH
B, T, R, L = 0.15, 0.96, 0.985, 0.015  # co-ords of Bottom, Top, Right and Left of axes
HSEP = 0.008  # horizontal separation of region panels
RESIDUAL_SHIFT = -3000
DISPS = iter(  # Label displacements from their default
    [
        # Plot 1
        -0.002,
        0.005,
        -0.002,
        0.005,
        -0.002,
        0.005,
        # Plot 2
        -0.002,
        -0.001,
        0.004,
        0.005,
        -0.002,
        -0.001,
        0.004,
        0.005,
        # Plot 3
        -0.002,
        -0.002,
        0.005,
        0.005,
        -0.002,
        -0.002,
        0.005,
        0.005,
    ]
)


def get_axes_geometries():
    region_widths = [r[0] - r[1] for r in regions]
    width_sum = sum(region_widths)
    widths = [
        (rw / width_sum) * (R - L - 2 * HSEP) for rw in region_widths
    ]
    lefts = [L + sum(widths[:i]) + (i * HSEP) for i in range(3)]
    return [(lf, B, w, T - B) for lf, w in zip(lefts, widths)]


def setup_fig_and_axes():
    fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    axs = [fig.add_axes(geom) for geom in get_axes_geometries()]
    axs[0].spines["right"].set_visible(False)
    axs[1].spines["left"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[2].spines["left"].set_visible(False)
    d = 3  # proportion of vertical to horizontal extent of the slanted line
    kwargs = {
        "marker": [(-1, -d), (1, d)],
        "markersize": 10,
        "linestyle": "none",
        "color": "k",
        "mec": "k",
        "mew": 1,
        "clip_on": False,
    }
    axs[0].plot([1, 1], [0, 1], transform=axs[0].transAxes, **kwargs)
    axs[1].plot([0, 0], [0, 1], transform=axs[1].transAxes, **kwargs)
    axs[1].plot([1, 1], [0, 1], transform=axs[1].transAxes, **kwargs)
    axs[2].plot([0, 0], [0, 1], transform=axs[2].transAxes, **kwargs)

    return fig, axs


def get_line_label_info(plots):
    full_info = []
    for i, plot in enumerate(plots):
        oscs = len(plot.oscillator_plots)
        info = {
            "x": plot.data_plot.get_xdata(),
            "ys": [
                plot.data_plot.get_ydata(),
            ] + [osc["line"].get_ydata() for osc in plot.oscillator_plots],
            "colors": ["k"] + oscs * [COLORS[i]] + ["#808080"],
            "lws": [1.4] + oscs * [0.8] + [1.4],
            "labelpos": [osc["label"].get_position() for osc in plot.oscillator_plots],
        }
        info["ys"].append(
            info["ys"][0] -
            sum(info["ys"][1:], start=np.zeros_like(info["ys"][0])) +
            RESIDUAL_SHIFT
        )

        full_info.append(info)

    return full_info


def plot_lines(axs, info):
    for ax, inf in zip(axs, info):
        x = inf["x"]
        for y, color, lw in zip(inf["ys"], inf["colors"], inf["lws"]):
            ax.plot(x, y, color=color, lw=lw)
        labelcolor = inf["colors"][1]
        for i, (x, y) in enumerate(inf["labelpos"], start=1):
            ax.text(x + next(DISPS), y, str(i), fontsize=8, color=labelcolor)


def finishing_touches(fig, axs):
    # Magnification factor to indicate relative scaling
    yscales = [ax.get_ylim()[1] - ax.get_ylim()[0] for ax in axs]
    max_yscale = max(yscales)
    max_axis = yscales.index(max_yscale)
    scale_factors = [max_yscale / yscale for yscale in yscales]
    for i, (ax, sf) in enumerate(zip(axs, scale_factors)):
        if i == max_axis:
            continue
        ax.text(0.05, 0.92, f"$\\times {sf:.2f}", transform=ax.transAxes)

    # Axis ticks
    for i, ax in enumerate(axs):
        low_lim, high_lim = ax.get_xlim()
        low_tick = low_lim + 0.02 - (low_lim % 0.02)
        high_tick = high_lim - (high_lim % 0.02)
        if i == 1:
            low_tick += 0.01
            high_tick -= 0.01
        nticks = round(((high_tick - low_tick) / 0.02) + 1)
        ticks = [low_tick + i * 0.02 for i in range(nticks)]
        ax.set_xticks(ticks)
        ax.set_yticks([])

        # Flip x-axis limits (NMR convention is for decrasing shifts from right to left)
        ax.set_xlim(reversed(ax.get_xlim()))

    fig.text(
        0.5,
        0.03,
        "$^{1}$H (ppm)",
        transform=fig.transFigure,
        horizontalalignment="center",
    )


fig, axs = setup_fig_and_axes()
info = get_line_label_info(plots)
plot_lines(axs, info)
finishing_touches(fig, axs)
fig.savefig(directory / "bespoke_fig.png")
