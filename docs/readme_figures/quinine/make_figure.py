# make_figure.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 14 Dec 2023 12:40:23 PM EST


from pathlib import Path
import sys
sys.path.insert(0, ".")

from bruker_utils import BrukerDataset
import nmrespy as ne
from matplotlib.patches import ConnectionPatch
import matplotlib as mpl
import numpy as np
from utils import (
    add_pure_shift_labels,
    get_pure_shift_labels,
    fix_linewidths,
    panel_labels,
    raise_axes,
)

estimator_path = Path("~/Documents/DPhil/results/cupid/quinine/estimator").expanduser()
save_path = "quinine/quinine.pdf"


residual_shift = 3e6
multiplet_shift = residual_shift + 1e6
onedim_shift = multiplet_shift + 1e6
ax_top = onedim_shift + 3e6

# =====================
colors = list(mpl.colormaps.get_cmap("plasma")(np.linspace(0.2, 0.95, 14)))
colors.insert(4, "#808080")

thold = 1.171 # 6 * default thold
if (path := Path("quinine/estimator.pkl")).is_file():
    estimator = ne.Estimator2DJ.from_pickle(path)
else:
    estimator = ne.Estimator2DJ.from_pickle(estimator_path)
    thold = 6. * estimator.default_multiplet_thold
    estimator.predict_multiplets(thold=thold, rm_spurious=True, max_iterations=1, check_neg_amps_every=1)
    estimator.to_pickle(path)

xaxis_ticks = [
    (0, (5.8, 5.75, 5.7, 5.65, 5.6)),
    (1, (4.95, 4.9,)),
    (2, (3.7, 3.65)),
    (3, (3.15, 3.1,)),
    (4, (2.75, 2.7, 2.65)),
    (5, (1.95, 1.9, 1.85, 1.8, 1.75)),
    (6, (1.6, 1.55, 1.5, 1.45, 1.4)),
]
xaxis_ticklabels = [[str(x) for x in entry[1]] for entry in xaxis_ticks]

fig, axs = estimator.plot_result(
    axes_right=0.975,
    axes_bottom=0.08,
    axes_top=0.975,
    axes_left=0.075,
    multiplet_thold=thold,
    region_unit="ppm",
    contour_base=1.7e4,
    contour_nlevels=10,
    contour_factor=1.6,
    contour_color="k",
    contour_lw=0.1,
    multiplet_colors=colors,
    marker_size=3.,
    multiplet_show_45=False,
    jres_sinebell=True,
    xaxis_label_height=0.01,
    xaxis_ticks=[
        (0, (5.8, 5.75, 5.7, 5.65, 5.6)),
        (1, (4.95, 4.9,)),
        (2, (3.7, 3.65)),
        (3, (3.15, 3.1,)),
        (4, (2.75, 2.7, 2.65)),
        (5, (1.95, 1.9, 1.85, 1.8, 1.75)),
        (6, (1.6, 1.55, 1.5, 1.45, 1.4)),
    ],
    ratio_1d_2d=(3., 1.),
    figsize=(5, 4),
)
raise_axes(axs, 1.1)
axs[1, 0].set_yticks([-20, -10, 0, 10, 20])
for ax, xticklabels in zip(axs[1], xaxis_ticklabels):
    ax.set_xticklabels(xticklabels)

fix_linewidths(axs, 1.)

_, shifts = estimator.get_shifts(unit="ppm", meshgrid=False)

# Remove H2O from CUPID spectrum
r1_params = estimator.get_params(indices=[1])
r1_params = np.delete(r1_params, (4), axis=0)
estimator._results[1].params = r1_params
r1_region = estimator.get_results(indices=[1])[0].get_region()
r1_slice = estimator.convert(r1_region, "hz->idx")[1]
cupid_spectrum = estimator.cupid_spectrum().real
shifts = estimator.get_shifts(unit="ppm", meshgrid=False)[1]
shifts_r1 = shifts[r1_slice[0]:r1_slice[1]]
spec_without_h2o = cupid_spectrum[r1_slice[0]:r1_slice[1]]
prev_spec = axs[0][1].get_lines()[-3]
vshift = prev_spec.get_ydata()[0] - spec_without_h2o[0]
prev_spec.remove()
specline = axs[0][1].plot(shifts_r1, spec_without_h2o + vshift, color="k", lw=0.8)

n_ax = axs[0].size
for i, (ax0, ax1) in enumerate(zip(axs[0], axs[1])):
    if i in [0, n_ax - 1]:
        lines = ax1.get_lines()[:-1]
    else:
        lines = ax1.get_lines()[:-2]
    for line in lines:
        line.remove()
        x = line.get_xdata()[0]
        con = ConnectionPatch(
            xyA=(x, ax1.get_ylim()[0]),
            xyB=(x, ax0.get_ylim()[0]),
            coordsA="data",
            coordsB="data",
            axesA=ax1,
            axesB=ax0,
            color=line.get_color(),
            lw=0.5,
            zorder=-1,
        )
        ax1.add_patch(con)
        ax0.axvline(x, color=line.get_color(), lw=0.5, zorder=-1)

# Label pure shift peaks
xs, ys, ss = get_pure_shift_labels(estimator, yshift=2.1e6, thold=1e5)
xs[5] += 0.015
xs[6] -= 0.015
ss[3] = "(D)*"
add_pure_shift_labels(axs[0], xs, ys, ss, fs=7)

ax_idx = 0
text_iter = iter(axs[0][0].texts)
for color in colors[:4] + colors[5:]:
    try:
        text = next(text_iter)
    except StopIteration:
        ax_idx += 1
        text_iter = iter(axs[0][ax_idx].texts)
        text = next(text_iter)
    txt = text.get_text()
    txt = txt.replace("(", "").replace(")", "")
    text.set_text(f"\\textbf{{{txt}}}")
    text.set_color(color)
    print(f"{ax_idx}, {color}")

fig.text(
    0.31,
    0.41,
    "H\\textsubscript{2}\\hspace{-0.7pt}O",
    color=colors[4],
    fontsize=7,
)

fig.savefig(save_path)
