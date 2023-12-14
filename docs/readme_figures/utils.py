# utils.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 13 Dec 2023 10:49:48 EST

from typing import Iterable
import matplotlib as mpl
from matplotlib import colors
import mpl_toolkits
import nmrespy as ne
import numpy as np
from scipy.signal import argrelextrema


def get_pure_shift_labels(estimator: ne.Estimator2DJ, yshift: float = 0., thold=None):
    shifts = estimator.get_shifts(unit="ppm", meshgrid=False)[1]
    cupid_spectrum = estimator.cupid_spectrum().real
    argmaxima = list(argrelextrema(cupid_spectrum, np.greater)[0])
    label_ys = [cupid_spectrum[idx] for idx in argmaxima]
    if thold is not None:
        for i, y in enumerate(label_ys):
            if y < thold:
                label_ys.pop(i)
                argmaxima.pop(i)

    label_xs = [shifts[idx] for idx in argmaxima]
    label_ys = [label_y + yshift for label_y in label_ys]
    label_texts = [f"({chr(65 + i)})" for i, _ in enumerate(argmaxima)]

    return label_xs, label_ys, label_texts


def add_pure_shift_labels(
    axs: np.ndarray[mpl.axes.Axes],
    xs: Iterable[float],
    ys: Iterable[float],
    ss: Iterable[float],
    fs=None,
) -> None:
    active_ax = 0
    for i, (x, y, s) in enumerate(zip(xs, ys, ss)):
        while True:
            l, r = axs[active_ax].get_xlim()
            if l > x > r:
                axs[active_ax].text(
                    x, y, s, ha="center", zorder=1000, fontsize=fs,
                    bbox={"facecolor": "w", "pad": 0.5, "edgecolor": "none"},
                )
                break
            else:
                active_ax += 1


def fix_linewidths(axs, lw):
    for ax_row in axs:
        for ax in ax_row:
            for line in ax.get_lines():
                line.set_linewidth(lw)


def panel_labels(fig, x, ys, start=97, **kwargs):
    for i, y in enumerate(ys):
        fig.text(x, y, f"\\textbf{{{chr(start + i)}.}}", **kwargs)


def raise_axes(axs, scale):
    new_top = scale * axs[0][0].get_ylim()[1]
    for ax in axs[0]:
        ax.set_ylim(top=new_top)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def transfer(old_ax, new_ax, new_fig):
    dim = "3d" if isinstance(new_ax, mpl_toolkits.mplot3d.axes3d.Axes3D) else "2d"
    for child in old_ax.get_children():
        to_add = False
        if dim == "2d":
            if isinstance(child, mpl.lines.Line2D):  #and child.get_xdata().shape[0] != 2:
                to_add = True
                func = new_ax.add_line
            elif isinstance(child, mpl.collections.PathCollection):
                to_add = True
                func = new_ax.add_collection

        elif dim == "3d":
            if (
                isinstance(child, mpl_toolkits.mplot3d.art3d.Line3D) and
                child.get_data_3d()[0].shape[0] != 2
            ):
                to_add = True
                func = new_ax.add_line

        if to_add:
            child.remove()
            child.axes = new_ax
            child.figure = new_fig
            child.set_transform(new_ax.transData)
            func(child)

    coords = ["x", "y"] if dim == "2d" else ["x", "y", "z"]
    for coord in coords:
        for obj in ["ticks", "ticklabels", "label", "lim"]:
            getter = getattr(old_ax, f"get_{coord}{obj}")
            setter = getattr(new_ax, f"set_{coord}{obj}")
            setter(getter())

    if dim == "3d":
        new_ax.view_init(elev=old_ax.elev, azim=old_ax.azim)


