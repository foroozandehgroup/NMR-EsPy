from nmrespy import freqfilter, sig

import matplotlib as mpl
mpl.use("pgf")
import matplotlib.pyplot as plt
plt.style.use("./lato.mplstyle")
from matplotlib import patches

import numpy as np

fig = plt.figure(figsize=(4,6))
bottom = 0
top = 1
left = 0.05
right = 0.95

total_height = top - bottom
total_width = right - left

axes = []
for i in range(2, -1, -1):
    axes.append(fig.add_axes(
        [left, bottom + (i * total_height / 3), total_width, total_height/3]
    ))
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])
    for orient in ["top", "bottom", "left", "right"]:
        axes[-1].spines[orient].set_visible(False)

n = [4096]
sw = [100.]
para = np.array([
    [1, 0, 40, 0.7],
    [2, 0, 41, 0.7],
    [2, 0, 42, 0.7],
    [1, 0, 43, 0.7],
    [0.7, 0, 29, 0.7],
    [1.4, 0, 30, 0.7],
    [0.7, 0, 31, 0.7],
    [1.5, 0, 9, 0.7],
    [3, 0, 8, 0.7],
    [1.5, 0, 7, 0.7],
    [1.5, 0, 6, 0.7],
    [3, 0, 5, 0.7],
    [1.5, 0, 4, 0.7],
    [1.8, 0, -10, 0.7],
    [1.8, 0, -11, 0.7],
    [1.8, 0, -13, 0.7],
    [1.8, 0, -14, 0.7],
    [0.9, 0, -28, 0.7],
    [1.8, 0, -29.5, 0.7],
    [0.9, 0, -31, 0.7],
])

spectrum = np.real(sig.ft(sig.make_fid(para, n, sw)[0]))
spectrum = spectrum - np.mean(spectrum[0:10])

region = [[1600, 1950]]
sg, c, bw = freqfilter.super_gaussian(region, n, p=40.0)
c = c[0]
bw = bw[0]

filter_spectrum = spectrum * sg

cut_ratio = 3
cut_region = [int(np.floor(c - (bw * cut_ratio / 2))),
              int(np.ceil(c + (bw * cut_ratio / 2)))]

axes[0].plot(spectrum)
axes[0].plot(sg * 1.1 * np.amax(spectrum))
filter_span = patches.ConnectionPatch(
    xyA=(region[0][0], -30), xyB=(region[0][1], -30),
    coordsA=axes[0].transData, coordsB=axes[0].transData,
    arrowstyle="|-|,widthA=0.2,widthB=0.2", axesA=axes[0], axesB=axes[0],
)
axes[0].add_artist(filter_span)
axes[0].text(
    c, -70, "Filter width",
    horizontalalignment="center", verticalalignment="center",
)
axes[0].set_ylim(-100, 1.2 * np.amax(spectrum))


axes[1].plot(filter_spectrum)
axes[1].set_ylim(-140, 1.2 * np.amax(spectrum))
filter_span = patches.ConnectionPatch(
    xyA=(cut_region[0], -30), xyB=(cut_region[1], -30),
    coordsA=axes[1].transData, coordsB=axes[1].transData,
    arrowstyle="|-|,widthA=0.2,widthB=0.2", axesA=axes[1], axesB=axes[1],
)
axes[1].add_artist(filter_span)
axes[1].text(
    c, -70, "Cut ratio $\\times$ Filter width ",
    horizontalalignment="center", verticalalignment="center",
)
axes[1].set_ylim(axes[0].get_ylim())

axes[2].plot(
    np.arange(n[0])[cut_region[0]:cut_region[1]],
    filter_spectrum[cut_region[0]:cut_region[1]],
)
axes[2].set_xlim(axes[0].get_xlim())
axes[2].set_ylim(axes[0].get_ylim())


filter_arrow = patches.ConnectionPatch(
    xyA=(0.6, 0.7), xyB=(0.6, 0.6),
    connectionstyle="arc3,rad=-0.5",
    coordsA="figure fraction", coordsB="figure fraction",
    arrowstyle="->"
)
fig.add_artist(filter_arrow)
fig.text(
    0.69, 0.65, "Filter",
    horizontalalignment="center", verticalalignment="center",
)

cut_arrow = patches.ConnectionPatch(
    xyA=(0.6, 0.32), xyB=(0.6, 0.22),
    connectionstyle="arc3,rad=-0.5",
    coordsA="figure fraction", coordsB="figure fraction",
    arrowstyle="->"
)
fig.add_artist(cut_arrow)
fig.text(
    0.69, 0.27, "Cut",
    horizontalalignment="center", verticalalignment="center",
)

to_td_arrow = patches.ConnectionPatch(
    xyA=(0.62, 0.1), xyB=(0.9, 0.1),
    coordsA="figure fraction", coordsB="figure fraction",
    arrowstyle="->"
)
fig.add_artist(to_td_arrow)
fig.text(
    0.62, 0.14, "Generate time-domain\nsignal...",
    horizontalalignment="left", verticalalignment="center",
)

fig.savefig("filter_cut.png")
