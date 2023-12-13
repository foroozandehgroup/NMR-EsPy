# make_figure.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 12 Dec 2023 08:44:01 PM EST

from pathlib import Path
assert (utils_dir := Path("~/Documents/DPhil/thesis/utils").expanduser()).is_dir()
import sys
sys.path.insert(0, str(utils_dir))
from utils import transfer

import nmrespy as ne
import matplotlib as mpl
import matplotlib.pyplot as plt


height = 2.5
width_factor = 2.

colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
estimator = ne.Estimator1D.from_pickle("~/Documents/DPhil/results/onedim/cyclosporin/estimator")

data_ylim = (3e3, 2.3e5)
mpm_ylim = (-2e4, 3.4e5)
nlp_ylim = (-1.5e4, 3.1e5)

colors.append("#808080")

cols = [
    colors[i] for i in [
        -1, 5, 5, -1, 5, 4, 5, 4, 4, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3,
        2, -1, 2, 2, 2, 2, -1, -1, 2, -1, -1,
        1, 1, 1, 1, 0, 0, 0, 0,
    ]
]

kwargs = dict(
    axes_top=0.98,
    axes_bottom=0.15,
    axes_left=0.02,
    axes_right=0.98,
    xaxis_unit="ppm",
    axes_region_separation=0.02,
    oscillator_line_kwargs={"lw": 0.7},
    xaxis_ticks=[
        [0, [5.52 - 0.02 * i for i in range(5)]],
        [1, [5.28 - 0.02 * i for i in range(5)]],
        [2, [5.02 - 0.02 * i for i in range(8)]],
    ],
    figsize=(width_factor * height, height),
    label_peaks=False,
    plot_model=False,
)

fig, axs = estimator.plot_result(oscillator_colors=cols, **kwargs)

fig.savefig("cyclosporin/cyclosporin.pdf")
