# good_bad_noise_regions.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 09 Jan 2023 15:26:29 GMT

import nmrespy as ne
import numpy as np
import matplotlib.pyplot as plt

shifts = [3.7, 3.92, 4.5]
couplings = [(1, 2, -10.1), (1, 3, 4.3), (2, 3, 11.3)]

params = np.array([
    [1, 0, 1864.4, 7],
    [1, 0, 1855.8, 7],
    [1, 0, 1844.2, 7],
    [1, 0, 1835.6, 7],
    [1, 0, 1981.4, 7],
    [1, 0, 1961.2, 7],
    [1, 0, 1958.8, 7],
    [1, 0, 1938.6, 7],
    [1, 0, 2265.6, 7],
    [1, 0, 2257.0, 7],
    [1, 0, 2243.0, 7],
    [1, 0, 2234.4, 7],
])
pts = 2048
sfo = 500.
sw = sfo * 1.
offset = sfo * 4.1

estimator = ne.Estimator1D.new_synthetic_from_parameters(
    params=params,
    pts=pts,
    sw=sw,
    offset=offset,
    sfo=sfo,
    snr=40.,
)

shifts = estimator.get_shifts(unit="ppm")[0]
spectrum = estimator.spectrum.real

fig, ax = plt.subplots(
    gridspec_kw=dict(
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.17,
    ),
    figsize=(4.5, 2.5),
)

ax.plot(shifts, spectrum, color="k")
ax.set_xlim(reversed(ax.get_xlim()))

ax.axvspan(xmin=4.47, xmax=4.53, color="#ffb2b2")
ax.axvspan(xmin=4.02, xmax=3.82, color="#b2d8b2")
ax.axvspan(xmin=4.3, xmax=4.25, color="#b0b0b0")
for x in ("top", "left", "right"):
    ax.spines[x].set_visible(False)
ax.set_yticks([])
ax.set_xlabel("\\textsuperscript{1}H (ppm)", labelpad=1)

fig.savefig("media/good_bad_noise_regions.png")
