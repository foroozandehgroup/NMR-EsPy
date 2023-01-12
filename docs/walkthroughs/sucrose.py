# sucrose.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 12 Jan 2023 17:40:19 GMT

from pathlib import Path
import pickle
import nmrespy as ne
import numpy as np
import matplotlib.pyplot as plt

# fid_path = Path(ne.__file__).expanduser().parents[1] \
#     / "samples/jres_sucrose_sythetic/sucrose_jres_fid.pkl"

# with open(fid_path, "rb") as fh:
#     fid = pickle.load(fh)

# expinfo = ne.ExpInfo(
#     dim=2,
#     sw=(30., 2200.),
#     offset=(0., 1000.),
#     sfo=(None, 300.),
#     nuclei=(None, "1H"),
#     default_pts=(64, 4096),
# )
# estimator = ne.Estimator2DJ(fid, expinfo)

# regions = (
#     (6.08, 5.91),
#     (4.72, 4.46),
#     (4.46, 4.22),
#     (4.22, 4.1),
#     (4.09, 3.98),
#     (3.98, 3.83),
#     (3.58, 3.28),
#     (2.08, 1.16),
#     (1.05, 0.0),
# )
# n_regions = len(regions)
# initial_guesses = n_regions * [None]
# initial_guesses[1:3] = [16, 16]
# # kwargs common to estimation of each region
# common_kwargs = {
#     "noise_region": (5.5, 5.33),
#     "region_unit": "ppm",
#     "max_iterations": 200,
#     "phase_variance": True,
# }
# for init_guess, region in zip(initial_guesses, regions):
#     kwargs = {**{"region": region, "initial_guess": init_guess}, **common_kwargs}
#     estimator.estimate(**kwargs)

# # estimator.to_pickle("sucrose_backup")
# estimator.to_pickle("walkthroughs/sucrose")

estimator = ne.Estimator2DJ.from_pickle("walkthroughs/sucrose")

# Normal 1D spectrum
init_spectrum = estimator.spectrum_zero_t1.real
# Homodecoupled spectrum produced using CUPID
cupid_spectrum = estimator.cupid_spectrum().real
# Get direct-dimension shifts
shifts = estimator.get_shifts(unit="ppm", meshgrid=False)[-1]

fig, ax = plt.subplots(figsize=(4.5, 2.5))
ax.plot(shifts, init_spectrum, color="k")
ax.plot(shifts, cupid_spectrum, color="r")
# The most interesting region of the spectrum
ax.set_xlim(4.7, 3.8)

# ========================
# These lines are just for plot aesthetics
for x in ("top", "left", "right"):
    ax.spines[x].set_visible(False)
ax.set_xticks([4.7 - 0.1 * i for i in range(10)])
ax.set_yticks([])
ax.set_position([0.03, 0.175, 0.94, 0.83])
ax.set_xlabel(f"{estimator.latex_nuclei[1]} (ppm)")
# ========================

fig.savefig("media/cupid_spectrum.png")

indices = [1, 2, 3, 4, 5]
multiplets = estimator.predict_multiplets(indices=indices)
params_1d = estimator.get_params(indices=indices)[:, [0, 1, 3, 5]]
expinfo_1d = estimator.direct_expinfo
spectra = []
for (freq, idx) in multiplets.items():
    print(f"{freq / estimator.sfo[1]:.4f}ppm: {idx}")
    mp_params = params_1d[idx]
    fid = expinfo_1d.make_fid(mp_params)
    fid[0] *= 0.5
    spectrum = ne.sig.ft(fid).real
    spectra.append(spectrum)
from itertools import cycle
colors = cycle(["#84c757", "#ef476f", "#ffd166", "#36c9c6"])
_, shifts_f2 = estimator.get_shifts(unit="ppm", meshgrid=False)
fig, ax = plt.subplots(figsize=(4.5, 2.5))
for spectrum in spectra:
    ax.plot(shifts, spectrum, color=next(colors))
ax.set_xlim(4.7, 3.8)
# ========================
# These lines are just for plot aesthetics
for x in ("top", "left", "right"):
    ax.spines[x].set_visible(False)
ax.set_xticks([4.7 - 0.1 * i for i in range(10)])
ax.set_yticks([])
ax.set_position([0.03, 0.175, 0.94, 0.83])
ax.set_xlabel(f"{estimator.latex_nuclei[1]} (ppm)")
# ========================
fig.savefig("media/multiplets.png")

sheared_fid = estimator.sheared_signal(indirect_modulation="phase")
print(sheared_fid.shape)
# Generates 2rr, 2ri, 2ir, 2ii spectra
sheared_spectrum = ne.sig.proc_phase_modulated(sheared_fid)
print(sheared_spectrum.shape)
spectrum_2rr = sheared_spectrum[0]
shifts_f1, shifts_f2 = estimator.get_shifts(unit="ppm")
fig, ax = plt.subplots(figsize=(4.5, 2.5))
# Contour levels
base, factor, nlevels = 20, 1.4, 10
levels = [base * factor ** i for i in range(nlevels)]
ax.contour(
    shifts_f2,
    shifts_f1,
    spectrum_2rr,
    colors="k",
    levels=levels,
    linewidths=0.2,
)
ax.set_xlim(4.7, 3.8)
ax.set_ylim(10., -10.)
# ========================
# These lines are just for plot aesthetics
ax.set_xticks([4.7 - 0.1 * i for i in range(10)])
ax.set_position([0.12, 0.175, 0.85, 0.79])
ax.set_xlabel(f"{estimator.latex_nuclei[1]} (ppm)", labelpad=1)
ax.set_ylabel("Hz", labelpad=1)
# ========================
fig.savefig("media/sheared_spectrum.png")

fig, axs = estimator.plot_result(
    indices=[1, 2, 3, 4, 5],
    region_unit="ppm",
    marker_size=5.,
    figsize=(4.5, 2.5),
    # Number of points to construct homodecoupled signal
    # and multiplet structures from
    high_resolution_pts=16384,
    # There is a lot of scope for editing the colours of
    # multiplets. See the reference!
    # Here I specify a recognised name of a colourmap in
    # matplotlib.
    multiplet_colors="inferno",
    # Argumnets of the position of the plot in the figure
    axes_left=0.1,
    axes_bottom=0.15,
)
fig.savefig("media/plot_result_2dj.png")
