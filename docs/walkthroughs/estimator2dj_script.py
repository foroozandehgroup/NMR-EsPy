# jres_walkthrough_script.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 26 Aug 2022 15:44:53 BST

# A script which runs through what is mentionned in the 2DJ Walkthrough in the
# NMR-EsPy docs.
# If you have setup the MATLAB engine for Python, and have installed Spinach,
# set I_CAN_USE_SPINACH = True
# otherwise, set I_CAN_USE_SPINACH = False

from pathlib import Path
import pickle
import nmrespy as ne

I_CAN_USE_SPINACH = False
directory = Path(ne.__file__).resolve().parent / "data/jres/sucrose_synthetic/"

pts = (64, 4096)
sw = (40., 2200.)
offset = 1000.
field = 300.
field_unit = "MHz"

with open("est.pkl", "rb") as fh:
    estimator = pickle.load(fh)

# # --- Generate the Estimator ---
# if I_CAN_USE_SPINACH:
#     # Obtain the chemical shifts and scalar couplings of sucrose
#     with open(directory / "shifts.pkl", "rb") as fh:
#         shifts = pickle.load(fh)
#     with open(directory / "couplings.pkl", "rb") as fh:
#         couplings = pickle.load(fh)
#     estimator = ne.Estimator2DJ.new_spinach(
#         shifts, pts, sw, offset, couplings=couplings, field=field,
#         field_unit=field_unit,
#     )
# else:
#     # Load presimulated sucrose data, add noise, and create an ExpInfo object.
#     with open(directory / "fid.pkl", "rb") as fh:
#         fid = pickle.load(fh)
#     fid = ne.sig.add_noise(fid, 20.)
#     expinfo = ne.ExpInfo(
#         dim=2, sw=sw, offset=(0., offset), sfo=(None, field), nuclei=(None, "1H"),
#         default_pts=pts,
#     )
#     estimator = ne.Estimator2DJ(fid, expinfo)

# # --- Estimate the signal ---
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
# common_kwargs = {
#     "noise_region": (5.5, 5.33),
#     "region_unit": "ppm",
#     "max_iterations": 40,
#     "nlp_trim": 512,
#     "fprint": False,
#     "phase_variance": True,
# }
# for i, region in enumerate(regions, start=1):
#     print(f"---> {i} / {n_regions}: {region[0]} - {region[1]}ppm")
#     kwargs = {**{"region": region}, **common_kwargs}
#     estimator.estimate(**kwargs)

# # --- Remove spurious oscillators from the result ---
# estimator.remove_spurious_oscillators(
#     max_iterations=30,
#     phase_variance=True,
#     nlp_trim=512,
# )

# with open("est.pkl", "wb") as fh:
#     pickle.dump(estimator, fh)

import matplotlib as mpl
mpl.use("tkAgg")
import matplotlib.pyplot as plt
estimator.plot_multiplets(shifts_unit="ppm")
plt.show()
