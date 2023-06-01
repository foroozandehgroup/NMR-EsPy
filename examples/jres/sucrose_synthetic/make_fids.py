# make_fids.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 25 Aug 2022 16:55:16 BST

import pickle

import nmrespy as ne

with open("shifts.pkl", "rb") as fh:
    shifts = pickle.load(fh)
with open("couplings.pkl", "rb") as fh:
    couplings = pickle.load(fh)

pts = (64, 4096)
sw = (40., 2200.)
offset = 1000.
field = 300.
field_unit = "MHz"

est_coupled = ne.Estimator2DJ.new_spinach(
    shifts, pts, sw, offset, couplings=couplings, field=field, field_unit=field_unit,
)
with open("fid.pkl", "wb") as fh:
    pickle.dump(est_coupled.data, fh)

est_decoupled = ne.Estimator2DJ.new_spinach(
    shifts, pts, sw, offset, field=field, field_unit=field_unit,
)
with open("fid_decoupled.pkl", "wb") as fh:
    pickle.dump(est_decoupled.data, fh)
