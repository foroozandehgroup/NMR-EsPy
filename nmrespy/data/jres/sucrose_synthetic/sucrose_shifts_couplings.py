# sucrose_shifts_couplings.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 10 Aug 2022 16:22:45 BST

import pickle
import matlab.engine
import numpy as np


def proc_couplings(raw_output):
    couplings = []
    for cinfo in raw_output:
        idx1, idx2, coupling = int(cinfo[0]), int(cinfo[1]), cinfo[2]
        if idx1 < idx2:
            couplings.append((idx1, idx2, coupling))
    return couplings


eng = matlab.engine.start_matlab()
shifts, couplings = eng.sucrose_shifts_couplings(nargout=2)
shifts = list(shifts[0])
couplings = proc_couplings(couplings)

with open("sucrose_shifts.pkl", "wb") as fh:
    pickle.dump(shifts, fh)
with open("sucrose_couplings.pkl", "wb") as fh:
    pickle.dump(couplings, fh)
