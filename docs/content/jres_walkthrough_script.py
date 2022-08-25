# jres_walkthrough_script.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 25 Aug 2022 18:19:38 BST

from pathlib import Path
import pickle
import nmrespy as ne


directory = Path(ne.__file__).resolve().parent / "examples/jres/sucrose_synthetic/"
with open(directory / "shifts.pkl") as fh:
    shifts = pickle.load(fh)
with open(directory / "couplings.pkl") as fh:
    couplings = pickle.load(fh)
print(shifts)
