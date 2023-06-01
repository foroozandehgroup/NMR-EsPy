# bruker_1d_fid.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 06 Jan 2023 22:33:49 GMT

import nmrespy as ne


estimator = ne.Estimator1D.new_bruker("home/simon/nmr_data/andrographolide/1")
estimator.phase_data(p0=2.653, p1=-5.686, pivot=13596)
estimator.baseline_correction()
