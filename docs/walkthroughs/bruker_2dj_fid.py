# bruker_2dj_fid.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 10 Jan 2023 15:17:09 GMT

import nmrespy as ne

estimator = ne.Estimator2DJ.new_bruker("/home/parsley/mf/jesu2901/DPhil/data/dexamethasone/3")
estimator.phase_data(p0=0.041, p1=-6.383, pivot=1923)
estimator.baseline_correction()
print(estimator)
