# spinach_1d.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 06 Jan 2023 18:56:56 GMT

import nmrespy as ne

# 2,3-Dibromopropanoic acid
shifts = [3.7, 3.92, 4.5]
couplings = [(1, 2, -10.1), (1, 3, 4.3), (2, 3, 11.3)]
sfo = 500.
offset = 4.1 * sfo  # Hz
sw = 1. * sfo
estimator = ne.Estimator1D.new_spinach(
    shifts=shifts,
    couplings=couplings,
    pts=2048,
    sw=sw,
    offset=offset,
    sfo=sfo,
)
