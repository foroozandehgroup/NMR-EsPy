# synth_from_params.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 06 Jan 2023 23:39:15 GMT

import nmrespy as ne
import numpy as np
import matplotlib as mpl
mpl.use("tkAgg")

shifts = [3.7, 3.92, 4.5]
couplings = [(1, 2, -10.1), (1, 3, 4.3), (2, 3, 11.3)]

params = np.array([
    [1, 0, 1864.4, 5],
    [1, 0, 1855.8, 5],
    [1, 0, 1844.2, 5],
    [1, 0, 1835.6, 5],
    [1, 0, 1981.4, 5],
    [1, 0, 1961.2, 5],
    [1, 0, 1958.8, 5],
    [1, 0, 1938.6, 5],
    [1, 0, 2265.6, 5],
    [1, 0, 2257.0, 5],
    [1, 0, 2243.0, 5],
    [1, 0, 2234.4, 5],
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

estimator.view_data(freq_unit="ppm")
