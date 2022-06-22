# bbqchili.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 22 Jun 2022 20:27:29 BST

import matplotlib.pyplot as plt
import nmrespy as ne
import numpy as np
from scipy.io import loadmat


def gen_spectrum(data: np.ndarray) -> np.ndarray:
    data[0] /= 2
    return ne.sig.ft(data)


data = loadmat("data/chirp_sim.mat")["chirp"].reshape((-1,))
data += ne.sig._make_noise(data, snr=30.)
bbq = ne.BBQChili(
    data=data,
    expinfo=ne.ExpInfo(1, sw=500.e3),
    pulse_length=100.e-6,
    pulse_bandwidth=500.e3,
    prescan_delay=0.,
)
bbq.estimate()
original_fid = bbq.data
original_spectrum = gen_spectrum(original_fid)
fixed_fid = bbq.back_extrapolate()
fixed_spectrum = gen_spectrum(fixed_fid)
shifts, = bbq.get_shifts()

fig, ax = plt.subplots()
for spec in (original_spectrum, fixed_spectrum):
    ax.plot(shifts, spec.real)
ax.set_xlim(70000, -70000)
fig.savefig("bbq.pdf")
