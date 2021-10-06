# fixbaseline.py
# Simon Hulse, 6-10-2021
# simon.hulse@chem.ox.ac.uk
# Illustration of baseline correction for spectral data filtered using
# a super-Gaussian band-pass filter.

import pathlib
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path('../..').resolve()))
from nmrespy import freqfilter, load, sig  # noqa: E402
mpl.use('pgf')
plt.style.use('~/Documents/DPhil/formatting/SGH_custom.mplstyle')


p0 = (2.417,)
p1 = (0.,)
region = ((4.92, 4.63),)
noise_region = ((11.5, 11.),)

data, expinfo = load.load_bruker('../data/4/1111', ask_convdta=False)
data = sig.phase(data, p0, p1)
virtual_echo = sig.make_virtual_echo([data])
spectrum = sig.ft(virtual_echo)
filterinfo = freqfilter.filter_spectrum(
    spectrum, expinfo, region, noise_region, region_unit='ppm',
    sg_power=50.
)
fs, expinfo = \
    filterinfo.get_filtered_spectrum(cut_ratio=None, fix_baseline=False)
shifts = sig.get_shifts(expinfo, unit='ppm')[0]
boundaries = filterinfo._get_cf_boundaries()
sg_fix = filterinfo._fit_sg(fs, boundaries)
fs_w_sg = fs + sg_fix
quadratic_fix = filterinfo._fit_quadratic(fs_w_sg, boundaries)
fs_w_sg_and_quad = fs_w_sg + quadratic_fix

fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.05, 0.12, 0.9, 0.83])
ax.plot(shifts, fs, label='Initial filtered spectrum')
ax.plot(shifts, sg_fix, label='super-Gaussian fix')
ax.plot(shifts, fs_w_sg, label='After SG fix')
ax.plot(shifts, quadratic_fix, label='Quadratic fix')
ax.plot(shifts, fs_w_sg_and_quad, label='After quadratic fix')
ax.legend()
ax.set_xlim(region[0][0] + 0.05, region[0][1] - 0.05)
ax.set_xlabel('$^1$H (ppm)')
ax.set_ylim(-7000, 15000)
ax.set_yticks([])
fig.savefig('fix_baseline.pdf')
