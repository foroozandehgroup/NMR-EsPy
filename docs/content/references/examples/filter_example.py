import matplotlib.pyplot as plt
import numpy as np

from nmrespy import ExpInfo
from nmrespy.freqfilter import Filter

expinfo = ExpInfo(
    dim=1,
    sw=1000.0,
    sfo=400.0,
    nuclei="1H",
    default_pts=4096,
)

parameters = np.array(
    [
        [1, 0, 300, 3],
        [2, 0, 290, 3],
        [1, 0, 280, 3],
        [1, 0, -150, 3],
        [3, 0, -155, 3],
        [3, 0, -160, 3],
        [1, 0, -165, 3],
    ]
)

# Make FID with SNR of 30dB
fid = expinfo.make_fid(parameters, snr=30.0)
region = [324.0, 260.0]
noise_region = [100.0, 50.0]
filterobj = Filter(
    fid,
    expinfo,
    region,
    noise_region,
)

og_tp, = expinfo.get_timepoints()
full_spectrum = filterobj.spectrum
full_filtered_spectrum, full_expinfo = filterobj.get_filtered_spectrum(cut_ratio=None)
full_filtered_fid, _ = filterobj.get_filtered_fid(cut_ratio=None)
full_shifts, = full_expinfo.get_shifts(unit="ppm", pts=full_filtered_spectrum.shape)
full_tp, = full_expinfo.get_timepoints(pts=full_filtered_fid.shape)

cut_spectrum, cut_expinfo = filterobj.get_filtered_spectrum()
cut_filtered_fid, _ = filterobj.get_filtered_fid()
cut_shifts, = cut_expinfo.get_shifts(unit="ppm", pts=cut_spectrum.shape)
cut_tp, = cut_expinfo.get_timepoints(pts=cut_filtered_fid.shape)

sg = np.amax(full_spectrum) * filterobj.sg

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
axs[0, 0].plot(full_shifts, full_spectrum.real, label="Original spectrum")
axs[0, 0].plot(full_shifts, full_filtered_spectrum.real, lw=0.5, label="Filtered spectrum")
axs[0, 0].plot(full_shifts, sg.real, label="Super-Gaussian filter")
axs[0, 1].plot(cut_shifts, cut_spectrum.real, label="Filtered, cut spectrum")
axs[1, 0].plot(og_tp, fid, label="Original FID")
axs[1, 0].plot(full_tp, full_filtered_fid, label="Filtered FID")
axs[1, 1].plot(cut_tp, cut_filtered_fid, label="Filtered, cut FID")

for ax in axs[0]:
    ax.set_xlim(reversed(ax.get_xlim()))
    ax.set_xlabel(f"{expinfo.unicode_nuclei[0]} (ppm)")
    ax.legend(fontsize=8, loc=1)

for ax in axs[1]:
    ax.set_xlabel("$t$ (s)")
    ax.legend(fontsize=8, loc=1)

axs[0, 0].set_ylim((axs[0, 0].get_ylim()[0], axs[0, 0].get_ylim()[1] * 1.2))
axs[0, 1].set_ylim((axs[0, 1].get_ylim()[0], axs[0, 1].get_ylim()[1] * 1.08))
fig.savefig("filtered_example.png")
