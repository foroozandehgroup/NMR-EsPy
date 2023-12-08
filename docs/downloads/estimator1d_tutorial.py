# estimator1d_tutorial.py
# Simon Hulse
# simonhulse@protonmail.com
# Last Edited: Fri 08 Dec 2023 04:12:50 PM EST

from time import sleep
import nmrespy as ne
from nmrespy._colors import ORA, END
import numpy as np
import matplotlib.pyplot as plt

# Set this to True if you don't want to be prompted to press
# <ENTER> after every step of the function
SKIP_PRESS_KEY = True
# Set this to True to prevent interactive mpl figures appearing
SKIP_PLT_SHOW = True

def press_key():
    if not SKIP_PRESS_KEY:
        input(f"\n{ORA}Press <ENTER> to continue...{END}")
        print("\n")

print("\nTutorial for nmrespy.Estimator1D")
press_key()

params = np.array([
    [1., 0., 1864.4, 7.],
    [1., 0., 1855.8, 7.],
    [1., 0., 1844.2, 7.],
    [1., 0., 1835.6, 7.],
    [1., 0., 1981.4, 7.],
    [1., 0., 1961.2, 7.],
    [1., 0., 1958.8, 7.],
    [1., 0., 1938.6, 7.],
    [1., 0., 2265.6, 7.],
    [1., 0., 2257.0, 7.],
    [1., 0., 2243.0, 7.],
    [1., 0., 2234.4, 7.],
])
sfo = 500.
estimator = ne.Estimator1D.new_from_parameters(
    params=params,
    pts=2048,
    sw=1.2 * sfo,
    offset=4.1 * sfo,
    sfo=sfo,
    snr=40.,
)
print("Created an Estimator1D object from synthetic NMR data:")
print(estimator)
press_key()

if not SKIP_PLT_SHOW:
    print("An mpl plot of the data (frequency domain) has been generated")
    estimator.view_data(freq_unit="ppm")
    press_key()

fid = estimator.data
tp = estimator.get_timepoints()[0]
spectrum = estimator.spectrum
shifts = estimator.get_shifts(unit="ppm")[0]
fig, axs = plt.subplots(nrows=2)
axs[0].plot(tp, fid.real)
axs[0].set_xlabel("$t$ (s)")
axs[1].plot(shifts, spectrum.real)
axs[1].set_xlim(reversed(axs[1].get_xlim()))
axs[1].set_xlabel("$^1$H (ppm)")

if not SKIP_PLT_SHOW:
    print("An mpl plot of the time-domain an frequency-domain data has been generated")
    plt.show()
    press_key()

# === Estimating the dataset ===
regions = [(4.6, 4.4), (4.02, 3.82), (3.8, 3.6)]
noise_region = (4.3, 4.25)

region_string = ", ".join([f"{left} - {right} ppm" for (left, right) in regions])
print("The data will now be estimated. Three regions of the data will be considered in turn:")
print(region_string)
press_key()
sleep(3)
for region in regions:
    estimator.estimate(
        region=region, noise_region=noise_region, region_unit="ppm",
    )
print("\nEstimation complete!")
press_key()

# === Inspecting the dataset ===
params1 = estimator.get_params()
print(f"All estimated parameters, frequencies in Hz:\n{params1}\n")
press_key()

errors = estimator.get_errors()
print(f"All errors, frequencies in Hz:\n{errors}\n")
press_key()

params2 = estimator.get_params(indices=[0], funit="ppm")
print(f"Parameters for first region, frequencies in ppm:\n{params2}\n")
press_key()

params3 = estimator.get_params(indices=[1, 2], merge=False, funit="ppm")
print(f"Parameters for second and third regions, split up\n{params3}\n")
press_key()

path = "tutorial_1d"
desc = "Simulated 2,3-Dibromopropanoic acid signal."
for fmt in ("txt", "pdf"):
    estimator.write_result(path=path, fmt=fmt, description=desc)
    press_key()

for (txt, indices) in zip(("complete", "index_1"), (None, [1])):
    fig, ax = estimator.plot_result(
        indices=indices,
        figsize=(4.5, 3.),
        xaxis_unit="ppm",
        axes_left=0.03,
        axes_right=0.97,
        axes_top=0.98,
        axes_bottom=0.15,
    )
    path = f"tutorial_1d_{txt}_fig.pdf"
    fig.savefig(path)
    print(f"Saved figure {path}")
    press_key()

estimator.to_pickle("tutorial_1d")
press_key()

estimator.save_log("tutorial_1d")
press_key()

print("All done!")
