# estimator_1d_script.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 09 Jan 2023 17:01:54 GMT

import nmrespy as ne
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

params = np.array([
    [1, 0, 1864.4, 7],
    [1, 0, 1855.8, 7],
    [1, 0, 1844.2, 7],
    [1, 0, 1835.6, 7],
    [1, 0, 1981.4, 7],
    [1, 0, 1961.2, 7],
    [1, 0, 1958.8, 7],
    [1, 0, 1938.6, 7],
    [1, 0, 2265.6, 7],
    [1, 0, 2257.0, 7],
    [1, 0, 2243.0, 7],
    [1, 0, 2234.4, 7],
])
pts = 2048
sfo = 500.
sw = sfo * 1.2
offset = sfo * 4.1

estimator = ne.Estimator1D.new_synthetic_from_parameters(
    params=params,
    pts=pts,
    sw=sw,
    offset=offset,
    sfo=sfo,
    snr=40.,
)

regions = [(4.6, 4.4), (4.02, 3.82), (3.8, 3.6)]
noise_region = (4.3, 4.25)
for region in regions:
    estimator.estimate(
        region=region, noise_region=noise_region, region_unit="ppm",
    )

print("All params, frequencies in Hz:")
print(estimator.get_params())
print("All errors, frequencies in Hz")
print(estimator.get_errors())
print("Params for first region, frequencies in ppm")
print(estimator.get_params(indices=[0], funit="ppm"))
print("Params for second and third regions, split up")
print(estimator.get_params(indices=[1, 2], merge=False, funit="ppm"))

for fmt in ("txt", "pdf"):
    estimator.write_result(
        path="downloads/tutorial_1d",
        fmt=fmt,
        description="Simulated 2,3-Dibromopropanoic acid signal.",
    )

for (txt, indices) in zip(("complete", "index_1"), (None, [1])):
    fig, ax = estimator.plot_result(
        indices=indices,
        figure_size=(4.5, 3.),
        region_unit="ppm",
        axes_left=0.03,
        axes_right=0.97,
        axes_top=0.98,
        axes_bottom=0.09,
    )
    fig.savefig(f"downloads/tutorial_1d_{txt}_fig.pdf")

estimator.to_pickle("downloads/dpa_estimator")
estimator.save_log("downloads/dpa_estimator")
