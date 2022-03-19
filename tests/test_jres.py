# test_jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 17 Mar 2022 21:51:39 GMT

from nmr_sims.experiments.jres import JresSimulation
from nmr_sims.spin_system import SpinSystem
from nmrespy import ExpInfo
from nmrespy.freqfilter import Filter
from nmrespy import sig


def test_jres():
    system = SpinSystem(
        {
            1: {
                "shift": 3.7,
                "couplings": {
                    2: -10.1,
                    3: 4.3,
                },
                "nucleus": "1H",
            },
            2: {
                "shift": 3.92,
                "couplings": {
                    3: 11.3,
                },
                "nucleus": "1H",
            },
            3: {
                "shift": 4.5,
                "nucleus": "1H",
            },
        }
    )

    sw = ["30Hz", "1ppm"]
    offset = "4.1ppm"
    points = [128, 512]
    channel = "1H"

    simulation = JresSimulation(system, points, sw, offset, channel)
    simulation.simulate()
    _, fid = simulation.fid
    fid = fid.T
    fid += sig._make_noise(fid, snr=30.)
    expinfo = ExpInfo(
        sw=simulation.sweep_widths,
        offset=[0, simulation.offsets[0]],
        sfo=[simulation.sfo[0], simulation.sfo[0]],
        nuclei=[channel, channel],
    )
    region = [None, [4.59, 4.41]]
    noise_region = [None, [4.2, 4.15]]
    f = Filter(
        fid, expinfo, region, noise_region, region_unit="ppm", twodim_dtype="jres"
    )
    signal, exp = f.get_filtered_fid(cut_ratio=None)

    spectrum = sig.ft(signal)
    shifts = sig.get_shifts(exp, spectrum.shape)

    import matplotlib as mpl
    mpl.use("tkAgg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    import numpy as np
    ax.plot_wireframe(*shifts, np.abs(spectrum), rstride=1, cstride=1, lw=0.2)
    plt.show()
