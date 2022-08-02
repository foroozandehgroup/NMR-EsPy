# sucrose.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 02 Aug 2022 01:47:45 BST

import copy
import os
from pathlib import Path
import subprocess
from typing import Optional, Tuple
import nmrespy as ne
import numpy as np
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("tkAgg")
mpl.rcParams["font.size"] = 6


EXPINFO = ne.ExpInfo(
    dim=2,
    sw=(40., 2200.),
    offset=(0., 1000.),
    sfo=(None, 300.),
    nuclei=(None, "1H"),
)

REGIONS = (
    (6.08, 5.91),
    (4.72, 4.46),
    (4.46, 4.22),
    (4.22, 4.1),
    (4.09, 3.98),
    (3.98, 3.83),
    (3.58, 3.28),
    (2.08, 1.16),
    (1.05, 0.0),
)


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)


def simulate(
    pts: Tuple[int, int],
    homodecouple: bool = False,
) -> None:
    cmd = (
        "matlab -nodesktop -nosplash -nodisplay -r " +
        (
            "\"run("
            f"\'sucrose_jres({pts[0]},{pts[1]},{str(homodecouple).lower()})\'); "
            "exit\""
        )
    )
    with cd("spinach/"):
        print(
            "Simulating 2DJ signal: "
            f"TD1: {pts[0]}, TD2: {pts[1]}, decouple: {homodecouple}"
        )
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)


def get_fid(
    pts: Tuple[int, int],
    homodecoupled: bool = False,
    snr: Optional[float] = None,
    lb: Optional[float] = None,
) -> np.ndarray:
    path = f"spinach/sucrose_2dj_#{pts[0]}_{pts[1]}.mat"
    if homodecoupled:
        path = path.replace("#", "homo_")
    else:
        path = path.replace("#", "")
    path = Path(path)
    if not path.is_file():
        simulate(pts, homodecoupled)
    fid = loadmat(path)["fid"].T

    fid = ne.sig.phase(fid, (0., np.pi / 2), (0., 0.))

    if snr is not None:
        fid = ne.sig.add_noise(fid, snr)
    if lb is not None:
        fid = ne.sig.exp_apodisation(fid, lb)

    return fid


def get_estimator(
    pts: Tuple[int, int],
    snr: float,
    estimate_anyway: bool = False,
) -> ne.Estimator2DJ:
    path = Path(f"estimators/estimator_{pts[0]}_{pts[1]}")
    if not path.with_suffix(".pkl").is_file() or estimate_anyway:
        fid = get_fid(pts, snr=20., lb=5.)
        EXPINFO._default_pts = pts
        estimator = ne.Estimator2DJ(fid, EXPINFO)
        print(f"Estimating 2DJ signal: TD1: {pts[0]}, TD2: {pts[1]}")
        for region in REGIONS:
            print(f"\tregion: {region[0]} - {region[1]}ppm")
            estimator.estimate(
                region=region, noise_region=(5.5, 5.33), region_unit="ppm",
                max_iterations=40, nlp_trim=512, fprint=False,
                phase_variance=True,
            )
        if not path.parent.is_dir():
            os.mkdir(path.parent)
        estimator.to_pickle(path)
        return estimator
    else:
        return ne.Estimator2DJ.from_pickle(path)


def rm_spurious_oscillators(
    estimator: ne.Estimator2DJ,
    estimate_anyway: bool = False,
) -> ne.Estimator2DJ:
    pts = estimator.default_pts
    path = Path(f"estimators/estimator_{pts[0]}_{pts[1]}_rm_spurious")
    if not path.with_suffix(".pkl").is_file() or estimate_anyway:
        estimator.remove_spurious_oscillators(
            max_iterations=20, phase_variance=True, nlp_trim=512,
        )
        estimator.to_pickle(path)
        return estimator
    else:
        return ne.Estimator2DJ.from_pickle(path)


def make_spectrum(fid: np.ndarray) -> np.ndarray:
    """Create spectrum from FID."""
    new_fid = copy.deepcopy(fid)
    idx = tuple(fid.ndim * [0])
    new_fid[idx] *= 0.5
    return ne.sig.ft(new_fid).real


def plot_homodecoupled(estimator: ne.Estimator2DJ) -> mpl.figure.Figure:
    """Plot derived homodecoupled signal against synthesised homodecoupled signal."""
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    f2_shifts = estimator.get_shifts(unit="ppm", meshgrid=False)[1]
    homodecoupled_result = make_spectrum(estimator.negative_45_signal())
    homodecoupled_spectrum = make_spectrum(
        get_fid(estimator.default_pts, lb=5., homodecoupled=True)[0]
    )
    ax.plot(f2_shifts, homodecoupled_result)
    ax.plot(f2_shifts, homodecoupled_spectrum)

    ax.set_xlim(reversed(ax.get_xlim()))
    return fig


if __name__ == "__main__":
    # If False and an estimator corresponding to the specified `pts` and `snr`
    # variables already exists, skip re-running the estimation routine.
    # If True, generate a new estimator and run the estimation routine.
    estimate_anyway = False
    pts = (64, 8192)
    snr = 30.

    estimator = get_estimator(pts, snr, estimate_anyway=estimate_anyway)
    estimator = rm_spurious_oscillators(estimator, estimate_anyway=estimate_anyway)
    multifig = estimator.plot_multiplets(shifts_unit="ppm")
    contourfig = estimator.plot_contour(
        nlevels=20, base=10., factor=1.1, shifts_unit="ppm",
    )
    plot_homodecoupled(estimator)
    plt.show()
