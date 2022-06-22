# test_bbqchili.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 22 Jun 2022 20:03:28 BST

import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import nmrespy as ne
import numpy as np
from scipy.io import loadmat
import utils
mpl.use("tkAgg")


class DefaultEstimator:

    data = loadmat("tests/data/chirp_sim.mat")["chirp"].reshape((-1,))
    data += ne.sig._make_noise(data, snr=30.)
    _before_estimation = ne.BBQChili(
        data=data,
        expinfo=ne.ExpInfo(1, sw=500.e3),
        pulse_length=100.e-6,
        pulse_bandwidth=500.e3,
        prescan_delay=0.,
    )

    @classmethod
    def before_estimation(cls):
        return copy.deepcopy(cls._before_estimation)

    @classmethod
    def after_estimation(cls):
        if hasattr(cls, "_after_estimation"):
            return copy.deepcopy(cls._after_estimation)
        else:
            cls._after_estimation = cls.before_estimation()
            cls._after_estimation.estimate()

        return cls.after_estimation()


def test_basic_parameters():
    estimator = DefaultEstimator.before_estimation()
    assert estimator.dim == 1
    assert estimator.default_pts == (262145,)
    assert estimator.sw() == estimator.sw(unit="hz") == (500.e3,)
    assert estimator.offset() == estimator.offset(unit="hz") == (0.,)
    assert estimator.sfo is None
    assert estimator.nuclei is None
    assert estimator.latex_nuclei is None
    assert estimator.unicode_nuclei is None
    assert estimator.fn_mode is None


def test_timepoints():
    estimator = DefaultEstimator.before_estimation()
    tp = estimator.get_timepoints()
    assert isinstance(tp, tuple)
    assert len(tp) == 1
    assert utils.equal(tp[0], np.linspace(0, 262144 / 500e3, 262145))

    tp = estimator.get_timepoints(pts=8192, start_time="20dt")
    assert utils.equal(tp[0], np.linspace(20 / 500e3, 8211 / 500e3, 8192))


def test_shifts():
    estimator = DefaultEstimator.before_estimation()
    shifts = estimator.get_shifts()
    assert isinstance(shifts, tuple)
    assert len(shifts) == 1
    assert utils.equal(shifts[0], np.linspace(250e3, -250e3, 262145))

    shifts = estimator.get_shifts(pts=8192, flip=False)
    assert utils.equal(shifts[0], np.linspace(-250e3, 250e3, 8192))


def test_estimate():
    # `after_estimation` will run estimate on two different regions.
    # Afterwards, estimate is tested with `region` set to `None`, and with different
    # optimisation methods invoked.
    DefaultEstimator.after_estimation()


def test_back_extrapolate():
    estimator = DefaultEstimator.after_estimation()
    fid = estimator.back_extrapolate()
    fid[0] /= 2
    fig = plt.figure()
    ax = fig.add_subplot()
    shifts, = estimator.get_shifts()
    spectrum = ne.sig.ft(fid)
    ax.plot(shifts, spectrum)
    ax.set_xlim(shifts[0], shifts[-1])
    plt.show()
