# test_mpm.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 29 Mar 2022 17:22:58 BST

"""Test the nmrespy.mpm module."""

import copy
import sys

import numpy as np

from nmrespy import ExpInfo
from nmrespy.mpm import MatrixPencil
sys.path.insert(0, ".")
from utils import same  # noqa: E402


EXPINFO1D = ExpInfo(dim=1, sw=20., sfo=10., default_pts=2048)
PARAMS1D = np.array(
    [
        [4, 0, -4.5, 1],
        [3, 0, -2.5, 2],
        [1, 0, 2.5, 1],
        [2, 0, 5.5, 2],
    ]
)

PARAMS1D_PPM = copy.deepcopy(PARAMS1D)
PARAMS1D_PPM[:, 2] /= 10.


EXPINFO2D = ExpInfo(dim=2, sw=(20., 20.), sfo=(None, 10.), default_pts=(128, 128))
PARAMS2D = np.array(
    [
        [3, 0, -2.5, 4.5, 1, 1],
        [1, 0, 2.5, 2.5, 1, 1],
        [2, 0, 5.5, 3.5, 1, 1],
        [0.5, 0, 8, 1, 1, 1],
    ]
)

PARAMS2D_PPM = copy.deepcopy(PARAMS2D)
PARAMS2D_PPM[:, 3] /= 10.


def test_1d():
    fid = EXPINFO1D.make_fid(PARAMS1D)
    mpm = MatrixPencil(fid, EXPINFO1D, oscillators=4)
    assert same(mpm.get_result(), PARAMS1D)
    assert same(mpm.get_result("ppm"), PARAMS1D_PPM)


def test_1d_nonzero_start():
    fid = EXPINFO1D.make_fid(PARAMS1D)
    mpm = MatrixPencil(fid[20:], EXPINFO1D, oscillators=4, start_point=[20])
    assert same(mpm.get_result(), PARAMS1D)
    assert same(mpm.get_result("ppm"), PARAMS1D_PPM)


def test_mpm_2d():
    fid = EXPINFO2D.make_fid(PARAMS2D)
    mpm = MatrixPencil(fid, EXPINFO2D, oscillators=4)
    assert same(mpm.get_result(), PARAMS2D)
    assert same(mpm.get_result("ppm"), PARAMS2D_PPM)

    # test _remove_negative_damping
    neg_damping = np.vstack(
        (
            mpm.result,
            np.array(
                [
                    [1, 0, 4, 3, -1, 1],
                    [1, 0, 4, 3, 1, -1],
                ]
            ),
        )
    )
    mpm.result, mpm.oscillators = mpm._remove_negative_damping(neg_damping)
    assert same(mpm.get_result(), PARAMS2D)
