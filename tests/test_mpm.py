"""Test the nmrespy.mpm module."""

import copy

import numpy as np
from nmrespy import ExpInfo
from nmrespy.mpm import MatrixPencil
from nmrespy.sig import make_fid


def test_mpm_1d():
    """Test 1D MPM."""
    params = np.array(
        [
            [4, 0, -4.5, 1],
            [3, 0, -2.5, 2],
            [1, 0, 2.5, 1],
            [2, 0, 5.5, 2],
        ]
    )
    expinfo = ExpInfo(pts=1024, sw=20, sfo=10)
    fid = make_fid(params, expinfo)[0]
    mpm = MatrixPencil(fid, expinfo, M=4)
    result = mpm.get_result()
    assert np.allclose(result, params, rtol=0, atol=1e-8)
    ppm_params = copy.deepcopy(params)
    ppm_params[:, 2] /= 10.0
    assert np.allclose(mpm.get_result(freq_unit="ppm"), ppm_params, rtol=0, atol=1e-8)

    # test with FID not starting at t=0
    mpm = MatrixPencil(fid[20:], expinfo, M=4, start_point=[20])
    result = mpm.get_result()
    assert np.allclose(result, params, rtol=0, atol=1e-8)


def test_mpm_2d():
    """Test 2D MPM."""
    params = np.array(
        [
            [3, 0, -2.5, 4.5, 1, 1],
            [1, 0, 2.5, 2.5, 1, 1],
            [2, 0, 5.5, 3.5, 1, 1],
            [0.5, 0, 8, 1, 1, 1],
        ]
    )
    expinfo = ExpInfo(pts=128, sw=20, sfo=10, dim=2)

    fid = make_fid(params, expinfo)[0]
    mpm = MatrixPencil(fid, expinfo, M=4)
    assert np.allclose(mpm.get_result(), params, rtol=0, atol=1e-8)
    ppm_params = copy.deepcopy(params)
    ppm_params[:, 2:4] /= 10.0
    assert np.allclose(mpm.get_result(freq_unit="ppm"), ppm_params, rtol=0, atol=1e-8)

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
    mpm.result, mpm.M = mpm._remove_negative_damping(neg_damping)
    assert np.allclose(mpm.get_result(), params, rtol=0, atol=1e-8)
